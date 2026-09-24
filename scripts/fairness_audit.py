"""Fairness audit for all three models, by race, gender, and age.

Framing. Being selected (top 20% of risk) means being *offered support*. In the course's notation
Y=1 is the "good type" who deserves the favorable output and Yhat=1 is the favorable output; here
Y=1 is a person who would be re-arrested (needs support) and Yhat=1 is selection. A false positive
is therefore not the harm; the harm is a **false negative** (needed support, not selected). The
primary metrics are the FNR gap (equal opportunity), the selection-rate gap (statistical parity) and
within-group calibration (sufficiency). FPR (predictive equality) is reported as secondary.

1. Gaps at two operating points: threshold 0.5 and the deployed top-20% capacity rule, with
   bootstrap 95% CIs and a TOST equivalence test (course §8.3: "not significant" is not "fair").
2. The course's fairness test statistics (p246 layout): chi-squared / CMH p-values for statistical
   parity, conditional statistical parity, equal opportunity, predictive equality, equalized odds,
   and sufficiency.
3. Base rates and within-group calibration (impossibility result, per attribute).
4. Mitigation frontier: group-blind capacity sweep vs group-specific thresholds equalizing FNR
   (analytic device only: different thresholds by group are disparate treatment).
5. Age audit: age is a protected attribute in the course and our strongest predictor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.metrics import expected_calibration_error, capacity_selection

MODELS = ["logistic", "xgboost", "tabicl"]
DISPLAY = {"logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500", "tabicl": "#7B2CBF"}
# Gaps are reported as "first minus second".
ATTRIBUTES = {"Race": ("BLACK", "WHITE"), "Gender": ("M", "F"), "Age": ("Under 33", "33 or older")}
CAPACITY = 0.20
BOOT = 500
# TOST tolerance: a gap within +/- 5 percentage points is treated as practically equivalent.
# It is absolute: on a ~20% selection rate it still admits a group ratio near 0.78 (below the
# four-fifths rule), so selection-rate conclusions must also be read against rate_a / rate_b.
DELTA = 0.05
ALPHA = 0.05
AGE_ORDER = ["18-22", "23-27", "28-32", "33-37", "38-42", "43-47", "48 or older"]


def rates(y: np.ndarray, sel: np.ndarray) -> dict[str, float]:
    pos, neg = y == 1, y == 0
    return {
        "selection_rate": sel.mean(),
        "tpr": sel[pos].mean() if pos.any() else np.nan,
        "fpr": sel[neg].mean() if neg.any() else np.nan,
        "fnr": 1 - sel[pos].mean() if pos.any() else np.nan,
        "precision": y[sel].mean() if sel.any() else np.nan,
    }


def denominators(y: np.ndarray, sel: np.ndarray) -> dict[str, int]:
    """Sample size behind each rate, needed for the TOST variance."""
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return {"selection_rate": len(y), "tpr": n_pos, "fpr": n_neg, "fnr": n_pos, "precision": int(sel.sum())}


def selection(p: np.ndarray, rule: str) -> np.ndarray:
    return p >= 0.5 if rule == "threshold_0.5" else capacity_selection(p, CAPACITY)


def gaps(y, p, group, a, b, rule):
    sel = selection(p, rule)
    ra, rb = rates(y[group == a], sel[group == a]), rates(y[group == b], sel[group == b])
    return {k: ra[k] - rb[k] for k in ra}


def tost(pa: float, na: int, pb: float, nb: int) -> float:
    """Two one-sided tests (Schuirmann 1987) as in the course, p276-277.

    H0: |pa - pb| >= DELTA (unfair) vs H1: |pa - pb| < DELTA (fair within tolerance).
    Returns the TOST p-value, max of the two one-sided p-values.
    """
    sigma = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    if sigma == 0:
        return float(abs(pa - pb) >= DELTA)
    theta = pa - pb
    df = na + nb - 2
    p_lower = stats.t.sf((theta + DELTA) / sigma, df)
    p_upper = stats.t.sf((DELTA - theta) / sigma, df)
    return float(max(p_lower, p_upper))


def audit(pred: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    y = pred["actual"].to_numpy()
    n = len(y)
    boots = [rng.integers(0, n, n) for _ in range(BOOT)]
    rows = []
    for model in MODELS:
        p = pred[f"p_{model}"].to_numpy()
        for attr, (a, b) in ATTRIBUTES.items():
            g = pred[attr].to_numpy()
            for rule in ["threshold_0.5", "top_20pct"]:
                sel = selection(p, rule)
                ra, rb = rates(y[g == a], sel[g == a]), rates(y[g == b], sel[g == b])
                na, nb = denominators(y[g == a], sel[g == a]), denominators(y[g == b], sel[g == b])
                draws = pd.DataFrame([gaps(y[i], p[i], g[i], a, b, rule) for i in boots])
                for metric in ra:
                    lo, hi = draws[metric].quantile([0.025, 0.975])
                    p_tost = tost(ra[metric], na[metric], rb[metric], nb[metric])
                    # Group rates are kept so relative scale (e.g. selection-rate ratio) is traceable:
                    # the absolute ±DELTA tolerance means very different things at 0.2 and at 0.7.
                    rows.append({"model": model, "attribute": attr, "comparison": f"{a} minus {b}",
                                 "rule": rule, "metric": metric, "rate_a": ra[metric], "rate_b": rb[metric],
                                 "gap": ra[metric] - rb[metric],
                                 "ci_low": lo, "ci_high": hi, "significant": not (lo <= 0 <= hi),
                                 "n_a": na[metric], "n_b": nb[metric],
                                 "tost_p": p_tost, "equivalent_within_delta": p_tost < ALPHA})
    return pd.DataFrame(rows)


def _chi2_p(table: np.ndarray) -> tuple[float, float]:
    """Pearson chi-squared independence test on a contingency table (no continuity correction)."""
    table = table[:, table.sum(axis=0) > 0]
    table = table[table.sum(axis=1) > 0]
    if min(table.shape) < 2:
        return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p)


def _cmh_p(outcome: np.ndarray, d: np.ndarray, strata: np.ndarray) -> float:
    """Cochran-Mantel-Haenszel test of outcome independent of D given strata (course p244)."""
    num, var = 0.0, 0.0
    for s in np.unique(strata):
        m = strata == s
        o, g = outcome[m], d[m]
        n = len(o)
        if n < 2 or len(np.unique(o)) < 2 or len(np.unique(g)) < 2:
            continue
        a = np.sum(o & g)
        row1, col1 = o.sum(), g.sum()
        num += a - row1 * col1 / n
        var += row1 * (n - row1) * col1 * (n - col1) / (n ** 2 * (n - 1))
    if var == 0:
        return 1.0
    return float(stats.chi2.sf(num ** 2 / var, 1))


def course_tests(pred: pd.DataFrame) -> pd.DataFrame:
    """The course's fairness test table (p246), at the deployed top-20% operating point.

    Conditional statistical parity conditions on Georgia's own risk score (a legitimate factor);
    sufficiency conditions on deciles of the model score (Y independent of D given p(X)).
    """
    y = pred["actual"].to_numpy().astype(bool)
    risk = pred["Supervision_Risk_Score_First"].fillna(-1).to_numpy()
    rows = []
    for model in MODELS:
        p = pred[f"p_{model}"].to_numpy()
        sel = selection(p, "top_20pct")
        deciles = pd.qcut(p, 10, labels=False, duplicates="drop")
        for attr, (a, b) in ATTRIBUTES.items():
            g = pred[attr].to_numpy()
            keep = (g == a) | (g == b)
            d, s, yy = (g[keep] == b), sel[keep], y[keep]  # D=1 is the second group

            def tab(mask):
                return pd.crosstab(d[mask], s[mask]).to_numpy()

            chi_pos, p_eopp = _chi2_p(tab(yy))
            chi_neg, p_peq = _chi2_p(tab(~yy))
            rows.append({
                "model": model, "attribute": attr, "protected_group": b,
                "statistical_parity": _chi2_p(pd.crosstab(d, s).to_numpy())[1],
                "conditional_statistical_parity": _cmh_p(s, d, risk[keep]),
                "equal_opportunity": p_eopp,
                "predictive_equality": p_peq,
                "equalized_odds": float(stats.chi2.sf(chi_pos + chi_neg, 2)),
                "sufficiency": _cmh_p(yy, d, deciles[keep]),
            })
    return pd.DataFrame(rows)


def impossibility(pred: pd.DataFrame) -> pd.DataFrame:
    y = pred["actual"].to_numpy()
    rows = []
    for attr, groups in ATTRIBUTES.items():
        for grp in groups:
            m = pred[attr].eq(grp).to_numpy()
            row = {"attribute": attr, "group": grp, "n": int(m.sum()), "base_rate": y[m].mean()}
            for model in MODELS:
                p = pred[f"p_{model}"].to_numpy()
                row[f"mean_score_{model}"] = p[m].mean()
                row[f"ece_{model}"] = expected_calibration_error(y[m], p[m])
            rows.append(row)
    return pd.DataFrame(rows)


def age_bands(pred: pd.DataFrame) -> pd.DataFrame:
    """Selection rate and FNR per age band at the top-20% rule, next to the observed base rate."""
    y = pred["actual"].to_numpy()
    rows = []
    for model in MODELS:
        sel = selection(pred[f"p_{model}"].to_numpy(), "top_20pct")
        for band in AGE_ORDER:
            m = pred["Age_at_Release"].eq(band).to_numpy()
            r = rates(y[m], sel[m])
            rows.append({"model": model, "age_band": band, "n": int(m.sum()), "base_rate": y[m].mean(),
                         "selection_rate": r["selection_rate"], "fnr": r["fnr"]})
    return pd.DataFrame(rows)


def frontier(pred: pd.DataFrame) -> pd.DataFrame:
    """Per attribute: group-blind capacity sweep, plus group thresholds equalizing FNR at 20%.

    Group thresholds are an analytic device to trace the fairness/utility frontier, not a
    shipping recommendation (a different decision threshold by race or gender is disparate
    treatment). This search uses evaluation labels, so its result is an optimistic, in-sample
    illustration, not validated mitigation. Threshold changes leave probabilities (and their
    calibration) unchanged.
    """
    y = pred["actual"].to_numpy()
    rows = []
    for attr, (a, b) in ATTRIBUTES.items():
        g = pred[attr].to_numpy()
        base_gap = y[g == a].mean() - y[g == b].mean()
        for model in MODELS:
            p = pred[f"p_{model}"].to_numpy()
            for cap in np.round(np.arange(0.05, 0.51, 0.05), 2):
                sel = capacity_selection(p, cap)
                fa, fb = rates(y[g == a], sel[g == a]), rates(y[g == b], sel[g == b])
                rows.append({"model": model, "attribute": attr, "method": "group_blind_single_threshold",
                             "capacity": cap, "fnr_gap": fa["fnr"] - fb["fnr"], "fpr_gap": fa["fpr"] - fb["fpr"],
                             "captured_events": int(y[sel].sum()), "base_rate_gap": base_gap})
            # Per-group thresholds, same total capacity, FNR gap minimized by search.
            k = int(round(len(p) * CAPACITY))
            ma, mb = g == a, g == b
            best = None
            for share_a in np.linspace(0.05, 0.95, 181):
                ka = min(int(round(k * share_a)), int(ma.sum()))
                kb = min(k - ka, int(mb.sum()))
                sel = np.zeros(len(p), bool)
                sel[np.where(ma)[0][np.argsort(-p[ma])[:ka]]] = True
                sel[np.where(mb)[0][np.argsort(-p[mb])[:kb]]] = True
                fa, fb = rates(y[ma], sel[ma]), rates(y[mb], sel[mb])
                gap = fa["fnr"] - fb["fnr"]
                if best is None or abs(gap) < abs(best["fnr_gap"]):
                    best = {"model": model, "attribute": attr, "method": "group_thresholds_equal_fnr",
                            "capacity": CAPACITY, "fnr_gap": gap, "fpr_gap": fa["fpr"] - fb["fpr"],
                            "captured_events": int(y[sel].sum()), "base_rate_gap": base_gap}
            rows.append(best)
    return pd.DataFrame(rows)


def operating_figure(result: pd.DataFrame, panels: list[tuple[str, str]], filename: str) -> None:
    """One row per (metric, label) panel, one column per attribute, with the TOST tolerance band."""
    sns.set_theme(style="whitegrid", context="talk")
    attrs = list(ATTRIBUTES)
    fig, axes = plt.subplots(len(panels), len(attrs), figsize=(7 * len(attrs), 5.5 * len(panels) + 0.5),
                             sharey="row", squeeze=False)
    for r, (metric, label) in enumerate(panels):
        for ax, attr in zip(axes[r], attrs):
            part = result[(result.attribute == attr) & (result.metric == metric)]
            x = np.arange(len(MODELS))
            for j, rule in enumerate(["threshold_0.5", "top_20pct"]):
                rr = part[part.rule == rule].set_index("model").loc[MODELS]
                ax.errorbar(x + (j - 0.5) * 0.3, rr.gap, yerr=[rr.gap - rr.ci_low, rr.ci_high - rr.gap],
                            fmt="o", capsize=6, ms=9, label="Threshold 0.5" if j == 0 else "Deployed top 20%")
            ax.axhline(0, color="grey", lw=1)
            for s in (-DELTA, DELTA):
                ax.axhline(s, color="grey", lw=1, ls=":")
            ax.set_xticks(x, [DISPLAY[m] for m in MODELS], fontsize=11)
            a, b = ATTRIBUTES[attr]
            ax.set_title(f"{attr}: {a} minus {b}", fontsize=13)
        axes[r][0].set_ylabel(label, fontsize=12)
    axes[0][0].legend(fontsize=10)
    fig.suptitle(f"Fairness gaps with 95% bootstrap CIs; dotted lines = ±{DELTA:.0%} TOST tolerance", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def figures(result: pd.DataFrame, front: pd.DataFrame) -> None:
    # Primary for a support programme: missed support (FNR) and statistical parity.
    operating_figure(result, [("fnr", "FNR gap (missed support), primary"),
                              ("selection_rate", "Selection-rate gap (statistical parity)")],
                     "fairness_support_access.png")
    operating_figure(result, [("fpr", "FPR gap (predictive equality), secondary")],
                     "fairness_operating_point.png")
    attrs = list(ATTRIBUTES)
    fig, axes = plt.subplots(1, len(attrs), figsize=(7.5 * len(attrs), 6))
    for ax, attr in zip(axes, attrs):
        a, b = ATTRIBUTES[attr]
        for model in MODELS:
            f = front[(front.model == model) & (front.attribute == attr) & (front.method == "group_blind_single_threshold")]
            ax.plot(f.captured_events, f.fnr_gap, marker="o", color=PALETTE[model], label=f"{DISPLAY[model]} (group-blind)")
            s = front[(front.model == model) & (front.attribute == attr) & (front.method == "group_thresholds_equal_fnr")]
            ax.scatter(s.captured_events, s.fnr_gap, marker="*", s=350, color=PALETTE[model], edgecolor="black")
        ax.axhline(0, color="grey", lw=1)
        ax.set(xlabel="Recidivism events captured (more capacity →)", ylabel=f"FNR gap ({a} minus {b})",
               title=f"{attr}: fairness/utility frontier")
    axes[0].legend(fontsize=9)
    fig.suptitle("Stars = per-group thresholds equalizing FNR at 20% capacity (analytic device, not a shipping option)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fairness_frontier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def with_audit_columns(pred: pd.DataFrame) -> pd.DataFrame:
    """Add age band and Georgia's risk score (by ID) for the age audit and conditional tests."""
    split = load_official_split()
    extra = split.audit_test[["ID"]].assign(
        Age_at_Release=split.X_test["Age_at_Release"].to_numpy(),
        Supervision_Risk_Score_First=split.X_test["Supervision_Risk_Score_First"].to_numpy(),
    )
    out = pred.merge(extra, on="ID", how="left", validate="one_to_one")
    young = out["Age_at_Release"].isin(AGE_ORDER[:3])
    out["Age"] = np.where(young, "Under 33", "33 or older")
    return out


def main() -> None:
    pred = with_audit_columns(pd.read_csv(ARTIFACT_DIR / "test_predictions.csv"))
    result, tests, imp = audit(pred), course_tests(pred), impossibility(pred)
    ages, front = age_bands(pred), frontier(pred)
    result.to_csv(ARTIFACT_DIR / "fairness_inference.csv", index=False)
    tests.to_csv(ARTIFACT_DIR / "fairness_tests.csv", index=False)
    imp.to_csv(ARTIFACT_DIR / "fairness_impossibility.csv", index=False)
    ages.to_csv(ARTIFACT_DIR / "fairness_age_bands.csv", index=False)
    front.to_csv(ARTIFACT_DIR / "fairness_frontier.csv", index=False)
    figures(result, front)
    show = result[(result.rule == "top_20pct") & result.metric.isin(["fnr", "selection_rate", "fpr"])]
    print(show.drop(columns=["comparison", "rule"]).round(3).to_string(index=False))
    print("\nCourse fairness tests (p-values, top-20%):")
    print(tests.round(4).to_string(index=False))
    print(imp.round(3).to_string(index=False))
    print(ages.round(3).to_string(index=False))
    print(front[front.method == "group_thresholds_equal_fnr"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
