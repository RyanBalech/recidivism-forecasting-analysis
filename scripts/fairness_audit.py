"""Fairness audit for all three models, by race and gender.

1. Gaps at two operating points: threshold 0.5 and the deployed top-20% capacity rule.
2. Bootstrap 95% confidence intervals on every gap (inference test: is the gap real?).
3. Base rates and within-group calibration (impossibility result, per attribute).
4. Mitigation frontier: race-blind capacity sweep vs group-specific thresholds (analytic device only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, RANDOM_SEED
from recidivism.metrics import expected_calibration_error

MODELS = ["logistic", "xgboost", "tabicl"]
DISPLAY = {"logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500", "tabicl": "#7B2CBF"}
ATTRIBUTES = {"Race": ("BLACK", "WHITE"), "Gender": ("M", "F")}
CAPACITY = 0.20
BOOT = 500


def rates(y: np.ndarray, sel: np.ndarray) -> dict[str, float]:
    pos, neg = y == 1, y == 0
    return {
        "selection_rate": sel.mean(),
        "tpr": sel[pos].mean() if pos.any() else np.nan,
        "fpr": sel[neg].mean() if neg.any() else np.nan,
        "fnr": 1 - sel[pos].mean() if pos.any() else np.nan,
        "precision": y[sel].mean() if sel.any() else np.nan,
    }


def gaps(y, p, group, a, b, rule):
    thr = 0.5 if rule == "threshold_0.5" else np.quantile(p, 1 - CAPACITY)
    sel = p >= thr
    ra, rb = rates(y[group == a], sel[group == a]), rates(y[group == b], sel[group == b])
    return {k: ra[k] - rb[k] for k in ra}


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
                point = gaps(y, p, g, a, b, rule)
                draws = pd.DataFrame([gaps(y[i], p[i], g[i], a, b, rule) for i in boots])
                for metric, value in point.items():
                    lo, hi = draws[metric].quantile([0.025, 0.975])
                    rows.append({"model": model, "attribute": attr, "comparison": f"{a} minus {b}",
                                 "rule": rule, "metric": metric, "gap": value, "ci_low": lo, "ci_high": hi,
                                 "significant": not (lo <= 0 <= hi)})
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


def frontier(pred: pd.DataFrame) -> pd.DataFrame:
    """For race AND gender: group-blind capacity sweep, plus group thresholds equalizing FPR.

    Group thresholds are an analytic device to trace the fairness/utility frontier, not a
    shipping recommendation (applying a different decision threshold by race or gender is
    disparate treatment). We also record the calibration cost of equalizing, since forcing
    equal error rates when base rates differ decalibrates the minority group.
    """
    y = pred["actual"].to_numpy()
    rows = []
    for attr, (a, b) in ATTRIBUTES.items():
        g = pred[attr].to_numpy()
        base_gap = y[g == a].mean() - y[g == b].mean()
        for model in MODELS:
            p = pred[f"p_{model}"].to_numpy()
            for cap in np.round(np.arange(0.05, 0.51, 0.05), 2):
                sel = p >= np.quantile(p, 1 - cap)
                fa, fb = rates(y[g == a], sel[g == a]), rates(y[g == b], sel[g == b])
                rows.append({"model": model, "attribute": attr, "method": "group_blind_single_threshold",
                             "capacity": cap, "fpr_gap": fa["fpr"] - fb["fpr"], "captured_events": int(y[sel].sum())})
            # Per-group thresholds, same total capacity, FPR gap minimized by search.
            k = int(round(len(p) * CAPACITY))
            ma, mb = g == a, g == b
            best = None
            for share_a in np.linspace(0.2, 0.95, 151):
                ka = min(int(round(k * share_a)), int(ma.sum()))
                kb = min(k - ka, int(mb.sum()))
                sel = np.zeros(len(p), bool)
                sel[np.where(ma)[0][np.argsort(-p[ma])[:ka]]] = True
                sel[np.where(mb)[0][np.argsort(-p[mb])[:kb]]] = True
                gap = rates(y[ma], sel[ma])["fpr"] - rates(y[mb], sel[mb])["fpr"]
                if best is None or abs(gap) < abs(best["fpr_gap"]):
                    best = {"model": model, "attribute": attr, "method": "group_thresholds_equal_fpr",
                            "capacity": CAPACITY, "fpr_gap": gap, "captured_events": int(y[sel].sum()),
                            "base_rate_gap": base_gap}
            rows.append(best)
    return pd.DataFrame(rows)


def figures(result: pd.DataFrame, front: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, attr in zip(axes, ATTRIBUTES):
        part = result[(result.attribute == attr) & (result.metric == "fpr")]
        x = np.arange(len(MODELS))
        for j, rule in enumerate(["threshold_0.5", "top_20pct"]):
            r = part[part.rule == rule].set_index("model").loc[MODELS]
            ax.errorbar(x + (j - 0.5) * 0.3, r.gap, yerr=[r.gap - r.ci_low, r.ci_high - r.gap],
                        fmt="o", capsize=6, ms=9, label="Threshold 0.5" if j == 0 else "Deployed top 20%")
        ax.axhline(0, color="grey", lw=1)
        ax.set_xticks(x, [DISPLAY[m] for m in MODELS], fontsize=11)
        a, b = ATTRIBUTES[attr]
        ax.set_title(f"FPR gap by {attr.lower()} ({a} minus {b}), 95% CI")
    axes[0].set_ylabel("False-positive-rate gap")
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fairness_operating_point.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    for ax, attr in zip(axes, ATTRIBUTES):
        a, b = ATTRIBUTES[attr]
        for model in MODELS:
            f = front[(front.model == model) & (front.attribute == attr) & (front.method == "group_blind_single_threshold")]
            ax.plot(f.captured_events, f.fpr_gap, marker="o", color=PALETTE[model], label=f"{DISPLAY[model]} (group-blind)")
            s = front[(front.model == model) & (front.attribute == attr) & (front.method == "group_thresholds_equal_fpr")]
            ax.scatter(s.captured_events, s.fpr_gap, marker="*", s=350, color=PALETTE[model], edgecolor="black")
        ax.axhline(0, color="grey", lw=1)
        ax.set(xlabel="Recidivism events captured", ylabel=f"FPR gap ({a} minus {b})",
               title=f"{attr}: fairness/utility frontier")
    axes[0].legend(fontsize=9)
    fig.suptitle("Stars = per-group thresholds equalizing FPR (analytic device, not a shipping option)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fairness_frontier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pred = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    result, imp, front = audit(pred), impossibility(pred), frontier(pred)
    result.to_csv(ARTIFACT_DIR / "fairness_inference.csv", index=False)
    imp.to_csv(ARTIFACT_DIR / "fairness_impossibility.csv", index=False)
    front.to_csv(ARTIFACT_DIR / "fairness_frontier.csv", index=False)
    figures(result, front)
    show = result[result.metric.isin(["fpr", "tpr", "selection_rate"])]
    print(show.round(3).to_string(index=False))
    print(imp.round(3).to_string(index=False))
    print(front[front.method == "group_thresholds_equal_fpr"].round(3).to_string(index=False))
    print(front[(front.method != "group_thresholds_equal_fpr") & (front.capacity == CAPACITY)].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
