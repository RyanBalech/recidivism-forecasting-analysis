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
    """Race-blind capacity sweep, plus group thresholds that equalize FPR at 20% capacity."""
    y = pred["actual"].to_numpy()
    race = pred["Race"].to_numpy()
    rows = []
    for model in MODELS:
        p = pred[f"p_{model}"].to_numpy()
        for cap in np.round(np.arange(0.05, 0.51, 0.05), 2):
            sel = p >= np.quantile(p, 1 - cap)
            fb, fw = rates(y[race == "BLACK"], sel[race == "BLACK"]), rates(y[race == "WHITE"], sel[race == "WHITE"])
            rows.append({"model": model, "method": "race_blind_single_threshold", "capacity": cap,
                         "fpr_gap": fb["fpr"] - fw["fpr"], "captured_events": int(y[sel].sum())})
        # Analytic device: per-group thresholds, same total capacity, FPR equalized by search.
        k = int(round(len(p) * CAPACITY))
        best = None
        for share_b in np.linspace(0.3, 0.8, 101):
            kb = int(round(k * share_b))
            kw = k - kb
            sb, sw = race == "BLACK", race == "WHITE"
            sel = np.zeros(len(p), bool)
            sel[np.where(sb)[0][np.argsort(-p[sb])[:kb]]] = True
            sel[np.where(sw)[0][np.argsort(-p[sw])[:kw]]] = True
            gap = rates(y[sb], sel[sb])["fpr"] - rates(y[sw], sel[sw])["fpr"]
            if best is None or abs(gap) < abs(best[0]):
                best = (gap, int(y[sel].sum()))
        rows.append({"model": model, "method": "group_thresholds_equal_fpr", "capacity": CAPACITY,
                     "fpr_gap": best[0], "captured_events": best[1]})
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

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODELS:
        f = front[(front.model == model) & (front.method == "race_blind_single_threshold")]
        ax.plot(f.captured_events, f.fpr_gap, marker="o", color=PALETTE[model], label=f"{DISPLAY[model]} (race-blind)")
        g = front[(front.model == model) & (front.method == "group_thresholds_equal_fpr")]
        ax.scatter(g.captured_events, g.fpr_gap, marker="*", s=350, color=PALETTE[model], edgecolor="black")
    ax.axhline(0, color="grey", lw=1)
    ax.set(xlabel="Recidivism events captured by the service allocation", ylabel="Race FPR gap (Black minus White)",
           title="Fairness/utility frontier (stars = group thresholds, analytic only)")
    ax.legend(fontsize=10)
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
