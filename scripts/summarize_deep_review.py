"""Summarize training-only experiments and report the fixed blend exploratorily.

The 50/50 blend was specified in accuracy_protocol.json before outer scores were
computed. It is a research challenger; production model artifacts stay separate.
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.metrics import classification_metrics, economic_value, fairness_table

OUT = ROOT / "artifacts/deep_review"


def main():
    folds = pd.concat([pd.read_csv(OUT / "accuracy_outer_folds.csv"),
                       pd.read_csv(OUT / "native_outer_folds.csv")], ignore_index=True)
    summary = folds.groupby("model")[["roc_auc", "brier", "log_loss", "accuracy"]].mean().sort_values("brier")
    summary.to_csv(OUT / "combined_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = summary.index.tolist()
    for ax, metric in zip(axes, ["roc_auc", "brier"]):
        for i, name in enumerate(names):
            values = folds.loc[folds.model.eq(name), metric]
            ax.scatter(values, np.full(len(values), i), alpha=.55, s=35)
            ax.scatter([values.mean()], [i], marker="D", color="black", s=35)
        ax.set(yticks=range(len(names)), yticklabels=names,
               xlabel="AUC (higher better)" if metric == "roc_auc" else "Brier loss (lower better)")
        ax.invert_yaxis(); ax.grid(alpha=.2)
    fig.suptitle("Training partition: three outer folds (dots); means (black diamonds)")
    fig.text(.5, .015, "Internal evidence: historical configurations saw this cohort; native-tree follow-up was exploratory.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .035, 1, .94])
    fig.savefig(OUT / "accuracy_comparison.png", dpi=170)
    plt.close(fig)

    # Report a previously specified candidate, without selecting a blend weight on evaluation labels.
    pred = pd.read_csv(ROOT / "artifacts/test_predictions.csv")
    p = (pred.p_xgboost.to_numpy() + pred.p_tabicl.to_numpy()) / 2
    y = pred.actual.to_numpy()
    baseline = pred.p_xgboost.to_numpy()
    rows = []
    for name, scores in [("xgboost", baseline), ("tabicl", pred.p_tabicl.to_numpy()), ("blend_equal", p)]:
        rows.append({"model": name, **classification_metrics(y, scores), **economic_value(y, scores)})
    pd.DataFrame(rows).to_csv(OUT / "blend_evaluation.csv", index=False)
    groups = [fairness_table(y, p, pred[attr], attr).assign(model="blend_equal") for attr in ["Race", "Gender"]]
    pd.concat(groups).to_csv(OUT / "blend_groups_threshold05.csv", index=False)
    rng = np.random.default_rng(20260923)
    draws = []
    for _ in range(1000):
        idx = rng.integers(len(y), size=len(y))
        draws.append([roc_auc_score(y[idx], p[idx])-roc_auc_score(y[idx], baseline[idx]),
                      np.mean((y[idx]-p[idx])**2-(y[idx]-baseline[idx])**2)])
    intervals = {name: np.quantile(np.asarray(draws)[:, i], [.025, .975]).tolist()
                 for i, name in enumerate(["auc_blend_minus_xgb", "brier_blend_minus_xgb"])}
    intervals["scope"] = "Fixed predictions, reused evaluation partition, unadjusted; excludes training/selection uncertainty"
    (OUT / "blend_paired_intervals.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    paths = [ROOT / "scripts" / f"{name}.py" for name in
             ["accuracy_review", "native_accuracy_review", "leakage_stress", "leakage_positive_control",
              "thread_sensitivity", "deep_leakage_audit", "summarize_deep_review"]]
    paths += [ROOT / "src/recidivism/modeling.py", ROOT / "nij-challenge2021_full_dataset.csv"]
    paths += sorted(OUT.glob("*.csv"))
    paths += [p for p in sorted(OUT.glob("*.json")) if p.name != "source_and_result_hashes.json"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (OUT / "source_and_result_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(summary.to_string())
    print(pd.DataFrame(rows)[["model", "roc_auc", "brier", "accuracy"]].to_string(index=False))
    print(intervals)


if __name__ == "__main__":
    main()
