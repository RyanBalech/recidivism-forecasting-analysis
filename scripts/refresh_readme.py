"""Refresh the marked README results block from the current validated run."""
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    metrics = pd.read_csv(ROOT / "artifacts/model_metrics.csv").set_index("model")
    paired = pd.read_csv(ROOT / "artifacts/paired_comparisons.csv")
    manifest = json.loads((ROOT / "artifacts/validation_manifest.json").read_text())
    baseline = pd.read_csv(ROOT / "artifacts/validation_baselines.csv").set_index("model")
    lines = ["<!-- RESULTS:START -->", "| Model | ROC AUC | Brier ↓ | ECE (10 bins) ↓ | Fit + prediction |",
             "|---|---:|---:|---:|---:|"]
    for key, label in [("logistic", "Logistic regression (tuned L1)"), ("xgboost", "XGBoost"), ("tabicl", "TabICLv2")]:
        row = metrics.loc[key]
        lines.append(f"| {label} | {row.roc_auc:.4f} | {row.brier:.4f} | {row.ece_10:.4f} | {row.fit_predict_seconds:.2f} s |")
    comparison = paired.query("model_a == 'xgboost' and model_b == 'tabicl' and metric == 'roc_auc'").iloc[0]
    skill = baseline.loc["xgboost", "brier_skill_vs_prevalence"]
    lines.extend(["", f"Current run: TabICLv2 on **{manifest['device']}**, conventional models on CPU. Timings are hardware-specific. "
                  f"XGBoost minus TabICLv2 AUC is {comparison.difference_a_minus_b:.5f}, with paired 95% interval "
                  f"[{comparison.ci_low:.5f}, {comparison.ci_high:.5f}]; this does not establish superiority or equivalence. "
                  f"XGBoost reduces Brier loss by {skill:.1%} relative to training-prevalence probabilities.",
                  "<!-- RESULTS:END -->"])
    path = ROOT / "README.md"
    updated, count = re.subn(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: "\n".join(lines),
                            path.read_text(encoding="utf-8"), flags=re.S)
    if count != 1:
        raise ValueError("README must have exactly one marked results block")
    path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
