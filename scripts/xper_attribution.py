"""XPER (Hué, Hurlin, Pérignon, Saurin): which features drive the model's AUC?

Unlike SHAP, which splits a *prediction*, XPER splits a *performance metric* into feature
contributions plus a benchmark (the AUC of an uninformative model). Kernel approximation on a
test subsample keeps CPU runtime to ~10 minutes per model. TabICL is excluded: each coalition
needs a fresh in-context prediction pass, which is impractical on CPU (reported as a cost).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, MODEL_DIR, RANDOM_SEED, pretty
from recidivism.data import load_official_split

SAMPLE_SIZE = 150
COALITIONS = 60
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500"}
DISPLAY = {"logistic": "Logistic regression", "xgboost": "XGBoost"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-only", action="store_true", help="Rebuild labels and diagnostics from saved values")
    args = parser.parse_args()
    if args.refresh_only:
        write_outputs(pd.read_csv(ARTIFACT_DIR / "xper_values.csv"))
        return
    warnings.filterwarnings("ignore")
    from XPER.compute.Performance import ModelPerformance

    split = load_official_split()
    rows = []
    for name in ["logistic", "xgboost"]:
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        xp = ModelPerformance(split.X_train, split.y_train, split.X_test, split.y_test, model,
                              sample_size=SAMPLE_SIZE, seed=RANDOM_SEED)
        auc = float(xp.evaluate(["AUC"]))
        start = time.perf_counter()
        phi, _ = xp.calculate_XPER_values(["AUC"], N_coalition_sampled=COALITIONS, kernel=True, seed=RANDOM_SEED)
        print(f"{name}: sample AUC {auc:.4f}, XPER sum {phi.sum():.4f}, {time.perf_counter() - start:.0f}s", flush=True)
        rows.append({"model": name, "feature": "benchmark (uninformative model)", "xper": float(phi[0]), "sample_auc": auc})
        for feat, value in zip(split.X_train.columns, phi[1:]):
            rows.append({"model": name, "feature": feat, "xper": float(value), "sample_auc": auc})

    result = pd.DataFrame(rows)
    result.to_csv(ARTIFACT_DIR / "xper_values.csv", index=False)
    write_outputs(result)


def write_outputs(result):
    diagnostics = result.groupby("model").agg(contribution_sum=("xper", "sum"), sample_auc=("sample_auc", "first"))
    diagnostics["reconstruction_residual"] = diagnostics.contribution_sum - diagnostics.sample_auc
    diagnostics.to_csv(ARTIFACT_DIR / "xper_diagnostics.csv")
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, name in zip(axes, ["logistic", "xgboost"]):
        part = result[(result.model == name) & ~result.feature.str.startswith("benchmark")]
        top = part.reindex(part.xper.abs().sort_values(ascending=False).index).head(10)[::-1]
        ax.barh(top.feature.map(pretty), top.xper, color=PALETTE[name])
        bench = result[(result.model == name) & result.feature.str.startswith("benchmark")].xper.iloc[0]
        residual = diagnostics.loc[name, "reconstruction_residual"]
        ax.set(title=f"{DISPLAY[name]}: approximate XPER\n(benchmark {bench:.3f}; AUC reconstruction residual {residual:+.3f})",
               xlabel="Contribution to AUC")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "xper.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(result[~result.feature.str.startswith("benchmark")].sort_values("xper", ascending=False)
          .groupby("model").head(6).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
