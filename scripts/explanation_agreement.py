"""Do SHAP, permutation importance and XPER rank features the same way?

They are not interchangeable, and the course treats the disagreement itself as
the finding (printed slides 175-181):

- **SHAP** attributes the *prediction*. It answers "what moved this score away
  from the average score?" and is computed without any outcome label.
- **Permutation importance** attributes *loss*. It answers "how much worse does
  the model score when this feature is made uninformative?" - it needs labels and
  it charges a feature for being predictive, not for being used.
- **XPER** attributes *AUC*, decomposing a performance metric into per-feature
  contributions. Its reconstruction residual is reported here: the gap between
  the summed contributions and the achieved AUC bounds how literally the
  decomposition can be read.

So a feature can be first for SHAP and mid-table for permutation importance
without either being wrong: one is asking about scores, the other about errors.
This script measures the disagreement instead of asserting it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR

MODELS = ["logistic", "xgboost"]
METHODS = ["shap", "permutation_importance", "xper"]
TOP_K = 10


def load_rankings() -> pd.DataFrame:
    shap = pd.read_csv(ARTIFACT_DIR / "shap_importance.csv").rename(
        columns={"mean_abs_shap": "value"})[["model", "feature", "value"]]
    shap["method"] = "shap"

    perm = pd.read_csv(ARTIFACT_DIR / "permutation_importance.csv").rename(
        columns={"importance": "value"})[["model", "feature", "value"]]
    perm["method"] = "permutation_importance"

    xper = pd.read_csv(ARTIFACT_DIR / "xper_values.csv").rename(
        columns={"xper": "value"})[["model", "feature", "value"]]
    # The benchmark row is the uninformative-model baseline, not a feature.
    xper = xper[~xper.feature.str.contains("benchmark", case=False, na=False)]
    xper["method"] = "xper"

    return pd.concat([shap, perm, xper], ignore_index=True)


def main() -> None:
    long = load_rankings()
    long = long[long.model.isin(MODELS)].copy()
    # Rank 1 = most important, within each (model, method).
    long["rank"] = long.groupby(["model", "method"]).value.rank(ascending=False, method="min")
    long.to_csv(ARTIFACT_DIR / "explanation_rankings.csv", index=False)

    rows = []
    for model in MODELS:
        part = long[long.model == model]
        wide = part.pivot_table(index="feature", columns="method", values="rank")
        for i, a in enumerate(METHODS):
            for b in METHODS[i + 1:]:
                if a not in wide.columns or b not in wide.columns:
                    continue
                both = wide[[a, b]].dropna()
                if len(both) < 3:
                    continue
                rho, p = spearmanr(both[a], both[b])
                top_a = set(part[(part.method == a)].nsmallest(TOP_K, "rank").feature)
                top_b = set(part[(part.method == b)].nsmallest(TOP_K, "rank").feature)
                rows.append({
                    "model": model, "method_a": a, "method_b": b,
                    "features_compared": len(both),
                    "spearman_rank_correlation": float(rho), "p_value": float(p),
                    "top10_overlap": len(top_a & top_b),
                    "top10_only_a": ", ".join(sorted(top_a - top_b)),
                    "top10_only_b": ", ".join(sorted(top_b - top_a)),
                })
    agreement = pd.DataFrame(rows)
    agreement.to_csv(ARTIFACT_DIR / "explanation_agreement.csv", index=False)

    diag = pd.read_csv(ARTIFACT_DIR / "xper_diagnostics.csv")
    _figure(long, agreement, diag)

    print("Rank agreement between explanation methods:")
    print(agreement[["model", "method_a", "method_b", "features_compared",
                     "spearman_rank_correlation", "top10_overlap"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nXPER reconstruction residual (summed contributions vs achieved AUC):")
    print(diag.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _figure(long: pd.DataFrame, agreement: pd.DataFrame, diag: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9 * len(MODELS), 9))
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        part = long[long.model == model]
        wide = part.pivot_table(index="feature", columns="method", values="rank")
        methods = [m for m in METHODS if m in wide.columns]
        wide = wide.dropna(subset=methods)
        order = wide[methods[0]].sort_values().index[:TOP_K]
        wide = wide.loc[order]

        xs = np.arange(len(methods))
        for feature in wide.index:
            ys = [wide.loc[feature, m] for m in methods]
            ax.plot(xs, ys, marker="o", lw=1.8, ms=7)
            ax.annotate(feature.replace("_", " "), (xs[0], ys[0]), textcoords="offset points",
                        xytext=(-8, 0), ha="right", fontsize=8, va="center")
        ax.invert_yaxis()
        ax.set_xticks(xs, [m.replace("_", "\n") for m in methods], fontsize=10)
        ax.set_ylabel("Rank (1 = most important)")
        pair = agreement[(agreement.model == model)]
        caption = " · ".join(
            f"{r.method_a[:4]}/{r.method_b[:4]} rho={r.spearman_rank_correlation:.2f}"
            for r in pair.itertuples())
        resid = diag[diag.model == model].reconstruction_residual
        extra = f"\nXPER residual = {resid.iloc[0]:.3f}" if len(resid) else ""
        ax.set_title(f"{model}: where the methods disagree\n{caption}{extra}", fontsize=11)
        ax.margins(x=0.28)
    fig.suptitle("Explanation methods answer different questions, so their rankings differ",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "explanation_agreement.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
