"""Structural stability: does the model itself stay the same when the training data wiggles?

Each model is refitted on bootstrap resamples of the training set (different seed each time).
Between every pair of refits we measure, on the fixed test set:
- distance among models: mean |p_i - p_j| and Spearman rank correlation of scores;
- decision stability: overlap (Jaccard) of the top-20% selected individuals;
- slow variation in feature contributions (logistic, XGBoost): rank correlation of
  mean |SHAP| importance vectors across refits.
TabICL refits cost ~70 s each on CPU, so it gets fewer resamples (stated, not hidden).
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR
from recidivism.data import load_official_split
from recidivism.modeling import logistic_model, tabicl_frames, xgboost_model

REFITS = {"logistic": 8, "xgboost": 8, "tabicl": 4}
CAPACITY = 0.20
DISPLAY = {"logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500", "tabicl": "#7B2CBF"}


def refit(name: str, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, seed: int):
    if name == "tabicl":
        from tabicl import TabICLClassifier
        m = TabICLClassifier(n_estimators=2, random_state=seed, n_jobs=-1)
        train, test = tabicl_frames(X, X_test)
        m.fit(train, y.to_numpy())
        return m.predict_proba(test)[:, 1], None
    m = logistic_model(X) if name == "logistic" else xgboost_model(X, random_state=seed)
    m.fit(X, y)
    return m.predict_proba(X_test)[:, 1], m


def shap_importance(pipe, X_sample: pd.DataFrame, kind: str) -> pd.Series:
    sys.path.insert(0, str(ROOT / "scripts"))
    from interpretability import shap_for
    values, _ = shap_for(pipe, X_sample.iloc[:300], X_sample, kind)
    return values.abs().mean()


def main() -> None:
    split = load_official_split()
    rng = np.random.default_rng(7)
    X_sample = split.X_test.sample(600, random_state=0)
    k = int(round(len(split.X_test) * CAPACITY))
    pair_rows, contrib_rows = [], []
    for name, n_refits in REFITS.items():
        preds, importances = [], []
        for seed in range(n_refits):
            idx = rng.integers(0, len(split.X_train), len(split.X_train))
            X, y = split.X_train.iloc[idx].reset_index(drop=True), split.y_train.iloc[idx].reset_index(drop=True)
            start = time.perf_counter()
            p, model = refit(name, X, y, split.X_test, seed)
            preds.append(p)
            if model is not None:
                importances.append(shap_importance(model, X_sample, name))
            print(f"{name} refit {seed + 1}/{n_refits} ({time.perf_counter() - start:.0f}s)", flush=True)
        for i, j in itertools.combinations(range(n_refits), 2):
            top_i = set(np.argsort(-preds[i])[:k])
            top_j = set(np.argsort(-preds[j])[:k])
            pair_rows.append({
                "model": name,
                "mean_abs_prob_diff": float(np.mean(np.abs(preds[i] - preds[j]))),
                "p95_abs_prob_diff": float(np.quantile(np.abs(preds[i] - preds[j]), 0.95)),
                "rank_correlation": float(spearmanr(preds[i], preds[j]).statistic),
                "top20_jaccard": len(top_i & top_j) / len(top_i | top_j),
            })
            if importances:
                a, b = importances[i].align(importances[j], fill_value=0)
                contrib_rows.append({"model": name, "importance_rank_correlation": float(spearmanr(a, b).statistic)})
        if importances:
            frame = pd.DataFrame(importances)
            cv = (frame.std() / frame.mean()).rename("coef_of_variation")
            top = frame.mean().sort_values(ascending=False).head(8).index
            for feat in top:
                contrib_rows.append({"model": name, "feature": feat, "mean_abs_shap": frame[feat].mean(),
                                     "coef_of_variation": cv[feat]})

    pairs = pd.DataFrame(pair_rows)
    contrib = pd.DataFrame(contrib_rows)
    pairs.to_csv(ARTIFACT_DIR / "stability_pairs.csv", index=False)
    contrib.to_csv(ARTIFACT_DIR / "stability_contributions.csv", index=False)
    summary = pairs.groupby("model").mean().round(4)
    summary["refits"] = pd.Series(REFITS)
    summary.to_csv(ARTIFACT_DIR / "stability_summary.csv")
    print(summary.to_string())
    if "importance_rank_correlation" in contrib:
        print(contrib.groupby("model")["importance_rank_correlation"].mean().round(4).to_string())

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, col, title in [(axes[0], "mean_abs_prob_diff", "Distance among refits (mean |Δp|, lower = stabler)"),
                           (axes[1], "top20_jaccard", "Same people selected? (top-20% Jaccard)"),
                           (axes[2], "rank_correlation", "Score rank correlation across refits")]:
        sns.boxplot(data=pairs, x="model", y=col, hue="model", palette=PALETTE, ax=ax, legend=False)
        ax.set(title=title, xlabel="", ylabel="")
        ax.set_xticks(range(3), [DISPLAY[m] for m in pairs.model.unique()], fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "structural_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
