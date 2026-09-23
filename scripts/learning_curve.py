"""Learning curve: how each model's held-out performance grows with training size.

Trains logistic, XGBoost, and TabICLv2 on stratified subsamples of the official
training set and scores every run on the same untouched test set.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics
from recidivism.modeling import logistic_model, tabicl_frames, xgboost_model

SIZES = [1_500, 5_000, 10_000]
SEEDS = [0, 1, 2]
DISPLAY = {"logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500", "tabicl": "#7B2CBF"}


def subsample(X: pd.DataFrame, y: pd.Series, n: int, seed: int):
    if n >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X_sub, y_sub


def fit_predict(name: str, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
    if name == "tabicl":
        from tabicl import TabICLClassifier

        train, test = tabicl_frames(X, X_test)
        model = TabICLClassifier(n_estimators=2, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(train, y.to_numpy())
        return model.predict_proba(test)[:, 1]
    model = logistic_model(X) if name == "logistic" else xgboost_model(X)
    model.fit(X, y)
    return model.predict_proba(X_test)[:, 1]


def main() -> None:
    split = load_official_split()
    runs = [(n, s) for n in SIZES for s in SEEDS] + [(len(split.X_train), 0)]
    rows = []
    for n, seed in runs:
        X, y = subsample(split.X_train, split.y_train, n, seed)
        for name in DISPLAY:
            start = time.perf_counter()
            p = fit_predict(name, X, y, split.X_test)
            m = classification_metrics(split.y_test, p)
            rows.append({"model": name, "n_train": len(X), "seed": seed,
                         "roc_auc": m["roc_auc"], "brier": m["brier"],
                         "seconds": time.perf_counter() - start})
            print(f"n={len(X):>6} seed={seed} {name:9s} AUC={m['roc_auc']:.4f} "
                  f"Brier={m['brier']:.4f} ({rows[-1]['seconds']:.0f}s)", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(ARTIFACT_DIR / "learning_curve.csv", index=False)
    summary = results.groupby(["model", "n_train"])[["roc_auc", "brier"]].agg(["mean", "std"])
    print(summary.round(4).to_string())

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, label in [(axes[0], "roc_auc", "ROC AUC (higher is better)"),
                              (axes[1], "brier", "Brier score (lower is better)")]:
        for name in DISPLAY:
            g = results[results.model.eq(name)].groupby("n_train")[metric]
            mean, std = g.mean(), g.std().fillna(0)
            ax.plot(mean.index, mean, marker="o", lw=2.5, color=PALETTE[name], label=DISPLAY[name])
            ax.fill_between(mean.index, mean - std, mean + std, color=PALETTE[name], alpha=0.15)
        ax.set(xscale="log", xlabel="Training rows", ylabel=label)
    axes[0].set_title("Learning curve: discrimination")
    axes[1].set_title("Learning curve: calibration")
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "learning_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
