"""Training-only sensitivity check for the TabICLv2 ensemble-size choice.

The official evaluation labels are deliberately not touched. A fixed stratified
development split is carved from the training partition, and the final model is
later refit on all training rows. This is a sensitivity analysis, not an unbiased
estimate of deployment performance.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.modeling import TABICL_CHECKPOINT, TABICL_ESTIMATORS, tabicl_frames

GRID = [1, 2, 4, 8, 16, 32, 64]
CHOSEN = TABICL_ESTIMATORS


def main() -> None:
    from tabicl import TabICLClassifier
    split = load_official_split()
    X_fit, X_dev, y_fit, y_dev = train_test_split(
        split.X_train,
        split.y_train,
        test_size=0.20,
        stratify=split.y_train,
        random_state=RANDOM_SEED,
    )
    train, dev = tabicl_frames(X_fit.reset_index(drop=True), X_dev.reset_index(drop=True))
    y_fit, y_dev = y_fit.to_numpy(), y_dev.to_numpy()
    rows = []
    for n in GRID:
        start = time.perf_counter()
        m = TabICLClassifier(
            checkpoint_version=TABICL_CHECKPOINT,
            n_estimators=n,
            batch_size=1,
            kv_cache="repr",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        m.fit(train, y_fit)
        p = m.predict_proba(dev)[:, 1]
        rows.append({
            "n_estimators": n,
            "roc_auc": roc_auc_score(y_dev, p),
            "brier": brier_score_loss(y_dev, p),
            "seconds": time.perf_counter() - start,
            "fit_rows": len(train),
            "development_rows": len(dev),
        })
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACT_DIR / "estimator_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df.n_estimators, df.roc_auc, marker="o", lw=2.5, color="#7B2CBF")
    ax.axvline(CHOSEN, ls="--", color="#FB8500", lw=1.5)
    ax.text(CHOSEN * 1.05, df.roc_auc.min(), f"chosen = {CHOSEN}", color="#FB8500", fontsize=10)
    ax.set(xscale="log", xlabel="TabICLv2 ensemble members (log)", ylabel="Development ROC AUC",
           title="Training-only TabICLv2 ensemble sensitivity")
    for _, r in df.iterrows():
        ax.annotate(f"{r.seconds:.0f}s", (r.n_estimators, r.roc_auc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="grey")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "estimator_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved artifacts/figures/estimator_sweep.png")


if __name__ == "__main__":
    main()
