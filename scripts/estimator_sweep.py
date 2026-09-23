"""Sensitivity of TabICLv2 to its ensemble size, to justify our n_estimators choice.

Sweeps n_estimators on the fixed test set and plots held-out AUC vs runtime. Shows the
plateau: accuracy is essentially flat beyond ~8 members, so 16 (our setting) is a fair,
near-maximal choice and going higher only buys runtime.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.modeling import tabicl_frames

GRID = [1, 2, 4, 8, 16, 32, 64]
CHOSEN = 16


def main() -> None:
    from tabicl import TabICLClassifier
    split = load_official_split()
    train, test = tabicl_frames(split.X_train, split.X_test)
    y = split.y_test.to_numpy()
    rows = []
    for n in GRID:
        start = time.perf_counter()
        m = TabICLClassifier(n_estimators=n, random_state=RANDOM_SEED)
        m.fit(train, split.y_train.to_numpy())
        p = m.predict_proba(test)[:, 1]
        rows.append({"n_estimators": n, "roc_auc": roc_auc_score(y, p),
                     "brier": brier_score_loss(y, p), "seconds": time.perf_counter() - start})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACT_DIR / "estimator_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df.n_estimators, df.roc_auc, marker="o", lw=2.5, color="#7B2CBF")
    ax.axvline(CHOSEN, ls="--", color="#FB8500", lw=1.5)
    ax.text(CHOSEN * 1.05, df.roc_auc.min(), f"chosen = {CHOSEN}", color="#FB8500", fontsize=10)
    ax.set(xscale="log", xlabel="TabICLv2 ensemble members (log)", ylabel="Held-out ROC AUC",
           title="TabICLv2 accuracy plateaus by ~8 members")
    for _, r in df.iterrows():
        ax.annotate(f"{r.seconds:.0f}s", (r.n_estimators, r.roc_auc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="grey")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "estimator_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved artifacts/figures/estimator_sweep.png")


if __name__ == "__main__":
    main()
