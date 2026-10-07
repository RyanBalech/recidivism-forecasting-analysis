"""Permutation importance for TabICLv2 on fields the original run left out.

train_evaluate.py measured TabICLv2 on ten prespecified fields only (each shuffle needs a full
foundation-model pass over the 1,000-person sample). Several fields outside that list turned out
to rank high for logistic and XGBoost, so they are added here with the same protocol: the same
1,000 evaluation people, one shuffle per field, importance = Brier loss increase. Rows are
appended to artifacts/permutation_importance.csv (existing TabICL rows are kept).

    python scripts/tabicl_permutation_extra.py            # ~15-20 min per field on CPU
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.modeling import tabicl_frames, tabicl_model

warnings.filterwarnings("ignore")
EXTRA = ["_v1", "Prior_Conviction_Episodes_Misd", "Condition_MH_SA", "Prior_Arrest_Episodes_Property"]


def main() -> None:
    split = load_official_split()
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(split.X_test), size=min(1_000, len(split.X_test)), replace=False)
    X_sample = split.X_test.iloc[idx].reset_index(drop=True)
    y_sample = split.y_test.iloc[idx].reset_index(drop=True)

    path = ARTIFACT_DIR / "permutation_importance.csv"
    table = pd.read_csv(path)
    todo = [f for f in EXTRA if not ((table.model == "tabicl") & (table.feature == f)).any()]
    if not todo:
        print("nothing to do")
        return

    model = tabicl_model(RANDOM_SEED)
    model.fit(tabicl_frames(split.X_train, split.X_train)[0], split.y_train.to_numpy())
    baseline = brier_score_loss(y_sample, model.predict_proba(tabicl_frames(split.X_train, X_sample)[1])[:, 1])
    shuffle_rng = np.random.default_rng(RANDOM_SEED + 1)
    for feature in todo:
        permuted = X_sample.copy()
        permuted[feature] = shuffle_rng.permutation(permuted[feature].to_numpy())
        p = model.predict_proba(tabicl_frames(split.X_train, permuted)[1])[:, 1]
        row = {"model": "tabicl", "feature": feature, "importance": brier_score_loss(y_sample, p) - baseline, "std": np.nan}
        # Append one line per field: an interrupted run keeps its progress, earlier rows stay byte-identical.
        pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)
        print(f"{feature}: {row['importance']:.4f}", flush=True)


if __name__ == "__main__":
    main()
