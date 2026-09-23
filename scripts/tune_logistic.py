"""5-fold cross-validated grid search for the logistic-regression benchmark.

Runs on the training set only; the test set is never touched. Searches the
regularization strength C, the penalty (L2 shrinks, L1 can zero out columns),
and the encoding of count fields ("one-hot" learns one coefficient per level,
"ordinal" learns one slope per field, which is easier to read). The winning
setting is copied into LOGIT_PARAMS in src/recidivism/modeling.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.modeling import logistic_model

GRID = {
    "model__C": np.logspace(-3, 1, 13).round(5).tolist(),
    "model__penalty": ["l1", "l2"],
}


def main() -> None:
    split = load_official_split()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    rows, best = [], None
    for encoding in ["onehot", "ordinal"]:
        search = GridSearchCV(
            logistic_model(split.X_train, encoding=encoding), GRID, cv=cv, n_jobs=-1,
            scoring={"roc_auc": "roc_auc", "brier": "neg_brier_score"}, refit="roc_auc",
        )
        search.fit(split.X_train, split.y_train)
        res = search.cv_results_
        for i, params in enumerate(res["params"]):
            rows.append({
                "encoding": encoding,
                "C": params["model__C"],
                "penalty": params["model__penalty"],
                "cv_roc_auc": res["mean_test_roc_auc"][i],
                "cv_roc_auc_sd": res["std_test_roc_auc"][i],
                "cv_brier": -res["mean_test_brier"][i],
            })
        if best is None or search.best_score_ > best["cv_roc_auc"]:
            best = {"cv_roc_auc": round(float(search.best_score_), 4), "encoding": encoding,
                    "C": float(search.best_params_["model__C"]),
                    "penalty": search.best_params_["model__penalty"]}

    table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    table.to_csv(ARTIFACT_DIR / "logistic_tuning_grid.csv", index=False)
    (ARTIFACT_DIR / "logistic_tuning.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(table.head(10).round(4).to_string(index=False))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
