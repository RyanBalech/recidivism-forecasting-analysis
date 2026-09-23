"""5-fold cross-validated random search for the XGBoost hyperparameters.

Runs on the training set only; the test set is never touched. The winning values are
copied into XGB_PARAMS in src/recidivism/modeling.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.modeling import xgboost_model

SPACE = {
    "model__n_estimators": randint(300, 1400),
    "model__max_depth": randint(2, 7),
    "model__learning_rate": loguniform(5e-3, 1e-1),
    "model__min_child_weight": randint(1, 15),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__reg_lambda": loguniform(0.5, 10),
    "model__reg_alpha": loguniform(1e-3, 2),
    "model__gamma": uniform(0, 0.5),
}


def main() -> None:
    split = load_official_split()
    search = RandomizedSearchCV(
        xgboost_model(split.X_train), SPACE, n_iter=60, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1, random_state=RANDOM_SEED,
    )
    search.fit(split.X_train, split.y_train)
    best = {k.removeprefix("model__"): (round(float(v), 4) if isinstance(v, float) else int(v))
            for k, v in search.best_params_.items()}
    result = {"cv_roc_auc": round(float(search.best_score_), 4), "params": best}
    (ARTIFACT_DIR / "xgboost_tuning.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
