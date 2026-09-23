"""Which machine-learning model should fill the "ML" slot?

Every candidate gets the same inputs and preprocessing as XGBoost (ordinal counts,
imputation, one-hot for the rest), a random search with 5-fold CV on the training
set, and out-of-fold predictions for calibration. Selection uses CV only. The
official test set is scored once at the end, for reporting, never for choosing.

XGBoost reuses the 60-draw search from scripts/tune_xgboost.py; the others get
N_ITER draws each (EBM uses its well-tested defaults because each fit is slow).

    python scripts/compare_ml_models.py            # all candidates
    python scripts/compare_ml_models.py --quick    # 4 draws, for a smoke run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics, economic_value
from recidivism.modeling import logistic_model, ordinal_encode, preprocessor, xgboost_model

warnings.filterwarnings("ignore", category=UserWarning)
N_ITER = 20


def _pipeline(frame: pd.DataFrame, estimator) -> Pipeline:
    """Same encoding path as XGBoost so only the learner differs."""
    return Pipeline([
        ("ordinal", FunctionTransformer(ordinal_encode)),
        ("prepare", preprocessor(ordinal_encode(frame))),
        ("model", estimator),
    ])


def candidates(frame: pd.DataFrame) -> dict[str, tuple[Pipeline, dict | None]]:
    """Each entry: (pipeline, search space or None for fixed settings)."""
    from catboost import CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingClassifier
    from lightgbm import LGBMClassifier

    return {
        "logistic": (logistic_model(frame), None),
        "xgboost": (xgboost_model(frame), None),
        "lightgbm": (_pipeline(frame, LGBMClassifier(random_state=RANDOM_SEED, verbose=-1, n_jobs=1)), {
            "model__n_estimators": randint(200, 1500),
            "model__learning_rate": loguniform(5e-3, 1e-1),
            "model__num_leaves": randint(4, 48),
            "model__min_child_samples": randint(10, 200),
            "model__subsample": uniform(0.6, 0.4),
            "model__subsample_freq": [1],
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_lambda": loguniform(1e-2, 10),
        }),
        "catboost": (_pipeline(frame, CatBoostClassifier(random_seed=RANDOM_SEED, verbose=0, thread_count=1)), {
            "model__iterations": randint(300, 1200),
            "model__learning_rate": loguniform(1e-2, 1e-1),
            "model__depth": randint(3, 8),
            "model__l2_leaf_reg": loguniform(1, 20),
        }),
        "hist_gb": (_pipeline(frame, HistGradientBoostingClassifier(random_state=RANDOM_SEED)), {
            "model__max_iter": randint(100, 800),
            "model__learning_rate": loguniform(1e-2, 2e-1),
            "model__max_leaf_nodes": randint(4, 48),
            "model__min_samples_leaf": randint(10, 200),
            "model__l2_regularization": loguniform(1e-3, 10),
        }),
        "random_forest": (_pipeline(frame, RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1)), {
            "model__n_estimators": randint(300, 800),
            "model__max_depth": randint(6, 20),
            "model__min_samples_leaf": randint(5, 60),
            "model__max_features": uniform(0.1, 0.5),
        }),
        "ebm": (_pipeline(frame, ExplainableBoostingClassifier(random_state=RANDOM_SEED, n_jobs=1)), None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", nargs="*", help="subset of candidate names")
    args = parser.parse_args()
    n_iter = 4 if args.quick else N_ITER

    split = load_official_split()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    rows, params_out = [], {}
    for name, (pipe, space) in candidates(split.X_train).items():
        if args.only and name not in args.only:
            continue
        start = time.perf_counter()
        if space:
            search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc", cv=cv,
                                        n_jobs=-1, random_state=RANDOM_SEED)
            search.fit(split.X_train, split.y_train)
            pipe = search.best_estimator_
            params_out[name] = {k.removeprefix("model__"): (round(float(v), 4) if isinstance(v, float) else v)
                                for k, v in search.best_params_.items()}
        search_seconds = time.perf_counter() - start

        # Out-of-fold probabilities: every training row is scored by a model that never saw it.
        oof = cross_val_predict(clone(pipe), split.X_train, split.y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        cv_m = classification_metrics(split.y_train, oof)

        start = time.perf_counter()
        final = clone(pipe).fit(split.X_train, split.y_train)
        p_test = final.predict_proba(split.X_test)[:, 1]
        fit_seconds = time.perf_counter() - start
        test_m = classification_metrics(split.y_test, p_test)
        value = economic_value(split.y_test, p_test, capacity=0.2)

        rows.append({
            "model": name,
            "cv_roc_auc": cv_m["roc_auc"], "cv_brier": cv_m["brier"], "cv_ece": cv_m["ece_10"],
            "test_roc_auc": test_m["roc_auc"], "test_brier": test_m["brier"], "test_ece": test_m["ece_10"],
            "test_net_value_top20": value["assumed_net_value"],
            "fit_predict_seconds": fit_seconds, "search_seconds": search_seconds,
        })
        print(f"{name:14s} cv_auc={cv_m['roc_auc']:.4f} cv_brier={cv_m['brier']:.4f} "
              f"test_auc={test_m['roc_auc']:.4f} fit={fit_seconds:.1f}s", flush=True)

    table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    suffix = "_quick" if args.quick else ""
    table.to_csv(ARTIFACT_DIR / f"ml_model_comparison{suffix}.csv", index=False)
    (ARTIFACT_DIR / f"ml_model_params{suffix}.json").write_text(json.dumps(params_out, indent=2, default=str), encoding="utf-8")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
