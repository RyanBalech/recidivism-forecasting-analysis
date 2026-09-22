from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import feature_types


def preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = feature_types(frame)
    return ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
            ]), categorical),
        ],
        verbose_feature_names_out=False,
    )


def logistic_model(frame: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("prepare", preprocessor(frame)),
        ("model", LogisticRegression(C=0.25, max_iter=2_000, solver="liblinear", random_state=42)),
    ])


def xgboost_model(frame: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("prepare", preprocessor(frame)),
        ("model", XGBClassifier(
            n_estimators=550,
            max_depth=3,
            learning_rate=0.035,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )),
    ])


def tabicl_frames(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve pandas dtypes so TabICLv2 can apply its native mixed-data encoder."""
    return X_train.copy(), X_test.copy()
