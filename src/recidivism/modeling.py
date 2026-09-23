"""Define comparable preprocessing and the three required model families."""

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
    """Build transformations learned only from the training data.

    Numeric values are median-imputed, receive an explicit missingness flag,
    and are standardized. Categorical values are mode-imputed and one-hot
    encoded. Unknown categories are ignored so a new category at prediction
    time does not crash the application.
    """
    numeric, categorical = feature_types(frame)
    return ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                # Rare categories seen fewer than 10 times are pooled. This
                # reduces fragile columns learned from only a handful of people.
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
            ]), categorical),
        ],
        verbose_feature_names_out=False,
    )


def logistic_model(frame: pd.DataFrame) -> Pipeline:
    """White-box benchmark: preprocessing followed by regularized logit."""
    return Pipeline([
        ("prepare", preprocessor(frame)),
        # C is inverse regularization strength. 0.25 applies useful shrinkage to
        # the 108 transformed columns and limits extreme coefficients.
        ("model", LogisticRegression(C=0.25, max_iter=2_000, solver="liblinear", random_state=42)),
    ])


def xgboost_model(frame: pd.DataFrame) -> Pipeline:
    """Nonlinear tree model using exactly the same eligible raw information."""
    return Pipeline([
        ("prepare", preprocessor(frame)),
        ("model", XGBClassifier(
            # Many shallow, slowly learned trees capture interactions while the
            # sampling and penalties below reduce overfitting.
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
    """Preserve pandas dtypes so TabICLv2 can apply its native mixed-data encoder.

    Unlike the two pipelines above, TabICLv2 expects the mixed-type table and
    performs its own encoding and learned normalization. Copies protect the
    original split from in-place changes inside third-party code.
    """
    return X_train.copy(), X_test.copy()
