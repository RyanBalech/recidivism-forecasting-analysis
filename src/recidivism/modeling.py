"""Define comparable preprocessing and the three required model families."""

from __future__ import annotations

import re
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import feature_types

# Ordered categories whose order one-hot encoding would discard.
_ORDERED = {
    "Prison_Years": {"Less than 1 year": 0, "1-2 years": 1, "Greater than 2 to 3 years": 2, "More than 3 years": 3},
    "Age_at_Release": {"18-22": 0, "23-27": 1, "28-32": 2, "33-37": 3, "38-42": 4, "43-47": 5, "48 or older": 6},
}
_COUNT = re.compile(r"\d+( or more)?")

# From scripts/tune_xgboost.py (5-fold CV random search, 60 draws, CV AUC 0.7343).
XGB_PARAMS = dict(
    n_estimators=1196, max_depth=2, learning_rate=0.0188, min_child_weight=8,
    subsample=0.8217, colsample_bytree=0.6233, gamma=0.4576,
    reg_lambda=4.5603, reg_alpha=1.4192,
)

# From scripts/tune_logistic.py (5-fold CV grid over C, penalty, and count encoding;
# one-hot counts beat ordinal counts, 0.7324 vs 0.7315).
LOGIT_ENCODING = "onehot"
# scikit-learn 1.8+ deprecates ``penalty`` in favour of ``l1_ratio``, which 1.7 rejects
# for liblinear. Keep ``penalty`` so every version in requirements.txt behaves the same.
warnings.filterwarnings("ignore", message="'penalty' was deprecated", category=FutureWarning)
LOGIT_PARAMS = dict(C=0.2154, penalty="l1")  # CV AUC 0.7324; C in [0.05, 1] is flat within 0.0002

# Selected as a practical point on the training-only development sweep in
# scripts/estimator_sweep.py. Keep this in one place so the app, main fit and
# all secondary audits use the same foundation-model ensemble.
TABICL_ESTIMATORS = 16


def ordinal_encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn ordered categories and 'N or more' counts into numbers; leave the rest as is."""
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        if col in _ORDERED:
            out[col] = out[col].map(_ORDERED[col]).astype(float)
            continue
        # Decide per value, not per batch size, so a single row encodes like the training set.
        values = set(map(str, out[col].dropna().unique()))
        if values and all(_COUNT.fullmatch(v) for v in values):
            out[col] = out[col].astype(str).str.extract(r"(\d+)")[0].astype(float)
    return out


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


def logistic_model(frame: pd.DataFrame, encoding: str | None = None, **overrides) -> Pipeline:
    """White-box benchmark: preprocessing followed by regularized logit.

    ``encoding="onehot"`` gives every count level its own coefficient;
    ``"ordinal"`` turns counts into numbers so each field has one slope.
    Defaults come from LOGIT_PARAMS (5-fold CV grid in scripts/tune_logistic.py).
    """
    encoding = encoding or LOGIT_ENCODING
    params = {**LOGIT_PARAMS, **overrides}
    steps = []
    if encoding == "ordinal":
        steps.append(("ordinal", FunctionTransformer(ordinal_encode)))
        frame = ordinal_encode(frame)
    elif encoding != "onehot":
        raise ValueError(f"Unknown encoding: {encoding}")
    # C is inverse regularization strength: smaller C means stronger shrinkage.
    steps += [
        ("prepare", preprocessor(frame)),
        ("model", LogisticRegression(**params, max_iter=2_000, solver="liblinear", random_state=42)),
    ]
    return Pipeline(steps)


def xgboost_model(frame: pd.DataFrame, random_state: int = 42, **overrides) -> Pipeline:
    """Nonlinear tree model using exactly the same eligible raw information.

    Ordered-count columns are ordinal-encoded first so trees can split on rank.
    Hyperparameters come from XGB_PARAMS (5-fold CV search in scripts/tune_xgboost.py);
    pass overrides for smaller/faster fits (e.g. learning curves, stability refits).
    """
    params = {**XGB_PARAMS, "n_jobs": -1, **overrides}
    return Pipeline([
        ("ordinal", FunctionTransformer(ordinal_encode)),
        ("prepare", preprocessor(ordinal_encode(frame))),
        ("model", XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
        )),
    ])


def tabicl_frames(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve pandas dtypes so TabICLv2 can apply its native mixed-data encoder.

    Unlike the two pipelines above, TabICLv2 expects the mixed-type table and
    performs its own encoding and learned normalization. Copies protect the
    original split from in-place changes inside third-party code.

    Missing categorical values are filled with the training mode, exactly as the
    other two pipelines do. TabICL would otherwise encode NaN as its own category,
    and Gang_Affiliated is missing for every woman and no man in NIJ, so that
    category would hand the model the excluded Gender attribute.
    """
    categorical = X_train.select_dtypes(exclude="number").columns
    modes = X_train[categorical].mode().iloc[0]
    return X_train.fillna(modes), X_test.fillna(modes)


def tabicl_model(random_state=42):
    """One shared inference configuration for training, app and secondary audits."""
    from tabicl import TabICLClassifier
    return TabICLClassifier(n_estimators=TABICL_ESTIMATORS, batch_size=1, kv_cache="repr",
                            random_state=random_state, n_jobs=-1)
