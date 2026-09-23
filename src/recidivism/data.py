"""Load the NIJ data and keep prediction inputs separate from audit fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_PATH, FEATURE_COLUMNS, ID_COLUMN, PROTECTED_COLUMNS, SPLIT_COLUMN, TARGET


@dataclass(frozen=True)
class DatasetSplit:
    """All objects needed after recreating NIJ's official train/test split.

    ``X`` means model inputs, ``y`` means the binary outcome, and ``audit``
    contains ID/race/gender for evaluation without exposing them to the model.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    audit_train: pd.DataFrame
    audit_test: pd.DataFrame


def _binary_target(series: pd.Series) -> pd.Series:
    """Convert NIJ's Yes/No outcome to the 1/0 format classifiers expect."""
    mapped = series.map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    # Fail loudly instead of silently treating an unfamiliar label as zero.
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].dropna().astype(str).unique())
        raise ValueError(f"Unexpected target labels: {bad}")
    return mapped.astype("int8")


def load_official_split(path: str | Path = DATA_PATH) -> DatasetSplit:
    """Load the post-challenge data and recreate NIJ's untouched 70/30 split.

    We do not make a new random split. ``Training_Sample`` identifies the exact
    rows NIJ originally released for training, leaving the other 7,807 people as
    a genuinely held-out evaluation cohort.
    """
    data = pd.read_csv(path)

    # Validate the schema before fitting anything. This catches a wrong or
    # modified CSV early, when the error is still easy to understand.
    required = set(FEATURE_COLUMNS + PROTECTED_COLUMNS + [ID_COLUMN, TARGET, SPLIT_COLUMN])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if data[ID_COLUMN].duplicated().any():
        raise ValueError("ID must be unique")

    y = _binary_target(data[TARGET])
    train_mask = data[SPLIT_COLUMN].eq(1)
    audit_cols = [ID_COLUMN, *PROTECTED_COLUMNS]

    # reset_index prevents the original row numbers from causing accidental
    # misalignment between X, y, and the audit table later in the pipeline.
    return DatasetSplit(
        X_train=data.loc[train_mask, FEATURE_COLUMNS].reset_index(drop=True),
        X_test=data.loc[~train_mask, FEATURE_COLUMNS].reset_index(drop=True),
        y_train=y.loc[train_mask].reset_index(drop=True),
        y_test=y.loc[~train_mask].reset_index(drop=True),
        audit_train=data.loc[train_mask, audit_cols].reset_index(drop=True),
        audit_test=data.loc[~train_mask, audit_cols].reset_index(drop=True),
    )


def feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return column names for the numeric and categorical preprocessing paths."""
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    return numeric, categorical
