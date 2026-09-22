from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_PATH, FEATURE_COLUMNS, ID_COLUMN, PROTECTED_COLUMNS, SPLIT_COLUMN, TARGET


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    audit_train: pd.DataFrame
    audit_test: pd.DataFrame


def _binary_target(series: pd.Series) -> pd.Series:
    mapped = series.map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].dropna().astype(str).unique())
        raise ValueError(f"Unexpected target labels: {bad}")
    return mapped.astype("int8")


def load_official_split(path: str | Path = DATA_PATH) -> DatasetSplit:
    """Load the post-challenge data and recreate NIJ's untouched 70/30 split."""
    data = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + PROTECTED_COLUMNS + [ID_COLUMN, TARGET, SPLIT_COLUMN])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if data[ID_COLUMN].duplicated().any():
        raise ValueError("ID must be unique")

    y = _binary_target(data[TARGET])
    train_mask = data[SPLIT_COLUMN].eq(1)
    audit_cols = [ID_COLUMN, *PROTECTED_COLUMNS]
    return DatasetSplit(
        X_train=data.loc[train_mask, FEATURE_COLUMNS].reset_index(drop=True),
        X_test=data.loc[~train_mask, FEATURE_COLUMNS].reset_index(drop=True),
        y_train=y.loc[train_mask].reset_index(drop=True),
        y_test=y.loc[~train_mask].reset_index(drop=True),
        audit_train=data.loc[train_mask, audit_cols].reset_index(drop=True),
        audit_test=data.loc[~train_mask, audit_cols].reset_index(drop=True),
    )


def feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    return numeric, categorical

