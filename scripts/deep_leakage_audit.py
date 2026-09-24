"""Independent release checks; passing does not certify absence of all leakage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.config import DATA_PATH, EXCLUDED_FROM_MODEL, FEATURE_COLUMNS, ID_COLUMN, SPLIT_COLUMN, TARGET
from recidivism.data import load_official_split
from recidivism.modeling import tabicl_frames


def check_release_features(features, release_columns):
    """Independent release schema must permit every scoring field."""
    forbidden = {ID_COLUMN, SPLIT_COLUMN, TARGET, *EXCLUDED_FROM_MODEL}
    bad = set(features) & forbidden
    bad |= {c for c in features if c.startswith("Recidivism_")}
    bad |= set(features) - set(release_columns)
    if bad:
        raise ValueError(f"Forbidden or unavailable-at-first-release features: {sorted(bad)}")


def audit(include_prediction_sensitivity=True):
    split = load_official_split()
    released = pd.read_csv(ROOT / "nij-challenge2021_test_dataset_1.csv")
    raw = pd.read_csv(DATA_PATH)
    original_train = pd.read_csv(ROOT / "nij-challenge2021_training_dataset.csv")
    check_release_features(FEATURE_COLUMNS, released.columns)
    assert set(split.audit_test.ID) == set(released.ID)
    assert set(split.audit_train.ID) == set(original_train.ID)
    assert set(split.audit_train.ID).isdisjoint(set(split.audit_test.ID))
    for source in [released, original_train]:
        before = source.set_index("ID")[FEATURE_COLUMNS].sort_index()
        after = raw.set_index("ID").loc[before.index, FEATURE_COLUMNS]
        pd.testing.assert_frame_equal(before, after, check_dtype=False)
    outcome_columns = [TARGET, *[f"Recidivism_Arrest_Year{i}" for i in (1, 2, 3)]]
    original_outcomes = original_train.set_index("ID")[outcome_columns].sort_index()
    pd.testing.assert_frame_equal(
        original_outcomes, raw.set_index("ID").loc[original_outcomes.index, outcome_columns],
        check_dtype=False,
    )
    filled_train, filled_test = tabicl_frames(split.X_train, split.X_test)
    cat = split.X_train.select_dtypes(exclude="number").columns
    assert not filled_train[cat].isna().any().any()
    assert not filled_test[cat].isna().any().any()
    probe = split.X_test.head(5).copy()
    probe.loc[:, cat] = np.nan
    train_again, probe_filled = tabicl_frames(split.X_train, probe)
    pd.testing.assert_frame_equal(train_again, filled_train)
    modes = split.X_train[cat].mode().iloc[0]
    for c in cat:
        assert probe_filled[c].eq(modes[c]).all()
    missingness = []
    for c in split.X_train:
        miss = split.X_train[c].isna()
        if miss.any():
            missingness.append({"feature": c, "missing_n": int(miss.sum()),
                "female_share_missing": float(split.audit_train.loc[miss, "Gender"].eq("F").mean()),
                "female_share_all": float(split.audit_train.Gender.eq("F").mean()),
                "target_rate_missing": float(split.y_train[miss].mean()),
                "target_rate_present": float(split.y_train[~miss].mean())})
    def hashes(frame):
        return pd.util.hash_pandas_object(frame.astype("string").fillna("__MISSING__"), index=False)
    a, b = hashes(split.X_train), hashes(split.X_test)
    shared = set(a) & set(b)
    duplicate_sensitivity = None
    prediction_path = ROOT / "artifacts/test_predictions.csv"
    if include_prediction_sensitivity and prediction_path.exists():
        pred = pd.read_csv(prediction_path)
        np.testing.assert_array_equal(pred.ID, split.audit_test.ID)
        np.testing.assert_array_equal(pred.actual, split.y_test)
        keep = ~b.isin(shared)
        duplicate_sensitivity = {name: float(roc_auc_score(pred.actual[keep], pred[f"p_{name}"][keep]))
                                 for name in ["logistic", "xgboost", "tabicl"]}
    return {
        "checks_passed": ["official split IDs", "independent first-release feature schema",
            "baseline values unchanged since official releases", "training outcomes match original release",
            "training-only categorical modes",
            "no explicit protected/ID/split/target feature", "no categorical NaN into TabICL"],
        "train_rows": len(split.X_train), "evaluation_rows": len(split.X_test),
        "cross_split_identical_feature_patterns": len(shared),
        "evaluation_rows_with_training_pattern": int(b.isin(shared).sum()),
        "auc_excluding_shared_patterns": duplicate_sensitivity,
        "missingness": missingness,
        "limitations": [
            "Matching coarse features does not prove records are the same person; unique IDs are the available identity check",
            "Gender-aligned missingness is proxy exposure, not by itself future-target leakage",
            "Mode filling removes an explicit NaN category; other proxies may remain",
            "First release availability does not timestamp initial risk assessment or gang verification",
            "Evaluation labels have been inspected repeatedly; performance comparisons remain exploratory",
            "Permutation controls cannot exclude timing leakage or benchmark selection contamination",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
