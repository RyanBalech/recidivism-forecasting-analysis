"""Negative tests: audit must reject introduced leakage, not just accept our data."""
import runpy

import numpy as np
import pandas as pd
import pytest

from recidivism.config import DATA_DIR, DATA_PATH, FEATURE_COLUMNS, ROOT
from recidivism.data import load_official_split
from recidivism.modeling import logistic_model, tabicl_frames


@pytest.mark.parametrize("field", ["ID", "Training_Sample", "Recidivism_Within_3years",
                                    "Recidivism_Arrest_Year1", "Gender", "Race",
                                    "Percent_Days_Employed", "Program_Attendances"])
def test_independent_release_gate_rejects_injected_feature(field):
    check = runpy.run_path(str(ROOT / "scripts/deep_leakage_audit.py"))["check_release_features"]
    released = pd.read_csv(DATA_DIR / "nij-challenge2021_test_dataset_1.csv", nrows=0)
    with pytest.raises(ValueError):
        check([*FEATURE_COLUMNS, field], released.columns)


def test_evaluation_outcomes_and_post_release_values_cannot_change_inputs(tmp_path):
    raw = pd.read_csv(DATA_PATH)
    before = load_official_split()
    is_eval = raw.Training_Sample.eq(0)
    raw.loc[is_eval, "Recidivism_Within_3years"] = "Yes"
    raw.loc[is_eval, "Program_Attendances"] = "999 or more"
    changed = tmp_path / "changed.csv"
    raw.to_csv(changed, index=False)
    after = load_official_split(changed)
    pd.testing.assert_frame_equal(before.X_train, after.X_train)
    pd.testing.assert_frame_equal(before.X_test, after.X_test)
    pd.testing.assert_series_equal(before.y_train, after.y_train)


def test_preprocessing_does_not_learn_evaluation_categories_or_statistics():
    train = pd.DataFrame({"number": [1., 2., np.nan, 4.], "kind": ["A", "A", "B", None]})
    train["kind"] = train.kind.fillna(np.nan)
    test = pd.DataFrame({"number": [999., np.nan], "kind": ["ONLY_IN_EVAL", np.nan]})
    model = logistic_model(train).fit(train, [0, 1, 0, 1])
    prep = model.named_steps["prepare"]
    num = prep.named_transformers_["numeric"]
    assert num.named_steps["impute"].statistics_[0] == 2.
    categories = prep.named_transformers_["categorical"].named_steps["onehot"].categories_[0]
    assert "ONLY_IN_EVAL" not in categories
    model.predict_proba(test)
    assert num.named_steps["impute"].statistics_[0] == 2.
    _, filled = tabicl_frames(train, test)
    assert filled.kind.iloc[1] == "A"
    assert filled.kind.iloc[0] == "ONLY_IN_EVAL"
