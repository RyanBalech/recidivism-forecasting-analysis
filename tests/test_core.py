import numpy as np
import pandas as pd
import pytest

from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics, economic_value


@pytest.mark.parametrize("p", [[-0.1, 0.9], [0.1, 1.1], [np.nan, 0.8], [0.5]])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError):
        classification_metrics([0, 1], p)


def test_single_class_metrics_and_exact_endpoints():
    m = classification_metrics([0, 0], [0, 0])
    assert np.isnan(m["roc_auc"]) and m["brier"] == 0
    assert classification_metrics([0, 1], [0, 1], threshold=1)["accuracy"] == 1


def test_capacity_zero_and_ties():
    from recidivism.metrics import capacity_selection
    assert capacity_selection([1, 1, 1, 1], 0.5).tolist() == [True, True, False, False]
    assert economic_value([0, 1], [2, 8], capacity=0)["selected"] == 0
    assert economic_value([0, 1], [2, 8], capacity=1)["selected"] == 2
    with pytest.raises(ValueError):
        economic_value([0, 1], [2, 8], capacity=1.1)


def test_group_metrics_use_position_not_series_index():
    from recidivism.metrics import fairness_table
    table = fairness_table(pd.Series([0, 1], index=[8, 9]), [0.1, 0.9], ["A", "B"], "group")
    assert table.n.sum() == 2 and np.allclose(table.brier, 0.01)


@pytest.mark.parametrize("column,value", [("Training_Sample", 2), ("Training_Sample", np.nan), ("ID", np.nan)])
def test_invalid_dataset_rejected(tmp_path, column, value):
    from recidivism.config import DATA_PATH
    frame = pd.read_csv(DATA_PATH).head(20)
    frame.loc[0, column] = value
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_official_split(path)


def test_official_split_is_complete_and_disjoint():
    split = load_official_split()
    assert len(split.X_train) == 18_028
    assert len(split.X_test) == 7_807
    assert set(split.audit_train.ID).isdisjoint(set(split.audit_test.ID))
    assert set(split.y_train.unique()) == {0, 1}


def test_perfect_predictions_have_expected_metrics():
    y = np.array([0, 0, 1, 1])
    metrics = classification_metrics(y, y)
    assert metrics["roc_auc"] == 1
    assert metrics["brier"] < 1e-12
    assert economic_value(y, y, capacity=0.5)["captured_events"] == 2



def test_ordinal_encode_keeps_order_and_leaves_binaries():
    import pandas as pd
    from recidivism.modeling import ordinal_encode

    frame = pd.DataFrame({
        "Prior_Arrest_Episodes_Felony": ["0", "3", "10 or more"],
        "Age_at_Release": ["18-22", "48 or older", "33-37"],
        "Gang_Affiliated": ["Yes", "No", "Yes"],
    })
    out = ordinal_encode(frame)
    assert out["Prior_Arrest_Episodes_Felony"].tolist() == [0, 3, 10]
    assert out["Age_at_Release"].tolist() == [0, 6, 3]
    assert out["Gang_Affiliated"].tolist() == ["Yes", "No", "Yes"]


def test_xgboost_pipeline_scores_raw_rows():
    from recidivism.modeling import xgboost_model

    split = load_official_split()
    model = xgboost_model(split.X_train, n_estimators=20).fit(split.X_train.head(2_000), split.y_train.head(2_000))
    p = model.predict_proba(split.X_test.head(50))[:, 1]
    assert p.shape == (50,) and ((p > 0) & (p < 1)).all()


def test_single_row_scores_like_a_batch():
    """Regression: encoding must not depend on how many rows are scored (app scores one row)."""
    import joblib
    from recidivism.config import MODEL_DIR

    split = load_official_split()
    model = joblib.load(MODEL_DIR / "xgboost.joblib")
    batch = model.predict_proba(split.X_test.head(20))[:, 1]
    single = [model.predict_proba(split.X_test.iloc[[i]])[:, 1][0] for i in range(20)]
    assert np.allclose(batch, single)


def test_published_predictions_match_saved_models():
    """Prevent app/report drift from stale models or changed preprocessing."""
    import joblib
    from recidivism.config import ARTIFACT_DIR, MODEL_DIR
    split = load_official_split()
    predictions = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    assert np.array_equal(predictions.ID, split.audit_test.ID)
    for name in ["logistic", "xgboost"]:
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        actual = model.predict_proba(split.X_test)[:, 1]
        np.testing.assert_allclose(actual, predictions[f"p_{name}"], atol=1e-7)


def test_tabicl_frames_do_not_expose_gender_through_missingness():
    """Regression: Gang_Affiliated is NaN for exactly the women, and TabICL encodes NaN as its own category."""
    from recidivism.modeling import tabicl_frames

    split = load_official_split()
    assert split.X_train.Gang_Affiliated.isna().any(), "source data changed; revisit this test"
    for frame in tabicl_frames(split.X_train, split.X_test):
        assert not frame.select_dtypes(exclude="number").isna().any().any()
