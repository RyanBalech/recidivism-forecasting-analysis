import numpy as np

from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics, economic_value


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


def test_tabicl_frames_do_not_expose_gender_through_missingness():
    """Regression: Gang_Affiliated is NaN for exactly the women, and TabICL encodes NaN as its own category."""
    from recidivism.modeling import tabicl_frames

    split = load_official_split()
    assert split.X_train.Gang_Affiliated.isna().any(), "source data changed; revisit this test"
    for frame in tabicl_frames(split.X_train, split.X_test):
        assert not frame.select_dtypes(exclude="number").isna().any().any()
