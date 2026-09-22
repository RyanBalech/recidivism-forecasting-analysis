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

