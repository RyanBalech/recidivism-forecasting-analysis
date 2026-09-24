"""Regression guards for clean-start leakage checks and stale deliverables."""
import runpy

import pandas as pd
import pytest

from recidivism.config import ARTIFACT_DIR, ROOT


def test_leakage_gate_does_not_need_prediction_artifacts(monkeypatch):
    module = runpy.run_path(str(ROOT / "scripts/deep_leakage_audit.py"))
    original = pd.read_csv

    def read_without_predictions(path, *args, **kwargs):
        if str(path).endswith("test_predictions.csv"):
            raise AssertionError("A pre-training gate cannot depend on saved predictions")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_without_predictions)
    assert module["audit"](include_prediction_sensitivity=False)["auc_excluding_shared_patterns"] is None


def test_artifact_audit_rejects_stale_performance(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = runpy.run_path(str(ROOT / "scripts/end_to_end_audit.py"))
    pred = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    metrics = pd.read_csv(ARTIFACT_DIR / "model_metrics.csv")
    module["verify_metrics"](pred, metrics)
    metrics.loc[0, "brier"] += 0.001
    with pytest.raises(ValueError, match="Stale metric"):
        module["verify_metrics"](pred, metrics)


def test_holm_adjustment_restores_original_order(monkeypatch):
    import numpy as np
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = runpy.run_path(str(ROOT / "scripts/end_to_end_audit.py"))
    np.testing.assert_allclose(module["holm_adjust"]([.04, .01, .03]), [.06, .03, .06])
