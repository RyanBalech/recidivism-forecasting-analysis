"""Guards for the explanation fixes: valid perturbation space, honest fidelity, sign conventions."""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from recidivism.config import ARTIFACT_DIR, ROOT

sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lime_reference_space_round_trips_categories():
    """Encoding then decoding must return the original categories.

    This is the property the previous implementation lacked: it perturbed one-hot
    columns independently, so a perturbed row could decode to two categories or none.
    """
    lime_module = _load("lime_local_fidelity")
    train = pd.DataFrame({
        "count": [1.0, 2.0, 3.0, np.nan],
        "band": ["18-22", "23-27", "18-22", "48 or older"],
        "flag": ["Yes", "No", "Yes", None],
    })
    encoded_train, encoded_eval, decode, cat_idx, cat_names = lime_module.reference_space(train, train)

    assert cat_idx == [1, 2], "categorical columns must be declared to LIME by position"
    assert set(cat_names[1]) == {"18-22", "23-27", "48 or older"}

    decoded = decode(encoded_eval)
    # Missing values are filled with training statistics, exactly as the pipeline does.
    assert decoded["band"].tolist() == train["band"].tolist()
    assert decoded["flag"].tolist() == ["Yes", "No", "Yes", "Yes"]
    assert decoded["count"].tolist() == [1.0, 2.0, 3.0, 2.0]


def test_decoded_perturbation_is_always_one_real_category():
    """Any code in range must decode to exactly one level that exists in training."""
    lime_module = _load("lime_local_fidelity")
    train = pd.DataFrame({"band": ["18-22", "23-27", "48 or older"] * 3,
                          "count": [1.0, 2.0, 3.0] * 3})
    _, encoded, decode, _, cat_names = lime_module.reference_space(train, train)
    rng = np.random.default_rng(0)
    perturbed = encoded.copy()
    perturbed[:, 0] = rng.integers(0, len(cat_names[0]), len(perturbed))
    decoded = decode(perturbed)
    assert decoded["band"].isin(cat_names[0]).all()
    assert decoded["band"].notna().all()


@pytest.mark.skipif(not (ARTIFACT_DIR / "lime_fidelity.csv").exists(),
                    reason="run scripts/lime_local_fidelity.py first")
def test_lime_fidelity_is_reported_and_beats_the_one_hot_version():
    fidelity = pd.read_csv(ARTIFACT_DIR / "lime_fidelity.csv")
    assert {"model", "case", "seed", "local_r2"} <= set(fidelity.columns)
    # Every explanation carries its own fidelity, and the cases/seeds are plural.
    assert fidelity.local_r2.notna().all()
    assert fidelity.case.nunique() >= 3 and fidelity.seed.nunique() >= 3
    # The transformed-space implementation scored ~0.25; the valid space must do better.
    assert fidelity.local_r2.mean() > 0.30


@pytest.mark.skipif(not (ARTIFACT_DIR / "probability_contrasts.csv").exists(),
                    reason="run scripts/logistic_effects.py first")
def test_probability_contrasts_are_against_a_single_reference():
    contrasts = pd.read_csv(ARTIFACT_DIR / "probability_contrasts.csv")
    # One reference level per feature, and no feature contrasted against itself.
    for (model, feature), part in contrasts.groupby(["model", "feature"]):
        assert part.reference.nunique() == 1, f"{model}/{feature} has multiple references"
        assert (part.level != part.reference).all()


@pytest.mark.skipif(not (ARTIFACT_DIR / "explanation_agreement.csv").exists(),
                    reason="run scripts/explanation_agreement.py first")
def test_explanation_agreement_compares_three_methods():
    agreement = pd.read_csv(ARTIFACT_DIR / "explanation_agreement.csv")
    pairs = set(zip(agreement.method_a, agreement.method_b))
    assert len(pairs) == 3, "all three method pairs must be compared"
    assert agreement.spearman_rank_correlation.between(-1, 1).all()
    # XPER answers a different question, so it should agree less than SHAP and PI do.
    shap_pi = agreement[agreement.method_b == "permutation_importance"].spearman_rank_correlation.mean()
    with_xper = agreement[agreement.method_b == "xper"].spearman_rank_correlation.mean()
    assert shap_pi > with_xper


def test_fnr_gap_sign_marks_the_protected_group_as_disadvantaged():
    """Negative gap must mean the protected group misses more support."""
    nested = _load("mitigation_nested")
    y = np.array([1, 1, 1, 1])
    protected = np.array([0, 0, 1, 1])
    # Reference group fully selected; protected group entirely missed.
    selected = np.array([True, True, False, False])
    assert nested.fnr_gap(y, selected, protected) == pytest.approx(-1.0)
    assert nested.fnr_gap(y, ~selected, protected) == pytest.approx(1.0)


@pytest.mark.skipif(not (ARTIFACT_DIR / "mitigation_nested_summary.csv").exists(),
                    reason="run scripts/mitigation_nested.py first")
def test_nested_mitigation_separates_selection_from_assessment():
    summary = pd.read_csv(ARTIFACT_DIR / "mitigation_nested_summary.csv")
    folds = pd.read_csv(ARTIFACT_DIR / "mitigation_nested.csv")
    assert (summary.folds >= 5).all()
    # The reported improvement must be out-of-fold, so a mitigated column exists per fold.
    assert folds.mitigated_auc.notna().all()
    # Mitigation trades accuracy for a smaller gap; both directions must be recorded.
    assert (summary.auc_change < 0).all(), "dropping a feature should not raise AUC"
    assert (summary.mitigated_fnr_gap.abs() < summary.baseline_fnr_gap.abs()).all()
