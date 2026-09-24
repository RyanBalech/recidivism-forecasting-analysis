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


@pytest.mark.skipif(not (ARTIFACT_DIR / "calibration_tests.csv").exists(),
                    reason="run scripts/calibration_tests.py first")
def test_calibration_claims_are_backed_by_tests():
    """Guards the three claims the recommendation is NOT allowed to rest on."""
    cal = pd.read_csv(ARTIFACT_DIR / "calibration_tests.csv").set_index("model")
    paired = pd.read_csv(ARTIFACT_DIR / "calibration_paired_tests.csv")
    pair = paired[(paired.model_a == "logistic") & (paired.model_b == "xgboost")].set_index("metric")

    # 1. Calibration is a tie between the two conventional models, not a win.
    assert not pair.loc["abs_slope_error", "significant"]
    for model in ["logistic", "xgboost"]:
        assert not cal.loc[model, "slope_differs_from_1"]
        assert cal.loc[model, "calibrated_at_5pct"]

    # 2. TabICL is the one model whose probabilities are measurably too extreme.
    assert cal.loc["tabicl", "slope_differs_from_1"]
    assert cal.loc["tabicl", "slope"] < 1
    assert not cal.loc["tabicl", "calibrated_at_5pct"]

    # 3. The ranking edge does not become a difference in who is offered support.
    assert not pair.loc["captured_events", "significant"]
    assert not pair.loc["abs_gender_fnr_gap", "significant"]


@pytest.mark.skipif(not (ARTIFACT_DIR / "calibration_bin_sensitivity.csv").exists(),
                    reason="run scripts/calibration_tests.py first")
def test_ece_ranking_is_not_stable_across_binning():
    """The reason ECE alone cannot support 'best calibrated'."""
    bins = pd.read_csv(ARTIFACT_DIR / "calibration_bin_sensitivity.csv")
    winners = {
        scheme: {part.loc[part[scheme].idxmin(), "model"] for _, part in bins.groupby("bins")}
        for scheme in ["ece_equal_width", "ece_equal_frequency"]
    }
    assert len(winners["ece_equal_width"] | winners["ece_equal_frequency"]) > 1, (
        "if one model won every binning choice, the sensitivity caveat would be wrong"
    )


@pytest.mark.skipif(not (ARTIFACT_DIR / "selected_set_overlap.csv").exists(),
                    reason="run scripts/calibration_tests.py first")
def test_models_mostly_select_the_same_people():
    overlap = pd.read_csv(ARTIFACT_DIR / "selected_set_overlap.csv")
    row = overlap.query("model_a == 'logistic' and model_b == 'xgboost'").iloc[0]
    assert row.selected_a == row.selected_b, "capacity rule must select equal-sized sets"
    assert row.jaccard > 0.8, "the recommendation text claims ~85% overlap"


@pytest.mark.skipif(not (ARTIFACT_DIR / "individual_stability.csv").exists(),
                    reason="run scripts/individual_stability.py first")
def test_individual_stability_uses_the_shared_bootstrap_protocol():
    """Per-person refits must reuse the structural script's resamples, not new ones."""
    individual = _load("individual_stability")
    structural_seed = 7
    assert individual.BOOTSTRAP_SEED == structural_seed
    a = individual.bootstrap_samples(1000)
    b = individual.bootstrap_samples(1000)
    assert all((x == y).all() for x, y in zip(a, b)), "resamples must be deterministic"
    assert len(a) == individual.N_REFITS


@pytest.mark.skipif(not (ARTIFACT_DIR / "individual_stability.csv").exists(),
                    reason="run scripts/individual_stability.py first")
def test_decision_stability_counts_are_coherent():
    people = pd.read_csv(ARTIFACT_DIR / "individual_stability.csv")
    n = people.n_refits.iloc[0]
    assert people.times_selected.between(0, n).all()
    # Unanimous means selected by none or all of the refits, nothing else.
    unanimous = people.times_selected.isin([0, n])
    assert (people.decision_unanimous == unanimous).all()
    assert (people.max_probability >= people.min_probability).all()
    # Roughly a fifth of the cohort is selected by any single refit.
    for model, part in people.groupby("model"):
        assert 0.15 < (part.times_selected >= n / 2).mean() < 0.25


@pytest.mark.skipif(not (ARTIFACT_DIR / "abstention_curve.csv").exists(),
                    reason="run scripts/individual_stability.py first")
def test_abstention_trades_coverage_against_the_gender_gap():
    """The finding: abstaining buys precision and widens the gender gap."""
    curve = pd.read_csv(ARTIFACT_DIR / "abstention_curve.csv")
    for model, part in curve.groupby("model"):
        part = part.sort_values("max_contested_votes")
        strictest, full = part.iloc[0], part.iloc[-1]
        assert full.coverage == pytest.approx(1.0), "the last row must decide everyone"
        assert strictest.coverage < full.coverage
        assert strictest.precision_at_capacity > full.precision_at_capacity, (
            f"{model}: abstention should improve precision on what remains")
        assert abs(strictest.fnr_gap_gender) > abs(full.fnr_gap_gender), (
            f"{model}: the reported finding is that the gender gap WIDENS under abstention")


@pytest.mark.skipif(not (ARTIFACT_DIR / "proxy_recovery.csv").exists(),
                    reason="run scripts/proxy_inference_audit.py first")
def test_proxy_recovery_quantifies_the_leak_and_the_residual():
    recovery = pd.read_csv(ARTIFACT_DIR / "proxy_recovery.csv")
    get = lambda a, f: float(recovery[(recovery.attribute == a)
                                      & (recovery.feature_set == f)].recovery_auc.iloc[0])
    # The missingness channel reconstructed gender essentially perfectly.
    assert get("Gender", "with_missingness") > 0.99
    # Exclusion is not removal: both attributes stay well above chance in the shipped set.
    assert get("Gender", "shipped") > 0.6
    assert get("Race", "shipped") > 0.6
    # Dropping the strongest proxy helps, but does not close the channel.
    assert get("Gender", "without_gang") < get("Gender", "shipped")
    assert get("Gender", "without_gang") > 0.6
    assert recovery.recoverable_share.between(0, 1).all()
