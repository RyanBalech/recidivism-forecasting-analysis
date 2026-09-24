"""Build and execute the professor-facing submission notebook.

The notebook is the readable entry point for the complete project. Heavy fitting
remains in versioned modules/scripts so the notebook and application use exactly
the same implementation; a visible switch in the notebook runs the whole pipeline.
"""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "Recidivism_Project_Submission.ipynb",
        ROOT / "notebooks" / "recidivism_analysis.ipynb"]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells = [
    md("# Trustworthy recidivism forecasting — submission notebook\n\n**Client:** a community-supervision software vendor. **Decision:** prioritize voluntary re-entry support at supervision start. **Target:** cumulative new arrest within three years. **Models:** tuned L1 logistic regression, XGBoost and TabICLv2.\n\nThis notebook is the professor-facing entry point for the complete analysis required by the Interpretability, Stability and Algorithmic Fairness project brief. It covers data preparation, leakage controls, model configuration, predictive and economic performance, interpretability, stability, fairness, trade-offs and the deployment recommendation. The project source and generated evidence remain in versioned `src/`, `scripts/` and `artifacts/` directories so the notebook and interactive application use the same implementation.\n\nThe target differs from NIJ's annual conditional forecasting task; comparison with the challenge leaderboard would be invalid."),
    code("""from pathlib import Path
import inspect, json, runpy, subprocess, sys
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display
ROOT = Path.cwd()
if not (ROOT / 'artifacts').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from recidivism.data import load_official_split
split = load_official_split()
A = lambda name: pd.read_csv(ROOT / 'artifacts' / name)
print(f'Training: {len(split.X_train):,}; evaluation: {len(split.X_test):,}; features: {split.X_train.shape[1]}')
"""),
    md("## Reproduce the complete project\n\nSet `RUN_FULL_PIPELINE=True` and execute this notebook from the repository root to regenerate model predictions, audits, figures, this notebook, the slide deck and the pre-validation PDF. The default is `False` because the checked-in notebook is already executed and the TabICLv2 stability/explanation passes are GPU-intensive. The fitting code is shown through its effective configuration below and is fully available in `src/recidivism/` and `scripts/`."),
    code("""RUN_FULL_PIPELINE = False
if RUN_FULL_PIPELINE:
    subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce.py')], cwd=ROOT, check=True)
else:
    print('Using checked-in, hash-audited artifacts. Set RUN_FULL_PIPELINE=True for a complete rebuild.')
"""),
    md("## Data preparation\n\nOnly baseline fields are eligible. Outcomes, IDs, split flags, protected attributes, geography and post-release supervision activities are excluded from model inputs. Imputation and scaling are learned within training pipelines. Logistic uses tuned L1 with one-hot categories; XGBoost ordinal-encodes ordered fields; TabICL mode-imputes categorical missingness before its mixed-data encoder. This neutralizes the gender-aligned `Gang_Affiliated` missingness channel; `scripts/leakage_audit.py` enforces the check."),
    code("""from recidivism.config import FEATURE_COLUMNS, BASELINE_COLUMNS, DATA_PATH
raw = pd.read_csv(DATA_PATH)
display(pd.Series({'training_target_rate': split.y_train.mean(), 'evaluation_target_rate': split.y_test.mean()}))
display(pd.DataFrame({'feature': raw.columns, 'used_for_scoring': [c in FEATURE_COLUMNS for c in raw.columns]}))
display(raw.assign(gang_missing=raw.Gang_Affiliated.isna()).groupby('Gender').gang_missing.agg(['mean','sum','count']))
display(split.X_train.isna().mean().sort_values(ascending=False).head(10))
"""),
    md("### Data integrity and leakage gate\n\nThis gate runs before model fitting and independently compares IDs, eligible feature values and training outcomes with NIJ's original releases. It rejects explicit targets, annual outcomes, IDs, split flags, protected attributes and post-release activity fields. Passing establishes the tested boundaries; it cannot prove the exact measurement time of every field or undo historical inspection of the evaluation labels."),
    code("""from deep_leakage_audit import audit as leakage_audit
leakage_result = leakage_audit(include_prediction_sensitivity=True)
display(pd.Series({
    'checks_passed': len(leakage_result['checks_passed']),
    'cross_split_feature_patterns': leakage_result['cross_split_identical_feature_patterns'],
    'evaluation_rows_in_shared_patterns': leakage_result['evaluation_rows_with_training_pattern'],
}))
display(pd.DataFrame(leakage_result['missingness']))
print('Checks:', *leakage_result['checks_passed'], sep='\\n- ')
print('\\nLimits:', *leakage_result['limitations'], sep='\\n- ')
"""),
    md("## Model definitions and training boundary\n\nAll three models receive the same 29 eligible raw fields and fit only the 18,028 training records. Learned preprocessing sits inside the conventional-model pipelines. TabICL receives no evaluation labels at inference. The cells below expose the effective hyperparameters and device evidence used by the shared application and scripts."),
    code("""from recidivism.modeling import (
    LOGIT_ENCODING, LOGIT_PARAMS, TABICL_CHECKPOINT, TABICL_ESTIMATORS, XGB_PARAMS,
    logistic_model, tabicl_model, xgboost_model,
)
display(pd.DataFrame([
    {'model': 'logistic', 'configuration': {'encoding': LOGIT_ENCODING, **LOGIT_PARAMS}},
    {'model': 'xgboost', 'configuration': XGB_PARAMS},
    {'model': 'tabicl', 'configuration': {'estimators': TABICL_ESTIMATORS, 'checkpoint': TABICL_CHECKPOINT}},
]))
run_manifest = json.loads((ROOT / 'artifacts/run_manifest.json').read_text())
end_to_end = json.loads((ROOT / 'artifacts/end_to_end/audit.json').read_text())
display(pd.Series({
    'train_rows': run_manifest.get('train_rows'),
    'evaluation_rows': run_manifest.get('test_rows'),
    'tabicl_estimators': run_manifest.get('tabicl_estimators'),
    'tabicl_device': run_manifest.get('tabicl_device'),
    'tabicl_checkpoint': end_to_end['cached_checkpoint']['name'],
    'tabicl_checkpoint_sha256': end_to_end['cached_checkpoint']['sha256'],
    'xgboost_execution': 'CPU for published model; paired CPU/CUDA audit available',
}))
"""),
    md("## Performance and baselines\n\nThe incumbent is a historical recorded score, not evidence about tools agencies use today. Constant prevalence and incumbent calibration are learned on training records."),
    code("""from recidivism.metrics import classification_metrics
predictions = A('test_predictions.csv')
recomputed = pd.DataFrame([
    {'model': model, **classification_metrics(predictions.actual, predictions[f'p_{model}'])}
    for model in ['logistic', 'xgboost', 'tabicl']
])
published = A('model_metrics.csv')
check = published[['model','roc_auc','brier','average_precision']].merge(
    recomputed[['model','roc_auc','brier','average_precision']], on='model', suffixes=('_published','_recomputed'))
display(check)
assert np.allclose(check.filter(like='_published'), check.filter(like='_recomputed').to_numpy(), atol=1e-7)
display(A('validation_baselines.csv'))
"""),
    md("## Foundation-model compute sensitivity\n\nThe 16-member TabICLv2 configuration is checked on a fixed stratified development slice carved only from the training partition. This is a compute/robustness sensitivity check, not final validation."),
    code("display(A('estimator_sweep.csv')); display(Image(filename=str(ROOT / 'artifacts/figures/estimator_sweep.png'), width=900))"),
    md("## Paired uncertainty and training CV\n\nDifferences are A minus B: positive AUC favors A, negative Brier favors A. Paired bootstrap preserves the correlation between model errors. Marginal intervals are exploratory and unadjusted for development selection or multiple comparisons. Fixed-configuration CV is not nested evaluation of the earlier search."),
    code("display(A('paired_comparisons.csv')); display(A('validation_cv.csv').groupby('model')[['roc_auc','brier']].agg(['mean','std']))"),
    md("## Additional leakage and accuracy review\n\nThe independent original-release checks and nested accuracy experiments are documented in reports/deep_review.md. Selection occurs on inner folds; outer folds score the selected configuration. Historical defaults already saw this training cohort, so these scores do not replace new-cohort validation. A fixed equal-weight ensemble is a research challenger."),
    code("review = ROOT / 'artifacts/deep_review/combined_summary.csv'\nif review.exists(): display(pd.read_csv(review))"),
    md("## Economic scenarios\n\nEffectiveness and costs are assumed, not estimated causal savings. Exact capacity uses stable row-order ties; this matters particularly for the discrete incumbent."),
    code("display(A('incumbent_economics.csv'))"),
    md("## Interpretability\n\nSHAP and LIME explain conventional models; XPER approximates performance attribution on a small sample. Surrogate fidelity is imperfect. TabICLv2 has PDP/ICE, permutation and interactive sensitivity, but no implemented native additive attribution. These are model-response explanations, not causal effects."),
    code("display(A('shap_importance.csv').groupby('model').head(8)); display(A('xper_values.csv')); display(json.loads((ROOT/'artifacts/interpretability_summary.json').read_text()))"),
    md("The approximate XPER contributions do not exactly reconstruct the sample AUC. Inspection of XPER 0.0.92 found that its kernel approximation uses unconstrained weighted regression without empty/full endpoint constraints, so exact reconstruction is not guaranteed. The residual remains visible rather than being presented as an exact decomposition."),
    code("diagnostics = ROOT / 'artifacts/xper_diagnostics.csv'\nif diagnostics.exists(): display(pd.read_csv(diagnostics))"),
    md("Logistic odds ratios below are per transformed unit (numeric inputs are standardized), not causal effects. Local sensitivity changes one field of the same explained person for all three models; it is not an additive attribution."),
    code("display(A('logistic_coefficients.csv')); display(A('local_sensitivity.csv'))"),
    md("## Stability\n\nAll models use the same eight bootstrap samples and fixed algorithm seeds to isolate training-data sensitivity. Sample hashes are in artifacts/stability_protocol.json. Jaccard is intersection/union, not the fraction of people retaining status. At equal selected-set size, replaced fraction is (1-J)/(1+J). Refits do not measure temporal stability; pairwise comparisons share refits."),
    code("stability = A('stability_summary.csv'); stability['selected_set_replacement'] = (1-stability.top20_jaccard)/(1+stability.top20_jaccard); display(stability)"),
    md("## Fairness\n\nThresholds change decisions, not calibration of unchanged probabilities. Equal FPR alone is not equalized odds. The group-threshold frontier optimizes using evaluation labels: an optimistic in-sample illustration, not validated mitigation. Removing race does not remove proxies or prove counterfactual fairness."),
    code("display(A('fairness_inference.csv')); display(A('intersectional_audit.csv')); display(A('race_ab_test.csv'))"),
    md("### Equivalence tests and course test table\n\nFor a support programme the harm is a missed offer, so FNR (equal opportunity) and selection rate (statistical parity) are primary; FPR is secondary. A difference test that fails to reject does not show fairness: `equivalent_within_delta` is a TOST at a pre-set ±5-point tolerance, and an interval that is neither significant nor equivalent is inconclusive. The course table gives p-values at the top-20% rule; conditional statistical parity conditions on the historical supervision score."),
    code("inf = A('fairness_inference.csv'); display(inf[(inf.rule == 'top_20pct') & inf.metric.isin(['fnr', 'selection_rate', 'fpr'])]); display(A('fairness_tests.csv')); display(A('fairness_age_bands.csv')); display(A('fairness_frontier.csv').query(\"method == 'group_thresholds_equal_fnr'\"))"),
    md("### Fairness interpretability (FPDP) and mitigation\n\nFPDP varies one input and re-evaluates the equal-opportunity test for logistic and XGBoost (TabICLv2 is excluded for compute cost). Candidate variables are diagnostic associations, not causes. The mitigation table reports effect sizes next to the p-values: group FNRs and their gap, captured events, and the protected group's mean score against its base rate. Under the top-20% rule the logistic FPDP is flat: fixing a feature to any constant shifts every logit equally, so the selected set does not depend on the value, and Panel B is labelled value-independent."),
    code("display(A('fairness_candidates.csv')); display(A('fairness_mitigation.csv'))"),
    md("""### Fairness findings at the top-20% rule

Each gap is classified with the pre-set ±5-point tolerance: **equivalent** (TOST rejects a gap of 5 points or more), **different** (the 95% bootstrap interval excludes 0 and TOST does not show equivalence), or **inconclusive** (neither). A gap can be both significant and equivalent: nonzero, but within the tolerance. The 90+ intervals in this section are not adjusted for multiple testing. The TOST uses an analytic variance and the intervals use the bootstrap, so borderline cases can disagree (e.g. logistic race predictive equality: chi-squared p ≈ 0.049, bootstrap interval includes 0)."""),
    code("""inf = A('fairness_inference.csv')
top = inf[(inf.rule == 'top_20pct') & inf.metric.isin(['selection_rate', 'fnr', 'fpr'])].copy()
top['status'] = ['equivalent' if e else 'different' if s else 'inconclusive'
                 for e, s in zip(top.equivalent_within_delta, top.significant)]
display(top.pivot_table(index=['attribute', 'metric'], columns='model', values='status', aggfunc='first'))
display(top.pivot_table(index=['attribute', 'metric'], columns='model', values='gap').round(3))"""),
    md("""- **Race:** selection-rate, FNR and FPR gaps are equivalent within ±5 points for all three models; the selection-rate ratio is 0.87–0.93. The ±5-point tolerance is absolute, so on a 20% selection rate it alone would admit a ratio near 0.78; the ratio is the stricter check. For XGBoost and TabICLv2 the selection and FPR gaps (about +2 points, Black minus White) are also significant: real but small, and in the direction of more support offered to Black people. Equivalence concerns allocation errors, not predictive quality: within-group AUC is lower for Black people (0.718–0.721 vs 0.746–0.748).
- **Gender:** women who are later re-arrested miss support more often (FNR gap M − F about −0.10 to −0.12, different for all models). Every model over-predicts women (mean score about 0.52 vs observed 0.45; ECE about 0.07 vs about 0.01–0.02 for men), yet women are selected about half as often at the top 20%. Sufficiency is also rejected for gender, but the two findings point in opposite directions: in a support programme over-prediction favours women, and recalibrating by gender would widen the FNR gap. The harm for this use is the FNR gap.
- **Age:** the largest disparity. The FNR gap (under 33 minus 33+) is about −0.23 to −0.25; about 97% of re-arrested people aged 48+ are not selected, vs about 49% at 18–22. Age is a model input and a validated risk factor, but in a support programme that choice needs an explicit justification (need vs risk). The FPDP finds no candidate variable for the age gap: with `Age_at_Release` fixed for everyone the test still rejects, because criminal-history inputs and the Georgia score carry age. Dropping age would not remove the gap.
- **FPDP (gender):** candidate variables are `Gang_Affiliated` and `Age_at_Release`. Gang affiliation is never recorded for women and is imputed as "No", so it acts as a gender-aligned measurement artefact (Cramér's V with gender = 1.0 on the raw field). Dropping it and re-estimating shrinks the gender FNR gap from −0.096 to 0.000 (logistic) and from −0.112 to −0.010 (XGBoost). Women's FNR falls about 8 points while men's rises about 2 (partly levelling down); captured events fall by 27–41, AUC by about 0.014 (five to six times the XGBoost − logistic gap), and women's over-prediction worsens. Statistical parity is still rejected.
- **Caveats:** candidate selection, the neutral value in Panel B and the mitigation are all evaluated on the same evaluation labels, so they are illustrative, not validated. A p-value above 0.05 after mitigation is not evidence of fairness; equivalence would be. Group-specific thresholds remain an analytic device only."""),
]
for name in ["performance_calibration", "incumbent_benchmark", "learning_curve", "shap_individual",
             "pdp_ice", "structural_stability", "fairness_support_access", "fairness_operating_point", "fairness_frontier", "fairness_dependence",
             "tradeoff_matrix"]:
    nb.cells.append(code(f"display(Image(filename=str(ROOT / 'artifacts/figures/{name}.png'), width=1000))"))
nb.cells.extend([
    md("## Recommendation\n\nPilot L1 logistic regression prospectively with XGBoost as the challenger. XGBoost's AUC edge is +0.0025 (paired interval 0.0006 to 0.0044); logistic is equal or better on native explanation, refit drift, gender and race gaps, and runtime, so the small accuracy gain does not justify giving those up. Small-sample results do not establish suitability for smaller agencies elsewhere. Require independent validation, benefit evidence, corrections/appeals and monitoring before real allocation. No adverse use."),
    code("display(json.loads((ROOT/'artifacts/validation_manifest.json').read_text()))"),
    md("## End-to-end audit\n\nA fresh three-model run exactly reproduced the published probabilities. See reports/end_to_end_review.md for the data boundaries, actual GPU checks and CPU/CUDA experiment. Holm correction of 54 course difference tests reduces rejections from 41 to 32; no race test survives, while gender and age equal-opportunity differences remain. This does not adjust TOST or mitigation selection."),
    code("adjusted = ROOT / 'artifacts/end_to_end/fairness_tests_holm.csv'\nif adjusted.exists(): display(pd.read_csv(adjusted))"),
    md("## Submission checklist\n\n- Binary target and client decision defined\n- White-box, machine-learning and tabular-foundation model compared\n- Statistical and economic performance assessed\n- Local/global interpretability, structural stability and subgroup fairness analyzed\n- Trustworthy-AI trade-offs and a logistic-regression shadow-pilot recommendation stated\n- Executed outputs included; no code-cell errors\n- Interactive application in `app.py`; presentation in `reports/ISAF_Recidivism_Presentation.pptx`\n\nReferences and requirement coverage: `reports/research_review.md`. Detailed methodological limits: `reports/technical_report.md`. End-to-end reproduction evidence: `reports/end_to_end_review.md`."),
])
NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
for out in OUTS:
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print(out)
