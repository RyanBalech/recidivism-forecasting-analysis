"""Build and execute the evidence notebook from canonical artifacts."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "recidivism_analysis.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells = [
    md("# Trustworthy recidivism forecasting\n\nClient: a community-supervision software vendor. Compare logistic regression, XGBoost and TabICLv2 for voluntary support prioritization. Target: three-year cumulative new arrest. This differs from NIJ annual conditional forecasting; challenge leaderboard comparisons are invalid."),
    code("""from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import Image, display
ROOT = Path.cwd()
if not (ROOT / 'artifacts').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from recidivism.data import load_official_split
split = load_official_split()
A = lambda name: pd.read_csv(ROOT / 'artifacts' / name)
print(f'Training: {len(split.X_train):,}; evaluation: {len(split.X_test):,}; features: {split.X_train.shape[1]}')
"""),
    md("## Reproduce\n\nRun scripts/reproduce.py for the complete pipeline. The evaluation partition is excluded from fitting, but was repeatedly inspected during development. It is not an untouched final holdout. This notebook executes the reporting layer; source scripts provide all fitting/audits."),
    md("## Data preparation\n\nOnly baseline fields are eligible. Outcomes, IDs, split flags, protected attributes, geography and post-release supervision activities are excluded from model inputs. Imputation and scaling are learned within training pipelines. Logistic uses tuned L1 with one-hot categories; XGBoost ordinal-encodes ordered fields; TabICL mode-imputes categorical missingness before its mixed-data encoder. This neutralizes the gender-aligned `Gang_Affiliated` missingness channel; `scripts/leakage_audit.py` enforces the check."),
    code("""from recidivism.config import FEATURE_COLUMNS, BASELINE_COLUMNS, DATA_PATH
raw = pd.read_csv(DATA_PATH)
display(pd.Series({'training_target_rate': split.y_train.mean(), 'evaluation_target_rate': split.y_test.mean()}))
display(pd.DataFrame({'feature': raw.columns, 'used_for_scoring': [c in FEATURE_COLUMNS for c in raw.columns]}))
display(raw.assign(gang_missing=raw.Gang_Affiliated.isna()).groupby('Gender').gang_missing.agg(['mean','sum','count']))
display(split.X_train.isna().mean().sort_values(ascending=False).head(10))
"""),
    md("## Performance and baselines\n\nThe incumbent is a historical recorded score, not evidence about tools agencies use today. Constant prevalence and incumbent calibration are learned on training records."),
    code("display(A('model_metrics.csv')); display(A('validation_baselines.csv'))"),
    md("## Foundation-model compute sensitivity\n\nThe 16-member TabICLv2 configuration is checked on a fixed stratified development slice carved only from the training partition. This is a compute/robustness sensitivity check, not final validation."),
    code("display(A('estimator_sweep.csv')); display(Image(filename=str(ROOT / 'artifacts/figures/estimator_sweep.png'), width=900))"),
    md("## Paired uncertainty and training CV\n\nDifferences are A minus B: positive AUC favors A, negative Brier favors A. Paired bootstrap preserves the correlation between model errors. Marginal intervals are exploratory and unadjusted for development selection or multiple comparisons. Fixed-configuration CV is not nested evaluation of the earlier search."),
    code("display(A('paired_comparisons.csv')); display(A('validation_cv.csv').groupby('model')[['roc_auc','brier']].agg(['mean','std']))"),
    md("## Economic scenarios\n\nEffectiveness and costs are assumed, not estimated causal savings. Exact capacity uses stable row-order ties; this matters particularly for the discrete incumbent."),
    code("display(A('incumbent_economics.csv'))"),
    md("## Interpretability\n\nSHAP and LIME explain conventional models; XPER approximates performance attribution on a small sample. Surrogate fidelity is imperfect. TabICLv2 has PDP/ICE, permutation and interactive sensitivity, but no implemented native additive attribution. These are model-response explanations, not causal effects."),
    code("display(A('shap_importance.csv').groupby('model').head(8)); display(A('xper_values.csv')); display(json.loads((ROOT/'artifacts/interpretability_summary.json').read_text()))"),
    md("Logistic odds ratios below are per transformed unit (numeric inputs are standardized), not causal effects. Local sensitivity changes one field of the same explained person for all three models; it is not an additive attribution."),
    code("display(A('logistic_coefficients.csv')); display(A('local_sensitivity.csv'))"),
    md("## Stability\n\nJaccard is intersection/union, not the fraction of people retaining status. At equal selected-set size, replaced fraction is (1-J)/(1+J). Refits do not measure temporal stability; pairwise comparisons share refits."),
    code("stability = A('stability_summary.csv'); stability['selected_set_replacement'] = (1-stability.top20_jaccard)/(1+stability.top20_jaccard); display(stability)"),
    md("## Fairness\n\nThresholds change decisions, not calibration of unchanged probabilities. Equal FPR alone is not equalized odds. The group-threshold frontier optimizes using evaluation labels: an optimistic in-sample illustration, not validated mitigation. Removing race does not remove proxies or prove counterfactual fairness."),
    code("display(A('fairness_inference.csv')); display(A('intersectional_audit.csv')); display(A('race_ab_test.csv'))"),
    md("### Equivalence tests and course test table\n\nFor a support programme the harm is a missed offer, so FNR (equal opportunity) and selection rate (statistical parity) are primary; FPR is secondary. A difference test that fails to reject does not show fairness: `equivalent_within_delta` is a TOST at a pre-set ±5-point tolerance, and an interval that is neither significant nor equivalent is inconclusive. The course table gives p-values at the top-20% rule; conditional statistical parity conditions on the historical supervision score."),
    code("inf = A('fairness_inference.csv'); display(inf[(inf.rule == 'top_20pct') & inf.metric.isin(['fnr', 'selection_rate', 'fpr'])]); display(A('fairness_tests.csv')); display(A('fairness_age_bands.csv')); display(A('fairness_frontier.csv').query(\"method == 'group_thresholds_equal_fnr'\"))"),
    md("### Fairness interpretability (FPDP) and mitigation\n\nFPDP varies one input and re-evaluates the equal-opportunity test for logistic and XGBoost (TabICLv2 is excluded for compute cost). Candidate variables are diagnostic associations, not causes; removal and re-estimation report the AUC cost alongside the fairness change."),
    code("display(A('fairness_candidates.csv')); display(A('fairness_mitigation.csv'))"),
    md("""### Fairness findings at the top-20% rule

Each gap is classified with the pre-set ±5-point tolerance: **equivalent** (TOST rejects a gap of 5 points or more), **different** (the 95% bootstrap interval excludes 0 and TOST does not show equivalence), or **inconclusive** (neither). A gap can be both significant and equivalent: nonzero, but within the tolerance. The 90+ intervals in this section are not adjusted for multiple testing. The TOST uses an analytic variance and the intervals use the bootstrap, so borderline cases can disagree (e.g. logistic race predictive equality: chi-squared p ≈ 0.049, bootstrap interval includes 0)."""),
    code("""inf = A('fairness_inference.csv')
top = inf[(inf.rule == 'top_20pct') & inf.metric.isin(['selection_rate', 'fnr', 'fpr'])].copy()
top['status'] = ['equivalent' if e else 'different' if s else 'inconclusive'
                 for e, s in zip(top.equivalent_within_delta, top.significant)]
display(top.pivot_table(index=['attribute', 'metric'], columns='model', values='status', aggfunc='first'))
display(top.pivot_table(index=['attribute', 'metric'], columns='model', values='gap').round(3))"""),
    md("""- **Race:** selection-rate, FNR and FPR gaps are equivalent within ±5 points for all three models. For XGBoost and TabICLv2 the selection and FPR gaps (about +2 points, Black minus White) are also significant: real but small, and in the direction of more support offered to Black people.
- **Gender:** women who are later re-arrested miss support more often (FNR gap M − F about −0.10 to −0.12, different for all models). Every model over-predicts women (mean score about 0.52 vs observed 0.45; ECE about 0.07 vs about 0.01–0.02 for men), yet women are selected less at the top 20%. Sufficiency is also rejected for gender.
- **Age:** the largest disparity. The FNR gap (under 33 minus 33+) is about −0.23 to −0.25; about 97% of re-arrested people aged 48+ are not selected, vs about 49% at 18–22. Age is a model input and a validated risk factor, but in a support programme that choice needs an explicit justification (need vs risk).
- **FPDP (gender):** candidate variables are `Gang_Affiliated` and `Age_at_Release`. Gang affiliation is never recorded for women and is imputed as "No", so it acts as a gender-aligned measurement artefact (Cramér's V with gender = 1.0 on the raw field). Dropping it and re-estimating removes the equal-opportunity rejection (logistic p 0.000 → 0.995; XGBoost → 0.67) at about −0.014 AUC, roughly six times the XGBoost − logistic gap. Statistical parity is still rejected.
- **Caveats:** candidate selection, the neutral value in Panel B and the mitigation are all evaluated on the same evaluation labels, so they are illustrative, not validated. A p-value above 0.05 after mitigation is not evidence of fairness; equivalence would be. Group-specific thresholds remain an analytic device only."""),
]
for name in ["performance_calibration", "incumbent_benchmark", "learning_curve", "shap_individual",
             "pdp_ice", "structural_stability", "fairness_support_access", "fairness_operating_point", "fairness_frontier", "fairness_dependence",
             "tradeoff_matrix"]:
    nb.cells.append(code(f"display(Image(filename=str(ROOT / 'artifacts/figures/{name}.png'), width=1000))"))
nb.cells.extend([
    md("## Recommendation\n\nPilot XGBoost prospectively with logistic as a transparent challenger. Weigh errors, explanation cost, refit stability and runtime together. Small-sample results do not establish suitability for smaller agencies elsewhere. Require independent validation, benefit evidence, corrections/appeals and monitoring before real allocation. No adverse use."),
    code("display(json.loads((ROOT/'artifacts/validation_manifest.json').read_text()))"),
    md("References and requirement coverage: reports/research_review.md. Detailed methodological limits: reports/technical_report.md."),
])
OUT.parent.mkdir(exist_ok=True)
NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, OUT)
print(OUT)
