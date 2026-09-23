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
    md("## Data preparation\n\nOnly baseline fields are eligible. Outcomes, IDs, split flags, protected attributes, geography and post-release supervision activities are excluded from model inputs. Imputation and scaling are learned within training pipelines. Logistic uses tuned L1 with one-hot categories; XGBoost ordinal-encodes ordered fields; TabICL mode-imputes categorical missingness before its mixed-data encoder."),
    code("""from recidivism.config import FEATURE_COLUMNS, BASELINE_COLUMNS, DATA_PATH
raw = pd.read_csv(DATA_PATH)
display(pd.Series({'training_target_rate': split.y_train.mean(), 'evaluation_target_rate': split.y_test.mean()}))
display(pd.DataFrame({'feature': raw.columns, 'used_for_scoring': [c in FEATURE_COLUMNS for c in raw.columns]}))
display(raw.assign(gang_missing=raw.Gang_Affiliated.isna()).groupby('Gender').gang_missing.agg(['mean','sum','count']))
display(split.X_train.isna().mean().sort_values(ascending=False).head(10))
"""),
    md("## Performance and baselines\n\nThe incumbent is a historical recorded score, not evidence about tools agencies use today. Constant prevalence and incumbent calibration are learned on training records."),
    code("display(A('model_metrics.csv')); display(A('validation_baselines.csv'))"),
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
]
for name in ["performance_calibration", "incumbent_benchmark", "learning_curve", "shap_individual",
             "pdp_ice", "structural_stability", "fairness_support_access", "fairness_operating_point", "fairness_frontier", "tradeoff_matrix"]:
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
