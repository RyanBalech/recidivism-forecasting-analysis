from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "recidivism_analysis.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["cells"] = [
    md("""# Trustworthy recidivism forecasting

**Question:** At supervision start, which people are most likely to have a new arrest within three years, so a hypothetical agency can prioritize voluntary support?

This notebook compares logistic regression, XGBoost, and TabICLv2 across predictive and economic performance, interpretability, stability, and fairness. The outcome is arrest, not inherent criminality; this analysis is unsuitable for adverse decisions."""),
    code("""from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / 'artifacts').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from recidivism.data import load_official_split
from recidivism.metrics import economic_value

split = load_official_split(ROOT / 'nij-challenge2021_full_dataset.csv')
metrics = pd.read_csv(ROOT / 'artifacts/model_metrics.csv')
fairness = pd.read_csv(ROOT / 'artifacts/fairness_by_group.csv')
gaps = pd.read_csv(ROOT / 'artifacts/fairness_gaps.csv')
importance = pd.read_csv(ROOT / 'artifacts/permutation_importance.csv')
predictions = pd.read_csv(ROOT / 'artifacts/test_predictions.csv')
print(f'Train: {len(split.X_train):,} | Test: {len(split.X_test):,} | Features: {split.X_train.shape[1]}')"""),
    md("""## Design and leakage control

The official `Training_Sample` flag defines the untouched test set. Only fields available when supervision begins are eligible. Post-release violations, tests, programs, employment, and residence changes are excluded. Race, gender, and Residence PUMA are excluded from scoring; race and gender remain in the audit layer.

The white-box pipeline imputes and one-hot encodes fields, then fits regularized logistic regression. XGBoost uses the same prepared matrix. TabICLv2 receives deterministic one-column-per-field encoding and applies pretrained in-context inference. The complete training and audit implementation is in `scripts/train_evaluate.py`."""),
    code("""# Change to True to rebuild every artifact (a CUDA GPU is recommended for TabICLv2).
RUN_TRAINING = False
if RUN_TRAINING:
    import subprocess
    subprocess.run([sys.executable, str(ROOT / 'scripts/train_evaluate.py')], cwd=ROOT, check=True)"""),
    md("## Predictive performance"),
    code("""cols = ['model','roc_auc','average_precision','brier','log_loss','ece_10','fit_predict_seconds']
metrics[cols].round(4).sort_values('brier')"""),
    code("display(Image(filename=str(ROOT / 'artifacts/figures/performance_calibration.png'), width=1000))"),
    md("TabICLv2 has the best held-out Brier score and discrimination, but its gain over XGBoost is small. XGBoost has the lowest calibration error. Overlapping bootstrap intervals caution against treating the rank order as decisive."),
    code("""intervals = json.loads((ROOT / 'artifacts/bootstrap_intervals.json').read_text())
pd.concat({m: pd.DataFrame(v).T for m, v in intervals.items()}, names=['model','metric']).round(4)"""),
    md("## Economic scenario"),
    code("""rows = []
for model in ['logistic','xgboost','tabicl']:
    rows.append({'model': model, **economic_value(predictions.actual, predictions[f'p_{model}'], capacity=.20, intervention_cost=5_000, event_cost=50_000, effectiveness=.20)})
pd.DataFrame(rows).round(3)"""),
    md("The dollar result is a transparent scenario, not a causal estimate. The observational data cannot show whether a specific intervention prevents arrests."),
    md("## Interpretability"),
    code("display(Image(filename=str(ROOT / 'artifacts/figures/feature_importance.png'), width=1100))"),
    md("Permutation importance measures the held-out increase in Brier loss when a field is shuffled. Age at release, gang affiliation, prior felony arrests, and prison tenure recur across models. These associations are not causal explanations."),
    md("## Fairness"),
    code("gaps.round(4).sort_values(['attribute','fpr_gap'])"),
    code("display(Image(filename=str(ROOT / 'artifacts/figures/race_fairness.png'), width=1000))"),
    md("At threshold 0.5, XGBoost has the smallest observed race and gender false-positive-rate gaps. Fairness depends on the operating threshold and use case. Attribute exclusion does not remove proxy, label, or historical bias."),
    md("## Stability"),
    code("""stability = json.loads((ROOT / 'artifacts/stability.json').read_text())
pd.DataFrame(stability).T.round(4)"""),
    md("""The models have similar bootstrap interval widths. A temporal stability test is impossible because no suitable cohort date is available; later-cohort validation is a deployment gate.

## Recommendation

Pilot **XGBoost** for benefit-only service allocation, with logistic regression as a transparent challenger. It is nearly tied on predictive performance, calibrates best, runs cheaply, and has smaller observed subgroup gaps. Keep the system in shadow mode until a prospective study demonstrates benefit without unacceptable subgroup harm."""),
]

OUT.parent.mkdir(exist_ok=True)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, OUT)
print(OUT)

