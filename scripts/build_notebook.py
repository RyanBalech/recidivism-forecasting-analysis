from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "recidivism_analysis.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def fig(name, width=1000):
    return code(f"display(Image(filename=str(FIG / '{name}'), width={width}))")


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["cells"] = [
    md("""# Trustworthy recidivism forecasting

**Client:** a software vendor selling risk-assessment tools to US state community-supervision
agencies. **Task:** pick and justify a three-year re-arrest score to embed in its product, judged on
predictive performance (statistical + economic), interpretability, stability, and fairness — a
trustworthy AI system, not a leaderboard.

Models compared: logistic regression (white-box), XGBoost (ML), TabICLv2 (tabular foundation model).
The outcome is *arrest*, not inherent criminality; this analysis is unsuitable for adverse decisions,
and the score is scoped to allocating voluntary support only."""),
    code("""from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / 'artifacts').exists(): ROOT = ROOT.parent
FIG = ROOT / 'artifacts' / 'figures'
sys.path.insert(0, str(ROOT / 'src'))
from recidivism.data import load_official_split

split = load_official_split(ROOT / 'nij-challenge2021_full_dataset.csv')
A = lambda name: pd.read_csv(ROOT / 'artifacts' / name)
metrics = A('model_metrics.csv')
predictions = A('test_predictions.csv')
print(f'Train: {len(split.X_train):,} | Test: {len(split.X_test):,} | Features: {split.X_train.shape[1]}')
print(f'Base recidivism rate (test): {predictions.actual.mean():.1%}')"""),
    md("""## Design and leakage control

The official `Training_Sample` flag defines the untouched 18,028 / 7,807 split. Only fields available
at supervision start are eligible; post-release variables are excluded because they accrue after the
scoring moment. Race, gender and Residence PUMA (a race proxy) are excluded from inputs and kept for
audit only. Ordered counts are ordinal-encoded; XGBoost hyperparameters come from a 5-fold CV search
(`scripts/tune_xgboost.py`). Full pipeline in `scripts/train_evaluate.py`.

Set `RUN_TRAINING = True` to rebuild every artifact (TabICLv2 runs on GPU at 16 estimators; a CUDA GPU is recommended)."""),
    code("""RUN_TRAINING = False
if RUN_TRAINING:
    import subprocess
    for s in ['train_evaluate','incumbent_benchmark','learning_curve','fairness_audit',
              'interpretability','stability_structural','race_ab_test','tradeoff_matrix']:
        subprocess.run([sys.executable, str(ROOT / f'scripts/{s}.py')], cwd=ROOT, check=True)"""),
    md("""## 1. The client's real question: better than the incumbent?

`Supervision_Risk_Score_First` is Georgia's existing 1–10 actuarial tool, already in the data. We
benchmark it against our models and random allocation. (145 test rows miss the score; imputed at the
median.)"""),
    code("pd.merge(A('incumbent_discrimination.csv'), A('incumbent_economics.csv')[['ranker','assumed_net_value']], on='ranker').round(3)"),
    fig("incumbent_benchmark.png", 1100),
    md("The incumbent reaches only ~0.60 AUC; every model reaches ~0.73 and roughly doubles net value. "
       "This is the headline for the client — the choice among our three models is secondary."),
    md("## 2. Predictive performance (statistical)"),
    code("metrics[['model','roc_auc','average_precision','brier','log_loss','ece_10','fit_predict_seconds']].round(4).sort_values('brier')"),
    fig("performance_calibration.png"),
    code("""intervals = json.loads((ROOT / 'artifacts/bootstrap_intervals.json').read_text())
pd.concat({m: pd.DataFrame(v).T for m, v in intervals.items()}, names=['model','metric']).round(4)"""),
    md("TabICLv2 and XGBoost tie on discrimination and Brier (AUC 0.733 vs 0.733); XGBoost calibrates best (ECE). "
       "Bootstrap 95% intervals overlap, so the ranking among the three is not decisive."),
    md("""### Learning curve: which model for which agency size?

Train sizes 1,500 / 5,000 / 10,000 (3 seeds each) plus the full 18,028 (single seed), all scored on
the fixed test set."""),
    code("A('learning_curve.csv').groupby(['model','n_train']).roc_auc.agg(['mean','std']).round(4)"),
    fig("learning_curve.png", 1100),
    md("TabICLv2's few-shot advantage is real on small data (a small county) and shrinks to +0.004 AUC "
       "at full data (a large state). No crossover — but at scale the cheaper, explainable model suffices."),
    md("## 3. Economic performance"),
    code("""from recidivism.metrics import economic_value
pd.DataFrame([{'model': m, **economic_value(predictions.actual, predictions[f'p_{m}'])}
              for m in ['logistic','xgboost','tabicl']]).round(3)"""),
    md("A transparent scenario ($5k support, $50k event, 20% effectiveness), not a causal estimate. The "
       "app exposes a sensitivity sweep; models beat the incumbent at every capacity from 5% to 50%."),
    md("## 4. Interpretability (all three models)"),
    fig("shap_global.png"),
    fig("shap_individual.png"),
    md("SHAP: global drivers and an individual waterfall for logistic and XGBoost. Age at release, gang "
       "affiliation, prior felony arrests and prison tenure recur. A LIME explanation of the same person "
       "(`artifacts/lime_individual.csv`) tells a consistent local story via a different mechanism."),
    md("**XPER** (Hué–Hurlin–Pérignon–Saurin) decomposes the model's *AUC* into feature contributions — "
       "performance attribution, complementary to SHAP's prediction attribution."),
    code("A('xper_values.csv').query(\"~feature.str.startswith('benchmark')\", engine='python').sort_values('xper', ascending=False).groupby('model').head(6).round(4)"),
    fig("xper.png", 1100),
    md("**Global surrogate**: a depth-3 tree mimics XGBoost (test fidelity R² below) — enough to narrate "
       "the logic, not to replace the model. **PDP/ICE** cover all three models, including TabICLv2, which "
       "has no native attribution path (a deployment cost, stated openly)."),
    code("json.loads((ROOT / 'artifacts/interpretability_summary.json').read_text())"),
    fig("global_surrogate.png", 1100),
    fig("pdp_ice.png", 1100),
    md("## 5. Stability (structural)"),
    code("A('stability_summary.csv').round(4)"),
    fig("structural_stability.png", 1100),
    md("Each model refit on 8 bootstrap resamples of the training data. All three are comparably stable "
       "(drift ~0.034-0.036, decision overlap ~76-77%). About one person in four changes priority status "
       "across refits, so scores need governance. Event dates are unavailable, so temporal stability is a "
       "deployment gate on a later cohort."),
    md("## 6. Fairness"),
    md("### Gaps at the deployed operating point, with inference tests\n"
       "The product allocates the top 20% by risk, so we audit there (and at 0.5) with bootstrap 95% CIs."),
    code("""inf = A('fairness_inference.csv')
inf[(inf.rule=='top_20pct') & inf.metric.isin(['fpr','tpr','selection_rate'])].round(3).sort_values(['attribute','metric','model'])"""),
    fig("fairness_operating_point.png", 1100),
    md("Auditing at 0.5 would overstate gaps ~2–3× (TabICLv2 gender 0.106 at 0.5 vs 0.041 at top-20%)."),
    md("### The impossibility result, split by attribute"),
    code("A('fairness_impossibility.csv').round(3)"),
    md("""Race base rates barely differ (0.582 vs 0.564) → calibration and equal error rates are
near-jointly achievable, so the race FPR gap is a *model property*. Gender base rates differ by 13.7
points (0.591 vs 0.454) → the theorem binds; because race is excluded, logistic/XGBoost over-predict
women. This explains why gender gaps exceed race gaps."""),
    md("### Mitigation frontier (race and gender)\n"
       "Group-specific thresholds trace the fairness/utility frontier. They are an **analytic device**, "
       "not a shipping option — a per-race/gender threshold is disparate treatment (Ricci v. DeStefano)."),
    code("A('fairness_frontier.csv').query(\"method=='group_thresholds_equal_fpr'\").round(4)"),
    fig("fairness_frontier.png", 1100),
    md("Group thresholds drive both gaps to ~0 keeping nearly all captured events — but equalizing "
       "*gender* decalibrates women (base-rate gap 0.137), the trade-off the theorem forces."),
    md("### A/B test: does race add anything?"),
    code("A('race_ab_test.csv').round(4)"),
    md("Adding race changes AUC by ≤0.0007 for every model but makes identical twins score differently "
       "(up to 4.8 pts for XGBoost). Removing race is free and guarantees identical twins — though "
       "TabICLv2's race gap rises slightly, evidence of proxy leakage (the limit of unawareness)."),
    md("## 7. Trade-offs and recommendation"),
    fig("tradeoff_matrix.png", 1100),
    md("""Performance is a near-tie, so the decision turns on interpretability, fairness, stability and
cost. **Deploy XGBoost** (best calibration and net value, smallest gender gap, SHAP-explainable, ~10×
faster than the TFM), **logistic as transparent challenger**, **TabICLv2 only for very small
agencies** where its few-shot edge is real. Any deployment is benefit-only, with a prospective pilot,
an appeal route, quarterly subgroup audits at the deployed operating point, and stop rules."""),
]

OUT.parent.mkdir(exist_ok=True)
NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, OUT)
print(OUT)
