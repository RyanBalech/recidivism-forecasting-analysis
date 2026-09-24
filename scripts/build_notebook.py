"""Build and execute the professor-facing submission notebook.

The notebook is the readable entry point for the complete project. Heavy fitting
remains in versioned modules/scripts so the notebook and application use exactly
the same implementation; a visible switch in the notebook runs the whole pipeline.
"""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
# One canonical notebook. A second copy only ever drifts from this one.
OUTS = [ROOT / "Recidivism_Project_Submission.ipynb"]
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
    md("""### Train and evaluate here, in this notebook

The configurations above are not just described. The next cell fits the two
conventional models on the 18,028 training records and scores the 7,807 held-out
records inside this notebook. TabICLv2 is excluded from this cell only because it
needs a GPU; its published predictions are read from `test_predictions.csv`.

Two different things are being checked, and they have different tolerances:

1. **Artifact integrity** — do the *saved* models still produce the published
   probabilities? This must hold exactly, otherwise every number below is stale.
   It is an assertion.
2. **Fit reproducibility** — does fitting again from scratch land in the same
   place? For logistic regression, yes, to machine precision. For XGBoost, *not
   bit-identically*: histogram splits are accumulated in a thread-dependent order,
   so a different core count changes the last bits of a split gain and occasionally
   flips a tie. Performance is reproducible; individual probabilities are only
   reproducible to ~1e-2. This is a real property of the model worth knowing before
   a client asks why two runs disagree, so it is reported rather than hidden."""),
    code("""from recidivism.modeling import logistic_model, xgboost_model
from recidivism.metrics import classification_metrics
from recidivism.config import MODEL_DIR
import joblib, time

published = A('test_predictions.csv')

# --- 1. Artifact integrity: saved models must reproduce the published file ---
integrity = pd.DataFrame([
    {'model': label,
     'max_abs_difference': float(np.abs(
         joblib.load(MODEL_DIR / f'{label}.joblib').predict_proba(split.X_test)[:, 1]
         - published[f'p_{label}']).max())}
    for label in ['logistic', 'xgboost']
])
display(integrity)
assert (integrity.max_abs_difference < 1e-6).all(), 'saved models no longer match the published predictions'
print('Artifact integrity OK: saved models reproduce the published probabilities.\\n')

# --- 2. Fit reproducibility: refit from scratch and report how close it lands ---
rows = []
for label, builder in [('logistic', logistic_model), ('xgboost', xgboost_model)]:
    start = time.perf_counter()
    refit = builder(split.X_train).fit(split.X_train, split.y_train)
    p = refit.predict_proba(split.X_test)[:, 1]
    rows.append({
        'model': label,
        'fit_seconds': round(time.perf_counter() - start, 1),
        'auc_refit': classification_metrics(split.y_test, p)['roc_auc'],
        'auc_published': classification_metrics(published.actual, published[f'p_{label}'])['roc_auc'],
        'max_abs_probability_difference': float(np.abs(p - published[f'p_{label}']).max()),
    })
reproducibility = pd.DataFrame(rows)
reproducibility['auc_difference'] = reproducibility.auc_refit - reproducibility.auc_published
display(reproducibility)
assert reproducibility.auc_difference.abs().max() < 1e-3, 'refit performance drifted materially'
print('Refit performance matches to <0.001 AUC for both models.')
"""),
    md("""### What each metric means

Before the numbers, what they measure and which direction is good:

| Metric | Question it answers | Better |
|---|---|---|
| **ROC AUC** | Given one person who was re-arrested and one who was not, how often does the model score the first higher? 0.5 = coin flip. | higher |
| **Average precision** | Area under precision–recall; sensitive to performance on the positive class. | higher |
| **Brier** | Mean squared error of the probability itself. Rewards being both discriminating *and* honest about uncertainty. | lower |
| **Log loss** | Like Brier but punishes confident mistakes far harder. | lower |
| **ECE (10 bins)** | Of the people scored ≈0.7, were ≈70% actually re-arrested? Pure calibration, ignores ranking. | lower |
| **FNR** | Of people who were re-arrested, the share the rule did *not* select. For a support programme this is the harm: a missed offer. | lower |
| **Brier skill** | Brier improvement over always predicting the training prevalence. 0 = no better than the base rate. | higher |

For this product the ranking metrics (AUC, AP) decide *who* gets offered support at
a fixed capacity, and calibration (Brier, ECE) decides whether the score can be
spoken about honestly to a supervisee. FNR is the fairness-relevant one, because
the cost of the model being wrong falls on the person who needed support."""),
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
    md("""### Reading the model in probabilities, not log-odds

A coefficient of 0.087 per standardized unit is not something a client can act on.
Two readings in the units the decision actually uses (course reference: printed
slides 33–38):

- **Average marginal effect** — the mean change in predicted probability per one
  *raw* unit, computed by finite difference through the whole fitted pipeline.
- **Average probability contrast** — set a categorical field to one level for
  everybody, then to the reference level for everybody, and average the difference.
  This is valid because each counterfactual row carries exactly one real category;
  perturbing one-hot columns independently would create people who are simultaneously
  two age bands, or none.

Both are computed for the linear and the nonlinear model, so the same quantity can
be compared across them."""),
    code("""ame = A('logistic_marginal_effects.csv')
contrasts = A('probability_contrasts.csv')
display(ame.sort_values('average_marginal_effect', key=abs, ascending=False).head(8))
top_contrasts = contrasts[contrasts.model == 'logistic'].copy()
display(top_contrasts.reindex(top_contrasts.average_probability_contrast.abs()
                              .sort_values(ascending=False).index).head(10))
display(Image(filename=str(ROOT / 'artifacts/figures/marginal_effects.png'), width=1100))
"""),
    md("""Read the contrasts as the client would: moving a supervisee from the 23–27 band to
48-or-older lowers predicted three-year arrest risk by about 27 percentage points,
holding everything else fixed. A recorded gang affiliation raises it by about 17.
These are model responses under a counterfactual edit, not causal effects of ageing
or of gang membership."""),
    md("""### Explanations disagree, and the disagreement is the finding

SHAP, permutation importance and XPER are routinely quoted as if interchangeable.
They are not, and the course treats the discrepancy as the point (printed slides
175–181):

- **SHAP** attributes the *prediction* — what moved this score away from the average
  score. No outcome label is involved.
- **Permutation importance** attributes *loss* — how much worse the model scores when
  a feature is made uninformative. It needs labels, and it charges a feature for being
  predictive rather than for being used.
- **XPER** attributes *AUC*, decomposing a performance metric per feature. Its
  reconstruction residual is printed above: the decomposition is approximate, and the
  residual bounds how literally it can be read.

So a feature can lead one ranking and sit mid-table in another without either being
wrong. The table below measures the disagreement instead of asserting it."""),
    code("""agreement = A('explanation_agreement.csv')
display(agreement[['model','method_a','method_b','spearman_rank_correlation','top10_overlap',
                   'top10_only_a','top10_only_b']])
display(Image(filename=str(ROOT / 'artifacts/figures/explanation_agreement.png'), width=1200))
"""),
    md("""SHAP and permutation importance agree strongly on ranking (Spearman ≈ 0.77–0.81);
XPER diverges (≈ 0.53–0.60), which is expected because it decomposes a rank-based
performance metric rather than individual scores. The *top-10 sets* largely coincide
(8–10 shared features), so the headline drivers are robust while their ordering is
method-dependent. Quote the set, not the rank."""),
    md("""### LIME, and whether its explanation is faithful

An explanation that does not approximate the model locally is not evidence about the
model. The course makes fidelity the criterion (printed slides 110–116), so it is
reported here beside every explanation rather than left implicit.

The earlier implementation explained the *transformed* one-hot space. Perturbing those
columns independently produces rows no person could occupy, so the local surrogate was
fitted on a region the model never sees and scored R² ≈ 0.25. LIME now works on the raw
feature space with `categorical_features` declared, so a perturbation swaps a category
for another real category. Fidelity roughly doubles, to R² ≈ 0.41–0.43.

Three cases fixed by rule before any explanation was inspected — highest, median and
lowest predicted risk — each explained under three seeds, so the reported weights carry
a visible sampling spread."""),
    code("""fidelity = A('lime_fidelity.csv')
display(fidelity.groupby(['model','case']).local_r2.agg(['mean','min','max']).round(3))
spread = A('lime_seed_spread.csv')
print(f"sign-stable conditions across seeds: {int(spread.sign_stable.sum())} / {len(spread)}")
display(spread[spread.model == 'xgboost'].sort_values('mean_weight', key=abs, ascending=False).head(10))
display(Image(filename=str(ROOT / 'artifacts/figures/lime_individual.png'), width=1200))
"""),
    md("""Fidelity of about 0.41 is honest rather than impressive: a linear surrogate can only
partly track a 1,196-tree model in a 29-feature neighbourhood. Every condition keeps its
sign across seeds, so the *direction* of each contribution is stable even where the
magnitude moves. Quote LIME directionally, and quote SHAP when a magnitude is needed."""),
    md("""### One person, start to finish

Everything above is cohort-level. This section follows a single held-out individual
through the whole pipeline — raw record, what preprocessing does to it, what each of the
three models predicts, what the explanations say, and finally whether the deployed rule
offers them support. This is the path a caseworker would actually traverse."""),
    code("""from recidivism.metrics import capacity_selection
preds = A('test_predictions.csv')
anchor = preds.p_xgboost.to_numpy()
person_idx = int(np.argsort(anchor)[len(anchor) // 2])   # the median-risk person
person_raw = split.X_test.iloc[[person_idx]]
person_id = int(split.audit_test.iloc[person_idx]['ID'])

print(f'Record ID {person_id} — the median-risk person in the evaluation cohort')
display(person_raw.T.rename(columns={person_idx: 'raw value'}))
""" ),
    md("**Step 1 — what preprocessing does to this record.** The raw row is categorical and has missing values. The fitted pipeline imputes, ordinal-encodes ordered counts, one-hot encodes the rest and standardizes numerics. The model never sees the row above; it sees the row below."),
    code("""pipe = logistic_model(split.X_train).fit(split.X_train, split.y_train)
transformed = pipe[:-1].transform(person_raw)
transformed = transformed.toarray() if hasattr(transformed, 'toarray') else transformed
names = list(pipe[-2].get_feature_names_out())
nonzero = pd.Series(transformed[0], index=names)
display(nonzero[nonzero != 0].sort_values(key=abs, ascending=False).head(12).to_frame('encoded value'))
print(f'{len(names)} transformed columns; {int((nonzero != 0).sum())} non-zero for this person')
"""),
    md("**Step 2 — what the three models say.** Same person, three model families. Agreement here is evidence that the score is a property of the record rather than of one estimator."),
    code("""display(pd.DataFrame([{
    'model': m,
    'predicted_probability': float(preds.loc[person_idx, f'p_{m}']),
} for m in ['logistic','xgboost','tabicl']]).assign(
    cohort_base_rate=preds.actual.mean(),
    actually_rearrested=bool(preds.loc[person_idx, 'actual'])))
"""),
    md("**Step 3 — why.** The LIME conditions for this person, with their fidelity, and the marginal-effect reading of the same fields."),
    code("""case = A('lime_weights.csv')
case = case[(case.model == 'xgboost') & (case.case == 'median_risk')]
display(case.groupby('condition').weight.agg(['mean','std'])
        .sort_values('mean', key=abs, ascending=False).head(8))
r2 = A('lime_fidelity.csv').query("model == 'xgboost' and case == 'median_risk'").local_r2
print(f'local surrogate fidelity for this explanation: R2 = {r2.mean():.3f}')
"""),
    md("**Step 4 — the decision.** The product does not ship a probability; it ships a capacity-constrained offer. At 20% capacity the rule selects the highest-scoring 20% of the cohort. Whether this person is offered support depends on where they sit in that ranking, not on whether their probability exceeds 0.5."),
    code("""selected = capacity_selection(anchor, 0.20)
rank = int((anchor > anchor[person_idx]).sum()) + 1
print(f'Rank {rank:,} of {len(anchor):,} by predicted risk')
print(f'Inside the top-20% capacity: {bool(selected[person_idx])}')
print(f'Threshold score at 20% capacity: {np.quantile(anchor, 0.8):.3f} | this person: {anchor[person_idx]:.3f}')
display(pd.Series({'selected_count': int(selected.sum()),
                   'capacity_share': float(selected.mean()),
                   'cohort_size': len(anchor)}).to_frame('deployed rule'))
"""),
    md("""This is the gap between a model and a product. The median-risk person carries a
predicted probability close to the cohort base rate, yet the deployed rule gives a
binary answer, and that answer is driven by rank against everyone else rather than by
the probability itself. It is also why every fairness statistic in this project is
computed at the top-20% rule rather than at 0.5 — 0.5 is not a threshold this product
ever applies."""),
    md("## Stability\n\nAll models use the same eight bootstrap samples and fixed algorithm seeds to isolate training-data sensitivity. Sample hashes are in artifacts/stability_protocol.json. Jaccard is intersection/union, not the fraction of people retaining status. At equal selected-set size, replaced fraction is (1-J)/(1+J). Refits do not measure temporal stability; pairwise comparisons share refits."),
    code("stability = A('stability_summary.csv'); stability['selected_set_replacement'] = (1-stability.top20_jaccard)/(1+stability.top20_jaccard); display(stability)"),
    md("## Fairness\n\nThresholds change decisions, not calibration of unchanged probabilities. Equal FPR alone is not equalized odds. The group-threshold frontier optimizes using evaluation labels: an optimistic in-sample illustration, not validated mitigation. Removing race does not remove proxies or prove counterfactual fairness."),
    code("display(A('fairness_inference.csv')); display(A('intersectional_audit.csv')); display(A('race_ab_test.csv'))"),
    md("### Equivalence tests and course test table\n\nFor a support programme the harm is a missed offer, so FNR (equal opportunity) and selection rate (statistical parity) are primary; FPR is secondary. A difference test that fails to reject does not show fairness: `equivalent_within_delta` is a TOST at a ±5-point tolerance (fixed in code before TOST ran, but after the gap point estimates had been seen — so it is a stated tolerance, not a pre-registered one), and an interval that is neither significant nor equivalent is inconclusive. The course table gives p-values at the top-20% rule; conditional statistical parity conditions on the historical supervision score."),
    code("inf = A('fairness_inference.csv'); display(inf[(inf.rule == 'top_20pct') & inf.metric.isin(['fnr', 'selection_rate', 'fpr'])]); display(A('fairness_tests.csv')); display(A('fairness_age_bands.csv')); display(A('fairness_frontier.csv').query(\"method == 'group_thresholds_equal_fnr'\"))"),
    md("### Fairness interpretability (FPDP) and mitigation\n\nFPDP varies one input and re-evaluates the equal-opportunity test for logistic and XGBoost (TabICLv2 is excluded for compute cost). Candidate variables are diagnostic associations, not causes. The mitigation table reports effect sizes next to the p-values: group FNRs and their gap, captured events, and the protected group's mean score against its base rate. Under the top-20% rule the logistic FPDP is flat: fixing a feature to any constant shifts every logit equally, so the selected set does not depend on the value, and Panel B is labelled value-independent."),
    code("display(A('fairness_candidates.csv')); display(A('fairness_mitigation.csv'))"),
    md("""### Fairness findings at the top-20% rule

Each gap is classified with the ±5-point tolerance described above (fixed before TOST ran, after the point estimates were seen): **equivalent** (TOST rejects a gap of 5 points or more), **different** (the 95% bootstrap interval excludes 0 and TOST does not show equivalence), or **inconclusive** (neither). A gap can be both significant and equivalent: nonzero, but within the tolerance. The 90+ intervals in this section are not adjusted for multiple testing. The TOST uses an analytic variance and the intervals use the bootstrap, so borderline cases can disagree (e.g. logistic race predictive equality: chi-squared p ≈ 0.049, bootstrap interval includes 0)."""),
    code("""inf = A('fairness_inference.csv')
top = inf[(inf.rule == 'top_20pct') & inf.metric.isin(['selection_rate', 'fnr', 'fpr'])].copy()
top['status'] = ['equivalent' if e else 'different' if s else 'inconclusive'
                 for e, s in zip(top.equivalent_within_delta, top.significant)]
display(top.pivot_table(index=['attribute', 'metric'], columns='model', values='status', aggfunc='first'))
display(top.pivot_table(index=['attribute', 'metric'], columns='model', values='gap').round(3))"""),
    md("""- **Race:** selection-rate, FNR and FPR gaps are equivalent within ±5 points for all three models; the selection-rate ratio is 0.87–0.93. The ±5-point tolerance is absolute, so on a 20% selection rate it alone would admit a ratio near 0.78; the ratio is the stricter check. For XGBoost and TabICLv2 the selection and FPR gaps (about +2 points, Black minus White) have unadjusted bootstrap intervals that exclude zero, in the direction of more support offered to Black people — but they do **not** survive Holm correction across the 54 course tests: after multiplicity control no race test is rejected for any model. Read them as small and unconfirmed, not as established disparities. Equivalence also concerns allocation errors, not predictive quality: within-group AUC is lower for Black people (0.718–0.721 vs 0.746–0.748).
- **Gender:** women who are later re-arrested miss support more often (FNR gap M − F about −0.10 to −0.12, different for all models). Every model over-predicts women (mean score about 0.52 vs observed 0.45; ECE about 0.07 vs about 0.01–0.02 for men), yet women are selected about half as often at the top 20%. Sufficiency is also rejected for gender, but the two findings point in opposite directions: in a support programme over-prediction favours women, and recalibrating by gender would widen the FNR gap. The harm for this use is the FNR gap.
- **Age:** the largest disparity. The FNR gap (under 33 minus 33+) is about −0.23 to −0.25; about 97% of re-arrested people aged 48+ are not selected, vs about 49% at 18–22. Age is a model input and a validated risk factor, but in a support programme that choice needs an explicit justification (need vs risk). The FPDP finds no candidate variable for the age gap: with `Age_at_Release` fixed for everyone the test still rejects, because criminal-history inputs and the Georgia score carry age. Dropping age would not remove the gap.
- **FPDP (gender):** candidate variables are `Gang_Affiliated` and `Age_at_Release`. Gang affiliation is never recorded for women and is imputed as "No", so it acts as a gender-aligned measurement artefact (Cramér's V with gender = 1.0 on the raw field). Dropping it and re-estimating shrinks the gender FNR gap from −0.096 to 0.000 (logistic) and from −0.112 to −0.010 (XGBoost). Women's FNR falls about 8 points while men's rises about 2 (partly levelling down); captured events fall by 27–41, AUC by about 0.014 (five to six times the XGBoost − logistic gap), and women's over-prediction worsens. Statistical parity is still rejected.
- **Caveats:** the Panel A/B numbers above select the candidate *and* measure the improvement on the same evaluation labels, so on their own they show the mitigation can be fitted, not that it generalises. The nested check below separates the two. A p-value above 0.05 after mitigation is still not evidence of fairness; equivalence would be. Group-specific thresholds remain an analytic device only."""),
    md("""### Does the mitigation survive out-of-sample?

The Panel A/B result above was produced by choosing the variable and scoring the
improvement on the same cohort. That is in-sample evidence. Following the scikit-learn
guidance on separating selection from evaluation, the check below runs entirely inside
the **training** partition:

    for each outer fold
        run the course FPDP candidate search on the outer-training part only
        pick the candidate from that part alone
        fit baseline and mitigated models on the outer-training part
        measure accuracy AND disparity on the held-out outer fold

The evaluation cohort is never touched. Two things matter in the output: how often the
procedure selects the **same** variable when it has never seen the evaluation labels
(selection stability), and how much of the disparity reduction survives out of fold."""),
    code("""nested = ROOT / 'artifacts/mitigation_nested.csv'
if nested.exists():
    folds = pd.read_csv(nested)
    display(folds[['fold','model','selected_feature','baseline_auc','mitigated_auc',
                   'baseline_p_equal_opportunity','mitigated_p_equal_opportunity',
                   'baseline_fnr_gap','mitigated_fnr_gap']].round(4))
    display(pd.read_csv(ROOT / 'artifacts/mitigation_nested_summary.csv').round(4))
else:
    print('Run scripts/mitigation_nested.py to generate the out-of-fold evidence.')
"""),
    md("""The procedure selects `Gang_Affiliated` in every fold for both models without ever
seeing the evaluation labels, so the candidate is a property of the training data rather
than an artefact of the cohort we audit. Out of fold the gender FNR gap shrinks by about
70% — from −0.118 to −0.035 (logistic) and −0.127 to −0.034 (XGBoost) — rather than to
zero, and the mean equal-opportunity p-value reaches 0.32 and 0.27 against the in-sample
0.99, at roughly 0.01 AUC. Two of five folds still reject. Report the out-of-fold
numbers; quote the in-sample ones only as the upper bound they are."""),
]
for name in ["performance_calibration", "incumbent_benchmark", "learning_curve", "shap_individual",
             "pdp_ice", "structural_stability", "fairness_support_access", "fairness_operating_point", "fairness_frontier", "fairness_dependence",
             "tradeoff_matrix"]:
    nb.cells.append(code(f"display(Image(filename=str(ROOT / 'artifacts/figures/{name}.png'), width=1000))"))
nb.cells.extend([
    md("""### How much of the protected attribute survives exclusion?

Every version of this project states that removing race, gender and geography takes
away the direct input but not the proxies. That is asserted throughout and measured
nowhere. The test is direct: try to predict the protected attribute **from the model's
own feature set**. The AUC of that attempt is the amount of protected information still
available to anything trained on these columns — 0.50 means genuinely unavailable, 1.00
means exclusion is cosmetic.

Three feature sets are compared, so the size of the channel the team closed is visible
rather than argued: the shipped 29 fields, the same fields plus explicit missing-value
indicators, and the shipped set minus `Gang_Affiliated` (the FPDP mitigation)."""),
    code("""display(A('proxy_recovery.csv').round(4))
display(Image(filename=str(ROOT / 'artifacts/figures/proxy_recovery.png'), width=1300))
"""),
    md("""Three findings, in order of importance.

**The leak was total.** With missingness indicators, gender is recovered at
**AUC = 1.0000**. `Gang_Affiliated` is missing for every woman and no man, so any model
that encodes NaN as a category — which is exactly what TabICLv2 does — had gender
available in full. That is the defect the mode-fill closed, now with a number on it
rather than an argument.

**Exclusion is not removal.** Even in the shipped feature set, gender is recovered at
AUC 0.776 and race at 0.708: 55% and 41% of the way from chance to perfect. The models
never see these attributes, but the information is there for the taking. This is the
measured version of the caveat attached to the race A/B test — the twins score
identically, and the group gaps persist anyway.

**The mitigation helps less than the headline suggests.** Dropping gang affiliation
moves gender recovery only from 0.776 to 0.747. It removes the strongest single proxy
and leaves most of the channel intact, which is consistent with the nested result that
the gap narrows without closing.

The named proxies are worth reading aloud: for gender, gun charges, mental-health and
substance-abuse conditions, and violent arrests; for race, mental-health and
substance-abuse conditions, age at release, violent arrests and dependents. None of them
can simply be dropped — they are the substance of the risk assessment, not incidental
fields. That is the honest ceiling on proxy removal as a mitigation strategy."""),
    md("""### Stability at the level of one person

Everything in the stability section so far is a cohort summary: mean |Δp| between
refits, Jaccard of the selected sets. Those cannot answer the question a caseworker
asks, which is about an individual:

> *Would this person still be offered support if we had drawn a slightly different
> training sample?*

The same eight bootstrap resamples are reused here (identical seed and draw order,
so the protocol hashes match), but the per-person predictions are kept instead of
collapsed. For each person that gives a spread of scores and a count of how many of
the eight refits would have selected them under the deployed top-20% rule.

Someone selected by 8/8 refits is a decision the product can stand behind. Someone
selected by 4/8 is a coin flip that happened to land one way in the published run."""),
    code("""stab = A('individual_stability.csv')
display(A('individual_stability_summary.csv').round(4))
display(Image(filename=str(ROOT / 'artifacts/figures/individual_stability.png'), width=1300))
"""),
    md("""About one person in seven has a **contested** decision, and — the number that
matters — roughly a third of the people the published run actually selects are
borderline rather than clearly above the line.

A natural response is selective prediction: abstain where the refits disagree and
send those cases to human review. The table below asks what that policy costs in
coverage and what it buys."""),
    code("""curve = A('abstention_curve.csv')
display(curve[['model','max_contested_votes','coverage','precision_at_capacity',
               'fnr_gap_gender','fnr_gap_race','abstained_share_F']].round(4))
"""),
    md("""**It buys reliability and costs equity, and that is the finding.** Keeping only
unanimous decisions raises precision at capacity from 0.822 to 0.846 (logistic) while
deciding 87% of the cohort. But the gender FNR gap *widens*, from −0.090 to −0.119.

The mechanism is visible in the data rather than assumed. Contested decisions are
spread evenly across gender (11.9% of women, 12.9% of men). The asymmetry is among
the people who are **selected**: 47% of the 118 selected women sit at the margin
against 30% of the 1,490 selected men. Abstaining therefore removes a larger share of
the few women who were being offered support, and the gap grows.

This is a second trade-off of the same family as the impossibility result: a standard
trustworthiness intervention — refuse to decide when uncertain — is not fairness-neutral.
If the client adopts abstention, the referred cases need a review process that is itself
audited, or the policy simply moves the disparity out of the model and into a queue."""),
    md("""### Testing the comparison instead of asserting it

Three claims about the model comparison are easy to carry as point estimates and hard to
defend when challenged: that XGBoost is better calibrated, that it captures more
re-arrests, and that logistic has the smaller gender gap. Each is checked below with a
paired bootstrap — every resample scores all three models on the same rows, so this is a
test of the difference, not two intervals eyeballed for overlap.

ECE is reported first at several binning choices, because it is a binned estimator and the
ranking it produces is not stable across them."""),
    code("""bins = A('calibration_bin_sensitivity.csv')
display(bins.pivot(index='bins', columns='model', values='ece_equal_width').round(4))
winners = bins.loc[bins.groupby(['bins'])['ece_equal_width'].idxmin(), ['bins','model']]
print('Lowest equal-width ECE by binning choice:')
print(winners.to_string(index=False))
"""),
    md("Bin-free instead: the Cox calibration regression (outcome on logit; perfect is intercept 0 and slope 1) and the Spiegelhalter z-test."),
    code("""cal = A('calibration_tests.csv')
display(cal[['model','intercept','slope','slope_differs_from_1','spiegelhalter_z','calibrated_at_5pct']].round(4))
"""),
    md("Then the paired differences, and how much the two selected sets actually differ at the deployed rule."),
    code("""display(A('calibration_paired_tests.csv').round(5))
display(A('selected_set_overlap.csv').round(4))
"""),
    md("""Logistic and XGBoost are both statistically indistinguishable from perfect calibration and
from each other; TabICLv2's slope of 0.913 is significantly below 1, so its probabilities are
too extreme. The captured-events difference has an interval including zero, and the two models
offer support to 85% of the same people. None of the three claims survives as an established
difference — which is why the recommendation below rests on refit stability and direct
interpretability rather than on any of them."""),
    md("""## Recommendation

**Pilot L1 logistic regression prospectively, with XGBoost as the challenger running in
parallel on the same cohort.**

This is a decision across four dimensions, not an accuracy ranking, so it is stated with
the counter-case attached.

**What the challenger buys.** XGBoost's AUC edge is +0.0025 with a paired 95% interval of
0.0006 to 0.0044, and its Brier score is lower by 0.0010 [0.0003, 0.0015]. Both are
detectable rather than noise — and they are the only two dimensions on which it is
significantly ahead.

**Calibration is a tie, not a win.** XGBoost has the lower ECE at 10 equal-width bins
(0.011 vs 0.013), but that ranking is an artefact of the binning: at 5 equal-width bins,
or 20 quantile bins, logistic wins instead. Bin-free, both are indistinguishable from
perfect calibration (Cox slopes 0.999 and 1.005, neither differing from 1; Spiegelhalter
z = 0.38 and 0.30) and from each other. TabICLv2 is the exception worth stating: slope
0.913, significantly below 1, z = 3.76 — its probabilities are measurably too extreme.

**The accuracy edge does not reach the decision.** At the deployed top-20% rule the two
models offer support to 85% of the same people (Jaccard 0.849); only 254 of 7,807 are
chosen by one and not the other. The difference in captured re-arrests is about 14 offers
with a 95% interval of −29 to 0 — it includes zero. Significant ranking metrics, no
detectable difference in who receives support.

**Why logistic is the recommendation.** Its refit stability is better on every one of the
28 resample pairs (mean |Δp| 0.0322 vs 0.0351; top-20% Jaccard 0.7725 vs 0.7468). Its
coefficients are read directly, with no second tool and no surrogate fidelity loss — the
depth-3 surrogate of XGBoost reaches only R² = 0.61. It runs about nine times faster. The
argument is not that logistic is fairer or better calibrated; on those it is tied. It is
that XGBoost's wins are confined to ranking metrics that do not change the allocation.

**A point estimate we do not claim as a difference.** Logistic's gender FNR gap is smaller
(−0.096 vs −0.112), but the paired difference in absolute gaps is −0.013 with a 95%
interval of −0.036 to +0.011. Not significant, so not offered as a reason.

**Why the accuracy gap is not the decisive number.** The disparity the FPDP mitigation
closes is far larger than the disparity separating the two models: dropping
`Gang_Affiliated` moves the gender FNR gap by about 0.08 out of fold, five times the 0.016
between XGBoost and logistic. The gap is a property of the feature set, not of the
estimator, so neither model choice resolves it and neither should be justified by it.

**What would reverse this.** XGBoost becomes correct if the client quotes the probability
numerically to supervisees rather than using it only to rank, in which case calibration
dominates; if capacity is large enough that twelve extra captured events per 1,561 offers
is material at their scale; or if a future feature set widens the margin. The challenger
arrangement is what keeps that reversible after the pilot.

**Preconditions either way.** Independent prospective validation, evidence that the
support programme actually benefits recipients, a corrections and appeals route, subgroup
monitoring after deployment, and no adverse use. The learning-curve result is evidence
about sample size in *this* cohort and does not establish suitability for smaller agencies
elsewhere."""),
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
