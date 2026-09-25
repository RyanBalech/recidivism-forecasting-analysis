"""Build the submission notebook with the complete course analysis in notebook cells.

The notebook uses the NIJ CSV files and installed third-party packages. It does not
import project modules, call project scripts, or read precomputed project artifacts.
"""
from pathlib import Path
import base64
import hashlib
import html
import zlib

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Recidivism_Project_Submission.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


EMBEDDED_CSVS = (
    "nij-challenge2021_full_dataset.csv",
    "nij-challenge2021_training_dataset.csv",
    "nij-challenge2021_test_dataset_1.csv",
)


def embedded_data_cell() -> nbf.NotebookNode:
    """Package the three original NIJ inputs in a verifiable notebook code cell."""
    lines = ["import base64, hashlib, io, zlib", "EMBEDDED_CSVS = {"]
    checksums = {}
    for name in EMBEDDED_CSVS:
        data = (ROOT / "data" / name).read_bytes()
        checksums[name] = hashlib.sha256(data).hexdigest()
        encoded = base64.b64encode(zlib.compress(data, level=9)).decode("ascii")
        lines.append(f"    {name!r}: (")
        lines.extend(f"        {encoded[i:i + 120]!r}" for i in range(0, len(encoded), 120))
        lines.append("    ),")
    lines.extend([
        "}",
        f"EMBEDDED_SHA256 = {checksums!r}",
        "def load_nij_csv(name):",
        "    data = zlib.decompress(base64.b64decode(EMBEDDED_CSVS[name]))",
        "    actual = hashlib.sha256(data).hexdigest()",
        "    assert actual == EMBEDDED_SHA256[name], f'Dataset checksum mismatch: {name}'",
        "    return pd.read_csv(io.BytesIO(data))",
        "print('Three original NIJ CSVs available inside this notebook; SHA-256 checked on load.')",
    ])
    result = code("\n".join(lines))
    result.metadata["jupyter"] = {"source_hidden": True}
    return result


def append_extended_evidence(notebook: nbf.NotebookNode) -> None:
    """Embed the earlier notebook's narrative, source and outputs in one file.

    The appendix consists only of markdown cells. Historical project-code calls are
    visible as collapsed reference text, never executed by the submission notebook.
    Figures are notebook attachments, so they remain visible without artifact paths.
    """
    if any(c.cell_type == "markdown" and c.source.startswith("# Appendix — complete earlier project review")
           for c in notebook.cells):
        return
    previous_path = ROOT / "notebooks/extended_artifact_review.ipynb"
    if not previous_path.is_file():
        raise FileNotFoundError(f"Build the extended review first: {previous_path}")
    previous = nbf.read(previous_path, as_version=4)
    notebook.cells.append(md("""# Appendix — complete earlier project review

The sections below preserve the **earlier 94-cell notebook** in this same file: its narrative, code listings, tables and figures. These are published exploratory results. The historical code is collapsed as reference text because it calls repository modules and saved artifacts; it is not executed when you run this notebook. The course workflow above is the independently executable analysis from raw NIJ CSVs. Embedded figures and tables remain visible even without the repository's `artifacts/` folder."""))
    for cell_index, old in enumerate(previous.cells):
        if cell_index == 0:
            continue  # the old title is redundant inside this appendix
        if old.cell_type == "markdown":
            notebook.cells.append(md(old.source))
            continue
        if old.cell_type != "code":
            continue
        escaped = html.escape(old.source)
        notebook.cells.append(md(
            f"<details><summary>Earlier analysis code cell {cell_index} (reference)</summary>"
            f"<pre><code>{escaped}</code></pre></details>"
        ))
        for output_index, output in enumerate(old.outputs):
            if output.output_type == "stream":
                notebook.cells.append(md(f"```text\n{output.text}\n```"))
                continue
            data = getattr(output, "data", {})
            if "image/png" in data:
                filename = f"earlier_figure_{cell_index}_{output_index}.png"
                figure = md(f"![Earlier analysis figure](attachment:{filename})")
                figure.attachments = {filename: {"image/png": data["image/png"]}}
                notebook.cells.append(figure)
            elif "text/html" in data:
                notebook.cells.append(md(str(data["text/html"])))
            elif "text/plain" in data:
                notebook.cells.append(md(f"```text\n{data['text/plain']}\n```"))


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [
        md("""# Trustworthy recidivism forecasting — self-contained submission

This notebook follows the familiar machine-learning workflow: **load data → explore the training data → choose and engineer eligible features → check leakage → train → evaluate → explain → test stability and fairness → recommend**. The three NIJ source CSVs are compressed and embedded in a collapsed cell, with SHA-256 checks on load. This `.ipynb` can stand alone as the analysis file; the course presentation and app are separate deliverables. It does not import code from `src/` or `scripts/`, run shell commands, or read saved models and result tables. It still requires installed third-party Python packages and TabICL's checkpoint, which TabICL can obtain through its normal package mechanism.

To rerun in a fresh Python environment, install `numpy`, `pandas`, `scikit-learn==1.7.2`, `xgboost`, `tabicl==2.2.0`, `torch`, `shap`, `lime`, `statsmodels`, `matplotlib` and `ipykernel`, then use **Run All**. A first TabICL checkpoint download may require internet access; CUDA speeds up its repeated fits.

**Client and decision.** A community-supervision software vendor is considering a voluntary re-entry support offer to the highest-risk 20% of people at supervision start. This is a retrospective course analysis, not an operationally validated tool. Our target is cumulative new arrest within three years. NIJ's challenge used conditional annual forecasts, so these results are not leaderboard-comparable.

**Reading map.** Each section states its question, shows the code that produces the result, and interprets the result. The training partition is the official NIJ release; the evaluation partition is excluded from fitting, although its labels were inspected during project development. Treat all evaluation comparisons as exploratory."""),
        code("""from pathlib import Path
import re, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
import torch

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore', message="'penalty' was deprecated", category=FutureWarning)
print('Python dependencies loaded; CUDA available:', torch.cuda.is_available())"""),
        md("""## 1. Load the dataset and define the prediction problem

The three original NIJ CSV inputs are embedded below as compressed data. The first is the full released dataset; the two earlier releases let us independently verify baseline fields and training outcomes. The outcome is cumulative new arrest within three years. We use the `Training_Sample` flag to recreate the official training and evaluation partitions; there is no new random split. The decision occurs at supervision start, before any future arrest or supervision activity."""),
        code("""TARGET = 'Recidivism_Within_3years'
raw = load_nij_csv('nij-challenge2021_full_dataset.csv')
assert raw.ID.is_unique and raw.ID.notna().all()
assert set(raw.Training_Sample.unique()) == {0, 1}
y_all = raw[TARGET].map({'Yes': 1, 'No': 0})
assert y_all.notna().all()
train_mask = raw.Training_Sample.eq(1)
train_rows = raw.loc[train_mask].reset_index(drop=True)
eval_rows = raw.loc[~train_mask].reset_index(drop=True)
y_train = y_all[train_mask].astype(int).reset_index(drop=True)
y_eval = y_all[~train_mask].astype(int).reset_index(drop=True)
audit_train = train_rows[['ID','Gender','Race']].copy()
audit_eval = eval_rows[['ID','Gender','Race']].copy()
display(pd.Series({'rows':len(raw),'columns':raw.shape[1],
                   'official_training_rows':len(train_rows),
                   'official_evaluation_rows':len(eval_rows),
                   'training_new_arrest_rate':y_train.mean()}))
display(train_rows[[TARGET,'Age_at_Release','Supervision_Risk_Score_First','Gang_Affiliated']].head())"""),
        md("""## 2. Explore the training data (EDA)

EDA is restricted to the training partition so choices about missingness, categories and feature transformations do not depend on evaluation outcomes. We inspect class balance, column types, missing values and category levels before modeling. Protected groups are shown here only to understand the data and later audit errors; they are excluded from scoring."""),
        code("""eda = train_rows.drop(columns=['ID','Training_Sample'])
display(pd.DataFrame({'outcome':["No new arrest","New arrest"],
                      'count':y_train.value_counts().sort_index().values,
                      'share':y_train.value_counts(normalize=True).sort_index().values}))
display(pd.Series({'numeric_columns':int(eda.select_dtypes(include='number').shape[1]),
                   'categorical_columns':int(eda.select_dtypes(exclude='number').shape[1]),
                   'duplicate_training_IDs':int(train_rows.ID.duplicated().sum())}))
missing_eda = train_rows.isna().mean().sort_values(ascending=False)
display(missing_eda.head(15).rename('missing_share').to_frame())
display(pd.DataFrame({'feature':['Age_at_Release','Supervision_Risk_Score_First',
                                 'Gang_Affiliated','Prison_Offense'],
                      'distinct_nonmissing':[train_rows[c].nunique(dropna=True) for c in
                                             ['Age_at_Release','Supervision_Risk_Score_First',
                                              'Gang_Affiliated','Prison_Offense']]}))"""),
        code("""fig,axes=plt.subplots(1,3,figsize=(16,4))
y_train.map({0:'No',1:'Yes'}).value_counts().reindex(['No','Yes']).plot.bar(ax=axes[0],title='Training outcome')
missing_eda[missing_eda>0].head(10).sort_values().plot.barh(ax=axes[1],title='Missing values (training)')
age_order=['18-22','23-27','28-32','33-37','38-42','43-47','48 or older']
age_rates=train_rows.assign(outcome=y_train).groupby('Age_at_Release').outcome.agg(['mean','size']).reindex(age_order)
age_rates['mean'].plot.bar(ax=axes[2],title='New-arrest rate by age (training)')
axes[0].set(ylabel='People'); axes[1].set(xlabel='Share missing'); axes[2].set(ylabel='Observed rate',ylim=(0,1))
plt.tight_layout(); plt.show()
display(age_rates.rename(columns={'mean':'new_arrest_rate','size':'people'}).round(3))
display(train_rows.assign(outcome=y_train).groupby(['Gender','Race']).outcome.agg(['mean','size']).round(3))"""),
        md("""The exploratory plots show why an accuracy score alone is insufficient: the outcome is common, arrest rates vary by age, and some baseline fields have structured missingness. These are associations, not causal effects. The evaluation partition remains untouched during the choices below."""),
        md("""## 3. Select and engineer eligible features

Only fields available in NIJ's first test release can enter a model. We keep 29 baseline fields and exclude ID, target, split flag, gender, race, geography and post-release supervision, employment or drug-test fields. `Gender` and `Race` stay in the audit tables. The feature list is explicit so a reader can see exactly what enters all three models."""),
        code("""FEATURES = [
    'Age_at_Release', 'Gang_Affiliated', 'Supervision_Risk_Score_First',
    'Supervision_Level_First', 'Education_Level', 'Dependents', 'Prison_Offense',
    'Prison_Years', 'Prior_Arrest_Episodes_Felony', 'Prior_Arrest_Episodes_Misd',
    'Prior_Arrest_Episodes_Violent', 'Prior_Arrest_Episodes_Property',
    'Prior_Arrest_Episodes_Drug', '_v1', 'Prior_Arrest_Episodes_DVCharges',
    'Prior_Arrest_Episodes_GunCharges', 'Prior_Conviction_Episodes_Felony',
    'Prior_Conviction_Episodes_Misd', 'Prior_Conviction_Episodes_Viol',
    'Prior_Conviction_Episodes_Prop', 'Prior_Conviction_Episodes_Drug',
    '_v2', '_v3', '_v4', 'Prior_Revocations_Parole',
    'Prior_Revocations_Probation', 'Condition_MH_SA', 'Condition_Cog_Ed',
    'Condition_Other',
]
assert len(FEATURES) == 29 and set(FEATURES).issubset(raw.columns)
assert not set(FEATURES) & {'ID', 'Gender', 'Race', 'Residence_PUMA', TARGET, 'Training_Sample'}
assert not any(c.startswith('Recidivism_') for c in FEATURES)
X_train = train_rows[FEATURES].copy()
X_eval = eval_rows[FEATURES].copy()
display(pd.DataFrame({'feature':FEATURES,
                      'dtype':X_train.dtypes.astype(str).values,
                      'missing_train':X_train.isna().mean().values,
                      'levels_train':X_train.nunique(dropna=True).values}))
print('Model matrix:',X_train.shape,'training;',X_eval.shape,'evaluation')"""),
        md("""## 4. Check leakage against the original releases

The full dataset was published after outcomes were known. We independently compare its baseline fields against the original training release and the first test release. We also verify the original training outcomes and disjoint IDs. These tests catch direct post-release fields and accidental row or label changes. They cannot establish the precise measurement timestamp of every nominally baseline field. The evaluation labels were used in later project development, so this is not an untouched research holdout."""),
        code("""first_test = load_nij_csv('nij-challenge2021_test_dataset_1.csv')
first_train = load_nij_csv('nij-challenge2021_training_dataset.csv')
assert set(FEATURES).issubset(first_test.columns)
assert set(first_test.ID) == set(audit_eval.ID)
assert set(first_train.ID) == set(audit_train.ID)
assert set(audit_train.ID).isdisjoint(audit_eval.ID)
indexed_full = raw.set_index('ID')
for release in (first_train, first_test):
    original = release.set_index('ID')[FEATURES].sort_index()
    current = indexed_full.loc[original.index, FEATURES]
    pd.testing.assert_frame_equal(original, current, check_dtype=False)
outcomes = [TARGET] + [f'Recidivism_Arrest_Year{i}' for i in (1,2,3)]
original_y = first_train.set_index('ID')[outcomes].sort_index()
pd.testing.assert_frame_equal(original_y, indexed_full.loc[original_y.index, outcomes], check_dtype=False)
missing = X_train.Gang_Affiliated.isna()
display(pd.Series({'released_features_verified':len(FEATURES), 'disjoint_IDs':True,
                   'training_outcomes_verified':True, 'gang_missing_rows':int(missing.sum()),
                   'women_among_gang_missing':float(audit_train.loc[missing,'Gender'].eq('F').mean())}))
print('Limitation: first-release availability does not prove that every assessment was measured before the decision.')"""),
        md("""### Missingness and protected-attribute proxies

`Gang_Affiliated` is missing for every woman and no man in the training set. A missingness indicator would reveal gender exactly even though Gender is excluded from model inputs. We use the **training mode** to fill missing categories, including for TabICL. Other correlations and proxies can still carry protected information; exclusion alone does not guarantee fairness."""),
        code("""display(pd.crosstab(audit_train.Gender, X_train.Gang_Affiliated.isna(), normalize='index'))
categorical = X_train.select_dtypes(exclude='number').columns
category_modes = X_train[categorical].mode().iloc[0]
tab_train, tab_eval = X_train.fillna(category_modes), X_eval.fillna(category_modes)
assert not tab_train[categorical].isna().any().any()
assert not tab_eval[categorical].isna().any().any()
print('Categorical missing values filled using training modes only.')"""),
        md("""## 5. Preprocess and train three model families

The feature engineering is explicit below. Ordered age/prison categories and count bands receive numeric ranks for XGBoost; logistic keeps categories one-hot. Numeric median imputation, categorical mode imputation, scaling and one-hot categories are learned inside the conventional-model pipelines from training rows only. TabICLv2 uses its own mixed-data encoding and the shared training-mode-filled table. The hyperparameters were selected earlier with training-only cross-validation; their exact values are visible. A CUDA GPU is used for TabICL when available."""),
        code("""ORDERED = {
    'Prison_Years': {'Less than 1 year':0, '1-2 years':1,
                     'Greater than 2 to 3 years':2, 'More than 3 years':3},
    'Age_at_Release': {'18-22':0, '23-27':1, '28-32':2, '33-37':3,
                       '38-42':4, '43-47':5, '48 or older':6},
}
COUNT = re.compile(r'\\d+( or more)?')
XGB_PARAMS = dict(n_estimators=1196, max_depth=2, learning_rate=0.0188,
                  min_child_weight=8, subsample=0.8217, colsample_bytree=0.6233,
                  gamma=0.4576, reg_lambda=4.5603, reg_alpha=1.4192)
LOGIT_PARAMS = dict(C=0.2154, penalty='l1')

def ordinal_encode(frame):
    out = frame.copy()
    for col in out:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        if col in ORDERED:
            out[col] = out[col].map(ORDERED[col]).astype(float)
        else:
            vals = set(map(str, out[col].dropna().unique()))
            if vals and all(COUNT.fullmatch(v) for v in vals):
                out[col] = out[col].astype(str).str.extract(r'(\\d+)')[0].astype(float)
    return out

def prepare(frame):
    num = frame.select_dtypes(include='number').columns.tolist()
    cat = [c for c in frame if c not in num]
    return ColumnTransformer([
        ('numeric', Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),
                              ('scale',StandardScaler())]), num),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                                 ('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=10))]),cat),
    ], verbose_feature_names_out=False)

def make_logistic(frame):
    return Pipeline([('prepare',prepare(frame)),
                     ('model',LogisticRegression(**LOGIT_PARAMS,max_iter=2000,
                                                  solver='liblinear',random_state=SEED))])

def make_xgboost(frame):
    return Pipeline([('ordinal',FunctionTransformer(ordinal_encode)),
                     ('prepare',prepare(ordinal_encode(frame))),
                     ('model',XGBClassifier(**XGB_PARAMS, n_jobs=-1,
                                            objective='binary:logistic',eval_metric='logloss',
                                            tree_method='hist',random_state=SEED))])

display(pd.DataFrame([{'model':'logistic','settings':LOGIT_PARAMS},
                      {'model':'xgboost','settings':XGB_PARAMS},
                      {'model':'tabicl','settings':{'n_estimators':16,
                       'checkpoint':'tabicl-classifier-v2-20260212.ckpt'}}]))
display(pd.DataFrame({'raw_age':X_train.Age_at_Release.head().values,
                      'xgboost_age_rank':ordinal_encode(X_train[['Age_at_Release']]).iloc[:,0].head().values,
                      'raw_prison_years':X_train.Prison_Years.head().values,
                      'xgboost_prison_rank':ordinal_encode(X_train[['Prison_Years']]).iloc[:,0].head().values}))
preview_transform = prepare(X_train).fit(X_train)
print('29 raw fields become',len(preview_transform.get_feature_names_out()),
      'prepared columns for logistic regression; fitted on training rows only.')"""),
        md("""### Fit on training rows and predict evaluation rows

The next cell trains all three models from scratch. The evaluation outcomes are not passed to `.fit()`. The training-mode category fill above is reused for TabICL evaluation rows."""),
        code("""from tabicl import TabICLClassifier

models, probabilities, timing = {}, {}, {}
for name, builder in [('logistic',make_logistic),('xgboost',make_xgboost)]:
    start = time.perf_counter()
    model = builder(X_train).fit(X_train,y_train)
    models[name] = model
    probabilities[name] = model.predict_proba(X_eval)[:,1]
    timing[name] = time.perf_counter()-start
    print(f'{name}: {timing[name]:.1f} s')

start = time.perf_counter()
tabicl = TabICLClassifier(n_estimators=16,batch_size=1,kv_cache='repr',
                          checkpoint_version='tabicl-classifier-v2-20260212.ckpt',
                          random_state=SEED,n_jobs=-1)
tabicl.fit(tab_train,y_train.to_numpy())
models['tabicl'] = tabicl
probabilities['tabicl'] = tabicl.predict_proba(tab_eval)[:,1]
timing['tabicl'] = time.perf_counter()-start
print(f'tabicl: {timing["tabicl"]:.1f} s; device: {"CUDA" if torch.cuda.is_available() else "CPU"}')
assert all(np.isfinite(p).all() and len(p)==len(y_eval) for p in probabilities.values())"""),
        md("""## 6. Evaluate probability quality and the support decision

ROC AUC measures ranking (0.5 is chance), average precision focuses on the positive class, Brier is mean squared probability error, and ECE is a binned calibration estimate. Lower Brier/ECE is better. We also select exactly 20% by descending risk, with stable row order for ties, to reflect a fixed service capacity."""),
        code("""def select_top(scores, capacity=.20):
    selected = np.zeros(len(scores),dtype=bool)
    selected[np.argsort(-np.asarray(scores),kind='stable')[:round(len(scores)*capacity)]] = True
    return selected

def ece(y,p,bins=10):
    y,p=np.asarray(y),np.asarray(p)
    bucket=np.digitize(p,np.linspace(0,1,bins+1)[1:-1])
    return sum((bucket==i).mean()*abs(y[bucket==i].mean()-p[bucket==i].mean())
               for i in range(bins) if (bucket==i).any())

def score(y,p):
    return dict(roc_auc=roc_auc_score(y,p), average_precision=average_precision_score(y,p),
                brier=brier_score_loss(y,p), log_loss=log_loss(y,p), ece_10=ece(y,p))

metrics = pd.DataFrame([{'model':name,**score(y_eval,p),'fit_predict_s':timing[name],
                         'captured_at_20pct':int(y_eval[select_top(p)].sum())}
                        for name,p in probabilities.items()]).set_index('model')
display(metrics.round(4))
base=np.repeat(y_train.mean(),len(y_eval))
print('Training-prevalence Brier baseline:',round(brier_score_loss(y_eval,base),4))
print('XGBoost relative Brier improvement:',
      round(1-metrics.loc['xgboost','brier']/brier_score_loss(y_eval,base),3))"""),
        code("""fig,axes=plt.subplots(1,2,figsize=(12,4))
for name,p in probabilities.items():
    bins=pd.qcut(p,10,duplicates='drop')
    curve=pd.DataFrame({'p':p,'y':y_eval,'bin':bins}).groupby('bin',observed=True)[['p','y']].mean()
    axes[0].plot(curve.p,curve.y,marker='o',label=name)
axes[0].plot([0,1],[0,1],'k--',alpha=.5)
axes[0].set(xlabel='Mean predicted probability',ylabel='Observed new-arrest rate',title='Calibration')
axes[0].legend()
metrics.roc_auc.plot.bar(ax=axes[1],ylim=(.70,.74),title='Evaluation ROC AUC')
axes[1].set_ylabel('AUC')
plt.tight_layout(); plt.show()"""),
        md("""### Paired uncertainty and a hypothetical economic scenario

The same bootstrap row sample is applied to both models, preserving the pairing. These intervals describe sampling variation conditional on the already selected models; they do not correct the repeated inspection of evaluation labels. Economic values assume 20% intervention effectiveness, a €50,000 event cost and €5,000 support cost. These are scenarios, not estimated causal savings."""),
        code("""rng=np.random.default_rng(SEED)
pair_draws=[]
for _ in range(500):
    ix=rng.integers(0,len(y_eval),len(y_eval))
    yy=y_eval.to_numpy()[ix]
    if len(np.unique(yy))<2: continue
    a=probabilities['xgboost'][ix]; b=probabilities['logistic'][ix]
    pair_draws.append((roc_auc_score(yy,a)-roc_auc_score(yy,b),
                       brier_score_loss(yy,a)-brier_score_loss(yy,b)))
pair_draws=np.asarray(pair_draws)
display(pd.DataFrame({'comparison':['XGBoost - logistic AUC','XGBoost - logistic Brier'],
                      'point':[metrics.loc['xgboost','roc_auc']-metrics.loc['logistic','roc_auc'],
                               metrics.loc['xgboost','brier']-metrics.loc['logistic','brier']],
                      'low':np.quantile(pair_draws,.025,axis=0),
                      'high':np.quantile(pair_draws,.975,axis=0)}))
economics=[]
for name,p in probabilities.items():
    chosen=select_top(p)
    captured=int(y_eval[chosen].sum())
    economics.append({'model':name,'offers':int(chosen.sum()),'observed_events_in_offers':captured,
                      'assumed_gross_eur':captured*50000*.20,
                      'assumed_cost_eur':chosen.sum()*5000,
                      'assumed_net_eur':captured*50000*.20-chosen.sum()*5000})
display(pd.DataFrame(economics).set_index('model'))"""),
        md("""## 7. Interpret what the models use

Logistic coefficients are effects on log-odds per transformed unit; they are not causal. Average marginal effects below instead perturb one raw field while holding all others fixed, then average the change in predicted probability. For categorical features we compare each level to its training mode. Permutation importance measures the decline in evaluation AUC when a raw field is shuffled. Correlated fields may share credit."""),
        code("""logit=models['logistic']
coefs=pd.DataFrame({'transformed_feature':logit[-2].get_feature_names_out(),
                    'coefficient':logit[-1].coef_[0]})
coefs['odds_ratio']=np.exp(coefs.coefficient)
display(coefs.reindex(coefs.coefficient.abs().sort_values(ascending=False).index).head(15))

base_p=logit.predict_proba(X_eval)[:,1]
effects=[]
for feature in FEATURES:
    observed=X_train[feature].dropna()
    if pd.api.types.is_numeric_dtype(X_train[feature]):
        scale=observed.std()
        changed=X_eval.copy()
        changed[feature]=changed[feature].fillna(observed.median())+scale
        effect=(logit.predict_proba(changed)[:,1]-base_p).mean()
        effects.append({'feature':feature,'contrast':'+1 training SD','mean_probability_change':effect})
    else:
        reference=observed.mode().iloc[0]
        ref=X_eval.copy(); ref[feature]=reference
        ref_p=logit.predict_proba(ref)[:,1]
        for level in observed.unique():
            if level==reference: continue
            changed=X_eval.copy(); changed[feature]=level
            effect=(logit.predict_proba(changed)[:,1]-ref_p).mean()
            effects.append({'feature':feature,'contrast':f'{level} vs {reference}',
                            'mean_probability_change':effect})
effects=pd.DataFrame(effects)
display(effects.reindex(effects.mean_probability_change.abs().sort_values(ascending=False).index).head(15))"""),
        code("""# Permute original columns, so one-hot levels move together.
importance=[]
for name in ['logistic','xgboost']:
    result=permutation_importance(models[name],X_eval,y_eval,scoring='roc_auc',
                                  n_repeats=3,random_state=SEED,n_jobs=1)
    importance.extend({'model':name,'feature':f,'auc_drop':v}
                      for f,v in zip(FEATURES,result.importances_mean))
importance=pd.DataFrame(importance)
display(importance.sort_values(['model','auc_drop'],ascending=[True,False]).groupby('model').head(10))
print('Permutation importance is descriptive on this repeatedly inspected evaluation cohort.')"""),
        md("""### SHAP, a global surrogate and partial dependence

SHAP attributes XGBoost's raw score (log-odds) across its transformed inputs. We sum one-hot levels back to their original fields for the global view. A depth-three decision tree is a readable approximation of XGBoost; its R² on evaluation probabilities tells us how much of the original model it can actually explain. Partial dependence then changes age for the same 200 records and averages each model's probabilities. These are model behavior descriptions, not causal effects."""),
        code("""import shap

sample_ix=np.random.default_rng(SEED).choice(len(X_eval),1000,replace=False)
X_shap=X_eval.iloc[sample_ix]
pipe=models['xgboost']
prepared=pipe[:-1].transform(X_shap)
prepared=prepared.toarray() if hasattr(prepared,'toarray') else prepared
shap_values=shap.TreeExplainer(pipe[-1]).shap_values(prepared)
transformed_names=pipe[-2].get_feature_names_out()
shap_importance=pd.DataFrame({'transformed_feature':transformed_names,
                               'mean_abs_shap':np.abs(shap_values).mean(axis=0)})
def original_field(name):
    name=str(name).removeprefix('missingindicator_')
    match=[f for f in FEATURES if name==f or name.startswith(f+'_')]
    return max(match,key=len) if match else name
shap_importance['feature']=shap_importance.transformed_feature.map(original_field)
shap_global=shap_importance.groupby('feature').mean_abs_shap.sum().sort_values(ascending=False)
display(shap_global.head(10).to_frame())
fig,ax=plt.subplots(figsize=(8,4))
shap_global.head(10).sort_values().plot.barh(ax=ax,title='XGBoost global mean absolute SHAP')
ax.set_xlabel('Mean absolute contribution to raw score'); plt.tight_layout(); plt.show()

encoded_train=pd.get_dummies(ordinal_encode(X_train)).fillna(-1)
encoded_eval=pd.get_dummies(ordinal_encode(X_eval)).reindex(columns=encoded_train.columns,fill_value=0).fillna(-1)
surrogate=DecisionTreeRegressor(max_depth=3,min_samples_leaf=200,random_state=SEED)
surrogate.fit(encoded_train,models['xgboost'].predict_proba(X_train)[:,1])
surrogate_r2=r2_score(probabilities['xgboost'],surrogate.predict(encoded_eval))
print('Depth-three surrogate R² on evaluation probabilities:',round(surrogate_r2,3))"""),
        code("""# Local SHAP for the predetermined median-risk person, grouped to raw features.
local_ix=int(np.argsort(probabilities['xgboost'])[len(X_eval)//2])
local_prepared=pipe[:-1].transform(X_eval.iloc[[local_ix]])
local_prepared=local_prepared.toarray() if hasattr(local_prepared,'toarray') else local_prepared
local_values=shap.TreeExplainer(pipe[-1]).shap_values(local_prepared)[0]
local_shap=(pd.DataFrame({'feature':[original_field(n) for n in transformed_names],
                          'contribution':local_values})
            .groupby('feature').contribution.sum().sort_values(key=abs,ascending=False).head(10))
display(local_shap.rename('raw-score contribution').to_frame())
fig,ax=plt.subplots(figsize=(9,5))
local_shap.sort_values().plot.barh(ax=ax,color=['#c1121f' if v>0 else '#2a9d8f'
                                            for v in local_shap.sort_values()])
ax.set(xlabel='Contribution to XGBoost raw score',
       title='Local SHAP: median-risk person'); plt.tight_layout(); plt.show()"""),
        code("""age_order=['18-22','23-27','28-32','33-37','38-42','43-47','48 or older']
ice_people=X_eval.iloc[np.random.default_rng(SEED).choice(len(X_eval),200,replace=False)]
pdp=[]
for name,model in models.items():
    for band in age_order:
        changed=ice_people.copy(); changed['Age_at_Release']=band
        if name=='tabicl':
            pred=model.predict_proba(changed.fillna(category_modes))[:,1]
        else:
            pred=model.predict_proba(changed)[:,1]
        pdp.append({'model':name,'age_band':band,'mean_probability':pred.mean()})
pdp=pd.DataFrame(pdp)
display(pdp.pivot(index='age_band',columns='model',values='mean_probability').round(3))
fig,ax=plt.subplots(figsize=(9,4))
for name,part in pdp.groupby('model'):
    ax.plot(part.age_band,part.mean_probability,marker='o',label=name)
ax.set(ylabel='Mean predicted probability',title='Age partial dependence')
ax.tick_params(axis='x',rotation=35); ax.legend(); plt.tight_layout(); plt.show()"""),
        md("""### Local explanation with a fidelity check

LIME samples nearby raw-feature values. Categorical variables are declared as categories so it does not create impossible combinations of independent one-hot flags. Its weighted local R² measures how faithfully the small surrogate approximates XGBoost around the chosen person. A low R² means the explanation should be treated cautiously."""),
        code("""from lime.lime_tabular import LimeTabularExplainer

def lime_space(training, evaluation):
    nums=training.select_dtypes(include='number').columns.tolist()
    cats=[c for c in training if c not in nums]
    med=training[nums].median()
    modes={c:training[c].mode().iloc[0] for c in cats}
    levels={c:sorted(map(str,training[c].fillna(modes[c]).unique())) for c in cats}
    def encode(frame):
        out=pd.DataFrame(index=frame.index)
        for c in training:
            if c in nums: out[c]=frame[c].fillna(med[c]).astype(float)
            else:
                lookup={v:i for i,v in enumerate(levels[c])}
                out[c]=frame[c].fillna(modes[c]).astype(str).map(lookup).astype(float)
        return out[training.columns].to_numpy(float)
    def decode(array):
        out=pd.DataFrame(array,columns=training.columns)
        for c in cats:
            ix=np.clip(np.rint(out[c]).astype(int),0,len(levels[c])-1)
            out[c]=[levels[c][i] for i in ix]
        return out
    cat_ix=[list(training.columns).index(c) for c in cats]
    names={i:levels[training.columns[i]] for i in cat_ix}
    return encode(training),encode(evaluation),decode,cat_ix,names

lime_train,lime_eval,decode,cat_ix,cat_names=lime_space(X_train,X_eval)
xgb_p=probabilities['xgboost']
case_indices={'low':int(np.argmin(xgb_p)),'median':int(np.argsort(xgb_p)[len(xgb_p)//2]),
              'high':int(np.argmax(xgb_p))}
lime_rows=[]
for case,ix in case_indices.items():
    for seed in (0,1,2):
        explainer=LimeTabularExplainer(lime_train,feature_names=FEATURES,
            categorical_features=cat_ix,categorical_names=cat_names,
            class_names=['no','yes'],discretize_continuous=True,random_state=seed)
        explanation=explainer.explain_instance(lime_eval[ix],
            lambda z:models['xgboost'].predict_proba(decode(z)),num_features=10,num_samples=3000)
        lime_rows.append({'case':case,'seed':seed,'local_r2':explanation.score,
                          'model_probability':xgb_p[ix],
                          'top_rules':explanation.as_list()[:3]})
lime_results=pd.DataFrame(lime_rows)
display(lime_results[['case','seed','local_r2','model_probability']])
display(lime_results.loc[(lime_results.case=='median')&(lime_results.seed==0),'top_rules'])
median_rules=lime_results.loc[(lime_results.case=='median')&(lime_results.seed==0),'top_rules'].iloc[0]
fig,ax=plt.subplots(figsize=(9,3))
pd.Series(dict(median_rules)).sort_values().plot.barh(ax=ax)
ax.set(xlabel='LIME contribution to local probability',
       title=f'Median-risk explanation (local R²={lime_results.loc[(lime_results.case=="median") & (lime_results.seed==0),"local_r2"].iloc[0]:.2f})')
plt.tight_layout(); plt.show()"""),
        md("""### One person's complete path

The median-risk evaluation person is fixed by XGBoost's rank. We show their raw fields, trained preprocessing, three probabilities, local explanation and support decision. These are predictions, not a diagnosis or a reason to impose a sanction."""),
        code("""ix=case_indices['median']
person=X_eval.iloc[[ix]]
print('Anonymized record ID:',int(audit_eval.ID.iloc[ix]))
display(person.T.rename(columns={ix:'raw value'}))
prepared=models['logistic'][:-1].transform(person)
print('Logistic prepared matrix:',prepared.shape,
      '; nonzero values:',int(prepared.nnz if hasattr(prepared,'nnz') else np.count_nonzero(prepared)))
display(pd.DataFrame({'model':list(probabilities),
                      'risk':[float(probabilities[m][ix]) for m in probabilities]}))
rank=int((xgb_p>xgb_p[ix]).sum())+1
print(f'XGBoost rank {rank} of {len(xgb_p)}; support offer at 20% capacity:',
      bool(select_top(xgb_p)[ix]))
print('LIME local R² for this case:',
      lime_results.loc[lime_results.case=='median','local_r2'].round(3).tolist())"""),
        md("""## 8. Stability under training resampling

Each model is refitted on the same bootstrap samples of the *training* partition. We compare its new evaluation scores and selected set to its original fit. This measures training-data sensitivity, not future population shift. TabICL refits are compute-heavy and therefore measured with three shared resamples here; the repository's expanded audit uses eight."""),
        code("""rng=np.random.default_rng(SEED)
stability=[]
for draw in range(3):
    sample_ix=rng.integers(0,len(X_train),len(X_train))
    xx,yy=X_train.iloc[sample_ix],y_train.iloc[sample_ix]
    modes=xx[xx.select_dtypes(exclude='number').columns].mode().iloc[0]
    for name,builder in [('logistic',make_logistic),('xgboost',make_xgboost)]:
        refit=builder(xx).fit(xx,yy)
        q=refit.predict_proba(X_eval)[:,1]
        original=probabilities[name]
        a,b=select_top(original),select_top(q)
        stability.append({'draw':draw,'model':name,'mean_abs_probability_change':np.mean(abs(q-original)),
                          'rank_correlation':pd.Series(q).corr(pd.Series(original),method='spearman'),
                          'top20_jaccard':(a&b).sum()/(a|b).sum()})
    refit=TabICLClassifier(n_estimators=16,batch_size=1,kv_cache='repr',
        checkpoint_version='tabicl-classifier-v2-20260212.ckpt',random_state=SEED,n_jobs=-1)
    refit.fit(xx.fillna(modes),yy.to_numpy())
    q=refit.predict_proba(X_eval.fillna(modes))[:,1]
    original=probabilities['tabicl']; a,b=select_top(original),select_top(q)
    stability.append({'draw':draw,'model':'tabicl','mean_abs_probability_change':np.mean(abs(q-original)),
                      'rank_correlation':pd.Series(q).corr(pd.Series(original),method='spearman'),
                      'top20_jaccard':(a&b).sum()/(a|b).sum()})
stability=pd.DataFrame(stability)
display(stability.groupby('model').agg(['mean','std']).round(4))
fig,axes=plt.subplots(1,2,figsize=(11,4))
stability.groupby('model').mean_abs_probability_change.mean().plot.bar(ax=axes[0],
    title='Mean probability change across refits')
stability.groupby('model').top20_jaccard.mean().plot.bar(ax=axes[1],
    title='Top-20% selected-set overlap',ylim=(0,1))
axes[0].set_ylabel('Mean |new p − original p|'); axes[1].set_ylabel('Jaccard similarity')
plt.tight_layout(); plt.show()"""),
        md("""## 9. Fairness at the actual support rule

At 20% capacity we compare selection rate (access), false negative rate among people later re-arrested (missed support) and false positive rate. Group gaps are descriptive and do not prove discrimination or fairness. Arrest itself is influenced by social and enforcement processes. Race and gender are audit fields only. Age is a model input, so age gaps are especially important to examine."""),
        code("""def group_rates(y, selected, groups, attribute, model):
    rows=[]
    for group in pd.Series(groups).fillna('Missing').unique():
        mask=np.asarray(pd.Series(groups).fillna('Missing').eq(group))
        positives=mask&(np.asarray(y)==1)
        negatives=mask&(np.asarray(y)==0)
        rows.append({'model':model,'attribute':attribute,'group':group,'n':int(mask.sum()),
                     'base_rate':np.mean(np.asarray(y)[mask]),
                     'selection_rate':selected[mask].mean(),
                     'fnr':1-selected[positives].mean() if positives.any() else np.nan,
                     'fpr':selected[negatives].mean() if negatives.any() else np.nan})
    return rows

audit_rows=[]
age=X_eval.Age_at_Release
for name,p in probabilities.items():
    chosen=select_top(p)
    audit_rows+=group_rates(y_eval,chosen,audit_eval.Gender,'Gender',name)
    audit_rows+=group_rates(y_eval,chosen,audit_eval.Race,'Race',name)
    audit_rows+=group_rates(y_eval,chosen,age,'Age_at_Release',name)
fairness=pd.DataFrame(audit_rows)
display(fairness.round(3))
gaps=fairness.groupby(['model','attribute'])[['selection_rate','fnr','fpr']].agg(
    lambda s:s.max()-s.min()).rename(columns=lambda c:c+'_range')
display(gaps.round(3))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,attribute in zip(axes,['Gender','Race']):
    part=fairness[fairness.attribute==attribute]
    part.pivot(index='model',columns='group',values='selection_rate').plot.bar(ax=ax)
    ax.set(title=f'{attribute}: support offers at 20% capacity',ylabel='Selection rate',ylim=(0,.3))
plt.tight_layout(); plt.show()
fig,ax=plt.subplots(figsize=(8,4))
fairness[fairness.attribute=='Gender'].pivot(index='model',columns='group',values='fnr').plot.bar(ax=ax)
ax.set(title='Gender: missed support among later re-arrested people',
       ylabel='False negative rate',ylim=(0,1))
plt.tight_layout(); plt.show()"""),
        md("""### Test multiplicity and a training-only mitigation check

We test group differences on the published decision rule and apply Holm correction across this notebook's test family. A non-rejection does not establish equivalence. The mitigation experiment drops `Gang_Affiliated`, refits within each training fold, and scores only its held-out fold. The candidate was chosen during prior research, so this is a validation of a fixed candidate, not a fresh search over all fields."""),
        code("""from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

tests=[]
for name,p in probabilities.items():
    chosen=select_top(p)
    for attr,groups in [('Gender',audit_eval.Gender),('Race',audit_eval.Race),
                        ('Age_at_Release',age)]:
        group=np.asarray(groups)
        for metric,mask in [('selection_rate',np.ones(len(y_eval),dtype=bool)),
                            ('fnr',y_eval.to_numpy()==1),('fpr',y_eval.to_numpy()==0)]:
            table=pd.crosstab(group[mask],chosen[mask])
            pval=chi2_contingency(table,correction=False)[1] if table.shape[0]>1 and table.shape[1]>1 else np.nan
            tests.append({'model':name,'attribute':attr,'metric':metric,'p_raw':pval})
tests=pd.DataFrame(tests)
valid=tests.p_raw.notna()
tests.loc[valid,'p_holm']=multipletests(tests.loc[valid,'p_raw'],method='holm')[1]
display(tests.round(4))
print('Holm-adjusted rejections:',int((tests.p_holm<.05).sum()),'of',int(valid.sum()))"""),
        code("""strata=y_train.astype(str)+'_'+audit_train.Gender.astype(str)
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
mitigation=[]
for fold,(tr,va) in enumerate(cv.split(X_train,strata),1):
    xtr,xva=X_train.iloc[tr],X_train.iloc[va]
    yy=y_train.iloc[tr]
    for name,builder in [('logistic',make_logistic),('xgboost',make_xgboost)]:
        for removed in (False,True):
            columns=[c for c in FEATURES if c!='Gang_Affiliated'] if removed else FEATURES
            model=builder(xtr[columns]).fit(xtr[columns],yy)
            p=model.predict_proba(xva[columns])[:,1]
            selected=select_top(p)
            group=audit_train.Gender.iloc[va].reset_index(drop=True)
            rates=pd.DataFrame(group_rates(y_train.iloc[va].reset_index(drop=True),selected,
                                           group,'Gender',name))
            fnr_f=float(rates.loc[rates.group=='F','fnr'].iloc[0])
            fnr_m=float(rates.loc[rates.group=='M','fnr'].iloc[0])
            mitigation.append({'fold':fold,'model':name,'drop_gang':removed,
                'auc':roc_auc_score(y_train.iloc[va],p),'female_minus_male_fnr':fnr_f-fnr_m})
mitigation=pd.DataFrame(mitigation)
display(mitigation.groupby(['model','drop_gang'])[['auc','female_minus_male_fnr']]
        .agg(['mean','std']).round(4))
print('Only training partition labels were used in this cross-validation.')"""),
        md("""### Four-dimensional comparison at a glance

The table below collects a predictive metric, an explanation-fidelity limit, a refit-stability metric and a fairness gap. It is a decision aid; these quantities have different units and should not be averaged into a single score."""),
        code("""tradeoff=metrics[['roc_auc','brier','captured_at_20pct']].copy()
tradeoff['mean_refit_top20_jaccard']=stability.groupby('model').top20_jaccard.mean()
tradeoff['gender_fnr_range']=gaps.xs('Gender',level='attribute').fnr_range
tradeoff['local_explanation_limit']=['direct coefficients' if name=='logistic'
    else ('LIME median R² ≈ '+str(round(lime_results.query("case == 'median'").local_r2.mean(),2))
          if name=='xgboost' else 'no native attribution') for name in tradeoff.index]
display(tradeoff.round(4))"""),
        md("""## 10. Recommendation and limits

The primary choice is **L1 logistic regression for a prospective shadow pilot**, with XGBoost running as a challenger. XGBoost has a small but measurable ranking advantage in this retrospective evaluation and captures 12 more observed events among 1,561 offers. Logistic offers direct coefficient interpretation, simpler maintenance and greater refit stability in these resamples. TabICL has the highest point AUC but uses more compute and has no equally direct native explanation. The median-risk person's LIME surrogate reached only about R² = 0.33, so that local rule list is a limited approximation.

The fairness gaps are material, particularly for women who are later re-arrested and across age bands. In the three training folds, dropping `Gang_Affiliated` reduced the female-minus-male FNR gap by about 0.08 but also lowered AUC by about 0.01. This is a real trade-off, not a free mitigation. The 27 tests above form this notebook's own family: TabICL's race selection-rate test survives Holm here, while none of the race tests survived correction across the broader 54-test course family in the repository audit. The conclusion is sensitive to the declared family; neither result proves race fairness. No model should allocate real services until intervention benefit, external validity, appeals, subgroup monitoring and stop rules are established. These data record arrests, not who would benefit most from help.

**Reproduction notes.** Notebook functions live above the cells that call them. The three original CSVs are embedded above, so no other repository file is needed to read or rerun this notebook. Installed public Python packages and TabICL's model checkpoint are still required; a first checkpoint download may need internet access. Numerical fit times and the final digits of XGBoost/TabICL outputs may vary by hardware and package version.

**What follows?** The appendix in this same notebook preserves the previous deep-dive tables and figures: estimator sweep, learning curve, incumbent benchmark, XPER, explanation agreement, FPDP/fairness frontier, proxy recovery, individual stability/abstention and detailed calibration tests. Those historical outputs are embedded snapshots with their earlier code visible as collapsed reference text. The main course analysis above is fully executable from raw CSVs without project functions."""),
    ]
    nb.cells.insert(3, embedded_data_cell())
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    append_extended_evidence(nb)
    nbf.write(nb, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
