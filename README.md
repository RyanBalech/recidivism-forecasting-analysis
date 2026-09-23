# Trustworthy Recidivism Forecasting

An end-to-end scoring analysis for the HEC Paris course **Interpretability, Stability, and Algorithmic Fairness**. The project compares a white-box model, a gradient-boosted model, and an open tabular foundation model on the [NIJ 2021 Recidivism Forecasting Challenge](https://nij.ojp.gov/funding/recidivism-forecasting-challenge).

The decision is framed as estimating three-year arrest risk at the start of parole supervision so that scarce, beneficial re-entry services can be offered. The score is not suitable for sanctions, detention, or increased surveillance.

## Results

> ⚠️ **Re-run pending.** The TabICLv2 figures below predate the gender-leak fix (PR #1:
> `Gang_Affiliated` is missing for every woman, and TabICL treated that as a category). Logistic
> figures predate its CV tuning (difference < 0.001 AUC). The recommendation is under review — see
> [`PLAN.md`](PLAN.md).

The official NIJ `Training_Sample` indicator creates an untouched 18,028/7,807 train/test split. Race, gender, and residence geography are excluded from model inputs and retained for subgroup audits. Only baseline variables available at supervision start are used.

**The client's real question — better than the tool agencies use today?** `Supervision_Risk_Score_First`, Georgia's existing 1–10 actuarial score, reaches only **0.60 ROC AUC**. Every candidate model reaches ~0.73 and roughly doubles the net value of a capacity-limited support programme (~$2.75M → ~$5.1–5.3M).

| Model | ROC AUC | Average precision | Brier ↓ | Calibration error ↓ | Runtime* |
|---|---:|---:|---:|---:|---:|
| Incumbent score | 0.600 | — | — | — | — |
| Logistic regression | 0.7295 | 0.7691 | 0.2055 | 0.0132 | 1.2 s |
| XGBoost | 0.7326 | 0.7722 | 0.2044 | **0.0109** | 2.9 s |
| TabICLv2 | **0.7338** | **0.7723** | **0.2038** | 0.0191 | 103.7 s |

\*Training plus one full held-out prediction run. XGBoost hyperparameters come from a 5-fold CV search; TabICLv2 runs on CPU here (a CUDA GPU is much faster). Bootstrap 95% intervals across the three models overlap.

**Recommendation:** deploy XGBoost for support allocation, with logistic regression as the transparent challenger; TabICLv2 only for very small agencies where its small-data edge is real (see the learning curve). XGBoost calibrates best, has the highest scenario net value and the smallest gender false-positive gap, runs ~35× faster than TabICLv2, and is SHAP-explainable (the foundation model has no native explanation path). Do not deploy before a prospective impact and fairness pilot. Full four-dimension comparison in [`artifacts/tradeoff_matrix.md`](artifacts/tradeoff_matrix.md) and the [technical report](reports/technical_report.md).

## Deliverables

- [Plain-language data and feature guide](reports/data_guide.md)
- [Analysis notebook](notebooks/recidivism_analysis.ipynb)
- [Interactive Streamlit app](app.py)
- [Presentation deck](reports/ISAF_Recidivism_Presentation.pptx)
- [Presentation notes and Q&A](reports/presentation_notes.md)
- [Technical report and model card](reports/technical_report.md)
- [Dataset pre-validation brief](reports/dataset_prevalidation.md)
- Reproducible source code in [`src/recidivism`](src/recidivism)

## Reproduce

Python 3.10+ is required. A CUDA GPU is recommended for TabICLv2.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1              # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/run_all.py               # every analysis + notebook + slides, in order
streamlit run app.py
```

`run_all.py` runs the steps below in dependency order, stops at the first failure, and prints how
to resume (`--from <step>`); `--only <step> ...` runs a subset. It checks the scikit-learn version
first: saved `.joblib` models only load with the exact version pinned in `requirements.txt`.

1. `train_evaluate` — train the three models; predictions, metrics, saved models
2. `interpretability` · `xper_attribution` · `stability_structural`
3. `fairness_audit` · `fairness_interpretability` — gaps, tests, age audit, FPDP, mitigation
4. `incumbent_benchmark` · `learning_curve` · `race_ab_test`
5. `tradeoff_matrix` · `improvement_journey` · `build_notebook` · `build_slides`

TabICLv2 takes about a minute per fit on CPU, so the full run takes 30–60 minutes without a GPU.
`python scripts/train_evaluate.py --skip-tabicl` is a **smoke test only**: it overwrites
`test_predictions.csv` and `model_metrics.csv` without the TabICL column, which breaks the step-2
scripts and the app until a full run restores it.

Model-selection scripts (run once; results are baked into `src/recidivism/modeling.py`):
`tune_xgboost.py`, `tune_logistic.py`, `compare_ml_models.py`.

Tests run with:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

## Repository map

```text
app.py                          Streamlit client prototype
artifacts/                      Held-out predictions, metrics, figures, saved models
notebooks/                      Executable analysis notebook
reports/                        Slides, report, notes, pre-validation brief
scripts/train_evaluate.py       Train the three models; baseline audit
scripts/interpretability.py     SHAP, LIME, global surrogate, PDP/ICE
scripts/xper_attribution.py     XPER decomposition of AUC
scripts/stability_structural.py Structural stability across refits
scripts/run_all.py              One command: every step below, in order
scripts/fairness_audit.py       FNR/selection gaps, CIs, TOST, course tests, age audit, frontier
scripts/fairness_interpretability.py  FPDP, candidate variables, proxy scatter, mitigation
scripts/incumbent_benchmark.py  Comparison with Georgia's existing risk score
scripts/learning_curve.py       Performance vs training size (agency size)
scripts/race_ab_test.py         Race as input vs not; counterfactual twins
scripts/tradeoff_matrix.py      3 models x 4 dimensions table
scripts/tune_*.py               CV hyperparameter searches (XGBoost, logistic)
scripts/compare_ml_models.py    CV comparison of six ML candidates
scripts/build_*.py              Notebook, slides, pre-validation PDF
scripts/improvement_journey.py  Figure for JOURNEY.md
src/recidivism/                 Config (features, labels), data, models, metrics
tests/                          Data-integrity, encoding, leak and metric tests
PLAN.md / JOURNEY.md            Work plan / process log
```

## Responsible-use boundary

The outcome is a new arrest, which combines behavior with policing and reporting processes. The cohort covers Georgia releases from 2013–2015 and cannot establish validity elsewhere or today. Dollar values in the app are transparent user-defined scenarios, not causal claims. Any real use needs prospective validation, an appeal process, benefit-only interventions, drift monitoring, and periodic subgroup audits.
