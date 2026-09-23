# Trustworthy Recidivism Forecasting

An end-to-end scoring analysis for the HEC Paris course **Interpretability, Stability, and Algorithmic Fairness**. The project compares a white-box model, a gradient-boosted model, and an open tabular foundation model on the [NIJ 2021 Recidivism Forecasting Challenge](https://nij.ojp.gov/funding/recidivism-forecasting-challenge).

The decision is framed as estimating three-year arrest risk at the start of parole supervision so that scarce, beneficial re-entry services can be offered. The score is not suitable for sanctions, detention, or increased surveillance.

## Results

The official NIJ `Training_Sample` indicator creates an untouched 18,028/7,807 train/test split. Race, gender, and residence geography are excluded from model inputs and retained for subgroup audits. Only baseline variables available at supervision start are used.

**The client's real question — better than the tool agencies use today?** `Supervision_Risk_Score_First`, Georgia's existing 1–10 actuarial score, reaches only **0.60 ROC AUC**. Every candidate model reaches ~0.73 and roughly doubles the net value of a capacity-limited support programme (~$2.75M → ~$5.1–5.3M).

| Model | ROC AUC | Average precision | Brier ↓ | Calibration error ↓ | Runtime* |
|---|---:|---:|---:|---:|---:|
| Incumbent score | 0.600 | — | — | — | — |
| Logistic regression | 0.7295 | 0.7691 | 0.2055 | 0.0132 | 1.2 s |
| XGBoost | 0.7326 | 0.7722 | 0.2044 | **0.0109** | 2.9 s |
| TabICLv2 | **0.7328** | 0.7722 | **0.2044** | 0.0199 | 23.1 s |

\*Training plus one full held-out prediction run. XGBoost hyperparameters come from a 5-fold CV search; TabICLv2 uses 16 ensemble members on a CUDA GPU (RTX 4060). Bootstrap 95% intervals across the three models overlap, so the three are a statistical tie.

**Recommendation:** deploy XGBoost for support allocation, with logistic regression as the transparent challenger; TabICLv2 only for very small agencies where its small-data edge is real (see the learning curve). XGBoost calibrates best, has the highest scenario net value and the smallest gender false-positive gap, runs ~10× faster than TabICLv2, and is SHAP-explainable (the foundation model has no native explanation path). Do not deploy before a prospective impact and fairness pilot. Full four-dimension comparison in [`artifacts/tradeoff_matrix.md`](artifacts/tradeoff_matrix.md) and the [technical report](reports/technical_report.md).

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
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python scripts/train_evaluate.py
python scripts/build_notebook.py
python scripts/build_slides.py
python scripts/build_prevalidation_pdf.py
streamlit run app.py
```

Use `python scripts/train_evaluate.py --skip-tabicl` for a CPU-only smoke run. The app expects the saved audit artifacts from a full run. Tests run with:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

## Repository map

```text
app.py                         Streamlit client prototype
artifacts/                     Held-out predictions, metrics, figures
notebooks/                     Executable analysis notebook
reports/                       Slides, report, pre-validation brief
scripts/train_evaluate.py      Training and four-dimension audit
scripts/build_deliverables.py  Notebook and slide generation
src/recidivism/                Data, model, and metric modules
tests/                         Data-integrity and metric tests
```

## Responsible-use boundary

The outcome is a new arrest, which combines behavior with policing and reporting processes. The cohort covers Georgia releases from 2013–2015 and cannot establish validity elsewhere or today. Dollar values in the app are transparent user-defined scenarios, not causal claims. Any real use needs prospective validation, an appeal process, benefit-only interventions, drift monitoring, and periodic subgroup audits.
