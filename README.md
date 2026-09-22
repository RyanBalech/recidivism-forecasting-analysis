# Trustworthy Recidivism Forecasting

An end-to-end scoring analysis for the HEC Paris course **Interpretability, Stability, and Algorithmic Fairness**. The project compares a white-box model, a gradient-boosted model, and an open tabular foundation model on the [NIJ 2021 Recidivism Forecasting Challenge](https://nij.ojp.gov/funding/recidivism-forecasting-challenge).

The decision is framed as estimating three-year arrest risk at the start of parole supervision so that scarce, beneficial re-entry services can be offered. The score is not suitable for sanctions, detention, or increased surveillance.

## Results

The official NIJ `Training_Sample` indicator creates an untouched 18,028/7,807 train/test split. Race, gender, and residence geography are excluded from model inputs and retained for subgroup audits. Only baseline variables available at supervision start are used.

| Model | ROC AUC | Average precision | Brier ↓ | Calibration error ↓ | Runtime* |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.7295 | 0.7691 | 0.2055 | 0.0132 | 0.59 s |
| XGBoost | 0.7299 | 0.7686 | 0.2054 | **0.0118** | 3.04 s |
| TabICLv2 | **0.7336** | **0.7719** | **0.2039** | 0.0197 | 7.13 s |

\*Training plus one full held-out prediction run on the development machine. TabICLv2 uses two ensemble members on an RTX 4050 Laptop GPU.

**Recommendation:** pilot XGBoost for support allocation, with logistic regression as the transparent challenger. Its predictive performance is statistically close to TabICLv2, it calibrates best, runs cheaply, and has the smallest observed race and gender false-positive gaps. Do not deploy before a prospective impact and fairness pilot.

## Deliverables

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
