# Trustworthy Recidivism Forecasting

A scoring analysis for HEC Paris **Interpretability, Stability, and Algorithmic Fairness**, comparing logistic regression, XGBoost, and TabICLv2 for a hypothetical community-supervision software vendor.

The decision is prioritizing voluntary re-entry support using **three-year cumulative new-arrest risk at supervision start**. This is a retrospective course project, not a validated operational risk tool.

## Scope and findings

We use NIJ's original 18,028 training / 7,807 evaluation partition and 29 baseline inputs. Race, gender, and geography are excluded from scoring; race and gender remain available for audits. Post-release measurements are excluded.

The [2021 NIJ challenge](https://nij.ojp.gov/funding/recidivism-forecasting-challenge) evaluated separate annual forecasts, restricting later years to people not previously rearrested. **Our cumulative-target results are not comparable to its leaderboard.** The course brief permits this binary target.

The original evaluation set was repeatedly inspected during development ([JOURNEY.md](JOURNEY.md)). It is excluded from fitting but is no longer an untouched model-selection holdout. Results are exploratory.

Models achieve approximately **0.73 ROC AUC and 0.20 Brier loss**, compared with approximately **0.60 AUC** for the historical recorded supervision score. This supports improvement on this dataset, not superiority over tools agencies use today.

<!-- RESULTS:START -->
| Model | ROC AUC | Brier ↓ | ECE (10 bins) ↓ | Fit + prediction |
|---|---:|---:|---:|---:|
| Logistic regression (tuned L1) | 0.7298 | 0.2054 | 0.0129 | 0.99 s |
| XGBoost | 0.7324 | 0.2045 | 0.0112 | 8.65 s |
| TabICLv2 | 0.7328 | 0.2044 | 0.0199 | 41.82 s |

Current run: TabICLv2 on **NVIDIA GeForce RTX 4050 Laptop GPU**, conventional models on CPU. Timings are hardware-specific. XGBoost minus TabICLv2 AUC is -0.00044, with paired 95% interval [-0.00230, 0.00127]; this does not establish superiority or equivalence. XGBoost reduces Brier loss by 16.4% relative to training-prevalence probabilities.
<!-- RESULTS:END -->

**Recommend XGBoost for a prospective shadow pilot, with logistic regression as the transparent challenger.** Model choice weighs explanation cost, subgroup errors, runtime and refit sensitivity. Small-sample TabICLv2 results do not establish transferability to smaller agencies elsewhere. Economic values are scenarios, not measured savings.

## Results and deliverables

- [Exact performance](artifacts/model_metrics.csv), [probability baselines and Brier skill](artifacts/validation_baselines.csv)
- [Paired model differences with bootstrap intervals](artifacts/paired_comparisons.csv), [fixed-configuration training CV](artifacts/validation_cv.csv)
- [Intersectional audit](artifacts/intersectional_audit.csv), [four-dimension trade-offs](artifacts/tradeoff_matrix.md)
- [Technical report](reports/technical_report.md), [research and requirement review](reports/research_review.md)
- [Executed notebook](notebooks/recidivism_analysis.ipynb), [interactive app](app.py)
- [Slide deck](reports/ISAF_Recidivism_Presentation.pptx), [presentation and Q&A notes](reports/presentation_notes.md)
- [Data guide](reports/data_guide.md), [pre-validation brief](reports/dataset_prevalidation.md)
- [Package versions, input/model hashes and consistency checks](artifacts/validation_manifest.json)

The supplied course brief requires dataset pre-validation by **24 September 2026, 09:40**, and deliverables by **28 September 2026, 09:40**; presentation: 15 minutes plus 10 minutes Q&A. Instructor submission remains the team's responsibility.

## Reproduce

Use Python **3.11–3.13** (verified on 3.13). Version ranges are not a complete environment lock; the validation manifest records the installed scientific stack. TabICLv2 may download its checkpoint on first use. Repeated explanations and refits take substantially longer than one training/prediction run.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1              # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/reproduce.py
streamlit run app.py
```

The complete run regenerates dependent analyses, validates saved-model/prediction agreement, and builds notebook/slides. Allow tens of minutes or longer depending on hardware. It runs a leakage gate and a training-only [TabICLv2 ensemble sensitivity check](scripts/estimator_sweep.py); the shared 16-member configuration is then fitted on all training rows. It does not repeat historical hyperparameter search. The merged team additions include [logistic tuning](scripts/tune_logistic.py), [six-candidate ML comparison](scripts/compare_ml_models.py), and readable feature labels; their historical search artifacts are retained. Run `scripts/tune_xgboost.py` separately to explore new configurations; its output is not automatically adopted.

`reproduce.py` first checks that scikit-learn matches the version pinned in `requirements.txt` (saved `.joblib` models only load with that version). It stops at the first failed step and prints how to resume (`--from-step <step>`); `--only <step> ...` reruns a subset whose inputs already exist. The fairness steps are [`fairness_audit.py`](scripts/fairness_audit.py) (FNR/selection gaps, bootstrap CIs, TOST equivalence at ±5 points, course test table, age audit, frontier) and [`fairness_interpretability.py`](scripts/fairness_interpretability.py) (FPDP, proxy dependence, candidate-variable removal and re-estimation; logistic and XGBoost only).

`python scripts/train_evaluate.py --skip-tabicl` writes a conventional-model smoke run to `artifacts/smoke/`, preserving published three-model artifacts. `--output-dir PATH` supports isolated training outputs. The app reads `artifacts/`; live TabICLv2 inference is opt-in. `python scripts/validate_project.py` audits existing predictions and runs conventional-model CV.

## Interpretation boundaries

Arrest reflects behavior, policing and reporting. Georgia releases from 2013–2015 cannot establish present-day or external validity. Removing protected columns does not remove proxies. Threshold changes alter decisions, not calibration of unchanged probabilities. The group-specific threshold frontier uses evaluation labels and is an in-sample illustration, not validated mitigation.

Capacity analyses select exactly `round(n × capacity)`, breaking ties by input row order. The discrete incumbent has many ties, so its scenario value depends on this convention. Real allocation requires a justified tie policy, prospective impact assessment, monitoring and appeals. This prototype must not drive sanctions, detention, sentencing or surveillance.
