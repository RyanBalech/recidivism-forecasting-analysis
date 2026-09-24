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

**Fairness at the proposed top-20% support rule** (FNR = re-arrested but not offered support): race allocation-error gaps are equivalent within a ±5-point tolerance for all three models (TOST; selection ratio 0.87–0.93), although within-group AUC is lower for Black people (about 0.72 vs 0.75); women who are re-arrested miss support more often (FNR gap about 10–12 points), and age is the largest disparity (about 23–25 points, older people selected far less; it persists when the age field is fixed). A fairness partial dependence analysis traces the gender gap mainly to gang affiliation, which is never recorded for women: dropping it shrinks the gap to 0.00 (logistic) / −0.01 (XGBoost), partly by raising men's FNR, at about 0.014 AUC and worse calibration for women. See the [technical report](reports/technical_report.md#fairness).

**Recommend L1 logistic regression for a prospective shadow pilot, with XGBoost as the challenger.** XGBoost's +0.0025 AUC edge is small; logistic is equal or better on explanation, refit stability, subgroup gaps and runtime. Small-sample TabICLv2 results do not establish transferability to smaller agencies elsewhere. Economic values are scenarios, not measured savings.

The [fresh accuracy review](reports/deep_review.md) found no material improvement from deeper trees, alternate encoding, native categorical trees or sigmoid calibration. A fixed XGBoost–TabICL average is a promising research challenger (evaluation AUC 0.7332, Brier 0.2042), but its paired improvement intervals include zero and its top-capacity capture is lower. The review also measured thread-dependent XGBoost fitting differences larger than the original model-family AUC gap.

## Results and deliverables

- **Professor submission notebook:** [`Recidivism_Project_Submission.ipynb`](Recidivism_Project_Submission.ipynb). It is fully executed and is the single readable entry point from data preparation and leakage controls through the three model families, performance, interpretability, stability, fairness and recommendation. Its saved outputs can be read as a standalone file; keep the repository folders beside it to rerun cells and resolve the relative data/source paths. Set `RUN_FULL_PIPELINE=True` inside the notebook only when a complete GPU-backed rebuild is required.
- [End-to-end review](reports/end_to_end_review.md): fresh three-model reproduction, actual GPU checks, CPU/CUDA comparison, paired stability refits, leakage gates and artifact integrity. Run `python scripts/end_to_end_audit.py` to check published results.
- [Fresh leakage and accuracy review](reports/deep_review.md): independent release checks, 144 inner-fold XGBoost fits, calibration, native categorical trees, and a fixed ensemble challenger. Reproduce with the commands in that report.
- [Exact performance](artifacts/model_metrics.csv), [probability baselines and Brier skill](artifacts/validation_baselines.csv)
- [Paired model differences with bootstrap intervals](artifacts/paired_comparisons.csv), [fixed-configuration training CV](artifacts/validation_cv.csv)
- [Intersectional audit](artifacts/intersectional_audit.csv), [four-dimension trade-offs](artifacts/tradeoff_matrix.md)
- [Technical report](reports/technical_report.md), [research and requirement review](reports/research_review.md)
- [Canonical executed notebook copy](notebooks/recidivism_analysis.ipynb), [interactive app](app.py)
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

[`proxy_inference_audit.py`](scripts/proxy_inference_audit.py) measures what exclusion actually removes by predicting each protected attribute from the model's own feature set; [`individual_stability.py`](scripts/individual_stability.py) keeps the per-person predictions across the shared bootstrap refits, so decision stability and a selective-prediction (abstention) policy can be evaluated for one person rather than for the cohort.

[`calibration_tests.py`](scripts/calibration_tests.py) tests the model comparison instead of asserting it: ECE at several binning choices (the ranking is not stable), bin-free Cox calibration and the Spiegelhalter z-test, paired bootstraps on calibration/captured events/gender gap, and the overlap between the selected sets at the deployed rule.

The interpretability steps are [`interpretability.py`](scripts/interpretability.py) (SHAP, global surrogate, PDP/ICE), [`lime_local_fidelity.py`](scripts/lime_local_fidelity.py) (category-aware LIME over fixed cases and seeds, with the local surrogate R² reported beside every explanation), [`logistic_effects.py`](scripts/logistic_effects.py) (average marginal effects and categorical probability contrasts, in probability units rather than transformed log-odds) and [`explanation_agreement.py`](scripts/explanation_agreement.py) (rank agreement between SHAP, permutation importance and XPER, plus the XPER reconstruction residual). [`mitigation_nested.py`](scripts/mitigation_nested.py) re-selects the FPDP candidate inside each training fold and scores the mitigation on held-out folds, so selection and assessment never share data; the evaluation cohort is not touched.

`python scripts/deep_leakage_audit.py` independently checks the original NIJ releases, including unchanged baseline values and training-only imputation. Passing these checks does not establish exact measurement timing or remove the bias from repeated evaluation-set inspection.

## Interpretation boundaries

Arrest reflects behavior, policing and reporting. Georgia releases from 2013–2015 cannot establish present-day or external validity. Removing protected columns does not remove proxies. Threshold changes alter decisions, not calibration of unchanged probabilities. The group-specific threshold frontier uses evaluation labels and is an in-sample illustration, not validated mitigation.

Capacity analyses select exactly `round(n × capacity)`, breaking ties by input row order. The discrete incumbent has many ties, so its scenario value depends on this convention. Real allocation requires a justified tie policy, prospective impact assessment, monitoring and appeals. This prototype must not drive sanctions, detention, sentencing or surveillance.
