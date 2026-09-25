# Trustworthy Recidivism Forecasting

**HEC Paris — Interpretability, Stability, and Algorithmic Fairness · Team 11**

Comparing a white-box model, a gradient-boosted model and a tabular foundation model for a
community-supervision software vendor, judged on predictive performance, interpretability,
stability and fairness rather than accuracy alone.

The decision is prioritising **voluntary re-entry support** using three-year cumulative
new-arrest risk at supervision start. Support allocation only — never sanctions, detention
or surveillance. This is a retrospective course project, not a validated operational tool.

---

## Start here

| If you want to… | Open |
|---|---|
| **Read and rerun the whole analysis** | [`Recidivism_Project_Submission.ipynb`](Recidivism_Project_Submission.ipynb) — fully executed and self-contained, data prep → recommendation |
| **Read the written report** | [`reports/technical_report.md`](reports/technical_report.md) |
| **See the deck** | [`reports/ISAF_Recidivism_Presentation.pptx`](reports/ISAF_Recidivism_Presentation.pptx) |
| **Run the client app** | `streamlit run app.py` |
| **Check a number** | [`artifacts/`](artifacts) — every figure and table regenerates from [`scripts/`](scripts) |

```bash
python -m pip install -r requirements.txt
streamlit run app.py                 # the client-facing lab
python -m pytest -q                  # automated tests
python scripts/reproduce.py          # full rebuild (GPU recommended)
python scripts/build_standalone_notebook.py  # rebuild only the submission notebook
```

---

## Headline results

<!-- RESULTS:START -->
| Model | ROC AUC | Brier ↓ | ECE (10 bins) ↓ | Fit + prediction |
|---|---:|---:|---:|---:|
| Logistic regression (tuned L1) | 0.7298 | 0.2054 | 0.0129 | 0.99 s |
| XGBoost | 0.7324 | 0.2045 | 0.0112 | 8.65 s |
| TabICLv2 | 0.7328 | 0.2044 | 0.0199 | 41.82 s |

Current run: TabICLv2 on **NVIDIA GeForce RTX 4050 Laptop GPU**, conventional models on CPU. Timings are hardware-specific. XGBoost minus TabICLv2 AUC is -0.00044, with paired 95% interval [-0.00230, 0.00127]; this does not establish superiority or equivalence. XGBoost reduces Brier loss by 16.4% relative to training-prevalence probabilities.
<!-- RESULTS:END -->

**Against the incumbent.** The historical recorded supervision score in this dataset reaches
≈0.60 AUC against our ≈0.73. That is improvement on this dataset, not superiority over tools
agencies run today.

**Recommendation: L1 logistic regression for a prospective shadow pilot, XGBoost as the
challenger.** XGBoost's +0.0025 AUC edge is real but does not reach the decision — at the
deployed rule the two models offer support to 85% of the same people, and the difference in
captured re-arrests has a confidence interval spanning zero. Calibration and subgroup gaps are
**tied**, not won by either. Logistic is chosen for refit stability and direct interpretability.

---

## Four findings worth reading

**1. Gender was fully recoverable through a missingness pattern.**
`Gang_Affiliated` is missing for every woman and no man. With missingness indicators, gender is
recovered from our own features at **AUC 1.000** — any model encoding NaN as a category had the
excluded attribute in full. Fixed by mode-filling, guarded by a regression test.
→ [`scripts/proxy_inference_audit.py`](scripts/proxy_inference_audit.py)

**2. Excluding an attribute does not remove it.**
Even in the shipped feature set, gender is recovered at 0.776 and race at 0.708. The twins in
the race A/B test score identically, and the group gaps persist anyway. Named proxies are gun
charges, mental-health and substance conditions, violent arrests, age and dependents — the
substance of the assessment, not incidental fields.

**3. Abstaining when uncertain is not fairness-neutral.**
13% of decisions flip across eight bootstrap refits, and a third of the people prioritised sit
at that margin. Referring contested cases to a human lifts precision 0.822 → 0.846 — and
**widens** the gender FNR gap −0.090 → −0.119, because 47% of the 118 selected women are at the
margin against 30% of 1,490 selected men.
→ [`scripts/individual_stability.py`](scripts/individual_stability.py)

**4. The impossibility result, on both sides.**
Race base rates barely differ (0.582 / 0.564), so those gaps are not forced by the data. Gender
base rates differ by 14 points (0.591 / 0.454), so calibration and equal error rates cannot both
hold — and the three models sit at different points on that trade-off.

**Fairness at the deployed top-20% rule.** Race gaps are equivalent within ±5 points for all
three models (TOST), and no race test survives Holm correction. Women who are re-arrested miss
support more often (FNR gap 10–12 points) and age is the largest disparity (23–25 points),
persisting even with the age field fixed.

---

## Repository map

```
Recidivism_Project_Submission.ipynb   executed notebook — complete course analysis in its own cells
app.py                                Streamlit client lab (6 tabs)
src/recidivism/                       data loading, modelling, metrics
scripts/                              analysis pipeline, one concern per file
tests/                                leakage and regression guards
data/                                 NIJ 2021 challenge datasets
artifacts/                            every generated table, figure and model
reports/                              deck, technical report, Q&A notes, reviews
docs/                                 plan, development log, assignment brief
```

**Submission notebook.** The root `.ipynb` imports only public Python packages and reads the
NIJ CSVs in `data/`. It defines preprocessing, all three models, evaluation, economic
scenario, interpretability, stability, fairness and the training-only mitigation check
in its own cells. It does not import `src/` or `scripts/`, call shell commands, or depend
on saved results. Running every cell is compute intensive, especially TabICL refits;
CUDA is used when available. The larger exploratory audit remains in the repository.

**Pipeline.** [`scripts/reproduce.py`](scripts/reproduce.py) runs every step in dependency order
and stops at the first failure (`--from-step` to resume, `--only` for a subset). It verifies that
scikit-learn matches the pinned version, since saved `.joblib` models load only with it.

| Area | Scripts |
|---|---|
| Training & performance | `train_evaluate`, `tune_logistic`, `tune_xgboost`, `incumbent_benchmark`, `learning_curve` |
| Leakage | `leakage_audit`, `deep_leakage_audit`, `leakage_stress`, `leakage_positive_control` |
| Interpretability | `interpretability`, `lime_local_fidelity`, `logistic_effects`, `explanation_agreement`, `xper_attribution` |
| Fairness | `fairness_audit`, `fairness_interpretability`, `mitigation_nested`, `proxy_inference_audit`, `race_ab_test` |
| Stability | `stability_structural`, `individual_stability` |
| Validation | `calibration_tests`, `validate_project`, `end_to_end_audit`, `accuracy_review` |
| Deliverables | `build_standalone_notebook`, `build_notebook` (extended artifact review), `build_slides`, `tradeoff_matrix`, `refresh_readme` |

---

## What this project does not establish

Stated plainly, because the analysis is only as good as its limits.

- **The evaluation set is not an untouched holdout.** It is excluded from fitting, but
  development decisions repeatedly inspected it ([`docs/JOURNEY.md`](docs/JOURNEY.md)). Results
  are exploratory; confirm on new data.
- **Not comparable to the NIJ leaderboard.** The [2021 challenge](https://nij.ojp.gov/funding/recidivism-forecasting-challenge)
  scored separate annual forecasts on conditional cohorts. Our cumulative binary target is a
  permitted course adaptation, not a reproduction.
- **No proven performance ceiling.** Three models converging is not evidence of one.
- **Dollar figures are scenarios.** Intervention effectiveness is assumed, not estimated. Risk
  ranking is not benefit ranking.
- **Arrest is a proxy for need,** institutionally mediated, and does not identify who benefits
  from support.
- **No temporal or external validation,** and no evidence that the support programme helps
  recipients. Both are preconditions for real allocation, alongside an appeals route, subgroup
  monitoring and pre-agreed stop rules.

---

## Further reading

| Document | What it covers |
|---|---|
| [Technical report & model card](reports/technical_report.md) | Full method, results and limitations |
| [Presentation & Q&A notes](reports/presentation_notes.md) | Talk sequence with timings, rehearsed answers |
| [Research & requirement review](reports/research_review.md) | Claims audited and corrected against sources |
| [End-to-end review](reports/end_to_end_review.md) | Reproduction, GPU checks, artifact integrity |
| [Deep leakage & accuracy review](reports/deep_review.md) | Release checks, 144 inner-fold fits, ensembles |
| [Data guide](reports/data_guide.md) | Every field, including the anonymised `_v1`–`_v4` |
| [Trade-off matrix](artifacts/tradeoff_matrix.md) | Three models × four dimensions |
| [Development log](docs/JOURNEY.md) · [Plan](docs/PLAN.md) | What we tried, including what we rejected |
| [Validation manifest](artifacts/validation_manifest.json) | Package versions, input and model hashes |
