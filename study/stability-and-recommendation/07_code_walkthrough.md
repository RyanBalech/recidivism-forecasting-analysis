# 7 · Code walkthrough: where stability and the recommendation live

> The brief says every member must answer questions on "any part of the analysis, code,
> models". These are the files behind your section, with what each block does. Paths are
> relative to the repo root.

---

## 7.1 The selection rule everything uses

[`src/recidivism/metrics.py:31`](../../src/recidivism/metrics.py#L31) `capacity_selection(scores, capacity=0.20)`
- It selects exactly `round(n × 0.20)`, which is 1,561 of 7,807 people.
- Ties are broken by input row order (`np.argsort(-scores, kind="stable")`), so the
  result is reproducible.
- Every Jaccard, abstention and fairness-at-top-20% number goes through this function.

## 7.2 Structural stability: `scripts/stability_structural.py`

| Lines | What happens |
|---|---|
| 35 | `REFITS = {"logistic": 8, "xgboost": 8, "tabicl": 8}`: the same number for all three models |
| 41–50 | `refit()` fits one model on one resample and predicts the 7,807 evaluation rows. TabICL goes through `tabicl_frames` (mode-fill, the gender-leak fix). |
| 53–57 | `shap_importance()` gives mean \|SHAP\| per feature on 300 evaluation rows, for explanation stability |
| 62–65 | **`rng = np.random.default_rng(7)`** draws 8 bootstrap index arrays *once*, so every model gets the same resamples |
| 66–69 | Writes `stability_protocol.json`: seeds plus the SHA-256 of each resample |
| 72–82 | Loop: for each model and each resample, refit with **model seed 42** fixed (line 78; only the data varies) |
| 83–93 | For each of the 28 pairs: mean \|Δp\|, p95 \|Δp\|, Spearman ρ, **top-20% Jaccard** (line 92) |
| 94–96 | Spearman correlation of the two SHAP importance vectors |
| 97–103 | Coefficient of variation of the top-8 importances across refits |
| 105–111 | Saves `stability_pairs.csv`, `stability_contributions.csv`, `stability_summary.csv` (mean over pairs) |
| 116+ | Box plots → `artifacts/figures/structural_stability.png` |

**Likely question: "Why fix the model seed?"**
To isolate *training-data* sensitivity. If the seed varied too, you couldn't tell data
noise from algorithm noise. The extras measure the seed separately.

## 7.3 One-person stability: `scripts/individual_stability.py`

| Lines | What happens |
|---|---|
| 45–48 | `N_REFITS = 8`, `BOOTSTRAP_SEED = 7`, which **must match** the structural script |
| 52–55 | `bootstrap_samples()` regenerates the *same* 8 resamples (same seed, same draw order) |
| 58–76 | `refit_predictions()` returns an 8 × 7,807 matrix of probabilities |
| 79–93 | `per_person()`: for each person, mean, SD, min/max, **range** of scores, **times_selected** (0–8), and `decision_unanimous` = selected by 0 or by 8 refits |
| 96–132 | `abstention_curve()`: "contested votes" = min(times selected, 8 − times selected). For limits 0…4, keep people with contested votes ≤ limit, decide by **majority vote** (≥ 4 of 8), then report coverage, precision and FNR gaps by gender (M − F) and race (B − W) |
| 135–188 | `main()`: runs logistic and XGBoost (TabICL only with `--with-tabicl` on a GPU) and saves the three CSVs |
| 191+ | Three-panel figure → `artifacts/figures/individual_stability.png` |

**Likely question: "Why is the abstention baseline −0.090 and not −0.096?"**
- Line 107: `selected = times_selected >= 4`. At full coverage the decision is the
  *majority vote of 8 refits*, not the single published model.
- A tie at 4 of 8 counts as selected, so the majority rule picks 1,608 people rather than
  exactly 1,561.
- Line 126 computes FNR as 1 − the selection rate among re-arrested people in each group.

## 7.4 Recommendation inputs

| File | What it gives slide 19 |
|---|---|
| [`scripts/tradeoff_matrix.py`](../../scripts/tradeoff_matrix.py) | Reads `model_metrics.csv`, `stability_summary.csv`, `fairness_inference.csv` and `interpretability_summary.json`, and draws the colour-coded 3 × 4 matrix. Ranks full-precision numbers, not rounded strings (a bug the deep review fixed). |
| [`scripts/calibration_tests.py:132`](../../scripts/calibration_tests.py#L132) | Paired bootstrap for calibration, captured events and gender-gap differences; also writes `selected_set_overlap.csv` (Jaccard 0.849 between logistic and XGBoost) |
| [`scripts/train_evaluate.py`](../../scripts/train_evaluate.py) | Main fit: `model_metrics.csv` (AUC, Brier, timing, net value), `paired_comparisons.csv` |
| [`scripts/incumbent_benchmark.py`](../../scripts/incumbent_benchmark.py) | Current Georgia score vs the models, and the capacity/effectiveness sweeps |
| [`scripts/learning_curve.py`](../../scripts/learning_curve.py) | AUC at 1,500 / 5,000 / 10,000 rows (TabICL best when data is small) |
| [`src/recidivism/modeling.py`](../../src/recidivism/modeling.py) | The three model definitions and their tuned parameters (`LOGIT_PARAMS`, `XGB_PARAMS`, `TABICL_ESTIMATORS = 16`) |

## 7.5 Where it appears in the submission notebook

`Recidivism_Project_Submission.ipynb`, by cell index (0-based):

| Cells | Content |
|---|---|
| 36–37 | **§8 Stability.** The *self-contained* version: **3** bootstrap draws, each refit compared with the **original** model. It gives Jaccard 0.84 / 0.82 / 0.78, higher than the deck's 0.77 / 0.75 / 0.78 because it compares refits with the original, not with each other. |
| 43–44 | Four-dimensional comparison table |
| 45 | §10 Recommendation and limits |
| 142–144 | Appendix: stability summary (the deck's numbers) plus the replaced share (1 − J)/(1 + J) |
| 197–204 | Appendix: one-person stability and the abstention table |
| 205–216 | Appendix: "Testing the comparison instead of asserting it" (calibration, captured events, gender gap) |
| 217 | Appendix: the full recommendation with the counter-case |

## 7.6 Where it appears in the app (`app.py`)

| Lines | Tab | What to show in the demo |
|---|---|---|
| 272–318 | **Stability** | Refit table, per-person figure, and the **abstention slider** (coverage / precision / gender and race gap) |
| 320–338 | Economics | Sliders for capacity, cost and effectiveness (you can demonstrate the break-even live: set effectiveness to 12%) |
| 340–351 | **Governance** | The operating policy. ⚠️ Its first bullet says logistic has "smaller subgroup gaps", which the team withdrew. Ask the team to change it to "lower refit drift and direct interpretability; calibration and fairness are tied". |

## 7.7 Your extra analysis

[`analysis/stability_course_aligned.py`](analysis/stability_course_aligned.py):

| Function | Purpose |
|---|---|
| `bootstrap_samples()` | The same 8 resamples as the published audit (checked by SHA-256) |
| `importance()` | Exact SHAP importance per raw feature, without the shap package. Logistic: linear SHAP. XGBoost: its own `pred_contribs` TreeSHAP. |
| `pair_distances()` | Every distance for one pair of models: \|Δp\|, Spearman, Jaccard, ‖φ₁ − φ₂‖₂, and for logistic ‖θ₁ − θ₂‖₂, zero pattern and sign agreement |
| `run_regimes()` | Seed-only, bootstrap and disjoint halves (10 stratified splits) for both models |
| `coefficient_stability()` | Per-coefficient range across refits, share non-zero, sign flips |
| `c_sweep()` | 9 values of C: 5-fold training-only CV AUC plus bootstrap stability |
| `contested_profile()` | Re-reads `individual_stability.csv`: contested share by group and by distance to the cut |

### The professor's methods: `analysis/course_stability_methods.py`

| Function | Purpose |
|---|---|
| `retrain_on_more_data()` | Slides 187/191: production model on a random 50% (D₁) vs all the data (D₂), 5 draws, logistic and XGBoost |
| `_anchored_logit()` | **Slide 186:** minimises Σ NLL + γ‖w‖² + λ‖w − w₁‖² with L-BFGS and an analytic gradient; the intercept is unpenalised |
| `stability_constrained()` | D₁ = 40%, D₂ = 60% (separate, n₂ > n₁). Fits θ̂₁ on D₁; for each λ runs 5-fold CV inside D₂, then refits on D₂; records ‖θ̂₂ − θ̂₁‖, Jaccard with the old model and AUC |
| `penalty_comparison()` | Slides 43–44: L1 / L2 / elastic net, C by 5-fold CV, then stability over the 8 published resamples |

## 7.8 How to regenerate

```bash
python scripts/stability_structural.py        # needs a GPU for TabICL (8 refits)
python scripts/individual_stability.py        # CPU; add --with-tabicl on a GPU
python scripts/tradeoff_matrix.py
python study/stability-and-recommendation/analysis/stability_course_aligned.py   # CPU, ~2.5 min
python study/stability-and-recommendation/analysis/course_stability_methods.py    # CPU, ~1 min
```
