# Project plan — Trustworthy Recidivism Forecasting (Team 11)

Course: HEC Paris — *Interpretability, Stability, and Algorithmic Fairness* (Pérignon / Saurin).
Deliverables due **Mon 28 Sep, 9:40 AM**: slide deck, notebook (data prep → evaluation), client app.
Presentation 15 min + Q&A 10 min. Every member must defend any section.

## Integration status — research review

The review preserves this team's pending course-aligned work list. Updated code includes the
merged L1 logistic configuration, ML comparison scripts and readable labels. All dependent
artifacts are being regenerated using that combined code. See reports/research_review.md.

Methodological corrections take precedence over historical claims below: no proven data ceiling;
NIJ annual challenge scores are not comparable to our cumulative target; repeated evaluation
inspection means no untouched holdout; thresholds do not change calibration of unchanged scores;
Jaccard is not a person-level switching rate. Intersectional results are descriptive with sample
sizes, not claims that small-cell gaps are reliably estimated. Group-specific policies require
independent validation and policy review, not a blanket legal conclusion.

Completed by this review: paired model intervals, conventional fixed-configuration CV, calibrated
incumbent/prevalence baselines, intersectional denominators, model/prediction consistency checks,
isolated smoke outputs, ordered full reproduction, missing-value-safe app, and corrected
report/notebook/slides. The original course-method TODOs below remain visible for team ownership.

## Client and framing

**Client (fictional):** a software vendor that sells risk-assessment tools to US state
community-supervision agencies. The tool is scoped to allocate **re-entry support services,
not detention or sanctions** — this scoping is the primary ethical defense and must appear
throughout.

Agencies vary in data volume (small county vs large state), which motivates the learning-curve
analysis of which model to ship to whom.

## Dataset

NIJ 2021 Recidivism Forecasting Challenge (Georgia parolees 2013–2015). Binary target:
re-arrest within 3 years. Official split 18,028 train / 7,807 test. Base recidivism rate 57.8%
(balanced — no resampling). Protected attributes Race (Black/White) and Gender (M/F) are kept
for audit only; Race, Gender, and Residence_PUMA (a race proxy) are excluded from model inputs.
Dynamic supervision variables are excluded to prevent leakage.

## Three models (brief requirement: white-box / ML / TFM)

- **Logistic regression** — white-box; one-hot + L1, C=0.2154 from a 5-fold CV grid
  (`scripts/tune_logistic.py`). The CV curve is flat for C in [0.05, 1], so tuning barely matters —
  but it is now backed by code, and L1 sparsity is the course's own argument for penalized LR.
- **XGBoost** — ML; count columns ordinal-encoded, hyperparameters from 5-fold CV random search.
  Kept after a CV comparison against LightGBM, CatBoost, HistGB, random forest and EBM
  (`scripts/compare_ml_models.py`): CatBoost 0.7343 vs XGBoost 0.7342 CV AUC is a tie, and
  XGBoost is already wired into every audit.
- **TabICLv2** — Tabular Foundation Model.

## Historical pre-review status — superseded by integrated rerun

`Gang_Affiliated` is missing for **all 3,167 women and no man**. TabICL encoded NaN as its own
category, so it could read the excluded Gender attribute. Fixed in PR #1 (`tabicl_frames` now
mode-fills categoricals, like the other two pipelines; regression test added). Logistic and
XGBoost were never affected (mode imputation, no categorical missing flag). No other feature has
gender- or race-dependent missingness (checked).

**Every TabICL number in `artifacts/`, the deck, the notebook, the report, README and JOURNEY.md
predates the fix** — including the 0.273 gender FPR gap, "TabICL drifts most", and the race A/B
"proxy leakage". Re-run the full pipeline before quoting any TabICL figure. The leak itself is a
strong Q&A story: our fairness audit found a protected attribute leaking through missingness.

## First post-fix numbers (CPU run, 23 Sep evening — historical; current values are in the status audit below)

| Model | AUC before fix → after | Brier | ECE |
|---|---|---:|---:|
| Logistic (tuned) | 0.7295 → 0.7298 | 0.2054 | 0.0129 |
| XGBoost | 0.7326 → 0.7325 | 0.2044 | 0.0103 |
| TabICLv2 | **0.7338 → 0.7317** | 0.2047 | 0.0208 |

TabICL's lead came partly from reading gender through missingness. After the fix it ranks
**below XGBoost** and calibrates worst; like the other two it now over-predicts women (mean score
0.52 vs observed 0.454). Fairness at top-20% (FNR = missed support, M − F / B − W / <33 − 33+):
gender −0.10 to −0.12 (women missed more, all significant), race within ±0.02 (TOST certifies
equivalence within ±5 pts for logistic and XGBoost), **age −0.23 to −0.25: 97% of re-arrested
people aged 48+ get no support vs ~50% at 18–22.** FPDP candidate variables for the gender gap:
`Gang_Affiliated` (never recorded for women) and `Age_at_Release`; dropping gang affiliation
removes the gender FNR gap for logistic (p 0.00 → 0.99) at −0.014 AUC.

## How to re-run

`python -m pip install -r requirements.txt` (scikit-learn is pinned to 1.7.2 so saved models
load for everyone), then `python scripts/reproduce.py` (`run_all.py` was renamed to
`reproduce.py` in the merge). It runs every step in order and prints how to resume if one
fails; `--from-step X` resumes and `--only A B` runs selected steps. Push the regenerated
`artifacts/`, notebook and deck afterwards.

**What needs the GPU and what does not.** Only steps that *fit or query TabICL* need the GPU:
`estimator_sweep`, `train_evaluate` (writes `test_predictions.csv`), `learning_curve`,
`interpretability` (TabICL PDP/ICE), `stability_structural`, `race_ab_test`. Everything that only
reads `test_predictions.csv` or uses logistic/XGBoost runs on CPU in minutes:
`fairness_audit`, `fairness_interpretability`, `xper_attribution`, `tradeoff_matrix`,
`improvement_journey`, `build_notebook`, `build_slides`. So the missing fairness outputs (see the
audit below) can be generated **without** the GPU:
`python scripts/reproduce.py --only fairness_audit fairness_interpretability tradeoff_matrix build_notebook build_slides`.

**Local environment (macOS, 23 Sep audit).** `pytest`: 12 passed, 8 failed, all failures are
environment, not assertions: missing `libomp` (`brew install libomp`), missing `shap`, and
scikit-learn 1.9.1 installed vs 1.7.2 pinned (saved `.joblib` models fail to load). The local venv
is Python 3.12; artifacts were produced on Windows / Python 3.13.5 / CUDA. Fix the environment
before any CPU re-run.

## Status audit — 23 Sep evening (code vs artifacts vs deliverables)

**Resolved (commit 4a076cf):** the published artifacts predated the course-aligned fairness code
(the GPU run 405b608 ran before 598b0dd was merged). The CPU-only steps `fairness_audit`,
`fairness_interpretability`, `tradeoff_matrix`, `build_notebook` and `build_slides` were re-run on the
unchanged `test_predictions.csv`, so all fairness outputs now share the GPU run's predictions. The
CPU numbers in "First post-fix numbers" are confirmed by these outputs; quote the artifact values
below, not that historical section.

**Authoritative current results (model outputs from GPU run 405b608; fairness outputs from
4a076cf on the same predictions) — safe to quote:**
- Performance: AUC 0.730 / 0.732 / 0.733 (logistic / XGBoost / TabICL), Brier ≈ 0.205 / 0.204 /
  0.204. XGBoost − logistic AUC +0.0025 [0.0006, 0.0044]; TabICL vs XGBoost not significant.
  Incumbent AUC 0.60; net value at 20%: $5.05M / $5.17M / $5.12M vs $2.73M incumbent.
- Interpretability: SHAP top features agree for logistic and XGBoost (age, gang affiliation,
  felony arrests, parole/probation-violation arrests, prison years). Surrogate R² = 0.61
  (moderate). LIME local R² = 0.25 (weak). PDP/ICE for 3 features × 3 models. XPER on 150 rows.
  TabICL permutation importance covers only 10 features.
- Stability (8 bootstrap refits each, 28 pairs): mean |Δp| 0.032 / 0.035 / 0.036; Spearman 0.975 /
  0.971 / 0.971; top-20% Jaccard 0.772 / 0.755 / 0.770 (≈13–15% of the selected set replaced);
  SHAP-importance rank correlation 0.86 (logistic) / 0.87 (XGBoost).
- Fairness at top-20%: gender FNR gap (M − F) −0.096 / −0.112 / −0.124, all significant (women who
  are re-arrested miss support more); women over-predicted (mean score ≈ 0.52 vs base rate 0.454,
  ECE 0.07–0.08; Black women n=339, ECE 0.106). Race: logistic nothing significant; XGBoost and
  TabICL selection-rate and FPR gaps ≈ +0.02 to +0.03, significant (Black people selected slightly
  more); race FNR gaps not significant for any model.
- Equivalence (TOST, δ = ±5 pts, top-20%): race selection, FNR and FPR gaps **equivalent** for all
  three models; gender and age gaps **different**; precision gaps inconclusive. Course test table:
  gender and age rejected on every criterion (incl. sufficiency for gender); race only borderline
  (logistic predictive equality p = 0.049 while its bootstrap CI includes 0).
- Age (<33 − 33+): FNR gap −0.245 / −0.228 / −0.237; about 97% of re-arrested people aged 48+ are
  not selected vs about 47–49% at 18–22.
- FPDP (gender, logistic and XGBoost): candidates `Gang_Affiliated` and `Age_at_Release`. Raw gang
  field has Cramér's V = 1.0 with gender (never recorded for women). Race proxies weak (max V ≈ 0.18).
- Mitigation: drop gang + re-estimate → equal-opportunity p 0.000 → 0.995 (logistic) / 0.67
  (XGBoost) at about −0.014 AUC (≈ 6× the XGBoost − logistic gap); statistical parity still
  rejected. Dropping age does not remove the gender gap. Panel B for XGBoost picks "gang = Yes for
  everyone" — an in-sample artefact, not a rule.

**Inconsistencies — status after the documentation pass (24 Sep):**
- ✅ Recommendation decided (24 Sep): **logistic in the shadow pilot, XGBoost as challenger**.
  Report, deck (slides 5, 8, 16), notebook, README, Q&A notes and app now say the same thing.
- ✅ Deck slide 11 now headlines gender FNR gap, race TOST equivalence and age FNR gap.
- ✅ Deck slide 6: the ECE "winner" card is replaced by the paired XGBoost − logistic AUC gap.
- ✅ New deck slide 13 (FPDP: gang affiliation as the gender-gap candidate); later slides renumbered.
- ✅ Report fairness section, notebook fairness findings, Q&A notes, README summary and JOURNEY.md
  (stability numbers, stale "TabICL proxy leakage") updated to the current artifacts.
- ❌ `surrogate_tree.txt` and raw CSVs still show `_v1`…`_v4`; the deck/app use readable labels.
- ❌ `run_manifest.json` has no package versions (they are in `validation_manifest.json`) and does
  not record the CPU-only fairness re-run (it is generated by `train_evaluate.py`, a GPU step; fix in
  code at the next full run rather than editing the JSON by hand).

## To-do order (revised 23 Sep evening after the status audit)

Tags: **[CPU]** = can be done now without the GPU once the local environment is fixed;
**[GPU]** = needs TabICL refits/predictions; **[TEXT]** = documents only, no code run.
Status: ✅ done · 🟡 partial · ❌ not started · ⚠️ code done, outputs missing.

### P0 — required before the deadline
1. ✅ **[CPU] Generate the fairness outputs** (commit 4a076cf) on the existing
   `test_predictions.csv`; notebook and deck rebuilt. Manifest entry still missing (see above).
2. ✅ **[TEXT] Justify δ for TOST.** Report and notebook state the ±5-point rationale, that δ was
   fixed in code before TOST ran but after the gap point estimates had been seen, the analytic-vs-
   bootstrap variance caveat, and the equivalent / different / inconclusive classification (the
   notebook now prints it). Optional [CPU]: switch TOST to the bootstrap 90% CI rule for consistency.
3. ✅ **[TEXT] Team decision on the recommendation (decided 24 Sep: logistic, XGBoost challenger), derived from the 3 × 4 matrix,** then make
   PLAN, report, deck and notebook say the same thing. Current evidence to weigh: XGBoost beats
   logistic by only +0.0025 AUC, has a larger gender FNR gap (−0.112 vs −0.096), the lowest
   top-20% overlap (0.755 vs 0.772), and significant (small) race selection/FPR gaps where logistic
   has none. "Logistic in production, XGBoost as challenger" needs serious consideration.
4. 🟡 **[TEXT] Fix deck/doc inconsistencies** listed above: FNR headline, ECE card, FPDP slide and
   narrative done; readable labels in raw artifacts and the manifest remain. **Check the rebuilt deck
   visually** (slides 6, 11, 12, 13): layout was not rendered during the edit.

### P1 — methodological gaps the jury is likely to probe
5. **[CPU] FPDP / mitigation logic.** "Candidate variable" and "mitigation worked" are currently
   decided by p > 0.05, which is the same misuse as "not significant = fair". Report gap magnitude
   with CI (or TOST) next to the p-value. Panel B picks the neutral value that maximizes the p-value
   **on the test set** (in-sample optimisation): say so, or choose it on a training-only validation
   split. Add calibration (ECE) and captured events to the mitigation table, not only AUC.
   Current evidence: XGBoost Panel B neutralizes gang to "Yes" for everyone, and dropping gang still
   leaves statistical parity rejected — both are already disclosed in the report.
6. **[CPU] Gang_Affiliated as a gender measurement problem.** It is missing for every woman and
   mode-imputed as "No", which pushes women's scores down. Make this the main FPDP story. The
   mitigation table now reports AUC, group FNRs and gap, captured events and women's mean score for
   dropping the feature. Do **not** add an explicit "not recorded" level: it is identical to gender
   (Cramér's V = 1.0), i.e. the leak we removed.
7. ❌ **[CPU] XAI disagreement audit.** SHAP vs permutation importance vs XPER (vs LIME where
   meaningful): Spearman rank correlation, top-5 overlap, sign agreement. Example already visible:
   SHAP ranks gang affiliation #2, XPER (logistic) puts it after misdemeanor arrests. Repeat XPER
   with 3–5 seeds and/or a larger sample (currently 150 rows, 60 coalitions) and report the spread.
8. 🟡 **[CPU] Complete the logistic white-box table.** The current table is coefficient + odds
   ratio per *transformed* unit. Add: original-feature statements (level vs reference), average
   marginal effects, bootstrap intervals, and the share of refits in which L1 keeps each
   coefficient non-zero. Note that L1 invalidates textbook standard errors; no causal wording.
9. ❌ **[CPU for logistic/XGBoost, GPU for TabICL] Stability, aligned with the course (§7):**
   - ✅ same bootstrap resamples for all three models (done in 1abf221; `stability_protocol.json`);
   - separate sampling from estimator randomness: data-only refits with a fixed seed, and seed-only
     refits on fixed data (XGBoost subsampling; TabICL ensemble seed);
   - two **disjoint halves** of the training set, next to the bootstrap refits;
   - ‖θ₁ − θ₂‖₂ on logistic coefficients and ‖φ(f₁) − φ(f₂)‖₂ on importance vectors;
   - per-person score SD and **selection flip rate** near the top-20% cutoff, overall and by
     gender/race;
   - report distributions/CIs, not only means (the 28 pairs are not independent);
   - note that bootstrap duplicates may affect TabICL's in-context learning;
   - optional: stability vs performance over C for logistic (course p192).
10. ❌ **[CPU] PDP limits.** Add ALE for age and prior arrests, or a correlation/support audit
    showing where PDP builds implausible records. Summarise ICE heterogeneity (e.g. SD of the ICE
    slopes) instead of only plotting 200 faint curves.
11. **[CPU] Multiple testing and conditioning.** More than 90 fairness intervals are unadjusted:
    disclose this or add a Holm correction. Conditional statistical parity conditions on the Georgia
    risk score, which is built from arrest history and may itself carry bias: justify it or add a
    second conditioning set.
12. 🟡 **[CPU] Explanations for a borderline person.** The explained person is the highest-risk case
    (p ≈ 0.94). Add one person near the top-20% cut where the models disagree, and state LIME's
    local R² next to its chart.

### P2 — if time permits
13. 🟡 **[GPU] TabICL per-person explanation** (sampled KernelSHAP, cached) and permutation
    importance on all 29 features (currently 10). Otherwise quantify the compute cost and keep it as
    a stated deployment cost.
14. **[GPU] Full re-run** only if any model-side code changes. Then regenerate everything from one
    manifest (include package versions in `run_manifest.json`).
15. 🟡 **[CPU] Notebook data-preparation section**: target balance and official split exist; add the
    excluded columns with reasons, encoding choices per model, and the gang/gender missingness table
    with its consequence.
16. **[TEXT] Q&A rehearsal**: each member defends a section they did not write.

### Status of the earlier to-do list (pre-audit numbering)
| Earlier item | Status |
|---|---|
| 1. Fairness, course-aligned | ✅ outputs generated (4a076cf) and written up; method caveats in P1.5, P1.11 |
| 2. Notebook data preparation | 🟡 → P2.15 |
| 3. Logistic table / per-person for 3 models / XPER vs PI vs SHAP | 🟡 / 🟡 / ❌ → P1.8, P2.13, P1.7 |
| 4. Stability: course distances, disjoint halves, TabICL seed-only | ❌ → P1.9 |
| 5. Paired model tests artifact | ✅ `paired_comparisons.csv` |
| 6. Full re-run + app update | ✅ GPU run + CPU fairness re-run on the same predictions; app now finds all fairness figures |
| 7. Recommendation decision | ✅ logistic, XGBoost challenger; all deliverables aligned |
| 8. Q&A rehearsal | ❌ → P2.16 |

## Previous to-do order (validated 23 Sep, kept for history)

Priority = what the brief grades and what the jury will ask. Details under each dimension below.
1. **Fairness, course-aligned** [CODE DONE — `fairness_audit.py`, `fairness_interpretability.py`; numbers from the GPU re-run] — FNR primary; course test table + TOST; FPDP/candidate variables
   (gender first); X/D vs X/Y scatter (= proxy answer); course mitigation; **age audit**.
2. **Notebook data-preparation section** (brief: "from data preparation to model evaluation").
3. **Interpretability** — logistic coefficient table; per-person explanations for all three models;
   XPER vs PI vs SHAP.
4. **Stability** — course distances; disjoint halves; TabICL seed-only variability.
5. Paired model tests saved as an artifact.
6. **Full re-run** with the leak fix (`python scripts/reproduce.py`, GPU teammate), then **app update** (FNR, TabICL per-person explanation).
7. Team decision on the recommendation → deck, report, notes, README, JOURNEY.md.
8. Q&A rehearsal.

## Key findings so far

- All three models plateau at ~0.73 AUC / ~0.20 Brier. Confirmed by CV search (best CV AUC
  0.7343) and by six ML candidates all landing at 0.728–0.734 CV AUC. This is not proof of a signal ceiling; NIJ's annual forecasts have a different target. Statistical performance is part of the brief.
- **Paired bootstrap on the test set** (same people, 1,000 resamples): XGBoost beats logistic by
  +0.003 AUC and −0.001 Brier, both significant but small; TabICL vs XGBoost is **not** significant
  on AUC, Brier or ECE. **ECE differences are not significant for any pair**, so "calibrates best"
  cannot justify a recommendation. (Pre-fix TabICL numbers; re-check after the re-run.)
- **Incumbent benchmark:** `Supervision_Risk_Score_First` is Georgia's existing 1–10 actuarial
  score that agencies use today. Standalone AUC = **0.60**, vs **0.73** for our models. The
  client's real question — "is any of this better than what we already deploy?" — answers itself:
  our models more than double the lift over random. This is the economic centerpiece.
- **Learning curve** (1,500 / 5,000 / 10,000 / 18,028 rows, 3 seeds, fixed test set): TabICL leads
  at every size but its edge over XGBoost shrinks from +0.021 AUC at 1,500 to +0.004 at full data.
  No crossover. XGBoost is ~35× faster and far more explainable. (Pre-fix TabICL numbers; the full-data
  TabICL point has 1 seed, not 3 — say so or add seeds.)

## Non-negotiable framing rule (from two external reviews)

The brief says "for each model, you assess four dimensions." Every dimension must be reported for
**all three models**, not just XGBoost. The notebook must contain a complete 3-models × 4-dimensions
matrix even if slide time is uneven. Deliver an explicit **trade-off comparison slide** immediately
before the recommendation.

## Work plan (value order)

### Business case / economic — the hook  [DONE]
1. Benchmark economic value + AUC against the **incumbent score** (`Supervision_Risk_Score_First`,
   Georgia's actuarial tool) and against random allocation. Sensitivity sweep over capacity and
   effectiveness. Result: incumbent AUC 0.60 vs models 0.73; net value $2.75M vs ~$5.1M at 20%
   capacity; models dominate at every capacity. **State missing-value handling: incumbent has 145
   missing test values, imputed at the median.**

### Fairness — the centerpiece (required dimension + course headline)
2. **Audit at the deployed operating point, not 0.5.** [DONE] The product allocates the top 20% by
   risk, so report every fairness gap at the top-20% capacity threshold AND at 0.5. (The quoted TabICL
   0.273 → 0.07 is pre-fix.)
3. **[CODE DONE; deck slide 11 still FPR — see P0.4] Make FNR / equal opportunity the primary metric.** Being flagged
   high-risk means *getting support*, so a false positive is not the harm; the harm is a false
   negative — someone who needed help and was not selected. Primary: FNR gap + selection rate
   (statistical parity) + within-group calibration (sufficiency). FPR becomes secondary. Update
   `fairness_audit.py`, the frontier, `tradeoff_matrix.py`, and the slides. Pre-fix data already
   shows the story: women have a significantly higher FNR at top-20% for all three models
   (+0.09 logistic, +0.11 XGBoost); race FNR gaps are not significant.
4. Bootstrap confidence intervals on every gap (syllabus: "metrics and inference tests"). Female =
   950 test rows, so show interval width, not point estimates.
5. **Impossibility result — split by attribute (corrected):**
   - Race: similar base rates do not prove joint attainability of fairness criteria or that the gap is fixable.
   - Gender: differing base rates motivate examining trade-offs, but descriptive rates do not prove why a particular model has a gap. Thresholds leave score calibration unchanged.
6. Mitigation + accuracy/fairness frontier. **Legal caveat (Q&A exposure):** group-specific
   group-aware policies need context-specific policy/legal review and independent validation. Present group thresholds as an *analytic device to trace the frontier*, then show a
   race-blind alternative (single threshold minimizing the gap, or pre/in-processing) and its cost.
   Note that scoping to service allocation (not sanctions) changes the legal calculus.
   **[CODE DONE, output not regenerated — see P0.1] Re-target the frontier to gender FNR**, and add the course's mitigation (below).

**[CODE DONE, outputs missing — see P0.1, P0.2, P1.5, P1.11] Align with the course slides (§8, pp. 238–277).** These are taught methods the jury will
look for:
- **Fairness test statistics** (χ² / CMH / z-tests) with p-values, in the course's table layout
  (statistical parity, conditional statistical parity, equal odds, equal opportunity, predictive
  equality, sufficiency) × 3 models × {race, gender}. Keep the bootstrap CIs alongside.
- **State the course's convention mapping on the slide:** course Y=1 = "good type" who deserves the
  favorable output, Ŷ=1 = favorable output. Here Y=1 = would be re-arrested (needs support) and
  Ŷ=1 = selected for support, so the course's *equal opportunity* is exactly our equal FNR.
- **Fairness interpretability: FPDP + candidate variables**, and the **X/D vs X/Y dependence
  scatter** (p260). This doubles as the **proxy-variable answer** (reviewer #7: are gang
  affiliation or prior arrests proxies for race?). A candidate variable is defined relative to a
  *rejected* fairness null, so run FPDP for **gender** (rejected for all models) and for race only
  where a race test rejects; for logistic at top-20% nothing rejects on race.
- **Age is a protected attribute in the course (p266)** and our top feature (test re-arrest rate 72%
  at 18–22 vs 44% at 48+). Add an age-group audit (selection rate, FNR by age band) and a stated
  justification: age is a legitimate, validated risk factor, and in a support-allocation product a
  higher score means *more* help for younger people. Expect this question.
- Intersectional cells are small (339 Black women in test): report descriptive counts/metrics with uncertainty caveats; avoid categorical fairness claims.
- **Course-style mitigation:** neutralize candidate variables with and without re-estimation,
  report SP p-value + AUC (pp. 261–262), instead of relying only on group thresholds.
- **Fairness equivalence (TOST, tolerance δ):** "not significant" ≠ "fair". Use a prespecified equivalence margin and TOST to assess whether
  that race gaps are within δ (e.g. 5 pts) — the only way to claim race fairness honestly.
- Explain the gender gap with the impossibility result + within-group calibration (pre-fix: logistic
  and XGBoost over-predict women, 0.52 vs 0.454 observed; re-check TabICL after the fix).

### Interpretability (required dimension — for ALL three models)
7. SHAP individual waterfall for logistic + XGBoost (notebook + app).
8. Global surrogate (shallow tree on XGBoost preds) + PDP/ICE (model-agnostic, covers all three) +
   one LIME example.
9. **TabICL interpretability story, said deliberately:** no native explanation path; KernelSHAP over
   7,807 rows is impractical. PDP/ICE cover it model-agnostically. Frame the missing native path as a
   *deployment cost*, not an oversight — it feeds the recommendation.
10. Attempt XPER (instructor's method); permutation importance is the documented fallback. [DONE on
    AUC, 150-row sample — noisy; consider a larger sample or XPER on the economic metric, p202]

Done since: SHAP global + individual (logistic, XGBoost), LIME, depth-3 surrogate (R²=0.61), PDP/ICE
for all three, XPER. Readable feature labels everywhere (`config.pretty`; no more `_v1`).
**[TODO — see P1.7, P1.8, P2.13]**
- **Logistic coefficient table**: coefficient, odds ratio, average marginal effect (course p66). The
  white-box advantage is invisible without it. With an L1 penalty the textbook standard errors and
  p-values do not apply, so report bootstrap intervals from the stability refits instead (and the
  share of refits in which L1 keeps each coefficient non-zero).
- **Individual explanations for all three models** on 1–2 test people (one clearly selected, one near
  the top-20% cut where models disagree). KernelSHAP on raw features makes TabICL explainable for a
  single person, so "no explanation path" becomes "explainable per person, but slow".
- **XPER vs permutation importance vs SHAP** comparison chart (course pp. 206/208) and a note on the
  disagreement problem (Krishna et al.).

### Stability (currently the weakest dimension)
11. Real structural stability: refit across seeds/resamples, measure distance between resulting models
    and drift in feature contributions — not just performance variance. [DONE, pre-fix TabICL]
    **[TODO — superseded by P1.9] align with the course definitions (§7.1):**
    - distance between models as **‖θ₁ − θ₂‖₂** on logistic coefficients, and **‖φ(f₁) − φ(f₂)‖₂** on
      feature-importance vectors (we currently report rank correlation);
    - the course's definition is literally "two datasets from the same population": add a refit on
      **two disjoint halves** of the training set next to the bootstrap refits;
    - per-person score SD across refits (how much one individual's score moves);
    - **TabICL seed-only variability** (same data, different seeds/ensemble views) separated from
      data resampling — course §7.2 on randomness: report seed, hardware (CPU vs GPU), version;
    - optional: stability–performance trade-off curve over C for logistic (course p192).

### Supporting
12. Learning curve — keep as support for the recommendation, not more slide time than fairness.
13. **Commit the CV search as real code.** [DONE] `tune_xgboost.py`, `tune_logistic.py`,
    `compare_ml_models.py`.
14. **[DONE on GPU (405b608), but before the fairness merge — see P0.1] Full re-run after the leak fix**, in this order (TabICL on CPU ≈ 30–60 min total):
    `train_evaluate` → `interpretability` → `xper_attribution` → `stability_structural` →
    `fairness_audit` → `incumbent_benchmark` → `learning_curve` → `race_ab_test` → `tradeoff_matrix`
    → `build_notebook` → `build_slides`. The review makes `--skip-tabicl` write to `artifacts/smoke/`, preserving published artifacts.
15. **[PARTIAL — see P0.3, P0.4] Refresh README, report, deck, notes and JOURNEY.md** with post-fix numbers; add the leak
    to JOURNEY.md as a fairness finding.
16. **[PARTIAL — see P2.15] Notebook data-preparation section.** The current notebook starts from saved artifacts.
    Add: target balance, the official split, excluded columns (protected, PUMA, post-release
    dynamics) and why, missingness by group (the Gang_Affiliated/gender finding), encoding choices.
17. **[PARTIAL — FNR captions done; fairness figures missing; TabICL explanation see P2.13] App update** after the re-run: FNR-based fairness tab, TabICL per-person explanation
    (KernelSHAP, cached), readable labels (done).
18. **[DONE in report] Incumbent comparison caveat:** our models use `Supervision_Risk_Score_First` as one input, so
    the benchmark is "incumbent alone vs incumbent + other baseline information". Say so. Its mean
    barely differs by race (6.2 vs 6.0) or gender (6.0 vs 6.1), so it is not an obvious proxy.

## Deliverables checklist

- Slide deck, incl. explicit 3-model trade-off slide before the recommendation.
- Complete notebook: 3 models × 4 dimensions matrix, data prep → evaluation.
- App: model selector (Logistic / XGBoost / TabICL), individual profile → probability → risk
  category → local explanation → **compare all three**. Tabs: Individual assessment | Model comparison
  | Fairness audit | Monitoring. Persistent banner: "Decision support for allocation of re-entry
  services — not for detention, sentencing, or sanctions."
- **Q&A rehearsal (10/25 points): schedule a session where each member defends a section they did not
  write.** Not optional.

## Explicitly not doing

- No SMOTE / class-weighting (data is balanced; would hurt calibration).
- No further tuning on the reused evaluation set; any new optimization requires training-only validation.
- LIME as a single illustrative example only, not a full pass.

## Recommendation to the client (trustworthy-AI wording)

**[DECIDED 24 Sep: logistic in production, XGBoost as challenger — first option below.]** The earlier deck said "deploy XGBoost" because it
"calibrates best" and has "the smallest gender FPR gap". Neither holds up: ECE differences are not
significant, and at top-20% logistic has *no* significant race gap while XGBoost's is borderline
significant. The reviewer's #8: in a justice setting with an appeal process, transparency and
contestability weigh heavily, and XGBoost's edge over logistic is +0.003 AUC. Two defensible options:
- **Logistic regression in production, XGBoost as challenger** — transparent, contestable, fairness no
  worse, performance practically equal.
- **XGBoost in production** — only if we argue explicitly that a significant +0.003 AUC / ~$0.1–0.25M
  is worth the loss of native interpretability, and show SHAP explanations are stable enough.
Whichever we choose, derive it from the 3 × 4 trade-off table, not from slide-by-slide accumulation.

Previous wording, kept for reference. Any of the three models is a large upgrade over the incumbent
score (0.60 → 0.73 AUC; ~$2.75M → ~$5.1M net value). Then, grounded in confidence intervals and
economics, not raw AUC:
- **Small-data agencies:** TabICL offers the strongest predictive performance and calibration;
  deployment is justified where its incremental benefit exceeds its computational and auditability
  costs (it has no native explanation path).
- **Large-data agencies:** once performance converges, XGBoost or logistic regression become more
  attractive because interpretability, efficiency, and auditability dominate marginal predictive gains.
Guardrails throughout: support allocation only, appeal process, drift monitoring, subgroup audits.

## Timeline

Due Mon 28 Sep 9:40 AM. **Dataset pre-validation is due Thu 24 Sep 9:40 AM** — confirm someone sent
`reports/dataset_prevalidation.pdf` and the instructor approved the NIJ dataset.
- Wed 23: leak fixed; logistic tuned; ML comparison; plan updated.
- Thu 24: course-aligned fairness / interpretability / stability additions; full re-run.
- Fri 25 – Sat 26: decide the recommendation; rebuild deck, notebook, report; app check.
- Sun 27: Q&A rehearsal — each member defends a section they did not write.
