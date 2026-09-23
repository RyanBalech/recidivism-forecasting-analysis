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

## To-do order (validated 23 Sep against the course slides, the brief, and the data)

Priority = what the brief grades and what the jury will ask. Details under each dimension below.
1. **Fairness, course-aligned** — FNR primary; course test table + TOST; FPDP/candidate variables
   (gender first); X/D vs X/Y scatter (= proxy answer); course mitigation; **age audit**.
2. **Notebook data-preparation section** (brief: "from data preparation to model evaluation").
3. **Interpretability** — logistic coefficient table; per-person explanations for all three models;
   XPER vs PI vs SHAP.
4. **Stability** — course distances; disjoint halves; TabICL seed-only variability.
5. Paired model tests saved as an artifact.
6. **Full re-run** with the leak fix, then **app update** (FNR, TabICL per-person explanation).
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
3. **[TODO — reviewer's #1 Q&A risk] Make FNR / equal opportunity the primary metric.** Being flagged
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
   **[TODO] Re-target the frontier to gender FNR**, and add the course's mitigation (below).

**[TODO] Align with the course slides (§8, pp. 238–277).** These are taught methods the jury will
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
**[TODO]**
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
    **[TODO] align with the course definitions (§7.1):**
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
14. **[TODO] Full re-run after the leak fix**, in this order (TabICL on CPU ≈ 30–60 min total):
    `train_evaluate` → `interpretability` → `xper_attribution` → `stability_structural` →
    `fairness_audit` → `incumbent_benchmark` → `learning_curve` → `race_ab_test` → `tradeoff_matrix`
    → `build_notebook` → `build_slides`. The review makes `--skip-tabicl` write to `artifacts/smoke/`, preserving published artifacts.
15. **[TODO] Refresh README, report, deck, notes and JOURNEY.md** with post-fix numbers; add the leak
    to JOURNEY.md as a fairness finding.
16. **[TODO] Notebook data-preparation section.** The current notebook starts from saved artifacts.
    Add: target balance, the official split, excluded columns (protected, PUMA, post-release
    dynamics) and why, missingness by group (the Gang_Affiliated/gender finding), encoding choices.
17. **[TODO] App update** after the re-run: FNR-based fairness tab, TabICL per-person explanation
    (KernelSHAP, cached), readable labels (done).
18. **Incumbent comparison caveat:** our models use `Supervision_Risk_Score_First` as one input, so
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

**[OPEN — team decision after the re-run]** The current deck says "deploy XGBoost" because it
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
