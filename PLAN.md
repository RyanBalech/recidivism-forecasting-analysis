# Project plan — Trustworthy Recidivism Forecasting (Team 11)

Course: HEC Paris — *Interpretability, Stability, and Algorithmic Fairness* (Pérignon / Saurin).
Deliverables due **Mon 28 Sep, 9:40 AM**: slide deck, notebook (data prep → evaluation), client app.
Presentation 15 min + Q&A 10 min. Every member must defend any section.

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

- **Logistic regression** — white-box.
- **XGBoost** — ML; count columns ordinal-encoded, hyperparameters from 5-fold CV random search.
- **TabICLv2** — Tabular Foundation Model.

## Key findings so far

- All three models plateau at ~0.73 AUC / ~0.20 Brier. Confirmed by CV search (best CV AUC
  0.7343). This is the data's signal ceiling, matching NIJ challenge winners. **Accuracy is not
  a grading criterion — report the ceiling in one slide and move on.**
- **Incumbent benchmark:** `Supervision_Risk_Score_First` is Georgia's existing 1–10 actuarial
  score that agencies use today. Standalone AUC = **0.60**, vs **0.73** for our models. The
  client's real question — "is any of this better than what we already deploy?" — answers itself:
  our models more than double the lift over random. This is the economic centerpiece.
- **Learning curve** (1,500 / 5,000 / 10,000 / 18,028 rows, 3 seeds, fixed test set): TabICL leads
  at every size but its edge over XGBoost shrinks from +0.021 AUC at 1,500 to +0.004 at full data.
  No crossover. TabICL calibrates best; XGBoost is ~35× faster and far more explainable.

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
2. **Audit at the deployed operating point, not 0.5.** The product allocates the top 20% by risk, so
   report every fairness gap at the top-20% capacity threshold AND at 0.5, and make the difference a
   slide (fairness is operating-point-dependent). Verified: TabICL gender FPR gap 0.273 @0.5 vs 0.07
   @top-20%.
3. Gaps by race and gender for all three models: demographic parity, equal opportunity, FPR/FNR gaps.
4. Bootstrap confidence intervals on every gap (syllabus: "metrics and inference tests"). Female =
   950 test rows, so show interval width, not point estimates.
5. **Impossibility result — split by attribute (corrected):**
   - Race: base rates nearly equal (0.582 vs 0.564) → calibration and equal error rates are jointly
     achievable → the observed FPR gap is a model property, therefore fixable.
   - Gender: base rates differ 13.7 pts (0.591 M vs 0.454 F) → the theorem binds → cannot equalize
     both, so we choose and justify. This explains why gender gaps exceed race gaps.
6. Mitigation + accuracy/fairness frontier. **Legal caveat (Q&A exposure):** group-specific
   thresholds by race = disparate treatment (Ricci v. DeStefano) and contradict excluding race from
   inputs. Present group thresholds as an *analytic device to trace the frontier*, then show a
   race-blind alternative (single threshold minimizing the gap, or pre/in-processing) and its cost.
   Note that scoping to service allocation (not sanctions) changes the legal calculus.

### Interpretability (required dimension — for ALL three models)
7. SHAP individual waterfall for logistic + XGBoost (notebook + app).
8. Global surrogate (shallow tree on XGBoost preds) + PDP/ICE (model-agnostic, covers all three) +
   one LIME example.
9. **TabICL interpretability story, said deliberately:** no native explanation path; KernelSHAP over
   7,807 rows is impractical. PDP/ICE cover it model-agnostically. Frame the missing native path as a
   *deployment cost*, not an oversight — it feeds the recommendation.
10. Attempt XPER (instructor's method); permutation importance is the documented fallback.

### Stability (currently the weakest dimension)
11. Real structural stability: refit across seeds/resamples, measure distance between resulting models
    and drift in feature contributions — not just performance variance. **Budget TabICL seeds
    explicitly (~73s/fit on CPU); decide the seed count now, not Sunday.**

### Supporting
12. Learning curve — keep as support for the recommendation, not more slide time than fairness.
13. **Commit the CV search as real code.** The repo still has hardcoded XGBoost hyperparameters and a
    one-hot preprocessor; the "tuned + ordinal, 5-fold CV" claim is currently unbacked. Add the search
    script + ordinal encoder and bake the result in, so every member can defend it.

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
- No further accuracy optimization (proven at ceiling — one slide, one sentence).
- LIME as a single illustrative example only, not a full pass.

## Recommendation to the client (trustworthy-AI wording)

Any of the three models is a large upgrade over the incumbent score (0.60 → 0.73 AUC; ~$2.75M →
~$5.1M net value). Then, grounded in confidence intervals and economics, not raw AUC:
- **Small-data agencies:** TabICL offers the strongest predictive performance and calibration;
  deployment is justified where its incremental benefit exceeds its computational and auditability
  costs (it has no native explanation path).
- **Large-data agencies:** once performance converges, XGBoost or logistic regression become more
  attractive because interpretability, efficiency, and auditability dominate marginal predictive gains.
Guardrails throughout: support allocation only, appeal process, drift monitoring, subgroup audits.

## Timeline

Due Mon 28 Sep 9:40 AM. Dataset pre-validation was due Thu 24 Sep — confirm the instructor approved
the NIJ dataset. ~5 days: economic case done; fairness next; keep TabICL refit counts tight.
