# Presentation and Q&A notes

Aim for 13 minutes plus a two-minute buffer. Every team member should rehearse every section. Use the rebuilt artifact values; historical JOURNEY.md numbers are not the current run.

## Talk sequence

The deck is **13 core slides plus a marked appendix**. Present the 13; the appendix
(learning curve, economic sensitivity, interpretability methods, explanation
disagreement, LIME fidelity, stability, process log) exists to answer questions, not
to be walked through. Budget 13 minutes plus a two-minute buffer.

Three findings carry the talk. Say each of them out loud at least twice:

1. **The three models are equally accurate and differently expensive.** Within ~0.003
   AUC of each other, so performance cannot pick the model - explanation cost,
   stability and fairness do.
2. **Access to support is substantially unequal by gender and age.** Not by race:
   race gaps are equivalent within ±5 points and none survives multiplicity control.
   Gender and age are rejected on every criterion.
3. **The recommendation is conditional, and we state what reverses it.**

| # | Slide | Time | The point to land |
|---|---|---|---|
| 1-2 | Title, engagement | 1.0 | A vendor allocating voluntary support. Recorded arrest is the outcome, not inherent offending. |
| 3 | Data design | 1.5 | Original 18,028/7,807 split, baseline-only fields; the cumulative target differs from NIJ's annual forecasting. Say plainly that the evaluation set was repeatedly inspected during development, and that we found and removed a gender-aligned missingness leak. |
| 4-5 | Model design, incumbent | 2.0 | Three families, same fields, 16-member TabICLv2. The incumbent historical score reaches 0.60 against our 0.73 - the client's real question, answered first. Prevalence and calibrated-incumbent baselines with Brier skill. |
| 6 | Predictive performance | 1.5 | **Finding 1.** All within ~0.003 AUC; ECE differences not significant. Use the paired bootstrap; do not infer "no difference" from overlapping individual intervals. |
| 7 | Subgroup audit | 2.0 | **Finding 2.** Selection means being offered support, so FNR is primary. Gender FNR gap -0.10 to -0.12, age -0.23 to -0.25, race equivalent within ±5 points. The age gap is **not** removed by fixing the age field: it runs through criminal-history inputs. |
| 8 | Fairness, sharpened | 1.0 | Women are over-predicted yet selected less. Thresholds change decisions, not the calibration of unchanged scores. |
| 9 | FPDP | 1.0 | Gang affiliation is never recorded for women - the gap has a named source. Dropping it shrinks the gender FNR gap from -0.10 to about 0.00 (logistic), partly by raising men's FNR, at about -0.014 AUC. |
| 10 | Mitigation, validated | 1.0 | Candidate re-selected inside every training fold; ~70% of the gap closes out of fold for ~0.01 AUC. Selection and assessment are separated; the evaluation cohort is untouched. |
| 11-12 | Trade-off matrix, recommendation | 2.0 | **Finding 3.** Walk the 3x4 matrix, then logistic for the shadow pilot with XGBoost as challenger - and say what would reverse it. |
| 13 | App | 1.0 | Demo one person end to end: score, explanation, support decision, editable assumptions; missing-value preservation and optional TFM inference. |

If time runs short, compress slides 4-5 and 9; do not compress 7, 10 or 12.

## Questions to rehearse

**Is this an NIJ competition reproduction?** No. Our cumulative three-year baseline task is a valid course adaptation. NIJ evaluated annual conditional forecasts on changing cohorts.

**Is your test set untouched?** No. It is excluded from fitting, but historical model-development decisions inspected it. Our comparisons are exploratory; confirm on new data.

**Does your new CV fix that?** It adds training-only evidence with fold-local preprocessing. Fixed parameters were previously tuned on that training set, so this is not nested validation of selection.

**Did you use the GPU to improve the result?** Yes. CUDA makes a 16-member TabICLv2 ensemble and eight stability refits feasible. The fresh review also tested native categorical XGBoost on CUDA without finding a material gain. Runtime settings matter: changing XGBoost's CPU thread count changed fitted predictions even at a fixed random seed. Hardware acceleration alone does not guarantee better accuracy.

**What leakage did you find?** `Gang_Affiliated` is missing for every woman in the training sample and no man. TabICL could treat that missing category as a direct gender marker, so categorical missingness is now filled from training modes and guarded by a standalone audit plus a regression test.

**Does passing the audit prove there is no leakage?** No. We verified feature values against original NIJ releases and tested fitting boundaries, but public data do not timestamp every measurement. Gender proxy exposure is distinct from future-target leakage. Shuffled labels cannot rule out temporal leakage. Deliberately adding ineligible post-release activities raises internal AUC to about 0.81; that is not a valid supervision-start result.

**Can we improve accuracy further?** The existing XGBoost configuration won all three inner-fold searches across 117 fits; native-category follow-ups added 27 fits with no material improvement. A fixed 50/50 XGBoost–TabICL blend modestly improves AUC/Brier point estimates, but intervals include zero, top-capacity capture is lower and inference costs increase. It remains a research challenger.

**Why did published numbers change?** The audit detected a saved-XGBoost/prediction mismatch. We rebuilt models and dependent results in one environment and added prediction-agreement checks. Hardware/software differences can also affect TFM outputs; current timings are for this run, not a universal speed ratio.

**Why logistic and not XGBoost?** XGBoost's AUC edge is +0.0025 (paired interval 0.0006 to 0.0044): real but operationally small. Logistic is equal or better on everything else: native coefficient explanations, the smallest refit drift (0.032 vs 0.035), the smaller gender FNR gap (−0.096 vs −0.112), no significant race gap, and the lowest runtime. The brief asks for a trustworthy-AI choice, not the top AUC. XGBoost stays as the shadow challenger; if a prospective pilot showed a materially larger gain, the choice could be revisited.

**Did you beat current agency tools?** We compared a historical recorded supervision score in this dataset. We do not know the performance of today's products.

**Is the model fair by race?** Within our ±5-point tolerance, yes for selection, FNR and FPR, for all three models (TOST). For XGBoost and TabICLv2 the selection and FPR gaps of about +2 points (in favour of Black people receiving support) have unadjusted intervals excluding zero, but **no race test survives Holm correction** across the 54 course tests - so call them small and unconfirmed, not established disparities. The tolerance was fixed in code before running TOST, but after we had seen the gap estimates; say so if asked. It is absolute: on a 20% selection rate, 5 points could still mean a ratio near 0.78, so we also check the ratio (0.87-0.93, above four-fifths). Precision gaps are inconclusive. All of this is about allocation errors at one rule: within-group AUC is lower for Black people (about 0.72 vs 0.75), so predictive quality is not equal.

**"Not significant" means fair, right?** No. A difference test that fails to reject can simply lack power. That is why we test equivalence against a tolerance and report equivalent / different / inconclusive.

**Where does the gender gap come from?** The FPDP flags gang affiliation and age. Gang affiliation is never recorded for women and is imputed as "No", so on the raw field it identifies gender perfectly. Dropping it and refitting shrinks the gender FNR gap from −0.096 to 0.000 (logistic) and from −0.112 to −0.010 (XGBoost). Women's FNR falls about 8 points, men's rises about 2, 27–41 fewer re-arrests are captured, AUC falls about 0.014 (five to six times the XGBoost − logistic difference), women's over-prediction worsens, and statistical parity is still rejected. For logistic the FPDP curve is flat: under a top-20% rule, fixing a feature to any constant shifts every logit equally and leaves the ranking unchanged, so it measures removing the feature, not a particular value. This is diagnostic and evaluated on the same labels, not a validated fix.

**Women are over-predicted, so why are they selected less?** Their mean score (≈ 0.52) exceeds their observed rate (0.454), but few women reach the high top-20% cutoff. Calibration (sufficiency) and equal FNR are different criteria and conflict when base rates differ. For a support programme the over-prediction works in women's favour: recalibrating by gender would select even fewer women and widen the FNR gap. So the harm to women here is the FNR gap, not the miscalibration.

**Isn't the age result age discrimination?** Age is the strongest input and a validated risk factor; about 97% of re-arrested people aged 48+ are not selected. In a support programme a higher score means more help for younger people, but ranking by risk rather than need is a policy choice the client must justify. Removing age would not fix it: with age fixed for everyone the gap persists, because prior-record inputs and the Georgia score carry age. We report it; we do not claim it is fair.

**Why not just remove gang affiliation?** It is the strongest mitigation candidate and it now has out-of-fold evidence, not only in-sample evidence. It still costs about 0.014 AUC and 27-41 captured re-arrests, raises men's FNR, worsens women's calibration, and leaves statistical parity rejected. Adding a "not recorded" category instead is not an option: that category is exactly the gender marker we removed. It remains a decision for the client with legal review.

**Didn't you find the fix and test it on the same data?** Originally yes, and that is why we added `scripts/mitigation_nested.py`. It reruns the FPDP candidate search inside each of five training folds, selects from that fold alone, refits there, and measures on the held-out fold - the evaluation cohort is never touched. `Gang_Affiliated` is selected in **all five folds for both models**, so the candidate is a property of the training data, not of the cohort we audit. Out of fold the gender FNR gap closes by about 70% (-0.118 → -0.035 logistic; -0.127 → -0.034 XGBoost) for roughly 0.01 AUC. Mean equal-opportunity p rises to 0.32 / 0.27, far below the in-sample 0.99, and two of five folds still reject. Quote the out-of-fold numbers; the in-sample figure is an upper bound.

**Does equalizing FPR decalibrate women?** No. Unchanged probability scores retain unchanged calibration. Thresholds change classifications, precision, recall and allocations. Equal FPR alone is not equalized odds.

**Is the threshold frontier valid mitigation?** It is an in-sample illustration optimized using evaluation labels. Real mitigation needs a separate validation protocol and policy review.

**Does excluding race establish fairness?** It guarantees invariance to that direct input while fixing the other fields. Proxies and group disparities remain; this is not causal counterfactual fairness.

**What does Jaccard 0.75 mean?** Intersection/union is 0.75. For equally sized selected sets, about 14.3% of each selected set is replaced, not 25% of the whole cohort.

**Can TabICL explain individuals?** Native additive attribution is not implemented. Feature edits and ICE examine local responses, with no causal interpretation. This limitation belongs in the trade-off assessment.

**Are dollars savings forecasts?** No. Intervention effects are assumed. Risk ranking may differ from treatment-benefit ranking.

**What remains before production?** New-cohort validation, verified input timing, evidence of support benefit, justified tie/eligibility policy, appeals, monitoring and pre-agreed suspension rules.

**Your LIME fit was weak - why should we believe it?** You should not believe the old one, and we replaced it. It explained the *transformed* one-hot space, so perturbations produced people who were two age bands at once or none; the surrogate fitted a region the model never sees and scored R² ≈ 0.25. LIME now runs on the raw features with `categorical_features` declared, so a perturbation swaps one real category for another. Fidelity roughly doubles to R² ≈ 0.41-0.43, reported per case next to every explanation. Three cases fixed by rule (highest / median / lowest risk), three seeds each, and all 63 conditions keep their sign. At 0.41 we quote LIME directionally and use SHAP when a magnitude is needed.

**SHAP, permutation importance and XPER disagree - which is right?** All three, about different questions. SHAP attributes the prediction, permutation importance attributes loss, XPER decomposes AUC. We measured the divergence rather than assuming it away: Spearman 0.77-0.81 between SHAP and permutation importance, 0.53-0.60 between either and XPER, with 8-10 features shared in the top ten. So the set of drivers is robust and the ordering is method-dependent - quote the set, not the rank. XPER's reconstruction residual (0.018 logistic, 0.023 XGBoost) stays visible because its kernel approximation has no endpoint constraints.

**What does a coefficient of 0.087 mean to my agency?** Nothing directly - it is per standardized unit. We report average marginal effects and average probability contrasts instead. Moving a supervisee from the 23-27 band to 48-or-older lowers predicted three-year arrest risk by about 27 points; a recorded gang affiliation raises it by about 17. Each contrast sets one real category for everyone, so there are no impossible rows. These are model responses to a counterfactual edit, not causal effects of ageing.

**XGBoost is measurably more accurate - why ship the weaker model?** Because the margin is +0.0025 AUC (paired interval 0.0006 to 0.0044) and twelve extra captured re-arrests out of 1,561 offers, and everything else points the other way: logistic has coefficient-level explanations with no surrogate fidelity loss (XGBoost's depth-3 surrogate reaches only R² = 0.61), the smallest refit drift (0.032 vs 0.035), the smallest gender FNR gap (-0.096 vs -0.112), no race test significant even before Holm correction where XGBoost has two, and about a ninth of the runtime. For a product sold on auditability that is the wrong trade. The decisive framing: the mitigation closes about 0.08 of gender gap out of fold, five times the 0.016 separating the two models - disparity is a feature-set property, not an estimator choice, so neither model should be justified by it. XGBoost is the right answer instead if the client quotes the probability numerically to supervisees (calibration then dominates: ECE 0.011 vs 0.013), or if their capacity is large enough that twelve extra offers per 1,561 is material. That is why it runs as the challenger.

**Two models fitted twice do not give identical probabilities - why?** Logistic does, to machine precision. XGBoost does not: histogram split gains accumulate in a thread-dependent order, so a different core count can flip a tie. Performance is reproducible (AUC within 0.0001); individual probabilities only to about 1e-2. The saved models reproduce the published file exactly, which is the check that matters for the artifacts; the notebook shows both.

**What does the team still need to do?** Submit the dataset for pre-validation by 24 September 2026 at 09:40, rehearse the demo/Q&A, and submit deliverables by 28 September at 09:40. The supplied brief requires a 15-minute presentation and 10-minute Q&A.
