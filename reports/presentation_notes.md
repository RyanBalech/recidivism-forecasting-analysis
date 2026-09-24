# Presentation and Q&A notes

Aim for 13 minutes plus a two-minute buffer. Every team member should rehearse every section. Use the rebuilt artifact values; historical JOURNEY.md numbers are not the current run.

## Talk sequence

1. **Client and decision (1 minute):** a software vendor assessing models for voluntary re-entry support. Recorded arrest is the outcome, not inherent offending.
2. **Data and validation (1.5 minutes):** original 18,028/7,807 partition; baseline-only features; cumulative target differs from NIJ annual forecasting. Explain that evaluation data were repeatedly inspected and how the gender-aligned missingness leak was removed.
3. **Models and performance (2 minutes):** logistic, XGBoost, 16-member TabICLv2; historical score and prevalence baselines; Brier skill and paired differences. The ensemble sweep uses a training-only development split. Do not infer no difference from overlapping individual intervals. All three are within about 0.003 AUC and ECE differences are not significant, so performance alone cannot pick the model.
4. **Economic scenario (1 minute):** transparent assumed costs/effectiveness and exact capacity. Historical captured outcomes are not causal savings.
5. **Interpretability (1.5 minutes):** explain one SHAP case, its log-odds scale, and surrogate fidelity. XPER attributes performance on a small sample. TabICL has model-response sensitivity but no implemented native additive attribution.
6. **Stability (1 minute):** probability drift and Jaccard; eight identical bootstrap samples for every model, with algorithm seeds fixed. This measures sensitivity to training data. Jaccard is not the fraction of people switching.
7. **Fairness (2.5 minutes):** selection means being offered support, so FNR (missed support) is primary. At the top-20% rule: race gaps are equivalent within ±5 points for all models (TOST); women who are re-arrested miss support more (FNR gap about −0.10 to −0.12); age is the largest gap (about −0.23 to −0.25). Then the FPDP slide: gang affiliation is never recorded for women, and dropping it removes the equal-opportunity rejection at about −0.014 AUC. Changing thresholds leaves score calibration unchanged; the frontier and the mitigation use evaluation labels.
8. **Recommendation and app (2 minutes):** prospective XGBoost shadow pilot, logistic challenger, clear deployment gates. Demo missing-value preservation, optional TFM inference, explanations and editable economic assumptions.

## Questions to rehearse

**Is this an NIJ competition reproduction?** No. Our cumulative three-year baseline task is a valid course adaptation. NIJ evaluated annual conditional forecasts on changing cohorts.

**Is your test set untouched?** No. It is excluded from fitting, but historical model-development decisions inspected it. Our comparisons are exploratory; confirm on new data.

**Does your new CV fix that?** It adds training-only evidence with fold-local preprocessing. Fixed parameters were previously tuned on that training set, so this is not nested validation of selection.

**Did you use the GPU to improve the result?** Yes. CUDA makes a 16-member TabICLv2 ensemble and eight stability refits feasible. The fresh review also tested native categorical XGBoost on CUDA without finding a material gain. Runtime settings matter: changing XGBoost's CPU thread count changed fitted predictions even at a fixed random seed. Hardware acceleration alone does not guarantee better accuracy.

**What leakage did you find?** `Gang_Affiliated` is missing for every woman in the training sample and no man. TabICL could treat that missing category as a direct gender marker, so categorical missingness is now filled from training modes and guarded by a standalone audit plus a regression test.

**Does passing the audit prove there is no leakage?** No. We verified feature values against original NIJ releases and tested fitting boundaries, but public data do not timestamp every measurement. Gender proxy exposure is distinct from future-target leakage. Shuffled labels cannot rule out temporal leakage. Deliberately adding ineligible post-release activities raises internal AUC to about 0.81; that is not a valid supervision-start result.

**Can we improve accuracy further?** The existing XGBoost configuration won all three inner-fold searches across 117 fits; native-category follow-ups added 27 fits with no material improvement. A fixed 50/50 XGBoost–TabICL blend modestly improves AUC/Brier point estimates, but intervals include zero, top-capacity capture is lower and inference costs increase. It remains a research challenger.

**Why did published numbers change?** The audit detected a saved-XGBoost/prediction mismatch. We rebuilt models and dependent results in one environment and added prediction-agreement checks. Hardware/software differences can also affect TFM outputs; current timings are for this run, not a universal speed ratio.

**Why XGBoost?** (Team decision still open — see PLAN.md P0.3.) Competitive probability quality, fast inference, available local explanations and auditable trade-offs. Its AUC edge over logistic is only +0.0025, and logistic has the smaller gender FNR gap and no significant race gap: be ready to argue why the edge is worth losing native interpretability, or switch the recommendation.

**Did you beat current agency tools?** We compared a historical recorded supervision score in this dataset. We do not know the performance of today's products.

**Is the model fair by race?** Within our pre-set ±5-point tolerance, yes for selection, FNR and FPR, for all three models (TOST). For XGBoost and TabICLv2 the selection and FPR gaps are also significant, about +2 points in favour of Black people receiving support: nonzero but small. The tolerance was fixed in code before running TOST, but after we had seen the gap estimates; say so if asked. Precision gaps are inconclusive.

**"Not significant" means fair, right?** No. A difference test that fails to reject can simply lack power. That is why we test equivalence against a tolerance and report equivalent / different / inconclusive.

**Where does the gender gap come from?** The FPDP flags gang affiliation and age. Gang affiliation is never recorded for women and is imputed as "No", so on the raw field it identifies gender perfectly. Dropping it and refitting removes the equal-opportunity rejection (logistic p 0.000 → 0.995) at about −0.014 AUC, roughly six times the XGBoost − logistic difference; statistical parity is still rejected. This is diagnostic and evaluated on the same labels, not a validated fix.

**Women are over-predicted, so why are they selected less?** Their mean score (≈ 0.52) exceeds their observed rate (0.454), but few women reach the high top-20% cutoff. Calibration (sufficiency) and equal FNR are different criteria and conflict when base rates differ.

**Isn't the age result age discrimination?** Age is the strongest input and a validated risk factor; about 97% of re-arrested people aged 48+ are not selected. In a support programme a higher score means more help for younger people, but ranking by risk rather than need is a policy choice the client must justify. We report it; we do not claim it is fair.

**Why not just remove gang affiliation?** Possibly, and it is the strongest mitigation candidate, but it costs about 0.014 AUC, leaves statistical parity rejected, and has to be validated on data not used to find it. It is a decision for the client with legal review.

**Does equalizing FPR decalibrate women?** No. Unchanged probability scores retain unchanged calibration. Thresholds change classifications, precision, recall and allocations. Equal FPR alone is not equalized odds.

**Is the threshold frontier valid mitigation?** It is an in-sample illustration optimized using evaluation labels. Real mitigation needs a separate validation protocol and policy review.

**Does excluding race establish fairness?** It guarantees invariance to that direct input while fixing the other fields. Proxies and group disparities remain; this is not causal counterfactual fairness.

**What does Jaccard 0.75 mean?** Intersection/union is 0.75. For equally sized selected sets, about 14.3% of each selected set is replaced, not 25% of the whole cohort.

**Can TabICL explain individuals?** Native additive attribution is not implemented. Feature edits and ICE examine local responses, with no causal interpretation. This limitation belongs in the trade-off assessment.

**Are dollars savings forecasts?** No. Intervention effects are assumed. Risk ranking may differ from treatment-benefit ranking.

**What remains before production?** New-cohort validation, verified input timing, evidence of support benefit, justified tie/eligibility policy, appeals, monitoring and pre-agreed suspension rules.

**What does the team still need to do?** Submit the dataset for pre-validation by 24 September 2026 at 09:40, rehearse the demo/Q&A, and submit deliverables by 28 September at 09:40. The supplied brief requires a 15-minute presentation and 10-minute Q&A.
