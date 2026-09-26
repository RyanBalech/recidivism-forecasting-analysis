# Presentation and Q&A notes

Aim for 14 minutes plus a one-minute buffer. Every team member should rehearse every section. Use the rebuilt artifact values; historical JOURNEY.md numbers are not the current run.

## Talk sequence

The deck follows [presentation_outline.md](presentation_outline.md): **19 core slides in six speaking
parts plus a 7-slide appendix**, about 13:45 of talk. Each slide's speaker note starts with the part
that speaks it and its timing, e.g. `[P3 · 3:00 · 50s]`.

| Part | Slides | Time | Topic |
|---|---|---|---|
| P1 | 1-3, 19 | 1:30 + 0:45 | Client, data and EDA; app demo at the end |
| P2 | 4-5 | 1:30 | Eligible features; the gender leak through missingness |
| P3 | 6-8 | 2:30 | Models and tuning; performance tie; historical score |
| P4 | 9-11 | 2:00 | Global drivers; one person and faithfulness; method disagreement |
| P5 | 12-15 | 3:00 | Race (exclusion, outcome audit); gender (over-prediction, FPDP, mitigation); age in one line |
| P6 | 16-18 | 2:30 | Structural and per-person stability; trade-offs and recommendation |

Three findings carry the talk: the models are equally accurate and differently expensive; access to
support is unequal by gender (and age, by design), not by race within tolerance; the recommendation is
conditional and we say what reverses it.

## Questions to rehearse

**Is this an NIJ competition reproduction?** No. Our cumulative three-year baseline task is a valid course adaptation. NIJ evaluated annual conditional forecasts on changing cohorts.

**Is your test set untouched?** No. It is excluded from fitting, but historical model-development decisions inspected it. Our comparisons are exploratory; confirm on new data.

**Does your new CV fix that?** It adds training-only evidence with fold-local preprocessing. Fixed parameters were previously tuned on that training set, so this is not nested validation of selection.

**Did you use the GPU to improve the result?** Yes. CUDA makes a 16-member TabICLv2 ensemble and eight stability refits feasible. The fresh review also tested native categorical XGBoost on CUDA without finding a material gain. Runtime settings matter: changing XGBoost's CPU thread count changed fitted predictions even at a fixed random seed. Hardware acceleration alone does not guarantee better accuracy.

**What leakage did you find?** `Gang_Affiliated` is missing for every woman in the training sample and no man. TabICL could treat that missing category as a direct gender marker, so categorical missingness is now filled from training modes and guarded by a standalone audit plus a regression test.

**Does passing the audit prove there is no leakage?** No. We verified feature values against original NIJ releases and tested fitting boundaries, but public data do not timestamp every measurement. Gender proxy exposure is distinct from future-target leakage. Shuffled labels cannot rule out temporal leakage. Deliberately adding ineligible post-release activities raises internal AUC to about 0.81; that is not a valid supervision-start result.

**Can we improve accuracy further?** The existing XGBoost configuration won all three inner-fold searches across 117 fits; native-category follow-ups added 27 fits with no material improvement. A fixed 50/50 XGBoost–TabICL blend modestly improves AUC/Brier point estimates, but intervals include zero, top-capacity capture is lower and inference costs increase. It remains a research challenger.

**Why did published numbers change?** The audit detected a saved-XGBoost/prediction mismatch. We rebuilt models and dependent results in one environment and added prediction-agreement checks. Hardware/software differences can also affect TFM outputs; current timings are for this run, not a universal speed ratio.

**Why logistic and not XGBoost?** XGBoost's AUC edge is +0.0025 (paired interval 0.0006 to 0.0044): real but operationally small — at the deployed rule the two models offer support to 85% of the same people, and the difference in captured re-arrests has an interval spanning zero. Logistic wins on what is left: native coefficient explanations with no surrogate fidelity loss, the lowest refit drift on all 28 resample pairs (0.032 vs 0.035), a more stable selected set (Jaccard 0.773 vs 0.747), and about a ninth of the runtime. Be careful not to claim more: on fairness and calibration the two are statistically **tied**, so the gender FNR gap and the race tests are not reasons. The brief asks for a trustworthy-AI choice, not the top AUC. XGBoost stays as the shadow challenger; if a prospective pilot showed a materially larger gain, the choice could be revisited.

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

**XGBoost is measurably more accurate - why ship the weaker model?** Because the two significant advantages it has (+0.0025 AUC, -0.0010 Brier) do not reach the decision. At the deployed top-20% rule the two models offer support to **85% of the same people** (Jaccard 0.849; only 254 of 7,807 differ), and the difference in captured re-arrests is about 14 offers with a 95% interval of -29 to 0, which includes zero. What logistic wins is refit stability - lower drift on all 28 resample pairs (0.0322 vs 0.0351) and a more stable selected set (Jaccard 0.773 vs 0.747) - plus coefficients read directly with no second tool and no surrogate fidelity loss, and about a ninth of the runtime. We are not trading accuracy for interpretability; the accuracy difference does not change who gets help.

**Isn't XGBoost better calibrated?** No - that was a point estimate we tested and withdrew. It has the lower ECE at 10 equal-width bins (0.011 vs 0.013), but the ranking flips with the binning: logistic wins at 5 equal-width bins and at 20 quantile bins. Bin-free, both are statistically indistinguishable from perfect calibration (Cox slopes 0.999 and 1.005, neither differing from 1; Spiegelhalter z = 0.38 and 0.30) and from each other (paired |slope - 1| difference -0.00003, CI -0.015 to +0.015). Calibration is a tie. The real calibration finding is TabICLv2: slope 0.913, significantly below 1, z = 3.76 - its probabilities are measurably too extreme. Everything is in `calibration_tests.csv` and `calibration_paired_tests.csv`.

**So logistic is fairer?** Not established, and we do not claim it. Its gender FNR gap is smaller as a point estimate (-0.096 vs -0.112), but the paired difference in absolute gaps is -0.013 with a 95% interval of -0.036 to +0.011. Neither model has a race test surviving Holm. Fairness does not separate these two models; the feature set drives the disparity, which is why the mitigation matters more than the model choice.

**Two models fitted twice do not give identical probabilities - why?** Logistic does, to machine precision. XGBoost does not: histogram split gains accumulate in a thread-dependent order, so a different core count can flip a tie. Performance is reproducible (AUC within 0.0001); individual probabilities only to about 1e-2. The saved models reproduce the published file exactly, which is the check that matters for the artifacts; the notebook shows both.

**You removed race and gender — doesn't that make it fair?** No, and we measured how far it falls short. Predicting the protected attribute from our own feature set recovers gender at AUC 0.776 and race at 0.708. The models never see those columns; the information is there anyway. That is the measured version of the caveat on the twin test: the twins score identically and the group gaps persist.

**How big was the leak you found?** Total. With missing-value indicators, gender is recovered at **AUC 1.000**. `Gang_Affiliated` is missing for every woman and no man, so any model encoding NaN as a category — TabICLv2 does — had gender in full. Mode-filling closes it. Dropping gang affiliation entirely only moves gender recovery from 0.776 to 0.747, which is why the FPDP mitigation narrows the gap without closing it.

**Would this person still get help if you had different training data?** For about one person in seven, not reliably. Refitting on eight bootstrap resamples, 13% of decisions flip, and about a third of the people actually prioritised sit at that margin. The app shows the score range and the vote count per person, so a contested case is visible rather than hidden behind a point estimate.

**Then abstain on the uncertain ones and refer them to a human.** We tested that, and it is not free. Keeping only unanimous decisions covers 87% of the cohort and lifts precision from 0.822 to 0.846 — but the gender FNR gap widens from −0.090 to −0.119. Contested decisions are evenly spread across gender (12.1% of women, 12.9% of men), but among people prioritised, 47% of the 118 selected women are at the margin against 30% of 1,490 men. Abstention removes more of the few offers women receive. It is the impossibility trade-off again in a different guise: refusing to decide when uncertain is not fairness-neutral. If the client wants it, the referral queue needs auditing too.

**What does the team still need to do?** Submit the dataset for pre-validation by 24 September 2026 at 09:40, rehearse the demo/Q&A, and submit deliverables by 28 September at 09:40. The supplied brief requires a 15-minute presentation and 10-minute Q&A.
