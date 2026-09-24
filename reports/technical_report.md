# Technical report and model card

## Recommendation and scope

Recommend **XGBoost for a prospective shadow pilot**, with logistic regression as a transparent challenger running in parallel on the same cohort. TabICLv2 supplies the required foundation-model comparison. No model is validated for operational decisions.

This is a decision across four dimensions rather than an accuracy ranking, so it is stated with the counter-case attached.

*What XGBoost buys.* Best calibration of the three (ECE 0.011 against 0.013 and 0.020), which matters because the score is quoted to a supervisee and used to rank under fixed capacity; 1,297 re-arrested people captured at 20% capacity against logistic's 1,285; and a paired-bootstrap AUC edge of +0.0025 whose interval excludes zero - small, but not noise.

*What it costs.* A wider gender FNR gap (-0.112 against -0.096), the least stable selected set across refits (Jaccard 0.747 against 0.772), dependence on SHAP plus a depth-3 surrogate of moderate fidelity (R² = 0.61) where logistic is read directly, and roughly nine times the runtime. Its race selection and FPR gaps have unadjusted intervals excluding zero where logistic's do not, though no race test survives Holm correction for either model, so this separates them less than it first appears.

*Why the trade is defensible.* The disparity that the FPDP mitigation closes is far larger than the disparity separating the two models: dropping `Gang_Affiliated` moves the gender FNR gap by about 0.08 out of fold, against the 0.016 that separates XGBoost from logistic. The gap is a property of the feature set rather than of the estimator, so paying in calibration and captured events to buy 0.016 would be addressing the wrong cause. Calibration cannot be recovered by a later intervention; this gap can.

*What would reverse it.* Logistic is the right answer if the client weights any of the following above calibration: a procurement or audit process requiring a model readable without a second explanation tool; deployment at agencies too small to maintain SHAP infrastructure; an auditor treating any unadjusted race interval excluding zero as disqualifying; or a contractual requirement on stability of the selected set between refits. Under the vendor framing used here these are live possibilities, and the challenger arrangement is what keeps the choice reversible after the pilot.

The hypothetical client is a vendor prioritizing voluntary re-entry support. The target is cumulative new arrest within three years of supervision start, using 29 baseline fields. The official partition contains 18,028 training and 7,807 evaluation records from Georgia releases during 2013–2015.

Our task differs from NIJ's annual conditional forecasts, which remove prior recidivists from later evaluation cohorts and permit later supervision variables. Challenge rankings are not directly comparable. See the [official protocol](https://nij.ojp.gov/funding/recidivism-forecasting-challenge) and [codebook](https://nij.ojp.gov/funding/recidivism-forecasting-challenge-appendix-2-codebook.pdf).

## Preparation and validation

Race, gender and PUMA are excluded from inputs; race and gender are audited. Post-release employment, tests, violations, attendance and residence changes are excluded. Actual operational availability of baseline fields still needs verification.

Imputation/scaling are fitted within pipelines. Logistic uses one-hot categories; XGBoost first ordinal-encodes selected ordered categories/counts. They use the same eligible raw information, **not the same transformed matrix**. TabICLv2 uses mixed inputs with categorical missing values filled from training modes. Filling missing gang affiliation removes a direct gender missingness marker, not all proxy information.

Logistic uses the merged team's L1 regularization with C=0.2154 and one-hot encoding, chosen by its five-fold grid. The team's six-candidate ML comparison and readable labels are retained. XGBoost's shallow-tree parameters come from a historical five-fold AUC search, not Brier optimization. TabICLv2 uses 16 ensemble members. Its 1–64 member sensitivity sweep uses only a fixed stratified development slice of the training partition; the final configuration is refitted on all training rows. This avoids another round of configuration choice on evaluation labels, but the single development split is still not an unbiased performance estimate.

The fresh five-fold conventional-model CV in `validation_cv.csv` evaluates fixed configurations with fold-local preprocessing. Earlier tuning used the same training data, so this is **not nested validation of the search procedure**. The TabICLv2 ensemble sensitivity check is not a full cross-validation study.

The original test set was repeatedly inspected for model variants, learning curves and audits. Historical choices in JOURNEY.md reference its results. It is therefore an evaluation set, **not an untouched final holdout**. New temporal/external data are required for confirmation; pipeline leakage control cannot undo adaptive evaluation reuse. See [scikit-learn's evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

## Predictive and economic performance

Exact current results are generated in [model_metrics.csv](../artifacts/model_metrics.csv), [validation_baselines.csv](../artifacts/validation_baselines.csv) and [paired_comparisons.csv](../artifacts/paired_comparisons.csv). Model AUC is approximately 0.73 against approximately 0.60 for the historical recorded score. All candidates already include that score as an input. This does not establish superiority over present-day agency products.

The constant benchmark uses training prevalence. The calibrated incumbent fits a one-feature logistic pipeline and imputation on training records. Brier skill measures improvement relative to the prevalence benchmark.

Paired bootstrap resamples the same people for both models, reporting A minus B. Positive AUC differences favor A; negative Brier differences favor A. These marginal 95% intervals use 1,000 draws, are exploratory and unadjusted for multiple comparisons/development selection. They condition on fixed predictions and do not include training or temporal uncertainty. Overlapping separate intervals do not establish equivalence.

Brier combines calibration and resolution; it is not calibration alone. ECE depends on binning. A constant model can have low ECE with no useful ranking.

Economic scenarios assume 20% capacity, $5,000 support cost, $50,000 event cost and 20% effectiveness. Historical captured outcomes are multiplied by those assumptions. This is not an identified treatment effect or causal saving; highest risk need not mean highest treatment benefit. Input-order tie breaking particularly affects the discrete incumbent. Sensitivity sweeps do not resolve these causal limitations.

## Interpretability

SHAP explains logistic/XGBoost predictions in log-odds, aggregated to raw features. Reconstruction tests verify that summed SHAP values recover saved-model probabilities. Neither SHAP nor LIME identifies causal effects. The shallow global surrogate has imperfect fidelity and cannot replace the original model.

**LIME and its fidelity.** The earlier implementation explained the *transformed* one-hot space. Perturbing one-hot columns independently produces rows no person could occupy - two categories at once, or none - so the local surrogate was fitted over a region the model never sees, and scored R² ≈ 0.25. `scripts/lime_local_fidelity.py` moves LIME to the raw feature space with `categorical_features` declared, so a perturbation swaps one real category for another. Local fidelity rises to R² ≈ 0.41-0.43 (`lime_fidelity.csv`). Three cases fixed by rule before inspection - highest, median and lowest predicted risk - are each explained under three seeds; all 63 reported conditions keep their sign across seeds (`lime_seed_spread.csv`). A fidelity near 0.41 still means a linear surrogate only partly tracks a 1,196-tree model in a 29-feature neighbourhood: quote LIME directionally and use SHAP where a magnitude is required.

**Effects in probability units.** Logistic coefficients are per transformed unit and are not actionable for a client. `scripts/logistic_effects.py` adds average marginal effects (mean dP/dx per raw unit, by finite difference through the whole pipeline) and average categorical probability contrasts (set the field to a level for everyone, then to the training-mode reference, and average the difference). The contrast is valid because each counterfactual row carries exactly one real category. Largest contrasts for logistic: moving from the 23-27 band to 48-or-older lowers predicted risk by about 27 points; a recorded gang affiliation raises it by about 17. These are model responses to a counterfactual edit, not causal effects.

**Methods disagree, and the disagreement is reported.** SHAP attributes the prediction, permutation importance attributes loss, XPER decomposes AUC - three different questions. `scripts/explanation_agreement.py` measures the resulting divergence rather than asserting it: Spearman rank correlation is 0.77-0.81 between SHAP and permutation importance, and 0.53-0.60 between either and XPER, while the top-10 sets share 8-10 features (`explanation_agreement.csv`). The headline drivers are therefore robust; their ordering is method-dependent. Quote the set, not the rank.

XPER approximates performance attribution on 150 records and 60 sampled coalitions. Its sample AUC and approximate contributions must not be treated as definitive full-population feature rankings. The current contribution sums exceed the corresponding sample AUC by 0.01792 (logistic) and 0.02279 (XGBoost); see `xper_diagnostics.csv`. The chart now displays this reconstruction residual. The exact numerical cause has not been isolated; an exact AUC decomposition is not established by these outputs.

TabICLv2 has permutation importance, PDP/ICE and interactive feature edits for local sensitivity. Native additive attribution is not implemented; this is an implementation limitation, not proof that the model cannot be explained. PDP/ICE can create implausible records and are not causal counterfactuals.

Local sensitivity for the same person across all three models and a transformed-unit logistic coefficient/odds-ratio table complement the explanations.

## Stability

The [end-to-end review](end_to_end_review.md) adds exact three-model reproduction evidence, a paired CPU/CUDA benchmark and a supplementary Holm correction of the 54 course fairness difference tests. Raw fairness results below remain exploratory. Inspection of XPER's installed kernel implementation found unconstrained weighted regression without endpoint constraints, explaining why reconstruction of sample AUC is not guaranteed.

Bootstrap refits compare probability drift, rank correlation and top-capacity Jaccard overlap. All three models use the same eight bootstrap samples and fixed algorithm seeds, isolating training-data sensitivity. Sample hashes are recorded in `stability_protocol.json`. Pairwise comparisons share refits and are not independent samples.

Jaccard is intersection divided by union, **not the fraction of all people changing status**. For equal-size selected sets, Jaccard J implies a replaced fraction `(1-J)/(1+J)` of each selected set. The paired rerun gives mean Jaccard 0.7725 (logistic), 0.7468 (XGBoost), and 0.7758 (TabICL). These correspond approximately to 13–15% replacement among selected people. TabICL has the highest overlap point estimate; logistic has the smallest probability drift. Neither difference establishes a population ranking from eight refits.

Refit sensitivity does not measure temporal drift. Random subsets of one historical cohort do not prove suitability for smaller agencies elsewhere.

## Fairness

Report FPR, TPR, selection rates, precision, Brier and calibration by race, gender and age (under 33 vs 33 or older), plus descriptive race-by-gender intersections and denominators. Small intersections and single-class groups have less reliable or undefined statistics. The section contains more than 90 intervals and tests with no multiple-testing correction; read them as an exploratory audit.

The supplementary [Holm sensitivity analysis](../artifacts/end_to_end/fairness_tests_holm.csv) adjusts the family of 54 course difference tests: rejections decrease from 41 to 32. No race test survives; gender and age equal-opportunity differences remain. Subsequent significance statements describe the original unadjusted results. This correction does not cover TOST or the evaluation-driven mitigation search.

Read the race results accordingly. The XGBoost and TabICLv2 selection-rate and FPR gaps of about +2 points have unadjusted bootstrap intervals excluding zero, but do not survive multiplicity control, and TOST places all race gaps inside the ±5-point tolerance. They are small and unconfirmed, not established disparities. Gender and age are the disparities this project actually has to answer for.

**Out-of-fold validation of the mitigation.** The Panel A/B numbers select the candidate variable and measure the improvement on the same evaluation cohort, so on their own they establish only that the mitigation can be fitted. `scripts/mitigation_nested.py` separates the two steps entirely inside the training partition: for each of five outer folds it reruns the FPDP candidate search on the outer-training part, selects from that part alone, refits baseline and mitigated models there, and measures accuracy and disparity on the held-out fold. The evaluation cohort is never touched.

The procedure selects `Gang_Affiliated` in all five folds for both models (selection agreement 1.00), so the candidate is a property of the training data rather than of the audited cohort. Out of fold the gender FNR gap moves from -0.118 to -0.035 (logistic) and -0.127 to -0.034 (XGBoost) - roughly 70% of the gap - at an AUC cost of about 0.010 and 0.009. The mean equal-opportunity p-value rises to 0.32 and 0.27, far below the in-sample 0.99, and three of five folds no longer reject at 0.05. Report these numbers; the in-sample figures are an upper bound. A p-value above 0.05 after mitigation still is not evidence of fairness - equivalence would be.

For beneficial support, prioritize missed access (FNR = 1 − TPR) and selection rates; retain FPR as a secondary allocation-error measure. In the course notation, selection is the favorable output and a re-arrested person is the one who needs it, so the course's equal opportunity is our equal FNR. Arrest is only a proxy for need and does not identify who benefits from support.

**Equivalence, not just difference.** A nonsignificant difference does not show fairness. Each gap is also tested with two one-sided tests (TOST) against a tolerance of ±5 percentage points. The tolerance was fixed in code before any TOST was run, but after the team had seen the point estimates of the gaps; state this openly. Rationale: at 20% capacity a 5-point difference in the share of a group offered support is the smallest gap we treat as operationally material. Gaps are classified as *equivalent* (TOST rejects |gap| ≥ 5 points), *different* (95% bootstrap interval excludes 0, equivalence not shown) or *inconclusive*. The TOST uses an analytic binomial variance that ignores the data-dependent top-20% cutoff, while the displayed intervals are bootstrap, so borderline cases can disagree.

Results at the proposed top-20% rule ([fairness_inference.csv](../artifacts/fairness_inference.csv), [fairness_tests.csv](../artifacts/fairness_tests.csv), [fairness_age_bands.csv](../artifacts/fairness_age_bands.csv)):

- **Race (Black minus White):** selection, FNR and FPR gaps are equivalent within ±5 points for all three models. For XGBoost and TabICLv2 the selection and FPR gaps (about +2 points) are also significant: nonzero but small, in the direction of more support offered to Black people. Logistic has no significant race gap. The course chi-squared table agrees except for logistic predictive equality (p ≈ 0.049, while the bootstrap interval includes 0). Precision gaps are inconclusive.
- **Gender (men minus women):** FNR gap −0.096 / −0.112 / −0.124 (logistic / XGBoost / TabICLv2), different for all models: women who are later re-arrested miss support more often. All models over-predict women (mean score ≈ 0.52 vs observed 0.454; ECE ≈ 0.07–0.08 vs ≈ 0.01–0.02 for men), and sufficiency is rejected. Women are nonetheless selected less, because few of them reach the top-20% cutoff.
- **Age (under 33 minus 33+):** FNR gap −0.245 / −0.228 / −0.237. About 97% of re-arrested people aged 48 or older are not selected, vs about 47–49% at 18–22. Age is a model input and a validated risk factor; in a support programme, ranking by risk rather than need is a policy choice that must be justified to the client.

**Which inputs generate the gender gap (FPDP).** For logistic and XGBoost, each input is set to each of its values for everyone, the top 20% is re-selected, and the equal-opportunity test is recomputed. Candidate variables are `Gang_Affiliated` and `Age_at_Release` ([fairness_candidates.csv](../artifacts/fairness_candidates.csv)). Gang affiliation is never recorded for women and is imputed as "No"; on the raw field its Cramér's V with gender is 1.0 ([fairness_dependence.csv](../artifacts/fairness_dependence.csv)). It is therefore a gender-aligned measurement artefact, not only a behavioural signal. Race proxies are weak (largest Cramér's V with race ≈ 0.18). TabICLv2 is excluded from FPDP because each point needs a full in-context prediction pass.

**Mitigation** ([fairness_mitigation.csv](../artifacts/fairness_mitigation.csv)). Dropping gang affiliation and re-estimating removes the equal-opportunity rejection (logistic p 0.000 → 0.995; XGBoost → 0.67) at about −0.014 AUC, roughly six times the XGBoost − logistic difference. Statistical parity is still rejected. Dropping age does not remove the gender gap. Neutralizing without re-estimation uses the FPDP value with the highest p-value on the evaluation set (for XGBoost, everyone set to gang = "Yes"), which is in-sample optimization and not a deployable rule. A p-value above 0.05 after mitigation is not evidence of fairness. Candidates, neutral values and mitigation are all evaluated on the same labels; confirm them on a separate validation split before claiming a mitigation.

The proposed capacity rule selects exactly round(n × 0.20). Its gaps differ from those at probability 0.5 because decisions differ. A smaller gap at another operating point does not mean the model itself improved.

Calibration and error-rate balance can conflict when base rates differ ([Chouldechova](https://arxiv.org/abs/1703.00056)). Descriptive group averages do not establish that a theorem binds for one attribute but not another. Equal FPR alone is not equalization of both error rates. **Changing thresholds leaves the probabilities and their calibration unchanged**, while allocations, precision and recall may change.

The group-threshold frontier now equalizes FNR. It optimizes using evaluation outcomes, so its near-zero gaps are optimistic in-sample illustrations, not independently validated mitigation. Any group-aware policy needs separate policy/legal review; we make no categorical legal determination.

Conditional statistical parity conditions on the historical Georgia supervision score. That score is built from arrest history and may itself carry policing bias, so conditioning on it can hide part of a disparity.

Removing race guarantees invariance to changing that raw input while fixing included features. It does not establish causal counterfactual fairness, eliminate proxies or guarantee smaller disparities. A/B results do not prove removal is universally free or fairer.

## Reproducibility and deployment gates

The [fresh review](deep_review.md) independently matches baseline features and values against the original NIJ releases and evaluates additional accuracy candidates using inner selection and outer evaluation folds within training. It distinguishes gender proxy exposure from future-outcome leakage. Shuffled-label tests cannot rule out all leakage: even a future-derived predictor loses association when training outcomes are scrambled. Exact measurement dates and fresh external validation remain unresolved.

The audit verifies official IDs, annual/cumulative target consistency, saved probabilities, model/prediction agreement, package versions and SHA-256 hashes. The September review found an XGBoost saved-model/prediction mismatch; models and dependent results were rebuilt together. Smoke outputs now live separately.

Before real use: verify feature timing, validate on a fresh cohort, justify eligibility and ties, assess intervention benefit prospectively, provide corrections/appeals, and monitor calibration and subgroup allocations. Define suspension thresholds with the client before observing deployment results. No sanctions, detention, sentencing or surveillance use is supported.
