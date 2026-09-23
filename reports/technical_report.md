# Technical report and model card

## Recommendation and scope

Recommend **XGBoost for a prospective shadow pilot**, with logistic regression as a transparent challenger. TabICLv2 supplies the required foundation-model comparison. Consider accuracy, explanation cost, fairness, refit sensitivity and runtime together. No model is validated for operational decisions.

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

SHAP explains logistic/XGBoost predictions in log-odds, aggregated to raw features. Reconstruction tests verify that summed SHAP values recover saved-model probabilities. LIME is a local approximation; inspect its recorded local surrogate R² before trusting the explanation (the review run was about 0.25, a weak fit). Neither identifies causal effects. The shallow global surrogate has imperfect fidelity and cannot replace the original model.

XPER approximates performance attribution on 150 records and 60 sampled coalitions. Its sample AUC and approximate contributions must not be treated as definitive full-population feature rankings.

TabICLv2 has permutation importance, PDP/ICE and interactive feature edits for local sensitivity. Native additive attribution is not implemented; this is an implementation limitation, not proof that the model cannot be explained. PDP/ICE can create implausible records and are not causal counterfactuals.

## Stability

Bootstrap refits compare probability drift, rank correlation and top-capacity Jaccard overlap. All three models use eight refits. Pairwise comparisons share refits and are not independent samples.

Jaccard is intersection divided by union, **not the fraction of all people changing status**. For equal-size selected sets, Jaccard J implies a replaced fraction `(1-J)/(1+J)` of each selected set. Overlap around 0.73–0.77 implies about 13–15% replacement among selected people, not one quarter of the cohort changing status.

Refit sensitivity does not measure temporal drift. Random subsets of one historical cohort do not prove suitability for smaller agencies elsewhere.

## Fairness

Report FPR, TPR, selection rates, precision, Brier and calibration by race/gender, plus descriptive race-by-gender intersections and denominators. Small intersections and single-class groups have less reliable or undefined statistics. Bootstrap intervals are exploratory, with no multiple-testing correction.

For beneficial support, prioritize missed access (FNR = 1 − TPR) and selection rates; retain FPR as a secondary allocation-error measure. Arrest is only a proxy for need and does not identify who benefits from support. An age-group descriptive audit is also included because age is a model input. Local sensitivity for the same person across all three models and a transformed-unit logistic coefficient/odds-ratio table complement the explanations.

The proposed capacity rule selects exactly round(n × 0.20). Its gaps differ from those at probability 0.5 because decisions differ. A smaller gap at another operating point does not mean the model itself improved.

Calibration and error-rate balance can conflict when base rates differ ([Chouldechova](https://arxiv.org/abs/1703.00056)). Descriptive group averages do not establish that a theorem binds for one attribute but not another. Equal FPR alone is not equalization of both error rates. **Changing thresholds leaves the probabilities and their calibration unchanged**, while allocations, precision and recall may change.

The group-threshold frontier optimizes using evaluation outcomes. Its small gaps are optimistic in-sample illustrations, not independently validated mitigation. Any group-aware policy needs separate policy/legal review; we make no categorical legal determination.

Removing race guarantees invariance to changing that raw input while fixing included features. It does not establish causal counterfactual fairness, eliminate proxies or guarantee smaller disparities. A/B results do not prove removal is universally free or fairer.

## Reproducibility and deployment gates

The audit verifies official IDs, annual/cumulative target consistency, saved probabilities, model/prediction agreement, package versions and SHA-256 hashes. The September review found an XGBoost saved-model/prediction mismatch; models and dependent results were rebuilt together. Smoke outputs now live separately.

Before real use: verify feature timing, validate on a fresh cohort, justify eligibility and ties, assess intervention benefit prospectively, provide corrections/appeals, and monitor calibration and subgroup allocations. Define suspension thresholds with the client before observing deployment results. No sanctions, detention, sentencing or surveillance use is supported.
