# Research review and requirement audit

Reviewed 23 September 2026 against the supplied two-page course brief, the NIJ protocol/codebook, official results contextualization, and primary methodological sources.

## Course coverage

| Requirement | Evidence | Limitation to defend |
|---|---|---|
| Binary scoring problem and client | Three-year new-arrest target; support-allocation vendor | Arrest is institutionally mediated; not latent offending |
| White-box, ML, foundation model | Logistic, XGBoost, TabICLv2 | Two-member TFM ensemble is a compute choice |
| Statistical and economic performance | Probability metrics, baselines, paired bootstrap, CV, capacity/cost scenarios | Reused evaluation set; scenario effectiveness is assumed |
| Individual/global interpretation | SHAP, LIME, XPER, surrogate, PDP/ICE, editable app records | TFM lacks implemented native additive attribution; sensitivity is not causality |
| Stability | Bootstrap refits, rank/probability drift, allocation overlap | Unequal refit budgets; no temporal validation |
| Fairness | Race/gender and intersectional audits; two operating points; A/B race experiment | Multiple comparisons; threshold frontier fitted to evaluation labels |
| Recommendation across dimensions | Trade-off matrix and shadow-pilot recommendation | Needs external/prospective validation |
| Slides, code/notebook, app | Generated deck, executable notebook, Streamlit | Team must rehearse and submit deliverables |

The brief allocates 5 points each to technical skill, presentation and Q&A, and 10 to slides/code/app/notebook. No audit can guarantee a grade. Every team member must be able to defend every component.

## Findings and changes

1. **Artifact inconsistency:** saved XGBoost predictions differed from the published CSV by up to 0.07853. Rebuilt models and dependent outputs, and added an executable agreement check. The earlier saved model came from an older serialization environment; the exact historical cause is not proven.
2. **Adaptive evaluation:** development logs selected/rejected experiments using test results. Removed the untouched-holdout claim. Added fixed-configuration training CV, explicitly not nested search validation.
3. **Uncertainty:** replaced inference from overlapping individual intervals with paired AUC/Brier difference bootstraps.
4. **Baselines:** added training-prevalence probabilities, a training-calibrated one-feature incumbent, and Brier skill. Historical recorded-score comparisons cannot describe current agency tools.
5. **Fairness reasoning:** corrected the claim that thresholds decalibrate unchanged probabilities; removed unsupported legal conclusions. Labeled threshold search as in-sample because it uses evaluation labels.
6. **Allocation implementation:** unified main economics/fairness/A-B capacity selection and specified stable tie handling; added zero-capacity and invalid-input checks.
7. **Robustness:** reject malformed IDs/split flags/probabilities; allow meaningful single-class subgroup metrics; prevent pandas index misalignment. Added intersectional denominators.
8. **Reproducibility:** smoke runs use an isolated directory; the full runner includes downstream dependencies; provenance hashes and package versions are recorded. Reduced permutation-process fan-out after a local resource failure.
9. **Communication:** corrected Jaccard interpretation and unsupported data-ceiling/small-agency generalizations; regenerated notebook/slides and strengthened Q&A notes.
10. **Installation:** the original `XPER>=2,<3` requirement does not exist on PyPI. Pinned the actual package release `XPER==0.0.92` and verified its API; see the [package registry](https://pypi.org/project/XPER/). SHAP/LIME were missing from the active environment and were installed using the corrected requirements.

## Research basis

- [NIJ challenge protocol](https://nij.ojp.gov/funding/recidivism-forecasting-challenge): annual conditional cohorts, changing feature availability, Brier scoring. The cumulative baseline task is a deliberate course-project adaptation, not a challenge reproduction.
- [NIJ codebook](https://nij.ojp.gov/funding/recidivism-forecasting-challenge-appendix-2-codebook.pdf): feature definitions support the baseline/post-release separation. It does not establish availability in a new operational workflow.
- [NIJ contextualization report](https://www.ojp.gov/pdffiles1/nij/304110.pdf): motivates explicit naive baselines and careful interpretation of fairness/accuracy indices. Its annual results cannot be compared directly to ours.
- [NIJ synthesis of winning reports](https://www.ojp.gov/pdffiles1/nij/309826.pdf): reinforces attention to model design, feature construction, and limits of narrow fairness/accuracy measures. We do not infer a proven performance ceiling from convergence of three models.
- [Chouldechova, Fair prediction with disparate impact](https://arxiv.org/abs/1703.00056): explains incompatibilities among fairness criteria when group prevalence differs. Equalizing only FPR is not equalizing both error rates; thresholding does not alter the original probability scores.
- [scikit-learn cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html): selection and evaluation should be separated; preprocessing belongs inside folds. Repeated holdout inspection remains a limitation even when fitting excludes test rows.

## Remaining research needs

Team merge integration preserved the tuned L1 logistic configuration, additional ML candidates,
feature labels and the detailed pending course-method plan. All downstream results were rerun
after integration. The review also added FNR support-access figures, age-group descriptions,
logistic coefficients and a common-person local-sensitivity table. Advanced tests/mitigations
listed as TODO in PLAN.md are not claimed as completed.

Prospective temporal/external evaluation, pre-registered policy thresholds, causal intervention-benefit estimation, stronger local TFM explanations, and independently validated mitigation are not established here. New modeling experiments should use training-only nested validation rather than further optimizing this familiar evaluation set. A support policy must ultimately be evaluated on service benefit and access, not only arrest prediction.
