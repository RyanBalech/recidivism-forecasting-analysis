# Technical report and model card

## Executive recommendation

Pilot **XGBoost** for allocating voluntary re-entry support, with logistic regression as a transparent challenger and quarterly governance review. TabICLv2 produced the best held-out probability accuracy, but the improvement over XGBoost was small: Brier score improved from 0.2054 to 0.2039 and ROC AUC from 0.7299 to 0.7336. XGBoost had the best calibration error (0.0118), required less than half the runtime, and showed smaller observed subgroup gaps at the fixed 0.5 threshold.

This is a pilot recommendation, not an authorization for operational use. A prospective study must show that offering services from the score improves outcomes without creating harmful allocation gaps.

## Decision and population

- **Intended decision:** prioritize scarce, beneficial support at the start of parole supervision.
- **Population:** people released from Georgia prisons to parole supervision from 2013–2015.
- **Target:** any new felony or misdemeanor arrest within three years.
- **Train/test:** NIJ's official split, 18,028 training and 7,807 held-out records.
- **Threshold:** 0.5 for the primary error-rate audit; the client app also supports fixed-capacity ranking.

The target is an arrest, not latent offending. It reflects policing, reporting, and legal processes as well as behavior. The model must never be used to impose sanctions or reduce access to services.

## Data preparation and leakage control

Only 29 baseline fields available at supervision start are modeled. Gender, race, and Residence PUMA are excluded from the feature set. Gender and race remain in a separate audit table. Post-release variables covering violations, drug tests, employment, program attendance, and residential changes are excluded because they accrue after the scoring time and can be affected by the outcome or supervision intensity.

Numeric fields are median-imputed and standardized for logistic regression; categorical fields are most-frequent-imputed and one-hot encoded. XGBoost uses the same transformed matrix for a fair comparison. TabICLv2 receives the mixed-type pandas frame and applies its native categorical encoding and learned normalization.

## Models

**Logistic regression** uses L2 regularization (`C=0.25`) and provides a stable, auditable linear baseline. **XGBoost** uses 550 depth-three trees, conservative learning rate and regularization. **TabICLv2** is a pretrained in-context tabular transformer. Two ensemble views are used to fit within the available 6 GB GPU; this compute choice is recorded in the run manifest.

## Held-out results

| Model | ROC AUC | Average precision | Brier ↓ | Log loss ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.7295 | 0.7691 | 0.2055 | 0.5970 | 0.0132 |
| XGBoost | 0.7299 | 0.7686 | 0.2054 | 0.5969 | **0.0118** |
| TabICLv2 | **0.7336** | **0.7719** | **0.2039** | **0.5932** | 0.0197 |

Bootstrap 95% intervals are stored in `artifacts/bootstrap_intervals.json`. Their overlap means the ranking should not be oversold.

## Economic scenario

At a 20% service capacity, the three models identify 1,282–1,292 of the held-out positive outcomes, or about 28.6%–28.8% of all events. Under the illustrative assumptions of $5,000 support cost, $50,000 event cost, and 20% effectiveness, the calculated net value is about $5.0M–$5.1M. These are scenario outputs, not causal estimates. The application exposes every assumption and allows the client to change it.

## Interpretability

Held-out permutation importance identifies age at release, gang affiliation, prior felony arrests, and prison tenure as recurring drivers. Logistic regression adds signed coefficients through its fitted pipeline. For TabICLv2, a model-agnostic permutation audit covers ten prespecified high-relevance fields. Explanations describe how the model behaves; they do not imply that changing a field would causally change recidivism.

## Stability

Four hundred bootstrap resamples quantify sampling uncertainty. The 95% ROC AUC interval widths are 0.022–0.023 and Brier interval widths are 0.0079–0.0088 across models. A feature-destruction stress test confirms all models rely materially on the input signal. Because the source lacks usable event dates, temporal stability cannot be estimated. A real pilot must evaluate a later release cohort and define drift triggers before launch.

## Fairness

At threshold 0.5, XGBoost has the smallest observed race false-positive-rate gap (0.042) and gender FPR gap (0.107). Corresponding gaps are 0.062/0.121 for logistic regression and 0.068/0.273 for TabICLv2. Brier gaps by race remain under 0.009 for all models. Threshold metrics depend on the operating policy and subgroup base rates, so the app lets reviewers inspect several measures rather than reducing fairness to one number.

Excluding protected attributes does not create fairness by itself. Proxy variables, label bias, differential policing, and historical selection can remain. Fairness monitoring should cover both error rates and who receives useful services.

## Deployment controls

1. Run a prospective shadow-mode evaluation on a later cohort.
2. Use the score only to expand access to voluntary support.
3. Publish plain-language documentation and provide an appeal/correction route.
4. Log data quality, predictions, allocation decisions, overrides, and outcomes.
5. Review calibration, drift, and subgroup gaps quarterly; suspend scoring on breach.
6. Estimate causal program benefit through a randomized or strong quasi-experimental design.

## Limitations

The data is historical, from one state, and de-identified through aggregation. It does not include every factor relevant to re-entry. Arrest is an imperfect and institutionally mediated outcome. The foundation-model ensemble was limited for local hardware. The economic analysis is illustrative. None of the three models is validated for high-stakes adverse decisions.
