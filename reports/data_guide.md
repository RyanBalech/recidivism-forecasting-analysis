# Data guide: what is in the files and what the models use

This guide separates the original NIJ variables from the transformations created by the modeling pipeline. The official source is the [NIJ Recidivism Forecasting Challenge](https://nij.ojp.gov/funding/recidivism-forecasting-challenge), and the field definitions come from the [NIJ codebook](https://nij.ojp.gov/funding/recidivism-forecasting-challenge-appendix-2-codebook.pdf).

## The five CSV files

| File | Rows × columns | Purpose | Outcomes included? |
|---|---:|---|---|
| `nij-challenge2021_full_dataset.csv` | 25,835 × 54 | Post-challenge combined data. Adds `Training_Sample` so the original split can be reconstructed. | Yes |
| `nij-challenge2021_training_dataset.csv` | 18,028 × 53 | Original labeled training cohort. Contains baseline, supervision-activity, and outcome fields. | Yes |
| `nij-challenge2021_test_dataset_1.csv` | 7,807 × 33 | Original Year 1 forecasting cohort. Contains baseline fields only. | No |
| `nij-challenge2021_test_dataset_2.csv` | 5,460 × 49 | People remaining after Year 1, with supervision-activity fields added. | No |
| `nij-challenge2021_test_dataset_3.csv` | 4,146 × 49 | People remaining after Year 2. | No |

The project uses the full file only because it now contains the outcomes and the official `Training_Sample` flag. Records with `Training_Sample = 1` form the 18,028-row training set; records with `0` form the 7,807-row evaluation set (subsequently reused during development).

## Representative raw rows

These are the first three records in the full file, reduced to a readable subset of columns. The values are original NIJ values.

| ID | Gender | Race | Age at release | Gang affiliated | First risk score | Education | Prison offense | Prison years | Prior felony arrests | Mental-health/substance condition | Recidivism within 3 years | Training sample |
|---:|---|---|---|---|---:|---|---|---|---|---|---|---:|
| 1 | M | BLACK | 43–47 | No | 3 | At least some college | Drug | More than 3 years | 6 | Yes | No | 1 |
| 2 | M | BLACK | 33–37 | No | 6 | Less than HS diploma | Violent/Non-Sex | More than 3 years | 7 | No | Yes | 1 |
| 3 | M | BLACK | 48 or older | No | 7 | At least some college | Drug | 1–2 years | 6 | Yes | Yes | 1 |

The Year 1 test file begins with IDs 6, 8, and 12. It has the same baseline fields but no outcome columns, because contestants originally had to forecast those outcomes.

## Prediction target

The project predicts `Recidivism_Within_3years`:

- `Yes` becomes `1`.
- `No` becomes `0`.
- NIJ defines the outcome as a new Georgia fingerprintable felony or misdemeanor arrest recorded within three years of parole supervision starting.

The target is an arrest record, not a direct measure of offending or an inherent personal trait. This distinction matters for interpretation and fairness.

## The 29 raw fields used by the models

No new behavioral or demographic data was invented. The model receives 29 original baseline fields, grouped below.

### Release and supervision information

| Field | Plain-language meaning |
|---|---|
| `Age_at_Release` | Age band when released from prison |
| `Gang_Affiliated` | Whether an investigation verified gang affiliation |
| `Supervision_Risk_Score_First` | First parole risk score, from 1 (lowest) to 10 |
| `Supervision_Level_First` | Initial assignment: Standard, High, or Specialized |

### Prison information

| Field | Plain-language meaning |
|---|---|
| `Education_Level` | Education at prison entry |
| `Dependents` | Number of dependents at prison entry, capped at 3+ |
| `Prison_Offense` | Primary prison conviction group |
| `Prison_Years` | Banded length of the prison stay before release |

### Prior arrest history

| Field | Plain-language meaning |
|---|---|
| `Prior_Arrest_Episodes_Felony` | Prior arrest episodes whose most serious charge was a felony |
| `Prior_Arrest_Episodes_Misd` | Prior misdemeanor arrest episodes |
| `Prior_Arrest_Episodes_Violent` | Prior violent arrest episodes |
| `Prior_Arrest_Episodes_Property` | Prior property arrest episodes |
| `Prior_Arrest_Episodes_Drug` | Prior drug arrest episodes |
| `_v1` | Prior arrests containing probation/parole-violation charges |
| `Prior_Arrest_Episodes_DVCharges` | Whether any prior arrest contained domestic-violence charges |
| `Prior_Arrest_Episodes_GunCharges` | Whether any prior arrest contained gun charges |

### Prior conviction history

| Field | Plain-language meaning |
|---|---|
| `Prior_Conviction_Episodes_Felony` | Prior felony conviction episodes |
| `Prior_Conviction_Episodes_Misd` | Prior misdemeanor conviction episodes |
| `Prior_Conviction_Episodes_Viol` | Whether any prior conviction's most serious charge was violent |
| `Prior_Conviction_Episodes_Prop` | Prior property conviction episodes |
| `Prior_Conviction_Episodes_Drug` | Prior drug conviction episodes |
| `_v2` | Whether a prior conviction contained probation/parole-violation charges |
| `_v3` | Whether a prior conviction contained domestic-violence charges |
| `_v4` | Whether a prior conviction contained gun charges |

The four `_v` names are present in the downloaded CSV. Their meanings above come from their positions in the official codebook.

### Prior supervision and release conditions

| Field | Plain-language meaning |
|---|---|
| `Prior_Revocations_Parole` | Whether parole had previously been revoked |
| `Prior_Revocations_Probation` | Whether probation had previously been revoked |
| `Condition_MH_SA` | Mental-health or substance-abuse programming condition |
| `Condition_Cog_Ed` | Cognitive-skills or education programming condition |
| `Condition_Other` | Other condition such as no victim contact, electronic monitoring, restitution, or registration/programming |

## Fields deliberately excluded from scoring

| Field or group | How it is used | Reason |
|---|---|---|
| `ID` | Record matching only | Identifier, not a predictor |
| `Gender` | Fairness audit only | Protected attribute |
| `Race` | Fairness audit only | Protected attribute |
| `Residence_PUMA` | Excluded | Geography can act as a strong proxy for race |
| `Training_Sample` | Defines the split | Administrative flag |
| Four recidivism fields | Outcomes only | Labels cannot be predictors |
| All 16 supervision-activity fields | Excluded | They occur after supervision begins and would violate the baseline prediction time |

The excluded activity fields are violations, delinquency reports, program attendance and absences, residence changes, drug-test timing and results, employment, jobs per year, and employment exemption. They are useful for later-year forecasting in the original challenge, but not for a score made at the start of supervision.

## Features created by preprocessing

The phrase “created features” refers to machine-readable transformations of the 29 raw fields:

1. `Recidivism_Within_3years` is converted from Yes/No to 1/0.
2. Logistic regression and XGBoost median-impute the numeric risk score. A separate missing-risk-score indicator is created.
3. Numeric inputs are standardized using training-set means and standard deviations.
4. Missing categorical values are filled with the training-set mode.
5. Each categorical value becomes a binary one-hot column. Examples include `Age_at_Release_18-22`, `Gang_Affiliated_Yes`, and `Prison_Offense_Drug`.

Logistic regression uses this one-hot representation. XGBoost first converts ordered age/prison bands and capped counts into numeric ranks, so its transformed matrix differs. The models share 29 raw inputs, not an identical number of transformed columns. TabICLv2 receives the mixed-type table after training-mode categorical imputation and uses its own encoder/normalization.

No synthetic people, external census fields or post-release variables were added. Existing baseline inputs may nevertheless carry proxy information about protected attributes.

## Features created for evaluation, not prediction

The pipeline also creates outputs used to understand the models:

| Output | Meaning |
|---|---|
| `p_logistic`, `p_xgboost`, `p_tabicl` | Each model's predicted three-year probability |
| Brier score and log loss | Probability accuracy |
| ROC AUC and average precision | Ranking/discrimination |
| Calibration error | Difference between predicted and observed risk |
| Selection, true-positive, and false-positive rates | Threshold behavior |
| Race and gender gaps | Fairness audit comparisons |
| Permutation importance | Change in held-out Brier loss after shuffling one field |
| Bootstrap intervals | Sampling uncertainty across 400 resamples |
| Economic scenario fields | Results under editable capacity, cost, and effectiveness assumptions |

These fields are not model inputs. Historical research decisions did inspect evaluation results, so the evaluation cohort is not an untouched model-selection holdout.

## Where this is implemented

- Raw feature selection: [`src/recidivism/config.py`](../src/recidivism/config.py)
- Split and target conversion: [`src/recidivism/data.py`](../src/recidivism/data.py)
- Preprocessing and model definitions: [`src/recidivism/modeling.py`](../src/recidivism/modeling.py)
- Metrics and economic scenario: [`src/recidivism/metrics.py`](../src/recidivism/metrics.py)
- Saved held-out predictions: [`artifacts/test_predictions.csv`](../artifacts/test_predictions.csv)
