# Dataset proposal for pre-validation

**Course:** Interpretability, Stability, and Algorithmic Fairness  
**Dataset:** NIJ 2021 Recidivism Forecasting Challenge, Georgia parole cohort  
**Source:** https://nij.ojp.gov/funding/recidivism-forecasting-challenge

## Proposed client question

At the start of community supervision, which people are most likely to experience a new arrest within three years? A hypothetical Georgia re-entry agency would use the estimate only to prioritize voluntary, beneficial support such as employment, housing, or treatment services. It would not be used to increase surveillance, restrictions, detention, or punishment.

## Why the dataset fits the project

The post-challenge file contains 25,835 records, a binary three-year target, an official 70/30 train/test flag, and 32 baseline person-level variables. It supports the required comparison of:

1. Logistic regression as the white-box model;
2. XGBoost as the conventional machine-learning model;
3. TabICLv2 as the open tabular foundation model.

It also contains race and gender, enabling explicit subgroup audits. Those attributes will be used only for evaluation, not as model inputs. Residence geography will also be excluded because it can proxy protected status.

## Evaluation design

Predictive performance will include ROC AUC, average precision, Brier score, log loss, calibration error, and bootstrap confidence intervals. Economic performance will be a transparent scenario analysis based on support capacity, intervention cost, avoided-event cost, and assumed effectiveness. Interpretability will combine logistic coefficients and model-agnostic held-out permutation importance. Stability will cover bootstrap uncertainty and sensitivity stress tests. Fairness will compare selection, false-positive and true-positive rates, calibration/Brier error, and gaps across race and gender.

## Key validity guardrail

Variables describing violations, drug tests, employment, programs, and residence changes accrue after supervision begins. They will be excluded from the prediction because using them for a baseline score would leak post-decision information. The outcome is arrest rather than offending, so the report will discuss construct bias, exposure to policing, historical context, and limited transportability beyond Georgia releases from 2013–2015.

