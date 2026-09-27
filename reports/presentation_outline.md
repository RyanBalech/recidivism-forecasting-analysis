# Presentation outline — Trustworthy Recidivism Forecasting (Team 11)

15 minutes of presentation + 10 minutes of Q&A. Target **14 minutes** of talk, leaving 1 minute of buffer.
Every member must be able to answer questions on any part, so each person has a backup partner.

Paths below are relative to this file (`reports/`). Figures are in `artifacts/figures/`.

## Split and timing

| # | Section | Time | Slides | Backup |
|---|---|---|---|---|
| P1 | Intro + EDA | 1.5 min | 1–3 | P4 |
| P2 | Features + Leakage | 1.5 min | 4–5 | P5 |
| P3 | Models + Performance | 2.5 min | 6–8 | P6 |
| P4 | Interpretability | 2.7 min | 9–12 | P1 |
| P5 | Fairness (race as the main line, gender as the problem we found) | 3.0 min | 13–16 | P2 |
| P6 | Stability + Trade-offs + Recommendation | 2.5 min | 17–19 | P3 |
| P1 | App demo | 0.75 min | 20 | — |

Interpretability was extended from 3 to 4 slides (26 Sep): PDP/ICE and XPER moved from the appendix
into the core so every course interpretability method has an evidence slide. Total ≈ 14:25.

P2 and P5 back each other up because both deal with proxies and leakage.

### Choices behind this structure

- **Fairness focuses on race and gender. Age appears only in the appendix.** Age is a validated and legally accepted risk factor in US risk assessment; the age–crime curve is one of the most robust findings in criminology. Its large FNR gap mostly follows from different base rates. The notebook and the app still show age results, so keep the appendix slide ready for Q&A.
- **Race is the main fairness storyline.** It is the question the audience expects: the COMPAS / ProPublica debate over whether risk tools treat Black people as more dangerous.
- **Gender is presented openly, as the problem we found and worked on.** All three models share the gap, so it comes from the feature set rather than from any one model. It also produced our strongest technical work: the missingness leak, the FPDP diagnosis and the nested mitigation.
- **Fairness comes before Stability.** The stability section's key finding is that abstention widens the gender FNR gap, and that only makes sense once the audience knows what the gap is.

---

## P1 · Intro + EDA (1.5 min)

### Slide 1 · Title
- Trustworthy Recidivism Forecasting, Team 11

### Slide 2 · Client and decision
- Client: a software vendor that sells risk tools to US community-supervision agencies.
- Decision: rank people by risk and prioritise the top 20% for **voluntary re-entry support**.
- Scope: support allocation only. Never sanctions, detention or surveillance.
- The target is *recorded re-arrest*, which is only a proxy for need.

### Slide 3 · Data and EDA
- NIJ 2021 Recidivism Forecasting Challenge data: people released in Georgia in 2013–2015. Official split of 18,028 training and 7,807 evaluation records.
- Target: new arrest within 3 years (cumulative). NIJ scored annual forecasts instead, so our results cannot be compared with the challenge leaderboard.
- EDA uses training data only:
  - 57.8% of people are re-arrested.
  - `Gang_Affiliated` is 12.3% missing and `Prison_Offense` is 12.9% missing.
  - Re-arrest rate by race is almost the same for Black and White people. By gender it is about 0.45 for women and 0.59 for men.
- Figure: the EDA panel in notebook section 2 (not saved to `artifacts/`, so export it from the notebook). Replace the age sub-panel with re-arrest rate by race × gender.
- Transition line: "The missingness is structured" → hand over to P2.

## P2 · Features + Leakage (1.5 min)

### Slide 4 · Only information known when supervision starts
- 29 baseline fields.
- Race, gender and residence PUMA are excluded from the models and used only for auditing.
- Everything recorded after release is excluded: employment, drug tests, violations, etc.
- Positive control: adding the post-release fields back raises AUC from 0.729 to 0.813. They really do leak outcome information, which is why they are excluded.
- Sources: [config.py](../src/recidivism/config.py), [INELIGIBLE_timing_positive_control.csv](../artifacts/deep_review/INELIGIBLE_timing_positive_control.csv)

### Slide 5 · The leak we found: gender was encoded in missingness
- `Gang_Affiliated` is missing for every woman and for no man.
- If missingness is kept as a feature, gender can be recovered from the model's own inputs with **AUC 1.000**. TabICL treats NaN as its own category, so it effectively received gender.
- Fix: fill missing values with the mode, plus a regression test so the leak cannot return.
- After the fix, gender is still recoverable at AUC 0.776. P5 picks this up.
- Sources: [proxy_recovery.png](../artifacts/figures/proxy_recovery.png), [proxy_recovery.csv](../artifacts/proxy_recovery.csv)
- Mention only if asked: the other audits cross-checked every field against the original NIJ release and ran shuffled-label stress tests ([leakage_audit.json](../artifacts/deep_review/leakage_audit.json)).

## P3 · Models + Performance (2.5 min)

### Slide 6 · Three models and how they were tuned

| Model | Encoding | Tuning |
|---|---|---|
| L1 logistic regression | one-hot | 5-fold grid search, C = 0.2154 |
| XGBoost | ordinal encoding for count fields | 5-fold random search, 60 draws; deeper trees overfit and were rejected |
| TabICLv2 | mixed types | 16 ensemble members; results level off at about 8 |

- Five other ML candidates (CatBoost, LightGBM, EBM, HistGB, random forest) all land between test AUC 0.729 and 0.733.
- Sources: [ml_model_comparison.csv](../artifacts/ml_model_comparison.csv), [estimator_sweep.png](../artifacts/figures/estimator_sweep.png)

### Slide 7 · Statistical performance: effectively a tie

| Model | ROC AUC | Brier ↓ | ECE ↓ | Fit + predict |
|---|---:|---:|---:|---:|
| Logistic regression | 0.7298 | 0.2054 | 0.0129 | 1.0 s |
| XGBoost | 0.7324 | 0.2045 | 0.0112 | 8.7 s |
| TabICLv2 | 0.7328 | 0.2044 | 0.0199 | 41.8 s |

- **How good is 0.73?** Reviews of US recidivism tools grade AUC < .55 poor, .55–.63 fair, .64–.71 good and ≥ .71 excellent (Desmarais & Singh 2013, CSG Justice Center, Table 2; bands anchored to Cohen's d via Rice & Harris 2005). Our models sit in the "excellent" band; the historical Georgia score (0.60) is "fair".
- XGBoost's AUC is 0.0025 higher than logistic's, with a paired 95% CI of [0.0006, 0.0044]. The difference is real but small.
- Brier = mean of (predicted probability − outcome)², lower is better. It is the accuracy score NIJ used to rank this challenge. All three models are about 16% better than predicting the training base rate for everyone (0.245).
- Calibration gets one sentence, as part of Brier (Brier = calibration error − resolution + uncertainty): does a predicted 70% mean 70% re-arrested? Logistic and XGBoost are statistically indistinguishable from perfect calibration (Cox slopes 0.999 and 1.005); TabICL's slope is 0.913, so its probabilities are slightly too extreme. This sets up "calibration" on slides 15 and 19; in course terms, calibration within groups is *sufficiency*. Bin-dependent ECE and the Spiegelhalter test stay for Q&A.
- Sources: [performance_benchmark.png](../artifacts/figures/performance_benchmark.png) (from `scripts/deck_figures.py`), [paired_comparisons.csv](../artifacts/paired_comparisons.csv), [validation_baselines.csv](../artifacts/validation_baselines.csv), [calibration_tests.csv](../artifacts/calibration_tests.csv)

### Slide 8 · The client's real question: are we better than the current tool, in dollars?
- Scenario (assumed, not estimated): 20% capacity = 1,561 offers; support costs $5,000 per person; a re-arrest costs $50,000; support prevents 20% of re-arrests.
- Formula: net value = re-arrested among offers × $50,000 × 20% − offers × $5,000. Logistic: 1,285 × $10,000 − 1,561 × $5,000 = $5.045M.
- Programme cost is the same for every ranking, so only precision (share of offers reaching someone later re-arrested) differs.

| Ranking | AUC | Re-arrested / offers | Precision | Net value |
|---|---:|---:|---:|---:|
| Random | 0.51 | 926 / 1,561 | 0.59 | $1.455M |
| Historical Georgia score | 0.60 | 1,053 / 1,561 | 0.67 | $2.725M |
| Logistic | 0.73 | 1,285 / 1,561 | 0.82 | $5.045M |
| XGBoost | 0.73 | 1,297 / 1,561 | 0.83 | $5.165M |
| TabICL | 0.73 | 1,292 / 1,561 | 0.83 | $5.115M |

- Break-even: support pays for itself if precision × $50,000 × effect > $5,000, i.e. effect > 10% / precision. That is 12% with our ranking against 15% with the historical score.
- The models beat the historical score at every capacity from 5% to 50% (appendix A2). The dollar figures are scenarios, not causal estimates.
- Sources: [incumbent_economics.csv](../artifacts/incumbent_economics.csv), [incumbent_capacity_sweep.csv](../artifacts/incumbent_capacity_sweep.csv), [incumbent_effectiveness_sweep.csv](../artifacts/incumbent_effectiveness_sweep.csv)

## P4 · Interpretability (2 min)

### Slide 9 · Global drivers
- SHAP global importance.
- Logistic effects in probability points:
  - Moving from age 23–27 to 48+ lowers predicted risk by about 27 points. Here age appears as a risk factor, consistent with how we treat it in fairness.
  - A recorded gang affiliation raises predicted risk by about 17 points.
- Sources: [shap_global.png](../artifacts/figures/shap_global.png), [marginal_effects.png](../artifacts/figures/marginal_effects.png), [probability_contrasts.csv](../artifacts/probability_contrasts.csv)

### Slide 10 · Looking from outside: PDP/ICE for all three models
- PDP (average effect) and ICE (one curve per person) for age, prior felony arrests and the Georgia score, for all three models. All agree risk falls with age.
- The three explanation routes: logistic coefficients read directly; XGBoost needs SHAP, and a depth-3 surrogate reproduces only R² = 0.61; TabICL has no native attribution, so PDP/ICE is the only view.
- Sources: [pdp_ice.png](../artifacts/figures/pdp_ice.png), [interpretability_summary.json](../artifacts/interpretability_summary.json)

### Slide 11 · Explaining one person, and checking the explanation is faithful
- A SHAP waterfall explains one person's score.
- LIME was moved to the raw feature space. This raised its local fit (R²) from 0.25 to 0.41–0.43.
- Three cases (highest, median and lowest risk), each run with three random seeds, give 63 feature conditions. All 63 keep the same sign across seeds.
- Sources: [shap_individual.png](../artifacts/figures/shap_individual.png), [lime_individual.png](../artifacts/figures/lime_individual.png), [lime_fidelity.csv](../artifacts/lime_fidelity.csv)

### Slide 12 · Explaining performance with XPER; the methods disagree on order
- XPER (course method) splits the AUC into a benchmark (≈ 0.47) plus feature contributions; age adds ≈ 0.09. Figure: [xper.png](../artifacts/figures/xper.png)
- Spearman rank correlation is 0.77–0.81 between SHAP and permutation importance, but only 0.53–0.60 between either of them and XPER.
- The three methods share 8–10 of their top-10 features.
- Takeaway: quote the *set* of main drivers, not their exact ranking.
- Interpretability cost by model:
  - Logistic: read the coefficients directly.
  - XGBoost: a depth-3 surrogate tree reproduces it with R² = 0.61 only.
  - TabICL: no native attribution; only PDP/ICE.
- Sources: [explanation_agreement.png](../artifacts/figures/explanation_agreement.png), [global_surrogate.png](../artifacts/figures/global_surrogate.png)

## P5 · Fairness (3 min)

**Primary metric: FNR, the share of people who are later re-arrested but were not selected for support.** Being selected means receiving help, so FNR measures missed support, which is the real harm in this programme. All results are at the deployed top-20% rule.

### Slide 13 · Race (1): excluding race is not the same as being fair
- Opening: the COMPAS debate (ProPublica, 2016), where Black defendants were more often wrongly flagged as high risk.
- Here, **base rates are almost equal**: 0.582 for Black and 0.564 for White people. So the data does not force a gap.
- What exclusion guarantees: two people who differ only in race get exactly the same score, and adding race back changes AUC by at most about 0.001.
- What exclusion does not guarantee: race can still be recovered from the remaining features at AUC 0.708, because criminal-history variables carry part of the same information. These variables are the core of risk assessment and cannot simply be dropped.
- Suggested script: *"Excluding race guarantees the model never uses it directly. It cannot guarantee equal outcomes, because criminal history carries part of the same information. That is why we audit outcomes, not inputs."*
- Sources: [race_ab_test.csv](../artifacts/race_ab_test.csv), [proxy_recovery.csv](../artifacts/proxy_recovery.csv), [fairness_impossibility.csv](../artifacts/fairness_impossibility.csv)

### Slide 14 · Race (2): the outcome audit, with three caveats
- FNR gap, Black minus White: −0.005 / −0.017 / −0.023 (logistic / XGBoost / TabICL).
- Every race gap falls within ±5 percentage points by an equivalence test (TOST). After Holm multiple-testing correction, no race test remains significant.
- Selection-rate ratio of White to Black: 0.93 / 0.89 / 0.87, all above the four-fifths rule.
- Three caveats:
  1. **In a support programme the direction is reversed.** XGBoost and TabICL select about 2 points more Black people, which here means more help offered. The same model used for sanctions would turn this into harm.
  2. **The label may be biased.** We measure recorded arrest, not actual reoffending. If policing intensity differs by race, this data cannot reveal it.
  3. **Prediction quality differs.** Within-group AUC is 0.718–0.721 for Black people against 0.746–0.748 for White people.
- Sources: [race_fairness.png](../artifacts/figures/race_fairness.png), [fairness_inference.csv](../artifacts/fairness_inference.csv), [fairness_tests_holm.csv](../artifacts/end_to_end/fairness_tests_holm.csv), [fairness_by_group.csv](../artifacts/fairness_by_group.csv)
- ⚠️ [fairness_support_access.png](../artifacts/figures/fairness_support_access.png) includes an age column. Crop or regenerate it before use.

### Slide 15 · Gender (1): women are over-predicted, yet selected less
- FNR gap, men minus women: −0.096 / −0.112 / −0.124. All three models share it, so the gap comes from the feature set rather than from one model.
- Women are selected at about half the rate of men (ratio 0.55 / 0.52 / 0.49).
- Every model over-predicts women's risk: the mean score is about 0.52 against an observed rate of 0.454. This is the pattern criticised in *State v. Loomis* and in the gender-responsive assessment literature.
- Impossibility result: base rates differ by about 14 points (0.591 for men, 0.454 for women), so calibration and equal error rates cannot both hold.
- The over-prediction actually helps women here. Recalibrating by gender would lower their scores, select fewer of them, and widen the FNR gap.
- Sources: [fairness_impossibility.csv](../artifacts/fairness_impossibility.csv), [fairness_inference.csv](../artifacts/fairness_inference.csv)

### Slide 16 · Gender (2): cause located, mitigated, validated out of sample
- FPDP points to `Gang_Affiliated`. It is never recorded for women, and the raw field has Cramér's V = 1.0 with gender. It reflects how records were kept rather than behaviour.
- Mitigation: drop the field and refit. A 5-fold nested validation inside the training data picks the same field in every fold.
- Out-of-fold results:

| Model | Gender FNR gap (before → after) | AUC cost |
|---|---|---:|
| Logistic | −0.118 → −0.035 | −0.010 |
| XGBoost | −0.127 → −0.034 | −0.009 |

- The gap closes by about 70%.
- Costs: men's FNR rises by about 2 points, so part of the improvement is levelling down. Selection rates still differ between men and women.
- Sources: [fpdp_gender.png](../artifacts/figures/fpdp_gender.png), [fairness_dependence.png](../artifacts/figures/fairness_dependence.png), [mitigation_nested_summary.csv](../artifacts/mitigation_nested_summary.csv), [fairness_mitigation.csv](../artifacts/fairness_mitigation.csv)

## P6 · Stability + Trade-offs + Recommendation (2.5 min)

### Slide 17 · Structural stability: 8 bootstrap refits

| Model | Mean \|Δp\| between refits | Top-20% Jaccard |
|---|---:|---:|
| Logistic | 0.032 | 0.77 |
| XGBoost | 0.035 | 0.75 |
| TabICLv2 | 0.035 | 0.78 |

- A Jaccard of about 0.77 means roughly 13–15% of the selected people change between refits (not 23%).
- Logistic is more stable than XGBoost on all 28 refit pairs.
- Sources: [structural_stability.png](../artifacts/figures/structural_stability.png), [stability_summary.csv](../artifacts/stability_summary.csv)

### Slide 18 · Stability for one person: abstaining is not fairness-neutral
- About 13% of decisions flip between refits, and about a third of the people selected sit at that margin.
- Keeping only the decisions all 8 refits agree on:
  - Precision rises from 0.822 to 0.846.
  - But the gender FNR gap widens from −0.090 to −0.119.
- Why: 47% of the selected women sit at the margin, against 30% of the selected men. Abstention therefore removes a larger share of women's places. This slide links directly back to P5.
- Sources: [individual_stability.png](../artifacts/figures/individual_stability.png), [individual_stability_summary.csv](../artifacts/individual_stability_summary.csv)

### Slide 19 · Trade-off matrix and recommendation
- Three models × four dimensions, with only the race and gender rows under fairness.
- **Recommendation: L1 logistic regression for a prospective shadow pilot (run alongside current practice without driving decisions), with XGBoost as the challenger.**
  - The two models select 85% of the same people (Jaccard 0.849); only 254 people differ. The difference in re-arrests captured has a CI that includes zero.
  - Logistic is more stable, directly interpretable and about 9× faster.
  - Calibration and fairness are tied, so neither is a reason to choose.
- What would reverse it: the client values calibration more than direct interpretability (for example, scores quoted numerically to supervisees), or operates at a scale where a few extra captured events matter.
- Limits, in one line: the evaluation set was inspected repeatedly during development, there is no temporal or external validation, and there is no evidence yet that the support programme helps.
- ⚠️ [tradeoff_matrix.png](../artifacts/figures/tradeoff_matrix.png) contains an "Age FNR gap" row. Regenerate it without that row.

## P1 · App demo (0.5–1 min)

### Slide 20 · Streamlit app
- Walk one person through the app:
  1. Their scores from all three models.
  2. The explanation of their score.
  3. How many of the 8 refits select them.
  4. Edit an input and watch the score change.
- Have screenshots or a screen recording ready as a backup.

---

## Appendix (not presented, for Q&A only)

- **A1 Learning curve:** [learning_curve.png](../artifacts/figures/learning_curve.png)
- **A2 Economic sensitivity:** [incumbent_effectiveness_sweep.csv](../artifacts/incumbent_effectiveness_sweep.csv)
- **A3 Explanation disagreement:** [explanation_agreement.png](../artifacts/figures/explanation_agreement.png)
- **A4 Global surrogate:** [global_surrogate.png](../artifacts/figures/global_surrogate.png)
- **A5 Group-threshold mitigation frontier:** [fairness_frontier.png](../artifacts/figures/fairness_frontier.png). It was optimised on the evaluation set, so it is optimistic.
- **A6 Age.** Age is a validated and legally accepted risk factor. Its FNR gap of about 0.24 mostly follows from different base rates (0.646 for under-33s, 0.509 for 33+). Whether to rank by risk or by need is a policy choice for the client. Fixing the age field alone does not remove the gap, because it runs through criminal-history inputs that are correlated with age. Sources: [fairness_age_bands.csv](../artifacts/fairness_age_bands.csv), [fpdp_age.png](../artifacts/figures/fpdp_age.png)
- **A7 Process log:** [improvement_journey.png](../artifacts/figures/improvement_journey.png)

## Open to-dos

- [x] EDA figure with race × gender instead of age: `artifacts/figures/eda_overview.png` (`scripts/deck_figures.py`).
- [x] Race-only and gender-only fairness figures for slides 13-14: `fairness_race_gaps.png`, `fairness_gender_gaps.png`.
      The original figures with the age column are kept for the notebook and the app.
- [x] Readable FPDP for slide 15 and appendix A6: `fpdp_gender_focus.png`, `fpdp_age_focus.png` (3 panels instead of 29).
- [x] Deck rebuilt to follow this outline: 20 core slides + 7 appendix, speaker notes with [P1]-[P6] owner and timing.
- [x] Age decision (team, 26 Sep): one line on slide 14 plus appendix A6; the age row stays in the trade-off matrix.
- [ ] Open the deck in PowerPoint and check every slide (it was checked with a layout preview, not rendered in PowerPoint).
- [ ] Record a backup video of the app demo.
- [ ] Everyone reads the Q&A section of [presentation_notes.md](presentation_notes.md).
