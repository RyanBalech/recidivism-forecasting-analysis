# Technical report and model card

*Client: a software vendor that sells risk-assessment tools to US state community-supervision
agencies. Engagement: choose and justify a three-year re-arrest scoring model to embed in the
vendor's product, evaluated as a trustworthy AI system rather than on accuracy alone.*

## Executive recommendation

**Any of our three models is a large upgrade over the tool agencies use today.** The incumbent
actuarial score (`Supervision_Risk_Score_First`, a 1–10 scale already in the data) reaches only
**0.60 ROC AUC**; all three candidate models reach **~0.73** and roughly double the net value of a
capacity-limited support programme (about $2.75M → $5.1–5.3M under our scenario). That gap, not the
choice between our three models, is the finding that matters to the client.

Among the three, **deploy XGBoost as the production model, with logistic regression as the
transparent challenger.** XGBoost has the best calibration (ECE 0.0109), the highest scenario net
value ($5.27M), the smallest gender false-positive gap, and runs ~10× faster than TabICLv2. TabICLv2
ties on discrimination (AUC 0.7328 vs 0.7326) but has no native explanation path, slightly worse
calibration, and slightly larger subgroup gaps — so its costs outweigh a statistically-indistinguishable
accuracy for a government-facing tool. For very small agencies (< ~5,000 records) TabICLv2's few-shot
advantage on scarce data is real and can justify it there (see the learning curve).

This is a pilot recommendation, not authorization for operational use. Any deployment requires a
prospective study, an appeal route, drift monitoring, and quarterly subgroup audits.

## Decision and population

- **Intended decision:** prioritize scarce, beneficial support at the start of parole supervision.
  Never sanctions, detention, or surveillance.
- **Population:** people released from Georgia prisons to parole, 2013–2015.
- **Target:** any new arrest within three years (base rate 57.8%; balanced, so no resampling).
- **Train/test:** NIJ's official split, 18,028 training / 7,807 held-out.
- **Operating point:** the product allocates support to the **top 20% by risk**, so fairness is
  audited at that capacity threshold as well as at 0.5.

The target is an arrest, not latent offending: it reflects policing, reporting, and legal processes
as well as behaviour.

## Data preparation and leakage control

29 baseline fields available at supervision start are modeled. Race, gender, and Residence PUMA (a
race proxy) are excluded from inputs and kept only for auditing. Post-release variables (violations,
drug tests, employment, program attendance, residence changes) are excluded because they accrue
*after* the scoring moment.

Ordered count fields ("3 or more", age bands, prison-tenure bands) are ordinal-encoded so trees can
split on rank; remaining categoricals are one-hot encoded, numerics median-imputed with a
missingness flag and standardized. Logistic and XGBoost share this matrix; TabICLv2 receives the raw
mixed-type frame and applies its own encoding.

**A subtle leak we found and fixed.** `Gang_Affiliated` is missing for exactly the 2,217 women and no
men in NIJ. TabICLv2 encodes NaN as its own category, so an early version could reconstruct the
excluded Gender attribute through that missingness — a leak that inflated TabICL's apparent gender
false-positive gap (0.27 at threshold 0.5) and lent it ~0.001 AUC. We now fill categorical NaN with
the training mode before TabICL (matching the other two pipelines), with a regression test. No
numeric column's missingness is gender-aligned, so the fix is complete. This is exactly the kind of
representation-level leak a trustworthy-AI review must catch.

## Models

- **Logistic regression** — L2, `C=0.25`. Transparent linear baseline.
- **XGBoost** — hyperparameters from a 5-fold cross-validated random search on the training set only
  (`scripts/tune_xgboost.py`, CV AUC 0.7343); ordinal-encoded counts.
- **TabICLv2** — pretrained in-context tabular transformer, 16 ensemble members, run on GPU (RTX 4060).

## Predictive performance

| Model | ROC AUC | Avg precision | Brier ↓ | ECE ↓ | Net value @20% | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| **Incumbent score** | 0.600 | — | — | — | $2.75M | — |
| Logistic regression | 0.7295 | 0.7691 | 0.2055 | 0.0132 | $5.02M | 0.8 s |
| XGBoost | 0.7326 | **0.7722** | **0.2044** | **0.0109** | **$5.27M** | 2.4 s |
| TabICLv2 | **0.7328** | 0.7722 | **0.2044** | 0.0199 | $5.13M | 23.1 s |

TabICLv2 and XGBoost are a **statistical tie** (AUC 0.7328 vs 0.7326, identical Brier); bootstrap 95%
intervals overlap heavily (e.g. XGBoost AUC [0.7205, 0.7437]). The ranking among the three is not
decisive, which is exactly why the recommendation turns on interpretability, fairness, stability, and
cost. Full intervals in `artifacts/bootstrap_intervals.json`.

**Learning curve** (`scripts/learning_curve.py`, train sizes 1,500 / 5,000 / 10,000 with 3 seeds
each, plus one run at the full 18,028): at 1,500 rows TabICLv2 leads (0.720 vs XGBoost 0.712 and
logistic 0.703); the gap closes to a tie at full data. This is why model choice depends on an agency's
data volume. (The full-data point is a single seed, so its variance band is not reported.)

**Economic scenario.** At 20% capacity under illustrative assumptions ($5,000 support cost, $50,000
event cost, 20% effectiveness), net value is $5.0–5.3M for the models vs $2.75M for the incumbent and
$1.46M for random allocation. The app exposes every assumption and a sensitivity sweep; the models
dominate the incumbent at all capacities from 5% to 50%, so the conclusion does not rest on one
assumption. These are scenario outputs, not causal estimates.

## Interpretability

- **SHAP** (`scripts/interpretability.py`): global mean-|SHAP| and individual waterfalls for logistic
  and XGBoost. Recurring drivers are age at release, gang affiliation, prior felony arrests, prison
  tenure. **LIME** on the same individual gives a consistent local story via a different mechanism.
- **XPER** (Hué, Hurlin, Pérignon, Saurin — `scripts/xper_attribution.py`) decomposes the model's
  **AUC** into feature contributions. Age at release is the top performance driver (~0.088 AUC),
  then prior felony arrests. This complements SHAP, which splits predictions, not performance.
- **Global surrogate**: a depth-3 tree mimics XGBoost with test fidelity R² = 0.61 — enough to
  narrate the main logic, not enough to replace the model.
- **TabICLv2 has no native attribution path**, and KernelSHAP over 7,807 rows is impractical on CPU.
  It is covered only model-agnostically (PDP/ICE, permutation). **This is a deployment cost**: a
  caseworker cannot be told why the foundation model scored a person — a real strike against it for a
  government-facing tool.

## Stability

`scripts/stability_structural.py` refits each model on 8 bootstrap resamples of the training data and
measures, on the fixed test set, distance between refits, decision overlap, and drift in feature
contributions. All three are comparably stable (score drift ~0.034–0.036, top-20% decision overlap
~76–77%). Across refits about **one person in four changes priority status** for every model, so
scores need governance and monitoring regardless of model. Event dates are unavailable, so temporal
stability must be evaluated on a later cohort before launch.

## Fairness

Protected attributes: Race (Black 58%, White 42%) and Gender (M 88%, F 12%). Base recidivism rates:
Black 0.582 / White 0.564; **Male 0.591 / Female 0.454**. Gaps reported at the **deployed top-20%
operating point** with bootstrap 95% CIs (`scripts/fairness_audit.py`):

| Model | Race FPR gap | Gender FPR gap |
|---|---:|---:|
| Logistic | **0.017** (n.s.) | 0.040 |
| XGBoost | 0.020 | **0.036** |
| TabICLv2 | 0.024 | 0.041 |

Auditing at the deployed point matters: at threshold 0.5 the same gaps are ~2–3× larger (TabICLv2
gender 0.106 at 0.5 vs 0.041 at top-20%), so a 0.5 audit would describe an operating point we never
deploy. (Before the missingness leak was fixed, TabICL's gender gap read 0.27 at 0.5 — a further
reason the leak mattered.)

**Impossibility result, split by attribute — the key fairness insight.** Race base rates barely
differ (0.582 vs 0.564), so calibration and equal error rates are near-jointly achievable: the
observed race FPR gap is a *model property*, and group-specific thresholds drive it to ~0 while
keeping essentially all captured events. Gender base rates differ by **13.7 points** (0.591 vs
0.454), so the theorem binds: because gender is excluded from every model, **all three over-predict
women** (mean score ~0.52 vs actual 0.45). Group thresholds can still equalize the gender FPR gap
(→ ~0.00 for every model, keeping ~1,285 of ~1,295 events), **but doing so decalibrates women** —
selecting them at a rate inconsistent with their lower actual recidivism. That is the trade-off the
theorem forces; we surface it rather than hide it.

**Mitigation caveat (legal).** Applying a different decision threshold by race or gender is disparate
treatment (cf. *Ricci v. DeStefano*) and contradicts excluding the attribute from inputs. We present
group thresholds only as an **analytic device** to trace the fairness/utility frontier, alongside the
group-blind single threshold that a real deployment would use.

**Removing race is free and fairer** (`scripts/race_ab_test.py`). Adding race as an input changes AUC
by ≤ 0.0007 for every model but makes two otherwise-identical people receive different scores (up to
4.8 points for XGBoost). Excluding it guarantees identical "twins". It does **not** guarantee equal
group rates: TabICLv2's race FPR gap slightly rises when race is removed, evidence of proxy leakage —
the classic limit of fairness-through-unawareness.

## Deployment controls

1. Prospective shadow-mode evaluation on a later cohort before any live use.
2. Score used only to expand access to voluntary support.
3. Plain-language documentation and an appeal/correction route.
4. Log data quality, predictions, allocations, overrides, outcomes.
5. Review calibration, drift, and subgroup gaps quarterly at the deployed operating point; suspend on
   breach.
6. Estimate causal programme benefit via a randomized or strong quasi-experimental design.

## Limitations

Historical data from one state, de-identified by aggregation. Arrest is an institutionally mediated
outcome, not latent offending. The foundation-model ensemble was limited by local hardware. The
economic analysis is illustrative. None of the three models is validated for high-stakes adverse
decisions, and this tool must never drive them.
