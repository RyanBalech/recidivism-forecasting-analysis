# 4 · The recommendation (slide 19)

> The brief asks for "a clear recommendation to your client about which model should be
> deployed in production and why", reflecting "the requirements of a trustworthy AI system
> rather than predictive performance alone". This slide is where the whole talk lands.

---

## 4.0 The professor's frame: slide 16

The course opens (slide 16) with one picture: **predictive performance** on the x-axis,
**interpretability** on the y-axis, and **stability, fairness, frugality and data
privacy** as further axes. The brief asks us to place the three models on that picture
and pick one "reflecting the requirements of a trustworthy AI system rather than
predictive performance alone".

| Axis (slide 16) | Logistic | XGBoost | TabICLv2 |
|---|---|---|---|
| Predictive performance | ≈ (−0.0025 AUC) | ≈ best | ≈ best |
| Interpretability | **direct coefficients** | SHAP + surrogate (R² 0.61) | none native |
| Stability | **most stable decisions; slide 186 applies** | least stable | stable selection |
| Fairness | no detectable difference between models; equivalent within ±5 pts | same | same |
| Frugality | **1 s, CPU** | 9 s, CPU | 42 s, GPU |
| Data privacy | same inputs for all; the slide-186 anchor retrains without keeping old records | same inputs | same inputs |

**Say:** "On the course's own picture we move a fraction of a point left on predictive
performance and gain on interpretability, stability and frugality."

## 4.1 The recommendation in one sentence

**Pilot L1 logistic regression prospectively in shadow mode, with XGBoost running as the
challenger on the same cohort. Neither model allocates real services until the pilot
passes pre-agreed gates.**

- **Shadow pilot:** the model scores every new supervisee, but decisions still follow
  current practice. We compare what the model *would* have done against what happened.
- **Challenger:** XGBoost scores the same people in parallel, so if it proves materially
  better on new data, switching is cheap.

## 4.2 The argument, in five steps

1. **Performance is a near-tie.**
   - AUC is 0.730 / 0.732 / 0.733.
   - XGBoost beats logistic by +0.0025 AUC (95% CI 0.0006 to 0.0044) and by −0.0010 Brier
     (CI 0.0003 to 0.0015). That's real, but small.
   - All three beat the current Georgia score (AUC 0.60) by a wide margin.
2. **The accuracy edge doesn't reach the decision.**
   - At the top-20% rule, logistic and XGBoost select 85% of the same people (Jaccard 0.849).
   - Only 254 of 7,807 people (3.25%) are chosen by one model and not the other.
   - The difference in re-arrests captured is ≈ 14, with a CI that includes 0.
3. **Stability favours logistic.**
   - Lower score drift on all 28 refit pairs (0.032 vs 0.035).
   - A more stable selected set on 26 of 28 pairs (Jaccard 0.77 vs 0.75).
   - It stays ahead under the course's harsher disjoint-halves test (see `03_*`).
4. **Interpretability favours logistic.**
   - Its coefficients are read directly, with no second tool.
   - XGBoost needs SHAP, and its depth-3 surrogate tree reproduces only R² = 0.61.
   - TabICL has no native attribution at all.
5. **Calibration and fairness don't separate the models, so they are *not* reasons.**
   Word this carefully. Slide 263: failing to reject a difference "does not certify"
   equality.
   - **Fairness.** The logistic-minus-XGBoost difference in the gender FNR gap is −0.013,
     95% CI [−0.036, +0.011]. That interval lies entirely inside the team's ±5-point
     tolerance, so the two are **equivalent within ±5 points (TOST, slide 272)**, not just
     "not different". No race test survives Holm correction for either model.
   - **Calibration.** Both are indistinguishable from perfect calibration (Cox slopes 0.999
     and 1.005). The paired difference in |slope − 1| is within ±0.015. Call it "no
     detectable difference", since no tolerance was set in advance.
   - The gender disparity comes from the *feature set* (gang-affiliation missingness), not
     from the model choice.

**Plus cost:** logistic is ≈ 9× faster than XGBoost (1.0 s vs 8.7 s) and ≈ 40× faster than
TabICL (41.8 s, which needs a GPU).

## 4.3 Reading the trade-off matrix row by row

Source: `artifacts/tradeoff_matrix.md` and `artifacts/figures/tradeoff_matrix.png`.

| Dimension | Row | Logistic | XGBoost | TabICLv2 | Verdict |
|---|---|---|---|---|---|
| Performance | AUC | 0.730 | 0.732 | 0.733 | XGBoost +0.0025 (significant, small) |
| | Brier | 0.205 | 0.204 | 0.204 | XGBoost −0.0010 (significant, small) |
| | Net value @20% | $5.04M | $5.17M | $5.12M | Within noise; the ranking changes with capacity (4.5) |
| Interpretability | Local explanation | coefficients | SHAP + surrogate | none native | **Logistic** |
| | Surrogate fidelity | exact | R² = 0.61 | PDP/ICE only | **Logistic** |
| Stability | Score drift | **0.032** | 0.035 | 0.035 | **Logistic** |
| | Top-20% overlap | 77% | 75% | **78%** | Logistic ≈ TabICL > XGBoost |
| Fairness | Race FNR gap | −0.005 | −0.017 | −0.023 | All within ±5 pts (TOST) |
| | Gender FNR gap | −0.096 | −0.112 | −0.124 | Shared by all; logistic vs XGBoost equivalent within ±5 pts |
| | Age FNR gap | −0.245 | −0.228 | −0.237 | Shared; a policy choice (appendix A6) |
| Cost | Train + predict | 1.0 s | 8.7 s | 41.8 s | **Logistic** |
| | Auditability | high | medium | low | **Logistic** |

**The course's own evidence for this choice:**
- **Slide 61:** "there is no universally best model". Logistic regression suits cases where
  "interpretability and inference on coefficients are important".
- **Slide 52 (guest lecture):** across credit-scoring datasets, a well-regularised logistic
  regression matches **TabPFN** (a tabular foundation model) on AUC and beats it on
  calibration and sparsity. Our result is the same: 0.730 vs 0.733 AUC, and TabICL is
  the worse calibrated.
- **Slide 42:** post-hoc explainers can be "unfaithful, unstable and contradictory".
  Logistic doesn't need one.

**Why not TabICL?**
- It has the best point AUC, but its probabilities are measurably too extreme
  (calibration slope 0.913, z = 3.76).
- It has no native explanation, and it needs a GPU (42 s per run).
- It *does* win when data are scarce: at 1,500 training rows its AUC is 0.720 against
  0.712 for the others (`artifacts/learning_curve.csv`). Mention this for small agencies.

## 4.4 What would reverse the recommendation

Say these out loud. It shows the choice was reasoned, not assumed.

1. **The client quotes scores as probabilities** to supervisees or judges, rather than
   only ranking people. Then calibration and Brier matter more than interpretability,
   and XGBoost's small Brier edge counts.
2. **Scale.** For a client serving hundreds of thousands of people, ≈ 12 extra
   re-arrests captured per 1,561 offers could become material.
3. **A new feature set** gives the non-linear model a materially bigger margin.
4. **A small agency** with little training data: TabICL is the strongest there (learning
   curve).

The challenger design keeps these options open after the pilot.

## 4.5 Economics: extra material, not in the deck

Assumptions (from the brief's economic dimension, all editable in the app): support
costs $5,000 per person, a re-arrest costs $50,000, and support prevents 20% of
re-arrests among those helped.

**Net value = captured re-arrests × $50,000 × 20% − people selected × $5,000**

Logistic: 1,285 × $10,000 − 1,561 × $5,000 = **$5.045M**.

### Break-even effectiveness: new, and a strong line

How effective must the support be for the programme to pay for itself?
$$e^* = \frac{\text{selected} \times \$5{,}000}{\text{captured} \times \$50{,}000}$$

| Ranker | Captured | Break-even effectiveness | Cost per re-arrest reached |
|---|---:|---:|---:|
| Random | 926 | 16.9% | $8,428 |
| Current Georgia score | 1,053 | 14.8% | $7,412 |
| **Logistic** | 1,285 | **12.1%** | $6,073 |
| XGBoost | 1,297 | 12.0% | $6,017 |
| TabICLv2 | 1,292 | 12.1% | $6,041 |

**Say:** "With our model, support pays for itself if it prevents at least 12% of
re-arrests among the people helped. With the current tool it would need 15%. We don't
know the real effectiveness, and that is exactly what the pilot must measure." At 10%
effectiveness *every* ranker loses money (`incumbent_effectiveness_sweep.csv`).

### No model wins on value at every capacity: new

Source: `artifacts/incumbent_capacity_sweep.csv`, net value in $M.

| Capacity | 5% | 10% | 15% | 20% | 25% | 30% | 35% | 40% | 50% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic | **1.57** | 2.86 | 3.99 | 5.05 | **6.17** | **7.01** | **7.75** | 8.22 | 9.19 |
| XGBoost | 1.55 | **2.93** | 4.06 | **5.17** | 6.04 | 7.00 | 7.63 | **8.39** | **9.32** |
| TabICLv2 | 1.53 | **2.93** | **4.09** | 5.12 | 6.03 | 6.96 | 7.63 | 8.36 | 9.19 |
| Current score | 0.74 | 1.46 | 2.11 | 2.73 | 3.27 | 3.96 | 4.37 | 4.91 | 5.72 |

**Say:** "Which model is most valuable depends on the capacity the client chooses. The
differences between our models are at most about $0.17M. The gap to the current tool is
$2–3.5M at every capacity. Model choice is second-order; replacing the current tool is
first-order."

## 4.6 Linking stability to the recommendation

- **Retraining policy, using slide 186.** A naive retrain on new data swaps ≈ 19% of the
  selected people. Retraining with the slide-186 stability constraint (λ by CV) cuts that
  to ≈ 12%, halves the coefficient movement, and costs no accuracy (03, C2). This is only
  possible because logistic has a parameter vector θ; XGBoost would need slide 194's
  custom objective.
- **Retrain on a fixed schedule**, and don't withdraw support already offered just because
  a retrain moved someone below the cut.
- **Reproducibility.** Pin package versions, threads and model files (slides 208, 222). On
  another machine, logistic's decisions are identical and XGBoost's change for 80 people
  (03, E6).
- **Contested cases.** Show the refit vote count per person (the app already does:
  "selected by 6 of 8 refits"). Caseworkers then see which decisions are borderline.
- **Abstention.** Don't adopt abstention by default. If the client wants it, audit the
  human-review queue by gender, because abstention widens the gender gap.

## 4.7 Deployment gates before any real allocation

From the report and the app's governance tab:

1. Verify that every input really exists at supervision start (feature timing).
2. Validate on a new cohort, i.e. a temporal or external dataset.
3. Measure whether support actually helps, with a randomised or quasi-experimental design.
4. Provide an appeals and correction route.
5. Monitor calibration and subgroup allocation every quarter at the operating point.
6. Agree stop rules *before* seeing the results.
7. Scope: support only. Never sanctions, detention or surveillance.

## 4.8 What we do **not** claim

- ❌ "Logistic is fairer": not significant. The two are *equivalent within ±5 points*.
- ❌ "Logistic is better calibrated": no detectable difference.
- ❌ "The two are equally fair because the difference isn't significant": slide 263. Say
  "equivalent within ±5 points (TOST)".
- ❌ "We beat today's agency tools": we beat a *historical* score in this dataset.
- ❌ "The dollars are savings": the effectiveness is assumed.
- ❌ "The evaluation set is untouched": it was inspected during development.
