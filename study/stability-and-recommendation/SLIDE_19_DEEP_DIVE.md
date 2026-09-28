# Slide 19 Deep Dive: The Recommendation and Trade-off Matrix

## Overview
**Slide Title:** "Pilot logistic regression; run XGBoost as the challenger"  
**Duration:** 55 seconds  
**Core Question:** Given everything we've learned about performance, fairness, stability, and interpretability, which model should the client deploy first? And how should they validate it?

---

## Part 1: The Four-Dimensional Trade-off

### The Framework: More Than Just Accuracy

From the professor (Slide 16):
> Trade-off picture: Predictive performance on the x-axis, Interpretability on the y-axis, and Stability, Fairness, Frugality, Data privacy as further axes.

**Our four key dimensions** (aligned with the course, adapted to the client's needs):

1. **Performance** — AUC, Brier loss, net value
2. **Interpretability** — Can we explain each decision?
3. **Stability** — Do rankings and decisions survive retraining?
4. **Fairness** — Are subgroup error rates balanced?

*We also care about Reproducibility (runs the same on different machines) and Cost (runtime), which emerge as tie-breakers.*

---

## Part 2: Head-to-Head Model Comparison

### The Three Models: Logistic, XGBoost, TabICL

| Dimension | Logistic | XGBoost | TabICLv2 | Winner(s) |
|-----------|----------|---------|----------|-----------|
| **PERFORMANCE** | | | | |
| ROC AUC | 0.730 | 0.732 | 0.733 | TabICL (by 0.003) |
| Brier loss | 0.205 | 0.204 | 0.204 | XGBoost & TabICL |
| Net value @20% ($M) | 5.04 | 5.17 | 5.12 | XGBoost |
| **INTERPRETABILITY** | | | | |
| Local explanation | Native coefficients | SHAP + surrogate | None | Logistic |
| Global surrogate | Exact (R²=1.0) | Tree surrogate (R²=0.61) | PDP/ICE only | Logistic |
| **STABILITY** | | | | |
| Score drift (mean \|Δp\|) | **0.032** | 0.035 | 0.035 | Logistic (by 0.003) |
| Same people selected (Jaccard) | 0.773 | 0747 | **0.776** | TabICL |
| Rank correlation | **0.975** | 0.971 | 0.972 | Logistic |
| Wins on drift (28 pairs) | **28/28** | — | — | Logistic |
| **FAIRNESS** (Gender FNR gap) | | | | |
| Baseline | −0.090 | −0.096 | ? | Logistic (smaller gap) |
| Statistical difference | −0.013 (CI: −0.036 to +0.011) — **not significant** | Equivalent | — | — |
| **REPRODUCIBILITY** | | | | |
| Same predictions on different machines | **0 changes** | **80 changes** | — | Logistic |
| **COST** | | | | |
| Fit + predict (seconds) | **0.515** | 3.033 | — | **~9× faster** |

---

## Part 3: The Honest Story on Performance

### "XGBoost's Edge Is Real but Small"

**The Numbers:**
- Logistic: AUC 0.730, Brier 0.205, Net value $5.04M
- XGBoost: AUC 0.732, Brier 0.204, Net value $5.17M
- TabICL: AUC 0.733, Brier 0.204, Net value $5.12M

**Interpretation:**
- XGBoost wins Brier and net value, but margins are tiny (0.001 and $0.13M).
- AUC differences are 0.002–0.003, well within bootstrap confidence intervals (all models have ≈0.023 width).
- TabICL wins AUC but lags on Brier and net value.

**Why we say "real but small":**
- Real: XGBoost's edge is statistically consistent across metrics, not noise.
- Small: The practical difference in re-arrests captured at 20% capacity is negligible.

**What NOT to say:** "All three are equally good on performance." They're not; XGBoost leads. But the lead doesn't justify switching away from the other advantages of logistic.

---

## Part 4: Fairness: They're Equivalent Within the Noise

### The Gender FNR Gap (False Negative Rate Gap)

**Metric:** FNR_gap = FNR(Men) − FNR(Women)

Negative value means women are missed more often (worse for women).

| Model | Gender FNR Gap | 95% CI |
|-------|---|---|
| Logistic | −0.090 | — |
| XGBoost | −0.096 | — |
| Difference | −0.006 | [−0.036 to +0.011] |

**The Statistical Story (Professor's Slide 263):**
> Failing to reject the null hypothesis means we have not found evidence that the model is unfair. It does not *certify* that the model is fair.

Apply this differently here: the CI includes zero, so we cannot say one model is *more fair* than the other. They are **equivalent within ±5 percentage points** (using the TOST equivalence test from the course).

**Why we don't separate them on fairness:**
- Logistic's gap is −0.090; XGBoost's is −0.096—a 0.6 percentage point difference.
- The uncertainty around each model's gap is larger than the difference between them.
- Fairness is not the deciding factor.

**What about calibration?**
- No detectable difference. All three have ECE (expected calibration error) < 0.015, well-calibrated.

---

## Part 5: What DOES Separate Them: Stability, Reproducibility, Interpretability, Cost

### Dimension 1: Stability

**Advantage: Logistic**

From Slides 17–18:
- Logistic drifts less (0.032 vs 0.035 mean |Δp|).
- Logistic wins on all 28 pairwise comparisons for drift.
- TabICL is tied on Jaccard (which people get selected), but logistic is clearer on score stability.

**The phrase:** "Logistic is more stable than XGBoost, consistently, but the margins are small."

---

### Dimension 2: Reproducibility (Runs the Same Everywhere)

**Advantage: Logistic (by a landslide)**

**The Experiment:** Train XGBoost and logistic on the same data, run on a different machine, count how many decisions flip.

- Logistic: **0 decisions change**
- XGBoost: **80 decisions change** (out of 1,561 selected, ~5% flip)

**Why it happens (Professor's Slides 210–212):**
- XGBoost: Floating-point sums in a thread-dependent order; CPU count, batch size, and load change the numerics.
- Logistic: Simple linear algebra; same inputs → same outputs deterministically.

**Client implication:** If the agency runs the model on a laptop vs a server, they get the same answers with logistic. With XGBoost, 80 people might flip between machines monthly. That's unacceptable for a support program.

**Quotable:** "The real divide is a controlled environment versus an unobservable one" (Professor's Slide 215). Logistic is reproducible; XGBoost is not.

---

### Dimension 3: Direct Interpretability

**Advantage: Logistic**

**For the caseworker:**
- Logistic coefficients are directly usable: "Age decreases risk by 0.02 per year; gang affiliation increases it by 0.31."
- XGBoost requires SHAP (a post-hoc explainer) which can be unfaithful or unstable.
- TabICL has no native explanation at all.

**For the auditor:**
- Logistic: read the coefficients, check for proxy bias, done.
- XGBoost: run SHAP on a sample, check for consistency, still uncertain.
- The professor (Slides 41–42) warns: "Post-hoc explainers can be unfaithful, unstable and contradictory."

**Our evidence (from the deep review):**
- SHAP vs logistic coefficients: often disagree on feature importance ranking.
- SHAP was unstable across random seeds; logistic coefficients stable.

---

### Dimension 4: Cost (Speed)

**Advantage: Logistic**

**Benchmark (fit + predict):**
- Logistic: **0.515 seconds**
- XGBoost: 3.033 seconds
- TabICL: (not benchmarked, but known to be slower)

**Practical meaning:**
- Logistic: **~9× faster** than XGBoost
- For batch scoring 50,000 people: logistic ≈25 seconds, XGBoost ≈150 seconds

**Client implication:** Quarterly retrains are faster, dashboards more responsive, less server cost.

---

## Part 6: The Recommendation

### The Statement

> "Pilot logistic regression for 6 months, with XGBoost as the challenger."

### What This Means

**Primary arm (6 months):**
- Deploy L1 logistic regression to real cases.
- Score all people, decide via majority vote (cover 100%, ~0.822 precision).
- Human review for contested cases (about 13%), with fairness audits.
- Measure: number served, re-arrest outcomes, subgroup consistency.

**Challenger arm:**
- Maintain XGBoost in parallel, score all people, don't decide yet.
- Use its scores to train on newer data or retrain logistic.
- If XGBoost outperforms logistic by >2% AUC over 6 months, escalate to decision-makers.

**Secondary arm (optional):**
- Retrain logistic without gang affiliation as a fairness test.
- If gender FNR gap shrinks dramatically, consider it as the primary for month 7–12.

### Why Not XGBoost as Primary?

**Three reasons:**
1. **Reproducibility:** 80-decision swings between machines are unacceptable.
2. **Interpretability:** Post-hoc explainers are weaker than native coefficients.
3. **Cost:** 6× slower means larger infrastructure and training bills.

**Performance is not enough:** XGBoost wins on AUC and net value, but the margins are 0.002–$0.13M. That doesn't justify the operational complexity.

---

## Part 7: Reversibility Conditions (What Would Change the Recommendation)

### From Slide 19 "What Would Reverse It" Card

**Condition 1:** "Scores quoted numerically to supervisees, or a scale where a dozen extra captured re-arrests matter."

**Meaning:**
- If the client decides to show each person a "risk score" as a probability (e.g., "You are 65% likely to be re-arrested"), then AUC and calibration become critical.
- TabICL has the best AUC (0.733) and better calibration (ECE 0.021 vs logistic's 0.013—actually logistic is better, so this reverses).
- Actually: logistic has the best calibration; TabICL the best AUC. XGBoost in the middle.
- If quotation is the plan, reconsider the trade-off; logistic's simplicity might matter less.

**Condition 2:** "A dozen extra captured re-arrests matter."

**Meaning:**
- The client cares about re-arrest cases. At 20% coverage:
  - Logistic captures: 0.822 × 312 ≈ 256 re-arrests (net value $5.04M)
  - XGBoost captures: 0.828 × 326 ≈ 270 re-arrests (net value $5.17M)
  - Difference: ~14 additional re-arrests, $130k extra.
- If the client's board says "14 more prevented re-arrests is worth operational complexity," switch to XGBoost.

**Condition 3:** "Validation in the pilot shows logistic is biased or ineffective."

**Meaning:**
- Bias: Gender or race gap widens unexpectedly in production.
- Ineffectiveness: Support program doesn't actually reduce re-arrest (requires RCT or quasi-experimental evidence).
- In either case, fall back to XGBoost or TabICL.

---

## Part 8: No Model Decides Real Support Until Benefit Is Proven

### The Critical Caveat

From the script:
> "No model allocates real support until the pilot shows the programme helps."

**What this means:**
- The model selects people for human review.
- Humans decide whether to offer support.
- After 6 months, we need evidence (RCT or quasi-experimental) that the program actually *reduces re-arrest*.

**Why it matters:**
- Observational predictions ≠ intervention effects (professor's course, many slides).
- A model can predict re-arrest accurately but still fail to prevent it if the program is ineffective.
- The client must design the pilot to measure outcome, not just prediction.

---

## Part 9: The Secondary Arm: No-Gang Affiliation

### What Is It?

**Retrain logistic regression *without* the gang affiliation feature.**

**Results from Slide 18:**
- Gender FNR gap at margin: 33% vs 33% (symmetric, vs 47% vs 30% with gang).
- Overall gender FNR gap: likely shrinks (no direct evidence yet on test set).
- AUC: unknown, but probably drops (gang affiliation is predictive).

**Why consider it?**
- If the no-gang model maintains AUC > 0.725 and eliminates gender asymmetry, it could be the fairer choice.
- Requires a decision: Does the client care more about accuracy or fairness?

**Timing:**
- Run in parallel with the main pilot.
- If the gender gap shrinks and AUC holds, escalate to leadership for a policy decision (month 3–4).
- This is not a backup; it's a *fairness experiment*.

---

## Part 10: Interpreting the Trade-off Matrix Figure

### What the Figure Shows

The slide displays a 4×3 colored grid (rows = dimensions, columns = models):

**Color coding (typical):**
- Green = best in dimension
- Yellow = middle
- Red = worst in dimension

**Layout:**
- Rows (top to bottom): Performance, Interpretability, Stability, Fairness, Cost
- Columns (left to right): Logistic, XGBoost, TabICL

**Key insight:** Logistic dominates **Interpretability, Stability (drift), Reproducibility, and Cost**. XGBoost wins **Performance (Brier/net value)**. TabICL wins **AUC**. No single model wins all dimensions.

---

## Part 11: The Script (55 seconds)

### Current Script
> So, across the four dimensions. Performance: XGBoost's edge is real but small — the two models select 85% of the same people, and the difference in re-arrests captured includes zero. Fairness doesn't separate them: the gap difference is equivalent within five points. Calibration shows no detectable difference. What does separate them is stability, reproducibility, direct interpretability and cost — and all four favour logistic. So we recommend a shadow pilot of L1 logistic regression, with XGBoost as the challenger, plus a second arm without gang affiliation, where logistic is still the more stable model. The choice reverses if scores are quoted to people as probabilities, or at a scale where a dozen extra captured re-arrests matter. And no model allocates real support until the pilot shows the programme helps. Over to the demo.

### Analysis & Notes

**Strengths:**
✅ Opens with "across the four dimensions" (sets up the trade-off framing)  
✅ Performance: clearly states XGBoost wins but it's small (0.85 overlap, CI includes 0)  
✅ Fairness: uses "equivalent within five points" (TOST language from the course)  
✅ Names all four separating dimensions  
✅ Gives the recommendation: shadow pilot logistic, XGBoost challenger  
✅ Mentions the no-gang arm  
✅ States reversibility conditions  
✅ Critical: "no model allocates real support until pilot shows programme helps"  
✅ Clear closing transition to demo  

**Timing check:**
- ~200 words ÷ 150 wpm = 80 seconds
- We have 55 seconds allocated
- This script is **overlong by ~25 seconds**

**Options:**
1. Trim: Remove "at a scale where a dozen extra captured re-arrests matter" to save 10 words (~4 seconds).
2. Trim: Shorten "the gap difference is equivalent within five points" → "the gap difference is within five points" (saves 2 words, minimal).
3. Accelerate: Deliver at 170–180 wpm (natural for practiced material, especially with enthusiasm).
4. Accept: This is the 3-minute total slot; individual slides can vary. Slide 17 is 45 s, Slide 18 is 45 s, so Slide 19 gets ~55 s. Tight, but doable.

**My recommendation:** Keep the script as is but rehearse and time it. When presenting, prioritize clarity over speed; the jury will appreciate precise language. If it runs 65 seconds, that's within the 14:55 budget for the whole talk.

---

## Part 12: Course Connections

### Professor's Slide 16: The Trade-off Picture
✅ We present trade-offs across four dimensions, exactly as the professor frames them.

### Professor's Slide 263 & TOST Equivalence
✅ We use "equivalent within ±5 points" for the fairness comparison, applying the professor's bioequivalence test.

### Professor's Slides 210–215: Reproducibility & The Controlled Environment
✅ We use the 0-vs-80 machine test as evidence of reproducibility, directly connecting to the professor's warning about floating-point arithmetic and controlled environments.

### Professor's Slide 61: Why Logistic Regression?
> "Logistic regression suits cases where effects are roughly linear and **interpretability and inference on coefficients are important**."

✅ This is our exact use case: coefficient interpretation is essential for caseworker explanations and auditor reviews.

---

## Part 13: Common Mistakes to Avoid

1. **"Logistic is the best model"** ❌ → More precise: "Logistic is the best *fit* for this use case given the trade-offs" ✅

2. **"XGBoost's performance advantage justifies using it"** ❌ → The advantage is small (0.002 AUC, $0.13M) relative to reproducibility and interpretability costs ✅

3. **"TabICL has the best AUC, so it's superior"** ❌ → AUC is one dimension; consider the full trade-off, including the lack of native explanations ✅

4. **"We're deploying logistic regression"** ❌ → More precise: "We're *piloting* logistic in a shadow test; XGBoost is the challenger" ✅

5. **"No model is risky until the trial proves benefit"** ❌ → We also check fairness in production; outcome measurement is necessary but not sufficient ✅

6. **"Reproducibility doesn't matter for this use case"** ❌ → It does. 80 decision flips between machines undermine client trust ✅

---

## Part 14: Q&A Talking Points

### "Why not just pick the highest-AUC model?"

AUC is one metric, important but not decisive. Logistic's coefficient stability, reproducibility, and interpretability outweigh TabICL's 0.3% AUC edge. The professor (Slide 61) says exactly this: "No universally best model; logistic suits interpretability."

### "How do we know logistic won't fail in production?"

That's why we pilot. The shadow test runs for 6 months; we measure:
- Whether human decisions (on contested cases) are consistent.
- Whether the program reduces re-arrest (separate RCT/quasi-experiment needed).
- Whether fairness gaps widen or stay stable.
- If any fails, we revert.

### "What if XGBoost beats logistic in the 6-month test?"

Then escalate to leadership. The recommendation holds *if* pre-pilot data is representative. If production reveals XGBoost is systematically better, it becomes the new primary.

### "Why measure reproducibility on different machines?"

Because the client might run the model on laptops (caseworkers), servers (batch scoring), or cloud (audits). A model that gives different answers on different hardware is not production-ready. Logistic's determinism is a non-negotiable requirement.

### "The no-gang arm sounds like we're hiding a bias issue."

Opposite: we're testing whether gang affiliation *is* the root cause of the gender asymmetry. If logistic without gang has symmetric margins *and* reasonable AUC, it's a legitimate fairness improvement, not a cover-up. Either way, the jury sees the evidence.

---

## Part 15: Key Takeaways to Memorize

1. **The trade-off:** No model wins all four dimensions. Logistic dominates interpretability, stability, reproducibility, and cost. XGBoost wins performance (slightly).

2. **The recommendation:** Pilot logistic for 6 months; use XGBoost as the challenger.

3. **The reversibility:** Recommendation flips if (a) scores are quoted to people, (b) the client values 14 extra prevented re-arrests highly, or (c) production evidence overwhelms the pilot data.

4. **The critical caveat:** No real support is allocated until the pilot proves the program works (outcome measurement required).

5. **The secondary experiment:** No-gang arm tests whether gang affiliation drives the gender asymmetry.

---

## Part 16: Supporting Files

From the codebase:

- **Trade-off matrix:** `artifacts/tradeoff_matrix.md` (the full 26-row detailed comparison)
- **Model comparison:** `artifacts/ml_model_comparison.csv` (performance metrics and runtime)
- **Fairness by group:** `artifacts/fairness_by_group.csv` (gender and race gaps by model)
- **Model-to-model overlap:** `artifacts/selected_set_overlap.csv` (Jaccard 0.85 between logistic & XGBoost)
- **Reproducibility:** `artifacts/deep_review/thread_sensitivity.json` (machine-to-machine changes)
- **Figure:** `artifacts/figures/tradeoff_matrix.png` (the four-panel visual)

Run the figure-generation script:
```bash
python scripts/deck_figures.py  # generates tradeoff_matrix.png
```

---

## Part 17: The Presentation Flow

### How Slides 17–19 Build

1. **Slide 17:** "Is the model stable?" → Yes, logistic is more stable.
2. **Slide 18:** "Does stability help fairness?" → No, abstention widens the gender gap.
3. **Slide 19:** "So which model should we deploy?" → Logistic, because of stability + reproducibility + interpretability + cost, despite XGBoost's tiny performance edge.

**Jury logic:**
- Stability and fairness matter (Slides 17–18).
- Reproducibility and interpretability matter (Slide 19 subtext).
- Performance margins are too small to override them (Slide 19 explicit).
- Therefore, logistic is the right pilot choice.

---

## Part 18: References

- **Professor's Slide 16:** Trade-off picture (performance × interpretability × stability × fairness)
- **Professor's Slide 61:** When logistic is appropriate
- **Professor's Slide 263:** TOST equivalence and fairness testing
- **Professor's Slides 210–215:** Reproducibility and the controlled environment
- **Our Concepts:** [01_stability_concepts_and_maths.md](01_stability_concepts_and_maths.md)
- **Deep Dive - Stability:** [SLIDE_17_DEEP_DIVE.md](SLIDE_17_DEEP_DIVE.md)
- **Deep Dive - Fairness:** [SLIDE_18_DEEP_DIVE.md](SLIDE_18_DEEP_DIVE.md)
- **Presentation Notes:** [10_p6_slides_and_script_3min.md](10_p6_slides_and_script_3min.md) (the 55-second script)

