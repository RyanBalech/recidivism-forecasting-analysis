# Slide 17 Deep Dive: Structural Stability Study Guide

## Overview
**Slide Title:** "Would a different sample give the same model?"  
**Duration:** 45 seconds  
**Core Question:** If we retrain the model on a different random sample from the same population, do we get the same rankings and recommendations?

---

## Part 1: The Course Definition

### From Professor's Slide 182: What is Stability?

> "If we obtain two datasets from the same population (same underlying probability distribution), then the ML algorithm should induce approximately the same model from both datasets."
> — Turney (1995)

**What this means for you:**
- Two different samples from the same population should lead to approximately the same decisions.
- The model shouldn't be overly sensitive to which specific people happen to be in the training data.

**Real-world story (Professor's Slide 183-184):**
A doctor gives patient data to an analyst → returns a decision tree.
More patient data arrives → analyst retrains → returns a *different* tree.
Doctor asks: "?!"

**Our version:**
The supervision agency retrains every quarter. About **13% of people lose their spot on the support list** just because the data changed, even though nothing about them changed.

---

## Part 2: The Three Tests (Mildest to Harshest)

### Why Three Tests?

We want to separate different sources of randomness:

| Test | What Varies | What's Fixed | Harshness |
|------|------------|--------------|-----------|
| **Seed** | Algorithm's random seed | Same data, same size | Mildest |
| **Bootstrap** | Sample composition (with replacement) | Data size stays ~full (63%), seed fixed | Moderate |
| **Disjoint Halves** | Completely separate halves | No overlap, each is 50% size | Harshest |

---

## Part 3: The Methodology

### Test 1: Seed-Only (Baseline Randomness)

**What we did:**
- Trained each model 8 times on the *same* training data
- Changed only the random seed (42, 43, 44, … 49)
- Evaluated on the same 7,807 test people
- Measured how much predictions varied

**What it tells us:** The model's inherent algorithmic randomness

**Our results:**
- XGBoost's random seed alone changes which 2% of people it selects (not in the main table)
- This is a floor: the real variation is larger

---

### Test 2: Bootstrap Refits (Same Size, Different Content)

**What we did:**
- Created 8 bootstrap samples from the training set of 18,028 people
- "Bootstrap" = draw with replacement, so about 63% of people appear, others are duplicates
- Train each model on one of the 8 samples (with seed fixed at 42 across all three models)
- Evaluate all 8 models on the same test set

**Why bootstrap matters:**
- It's mild: each sample is nearly full-sized and overlaps heavily with others
- But it's realistic: this is what happens quarterly when the agency gets new releases

**What the professor says (Slide 195):**
> Treat the score as a draw from a distribution, not a fixed measurement.

**Our results from Table on Slide 17:**

| Model | Mean |Δp| (drift) | Jaccard (same top-20%) | # Pairs (of 28) where logistic wins |
|-------|-----|----------|-------------|---|
| Logistic | **0.032** | **0.77** | — |
| XGBoost | 0.035 | 0.75 | 28/28 |
| TabICLv2 | 0.035 | 0.78 | 26/28 |

**What these numbers mean:**

- **Mean |Δp| = 0.032 (logistic):** On average, a person's predicted risk shifts by 3.2 percentage points between bootstrap refits. That's the smallest.

- **Jaccard = 0.77 (logistic):** If we take the top 20% (1,561 people) from one refit and the top 20% from another:
  - 77% of them are the same people
  - **NOT** "23% changed" — instead, it's ≈ 13% swapped out
  - Formula: `replaced = (1 − J) / (1 + J) = (1 − 0.77) / (1 + 0.77) ≈ 12.8%`

- **28/28 pairs where logistic wins:** Across all C(8,2) = 28 pairwise comparisons, logistic had smaller drift than XGBoost in every single one.

---

### Test 3: Disjoint Halves (Harshest Test)

**What we did:**
- Split the 18,028 training people into two completely separate halves (9,014 each)
- Train each model on one half only
- Evaluate on the same 7,807 test people
- Compare predictions between the two models

**Why this test is harsh:**
- Each model sees only 50% of the data
- Zero overlap between the two halves (unlike bootstrap's 40% overlap)
- This is the literal interpretation of "two datasets from the same population"

**Our results:**
- Logistic shows about 18% turnover (compared to 13% on bootstrap)
- XGBoost and TabICL similar
- The pattern holds: logistic remains the most stable

---

## Part 4: Interpreting the Figure (Three Panels)

### Left Panel: Distance Among Refits
**Title:** "Distance among refits (mean |Δp|, lower = stabler)"

- Shows a box plot for each model across 8 refits
- The box shows the spread; the line in the middle is the median
- Logistic's median is lowest (~0.030)
- All three models have overlapping uncertainty bands

**Key insight:** Logistic drifts least, but the difference is small.

---

### Middle Panel: Same People Selected?
**Title:** "Same people selected? (top-20% Jaccard)"

- Box plot showing Jaccard scores across 28 pairs
- Logistic's median Jaccard ≈ 0.77
- XGBoost lower at 0.75
- TabICL median ~0.78

**Key insight:** Small differences, but consistent. Logistic's selections are the most stable.

---

### Right Panel: Score Rank Correlation
**Title:** "Score rank correlation across refits"

- Shows Spearman ρ (rank correlation)
- All three models highly correlated: 0.97–0.98
- This means the *rankings* are similar, even if the exact scores differ

**Key insight:** Rankings are more stable than absolute scores.

---

## Part 5: The Script (45 seconds)

> **The course defines stability simply:** two datasets from the same population should give approximately the same model. **We tested that three ways, from mildest to harshest.**
>
> Changing only the random seed barely moves logistic regression; XGBoost already swaps about 2% of the people it selects. 
>
> **With eight bootstrap refits** — the same eight for all three models — a person's risk moves by about three points, and about **13% of the people offered support change.** Not 23%: Jaccard is not the share that changed. 
>
> With two completely separate halves of the data, it's about 18%. 
>
> **In every test, logistic is more stable than XGBoost, and TabICL matches logistic on who gets selected.**

---

## Part 6: What the Numbers Actually Mean

### The "13%" Insight (Critical)

Many people misread Jaccard of 0.77 as "23% changed."

**Correct reading:**
- Jaccard = overlap / (set1 ∪ set2)
- 0.77 = 1,201 / 1,561 overlap out of 1,561 total selected each time
- Wait, that doesn't match — let me recalculate.

For two equal-sized sets of 1,561 people each with m people in common:
```
J = m / (1,561 + 1,561 - m) = m / (3,122 - m)
0.77 = m / (3,122 - m)
0.77 × (3,122 - m) = m
2,404 - 0.77m = m
2,404 = 1.77m
m ≈ 1,357
```

So 1,357 people appear in both selections out of 1,561, meaning **1,561 − 1,357 = 204 are swapped** each time (from one refit to another).

Share swapped: 204 / 1,561 ≈ 13% ✓

**Why this matters:**
- Caseworkers work with 1,561 people per quarter.
- About 200 of them appear and disappear.
- That's a problem, but not catastrophic.

---

## Part 7: Comparing to the Other Models

### Why Logistic Wins (But Not Decisively)

| Metric | Logistic | XGBoost | TabICL | Winner | Margin |
|--------|----------|---------|--------|--------|--------|
| Drift (mean \|Δp\|) | 0.032 | 0.035 | 0.035 | Logistic | 0.003 |
| Jaccard | 0.770 | 0.747 | 0.776 | TabICL | 0.006 |
| Rank correlation | 0.978 | 0.971 | 0.972 | Logistic | 0.006 |

**The honest story:**
- Logistic is the most stable by *drift* (the score movement).
- TabICL is actually *tied or better* on which people get selected (Jaccard).
- The differences are small—none of them are catastrophically unstable.

**Why we still recommend logistic:**
- Consistency on drift (smaller daily changes)
- Reproducibility across machines (see Slide 19)
- Interpretability (the main tie-breaker)

---

## Part 8: The Course Connection

### Slide 182: Definition
✅ We opened with the exact quote and explained our three-test approach.

### Slides 187–188: Comparing Models on 50% vs Full Data
✅ This is part of our harshness spectrum; full analysis in [03_beyond_the_deck_course_aligned_extras.md](03_beyond_the_deck_course_aligned_extras.md).

### Slide 191: The Trade-off
> A small loss in predictive power for a significant gain in stability.

- We don't sacrifice accuracy here—all three models are tied.
- We *do* gain interpretability and reproducibility for no accuracy loss.

### Slide 194: Importance-Distance Stability
✅ Covered in the deep review; our slide notes mention it for Q&A.

---

## Part 9: Common Mistakes to Avoid

1. **"Jaccard 0.77 means 23% changed"** ❌ → Use the formula: (1 − J) / (1 + J) ≈ 13% ✅

2. **"Bootstrap shows logistic is significantly more stable"** ❌ → The pairs aren't independent; say "consistent tendency" ✅

3. **"The halves test is unfair because each model sees less data"** ❌ → That's the point; it's a stress test ✅

4. **"All three models are equally stable"** ❌ → Logistic wins on drift and rank correlation, though the margin is small ✅

5. **"Stability means the model is fair"** ❌ → A stable model can still be unfair; see Slide 18 ✅

---

## Part 10: Talking Points for Q&A

### "Why not just use the most stable model on Jaccard (TabICL)?"

TabICL's Jaccard is slightly higher, but:
- Its drift is identical to XGBoost's (0.035 vs 0.032).
- It's far less interpretable (no native coefficients).
- On Slide 19, we see it has calibration and fairness issues.
- Stability is one dimension; the trade-off matrix decides.

### "Why three tests? Why not just one?"

Different tests isolate different sources of randomness:
- **Seed:** algorithmic randomness only.
- **Bootstrap:** realistic quarterly retrains with the same sample size.
- **Halves:** a worst-case stress test.

Logistic wins all three, so the finding is robust.

### "The differences look small. Are they meaningful?"

In absolute terms: 3 vs 3.5 percentage-point drift is small.
In practical terms: 200 people swapping in/out per quarter is a real support-program disruption.
In fairness terms: Slide 18 shows that small instability at the cut can widen gender gaps.

So yes, the 13% is meaningful.

---

## Part 11: Key Takeaways to Memorize

1. **Definition:** "Two datasets from the same population should give approximately the same model."

2. **The three tests:**
   - Seed-only: ~2% swaps (baseline)
   - Bootstrap: ~13% swaps (realistic)
   - Halves: ~18% swaps (stress test)

3. **Winner:** Logistic, consistently, but the margins are small.

4. **Jaccard trap:** Don't divide by 2; use (1 − J) / (1 + J).

5. **Stability is one dimension:** Performance, interpretability, reproducibility, and fairness matter too. Slide 19 weighs them all.

---

## Part 12: Supporting Files

From the codebase, these files back up every number:

- **Data & Protocol:** `artifacts/stability_protocol.json` (seeds, resampling, model versions)
- **Results:** `artifacts/stability_summary.csv` (the table numbers)
- **Figure source:** `artifacts/figures/structural_stability.png` (the three-panel plot)
- **Code:** `scripts/stability_structural.py` (how we computed it)

Run the script locally to see the full output:
```bash
python scripts/stability_structural.py
```

---

## References

- **Professor's Slide 182:** Definition of stability (Turney, 1995)
- **Our Concepts:** [01_stability_concepts_and_maths.md](01_stability_concepts_and_maths.md)
- **Beyond the Deck:** [03_beyond_the_deck_course_aligned_extras.md](03_beyond_the_deck_course_aligned_extras.md) (deeper analysis, other resampling tests)
- **Presentation Notes:** [10_p6_slides_and_script_3min.md](10_p6_slides_and_script_3min.md) (the 45-second script)

