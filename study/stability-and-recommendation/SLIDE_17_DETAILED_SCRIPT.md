# Slide 17: Detailed Educational Script
## "Structural stability: 8 bootstrap refits"

---

## Full Educational Script (Detailed Version)

### Part 1: Definition of Bootstrap

> **Bootstrap is a resampling technique.** We take our original training set of 18,028 people and draw samples of the same size *with replacement*. That means some people appear multiple times, and others don't appear at all. In fact, about 63% of the original people appear at least once in each bootstrap sample, and about 37% are missing. This creates eight slightly different datasets, all from the same population but with different compositions.

**Why bootstrap matters:** It's mild (the samples are nearly full-sized) but realistic—this is what happens in the real world when the agency gets new releases of people each quarter.

---

### Part 2: The Three Methods We Tried

> **We tested stability three ways, from mildest to harshest.**
>
> **First, the seed-only test:** We took the exact same training data and ran each model eight times, changing only the random seed (42, 43, 44, and so on). This tells us: how much does the algorithm's own randomness affect the decisions? The answer: XGBoost's random seed alone swaps about 2% of the people it selects. That's our floor.
>
> **Second, the bootstrap test—our main test:** We created eight different bootstrap samples of the training data. Each sample is the same size as the original, but with different people. We trained each model once on each bootstrap sample, keeping the seed fixed at 42 across all models so they're fair to compare. Then we evaluated all three models on the same test set of 7,807 people. This is our centerpiece because it's realistic: quarterly retrains look like this.
>
> **Third, the disjoint halves test:** We split the training data into two completely separate halves—9,014 people each, with no overlap. We trained each model on one half and evaluated on the same test set. This is the harshest test: each model sees only 50% of the training data, and the two halves share zero people.

---

### Part 3: What Mean Delta P Is

> **Mean delta P, or mean |Δp|, measures score drift.** For each person in the test set, we look at their predicted probability from two different refits—say, refit 1 and refit 2. We calculate the absolute difference: |p₁ − p₂|. We do this for every person and every pair of refits. Then we average all those differences.
>
> **What does this tell us?** On average, how much does one person's predicted risk score bounce around between refits? 
>
> **Our results:** 
> - Logistic: 0.032 — on average, a person's risk score moves by 3.2 percentage points
> - XGBoost: 0.035 — slightly more, 3.5 percentage points
> - TabICL: 0.035 — same as XGBoost
>
> **Why this matters:** If a person's score bounces around a lot, the decisions become uncertain. A person at 0.50 (right at the cut for the top 20%) who swings to 0.58 is sometimes selected, sometimes not. That's a problem.

---

### Part 4: What Top-20% Jaccard Is

> **Jaccard is a measure of overlap between two sets.** 
>
> Here's how it works: We take the top 20% of people (1,561 people) selected by model trained on refit 1. We take the top 20% selected by model trained on refit 2 (also 1,561 people). How many people appear in *both* lists? That's the overlap. 
>
> The Jaccard formula is: **overlap / (set1 ∪ set2)** — the intersection divided by the union.
>
> **Example:** If 1,357 people are in both lists out of a combined 1,561 (the larger of the two sets), then Jaccard = 1,357 / 1,561 ≈ 0.87. But our data gives us 0.77, which means fewer overlap.
>
> **What does this tell us?** Do the *same people* get selected across refits? Or does retraining shuffle who gets support?
>
> **Our results:**
> - Logistic: 0.77 — 77% overlap
> - XGBoost: 0.75 — 75% overlap
> - TabICL: 0.78 — 78% overlap
>
> **The trap:** People often misread this as "23% changed" (since 1 − 0.77 = 0.23). But that's wrong. The correct interpretation uses the formula: **(1 − J) / (1 + J) = (1 − 0.77) / (1 + 0.77) ≈ 0.128 = 12.8%**. So about 13% of the selected people swap, not 23%.

---

### Part 5: Why We See 13–15% (Between 9% and 20%)

> **The range 9%–20% comes from the fact that we have 28 pairs of refits.** 
>
> With 8 bootstrap refits, there are C(8,2) = 28 possible pairwise comparisons. Each pair gives a slightly different Jaccard value. Some pairs have 12% turnover, others have 15%. When we look across all 28 pairs:
>
> **For logistic:**
> - Minimum turnover: 9% (the best-case pair)
> - Maximum turnover: 20% (the worst-case pair)
> - **Median/average: 12.8%**
>
> **Why does it vary?** Because some bootstrap samples happen to overlap more with each other than others. Sample A and B might be very similar (high overlap, low turnover). Samples C and D might be quite different (low overlap, high turnover). The variation across 28 pairs shows us the range of realistic outcomes.
>
> **The headline number:** 13% is our central finding. But "13–15%" or "9–20%" tells the full story—there's some variation depending on which refits you compare.

---

### Part 6: The Box Plot Bit

> **The figure has three panels, each showing a box plot for the three models.**
>
> **Left panel: "Distance among refits (mean |Δp|, lower = stabler)"**
> - X-axis: Three models (logistic, XGBoost, TabICL)
> - Y-axis: Mean |Δp| values
> - Each box shows the distribution of this metric across the 28 pairs
> - The line in the middle of the box is the median
> - The box itself shows the middle 50% of values (25th to 75th percentile)
> - The whiskers show the range
> - **Interpretation:** Logistic's box is lowest, meaning it drifts least on average. All three models overlap, so differences are small but consistent.
>
> **Middle panel: "Same people selected? (top-20% Jaccard)"**
> - X-axis: Three models
> - Y-axis: Jaccard values (closer to 1.0 = more overlap)
> - Each box shows the distribution of Jaccard across the 28 pairs
> - **Interpretation:** Logistic's box is slightly lower (0.77), XGBoost is lowest (0.75), TabICL is highest (0.78). But all overlap. TabICL is actually best here, but logistic wins on drift.
>
> **Right panel: "Score rank correlation across refits"**
> - X-axis: Three models
> - Y-axis: Spearman ρ (rank correlation, 0 = no correlation, 1 = perfect)
> - **Interpretation:** All three models have high correlation (0.97–0.98), meaning the *rankings* are very stable even if scores bounce around.

---

## Condensed Script (45 Seconds - ~110 words)

> **Bootstrap resamples our 18,028 training people with replacement, creating eight slightly different datasets. We tested stability three ways: seed-only (baseline), bootstrap (realistic quarterly retrain), and disjoint halves (stress test).**
>
> **Mean delta P measures score drift.** A person's risk bounces around: logistic 3.2 points, XGBoost 3.5 points. That's small but matters at the cut.
>
> **Top-20% Jaccard measures overlap.** Logistic 0.77, XGBoost 0.75, TabICL 0.78. Here's the key: that's NOT "23% changed." The correct formula is (1−J)/(1+J) ≈ **13% swap out**—not 23%.
>
> **Why 13–15%?** We have 28 pairwise comparisons; turnover ranges 9–20% depending on which refits we compare. Median is 12.8%, which we round to 13%.
>
> **The box plots show logistic drifts least (left panel), rankings are stable across all three (right panel).**
>
> **In every test: logistic is more stable than XGBoost.**

---

## Which Version Do You Want?

**Option A:** Use the **Full Version (450 words)** for a complete, detailed 3-minute explanation  
**Option B:** Edit down to a **45-second version** for the actual presentation  
**Option C:** Use this as a **study guide** and create a separate tight script for delivery  

Which would work best for you?
