# Slide 18 Deep Dive: Individual Stability and the Fairness Paradox

## Overview
**Slide Title:** "Stability for one person: abstaining is not fairness-neutral"  
**Duration:** 45 seconds  
**Core Question:** When decisions are uncertain (contested across refits), should we send them to a human? And if we do, does that help or hurt fairness?

---

## Part 1: From Structural to Individual Stability

### The Shift in Scope

**Slide 17 asked:** Do the *overall* rankings stay the same when we retrain?  
**Slide 18 asks:** For *one specific person*, do they keep the same decision?

**From the professor (Slide 183–184):** The caseworker's "?!" isn't about global statistics—it's "Why did person X disappear from the list?"

### Decision Types: Unanimous vs Contested

For each person, count how many of the 8 refits select them (0 to 8):

| Outcome | Refits Selecting | What It Means | Count (Logistic) |
|---------|------------------|---------------|-----------------|
| **Never selected** | 0/8 | Clear no | 5,707 |
| **Contested** | 1–7/8 | Partly luck | 1,002 |
| **Always selected** | 8/8 | Clear yes | 1,098 |
| **Among those majority-selected** | 5–8/8 | (1,002 + 1,098 refits select them) | — |

**Key insight:** Among the 1,561 people we help (majority selected):
- **1,098 are unanimous** (everyone agrees)
- **463 are contested** (roughly "a third sit at the margin")

**Logistic:** 1,002 total contested = 12.8% of everyone  
**XGBoost:** 1,140 total contested = 14.6% of everyone

---

## Part 2: The "Nearly Tied" Problem

### Score Spread: How Much Does One Person's Risk Move?

Across the 8 refits, each person gets 8 predictions. The range (max − min) tells you how much uncertainty there is.

| Metric | Logistic | XGBoost |
|--------|----------|---------|
| **Median score range** | 0.079 | 0.085 |
| **90th percentile range** | 0.122 | 0.132 |
| **Example:** A person with median range 0.079 might score 0.48 in one refit and 0.56 in another—a 7.9 percentage point shift. |

**Why this matters (professor's Slide 212):**
> "The problem arises when the two leading candidates are nearly tied."

A person sitting at score 0.50 (right at the support threshold) with a range of 0.08 will sometimes clear the cut (0.58) and sometimes miss it (0.42). That's the contested decision.

---

## Part 3: The Gender Asymmetry Discovery

### The Surprising Finding: Instability Is Not Equally Distributed

**Overall contested rate (among ALL 7,807 people):**
- Women: 12.1%
- Men: 12.9%
- → Looks balanced.

**But among people we ACTUALLY SELECT:**
- Women: **47.5% are contested**
- Men: **30.5% are contested**
- → Highly asymmetric!

### Why Does This Happen?

**The mechanism:**
1. Few women are selected overall (118 women vs 1,490 men in the "majority selected" group).
2. Those women who ARE selected sit right at the cut.
3. Those men who ARE selected spread across a wider range of scores.

**Visualization:**
- Distribution of women's scores: concentrated near 0.50 (the cut)
- Distribution of men's scores: spread out, many well above 0.50
- Result: women's scores bounce around the cut; men's don't.

**Without gang affiliation (the control test):**
- Women contested: 33%
- Men contested: 33%
- → Perfectly symmetric!

**Implication:** The gender asymmetry at the margin is driven by gang affiliation's correlation with both gender and re-arrest risk.

---

## Part 4: The Abstention Policy (The Critical Finding)

### What Is Abstention?

**Policy:** For any person with a contested decision (1–7 out of 8 refits), send them to a human reviewer instead of deciding automatically.

**Example:** If only 4 out of 8 refits select person A, don't put them on the list; send to caseworker for manual review.

### The Trade-off: Coverage vs Precision vs Fairness

| Policy | Coverage | Precision | Gender FNR Gap | Effect |
|--------|----------|-----------|---|---|
| Decide all (majority vote, 4+) | 100% | 0.822 | −0.090 | Baseline |
| Unanimous only (8/8) | 87.2% | 0.846 | −0.119 | **Gap widens by 3 pts** |

**Translation:**
- **Coverage 100% → 87.2%:** We automatically decide 87% of people; 13% go to review.
- **Precision 0.822 → 0.846:** Among those we auto-decide, re-arrest detection improves from 82% to 85%.
- **Gender gap −0.090 → −0.119:** The FNR gap *widens* by 0.029 (about 3 percentage points).

---

## Part 5: Why Abstention Widens the Gender Gap

### The Mechanism (One Sentence)

Abstention removes contested cases, and 47% of the few women selected are contested against 30% of men, so women lose a bigger share of their offers and the gap widens.

### Detailed Logic

1. **Start:** gender FNR gap = −0.090 (men are missed 9 percentage points more often).

2. **Abstention removes contested cases:**
   - Removes 463 contested people (from 1,561 down to ~1,098 unanimous ones).
   - Of the women selected, 47.5% are removed (56 out of 118).
   - Of the men selected, 30.5% are removed (456 out of 1,490).
   - **Relative loss:** women lose 47.5% of their spots; men lose 30.5%.

3. **Among the remaining unanimous cases:**
   - Fewer women are left (62 instead of 118, a 47% cut).
   - Fewer men are left (1,034 instead of 1,490, a 31% cut).

4. **Recalculate FNR on the unanimous subset:**
   - Women now have higher FNR (we're missing more of them because we removed many from the list).
   - Men's FNR increases too, but less dramatically.
   - New gap: −0.119 (men missed 12% more than women).

---

## Part 6: Does Abstention Help on Race?

### The Answer: Yes, Slightly

**Race FNR gap (Black − White):**
- Baseline: −0.013 (Black people missed 1.3 percentage points more)
- After abstention: −0.006 (gap narrows)

### Why the Difference?

**Race IS balanced at the margin:**
- Black: 32.2% of selected are contested
- White: 31.0% of selected are contested
- → Nearly identical.

So abstention removes a balanced share from both groups, leaving the gap intact (or even narrowing it slightly due to sampling).

**Gender is NOT balanced at the margin:**
- Women: 47.5% contested
- Men: 30.5% contested
- → 17 percentage points apart.

So abstention disproportionately affects women, widening the gap.

---

## Part 7: The Critical Numbers for Slide 18

### The Four Red Statistics (From Slide Callout)

1. **≈13% of decisions flip** (logistic)
   - More precisely: 12.8% of 7,807 people are contested = 1,002 people
   - This is the share of people who might get different decisions in different refits

2. **0.822 → 0.846** (precision improves)
   - Baseline: among the 1,561 we select, 82.2% are re-arrested
   - Unanimous only: among the ~1,098 unanimous selections, 84.6% are re-arrested
   - Interpretation: abstention removes *uncertain* low-risk cases, leaving only *certain* high-risk cases

3. **−0.090 → −0.119** (gender FNR gap widens)
   - Baseline: men missed 9 percentage points more than women
   - Unanimous: men missed 12 percentage points more than women
   - Cost: fairness gets worse to gain precision

4. **47% vs 30%** (gender asymmetry at the margin)
   - Of the women selected, 47.5% are contested
   - Of the men selected, 30.5% are contested
   - Root cause: women selected are concentrated near the cut; men spread out

---

## Part 8: The Script (45 seconds)

### Current Script (Existing)
> Now one person. Across the eight refits, about one decision in eight is contested — some refits select the person, others don't — and all of them sit close to the cut. The natural fix is to send contested cases to a human. That raises precision from 0.822 to 0.846. But the gender gap widens, from minus 0.090 to minus 0.119, because the few women we select sit at the margin far more often than men: 47% against 30%. So abstaining is not fairness-neutral — and that asymmetry comes from gang affiliation: without it, women and men are equally often at the margin.

### Analysis & Potential Improvements

**Strengths:**
✅ Opens with the core concept (one decision in eight is contested)  
✅ Introduces abstention as the natural fix  
✅ Gives the precision number (0.822 → 0.846)  
✅ Names the fairness cost (gap widens)  
✅ Explains the mechanism (women sit at margin more often)  
✅ Connects to gang affiliation  

**Potential clarifications (if timing allows):**
- "One decision in eight is contested" could be slightly more precise: "about 13% of people are contested across the 8 refits" (matches Slide 17 language)
- "All sit close to the cut" is implied but not emphasized—could be explicit for the professor's Slide 212 connection

**Timing:** Current script ≈ 160 words ≈ 64 seconds at 150 wpm. We have 45 seconds, so we're slightly over. Options:
1. Keep it as is (listeners are often faster on practiced material; measure when rehearsing)
2. Trim slightly: remove "and that asymmetry comes from gang affiliation: without it, women and men are equally often at the margin" → move to Q&A

---

## Part 9: Interpreting the Figure (Three Panels)

### Left Panel: How Often Each Person Is Selected
**Title:** "How often is each person selected across 8 refits?"

- X-axis: number of refits (0 to 8)
- Y-axis: count of people (log scale, so steep is many)
- Two curves: logistic (blue) and XGBoost (orange)
- Sharp peaks at 0 (never) and 8 (always), long tail in the middle

**Key insight:** Most decisions are unanimous (0 or 8), but about 13% sit in the contested middle.

---

### Middle Panel: Spread of One Person's Score
**Title:** "Spread of one person's score across refits"

- Histogram showing the distribution of maximum − minimum score range across all people
- Median around 0.08 (8 percentage points)
- 90th percentile around 0.12 (12 percentage points)
- Both logistic (blue) and XGBoost (orange) similar

**Key insight:** Typical person's risk bounces around by 8 points; worst case ~12 points.

---

### Right Panel: Abstaining on Contested Decisions: Does the Gap Close?
**Title:** "Abstaining on contested decisions: does the gap close or widen?"

- Shows FNR gap (gender) on the y-axis vs coverage (share still decided) on the x-axis
- Two lines: logistic (blue) and XGBoost (orange)
- As coverage drops (more abstentions), gap *worsens* (becomes more negative)
- Logistic: −0.090 at 100% coverage → −0.119 at 87% coverage
- Orange similar pattern

**Key insight:** Abstaining to improve precision *hurts* gender fairness because women are overrepresented in contested cases.

---

## Part 10: Course Connections

### Professor's Slide 212: "The Problem Arises When the Two Leading Candidates Are Nearly Tied"

Our finding: Contested decisions (1–7/8 refits) are people whose scores hover around the cut (0.50).

Example person:
- Refit 1: score 0.52 → selected
- Refit 2: score 0.48 → not selected
- Refit 3: score 0.54 → selected
- ...
- Difference: just 0.06 (within our median range of 0.079)

**Connection:** This IS the "nearly tied" scenario the professor warns about. Abstention handles it by asking a human, which is wise—but only if the human reviewer is itself audited for fairness.

### Professor's Slide 263: "Failing to Reject Does Not Certify Fairness"

We initially thought: "Abstention improves precision, so it's good for the model."

The data showed: Precision improves, but gender fairness *worsens*.

**Implication:** We can't claim abstention is fair just because precision increases. We must audit the fairness impact explicitly—which we did on Slide 18.

---

## Part 11: Why Without Gang Affiliation It's Symmetric

### The Control Experiment

We retrained logistic regression *without* the "gang affiliation" feature and recomputed individual stability.

**Result:**
- Women contested at margin: 33%
- Men contested at margin: 33%
- → Perfectly symmetric

**Interpretation:**
- Gang affiliation is correlated with (1) lower re-arrest and (2) being male.
- When we remove it, the correlation with gender disappears, and the asymmetry at the cut vanishes.
- The model's fairness problem (gender gap at baseline) is partly driven by gang affiliation being a strong predictor.
- Removing it doesn't solve fairness, but it does make abstention symmetric.

**For Slide 18 context:** This tells the jury that the gender asymmetry is *not* a misclassification error or a bug—it's how the data itself is structured. It's real, and we found it.

---

## Part 12: Common Mistakes to Avoid

1. **"13% of decisions change randomly"** ❌ → Not random; they sit at the cut ✅

2. **"Abstention is unfair because the gender gap widens"** ❌ → More precise: "Abstention trades precision for gender fairness; the gap widens because women sit at the margin more often" ✅

3. **"Race is unaffected by abstention, so race is fair"** ❌ → Race is balanced at the margin for this data; gender is not. Don't generalize. ✅

4. **"Gang affiliation causes the gender gap"** ❌ → Gang affiliation is *correlated* with the gap, but causality requires intervention evidence. ✅

5. **"Send all contested cases to humans and the problem is solved"** ❌ → The human-review queue must also be audited for fairness. ✅

---

## Part 13: Q&A Talking Points

### "Why not just raise the threshold to avoid disputed cases?"

Raising the cutoff from 0.50 to 0.55 would mean fewer people selected overall. The contested people would mostly just be demoted, not removed. You'd still have a tail at the new cut.

Better to explicitly abstain and audit the human review.

### "Is the gender gap actually caused by gang affiliation?"

Gang affiliation is strongly predictive and skewed by gender. But correlation ≠ causation. We can show that *removing* gang affiliation makes the margin symmetric, but we can't say it *causes* the gap without intervention evidence (e.g., a no-gang model piloted in production).

### "Why does race stay balanced but gender doesn't?"

Race has similar re-arrest predictiveness as gender, but it's not as skewed in the data we see. Women selected are fewer and sit near the cut; Black people selected are more numerous and spread across a wider score range. The numbers work out differently.

### "Should we abstain or not?"

That's a policy choice, not a data choice. The data shows: abstention improves precision by 2.4 points (0.822 → 0.846) but widens the gender gap by 2.9 points (−0.090 → −0.119). The client decides whether that trade-off is acceptable.

---

## Part 14: Key Takeaways to Memorize

1. **Definition:** A decision is contested if 1–7 of the 8 refits select the person; unanimous if 0/8 or 8/8.

2. **The number:** 12.8% of people (logistic) are contested; 14.6% (XGBoost).

3. **The asymmetry:** 47% of women selected are contested; 30% of men. Root cause: women selected are concentrated near the cut.

4. **The trade-off:** Abstention on unanimous decisions raises precision (0.822 → 0.846) but widens the gender FNR gap (−0.090 → −0.119).

5. **The fairness question:** Abstention is not fairness-neutral. The human-review queue must be audited too.

6. **The control:** Without gang affiliation, women and men are equally contested at the margin (33% vs 33%), showing the asymmetry is data-driven.

---

## Part 15: Supporting Files

From the codebase:

- **Per-person data:** `artifacts/individual_stability.csv` (all 7,807 people × 8 scores)
- **Summary:** `artifacts/individual_stability_summary.csv` (contested counts and score ranges)
- **Abstention curves:** `artifacts/abstention_curve.csv` (coverage, precision, gender/race gaps at each threshold)
- **Figure:** `artifacts/figures/individual_stability.png` (the three panels)
- **Code:** `scripts/individual_stability.py` (how we computed it)

Run the script:
```bash
python scripts/individual_stability.py
```

---

## Part 16: References

- **Professor's Slide 212:** "The problem arises when the two leading candidates are nearly tied"
- **Professor's Slide 263:** "Failing to reject does not certify fairness"
- **Our Concepts:** [01_stability_concepts_and_maths.md](01_stability_concepts_and_maths.md) (decision stability section)
- **Full Audit:** [02_what_we_found.md](02_what_we_found.md) (sections 2.3 and 2.6 detail the asymmetries)
- **Presentation Notes:** [10_p6_slides_and_script_3min.md](10_p6_slides_and_script_3min.md) (the 45-second script)

