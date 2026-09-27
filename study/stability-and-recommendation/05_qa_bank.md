# 5 · Q&A bank: stability and recommendation

> Short answers first, then the detail if pushed. Practise saying the **bold** line out
> loud. The team's shared answers are in `reports/presentation_notes.md`, and these stay
> consistent with them.

---

## Stability: definitions

**Q1. What do you mean by stability?**
**"If we had drawn a different sample from the same population, would we build the same
model, and would the same people get help?"**
- It follows the course's §7 definition.
- We measure it at three levels: the scores, the selected set, and one person's decision.

**Q2. Why bootstrap and not something else?**
**"Bootstrap resamples are full-size, so they isolate sampling noise without shrinking the
data. We also ran the course's literal version, two disjoint halves, and the ordering
holds."**
- Disjoint halves: logistic Jaccard 0.70 vs XGBoost 0.68.
- Logistic is ahead on all 10 splits.

**Q3. Why only 8 refits?**
**"TabICL needs a GPU for each refit, and 8 was what made all three models feasible on
identical resamples."**
- 8 refits give 28 pairs. The pairs share refits, so we report a consistent tendency
  (28/28 on drift), not a p-value.

**Q4. Jaccard 0.77, so 23% of people change?**
**"No. About 13%. With equal-sized sets, the replaced share is (1 − J)/(1 + J)."**
- That's about 200 of the 1,561 people selected.

**Q5. Is this temporal stability?**
**"No. All resamples come from the same 2013–2015 Georgia cohort."**
- Drift over time needs new cohorts, and that's one of the pilot's gates.

**Q6. Isn't AUC stability enough?**
**"No. AUC can stay put while different people are selected."**
- The AUC interval width is about 0.023 for all three models, yet 13–15% of the selected
  set turns over.

## Stability: results

**Q7. Is logistic really more stable, or is that noise?**
**"Against XGBoost it's consistent: lower drift on all 28 pairs and a more stable selected
set on 26 of 28. Under disjoint halves, 9 of 10 and 10 of 10."**
- Against TabICL it's a tie on the selected set (14 of 28 pairs).
- So say "more stable than XGBoost", not "the most stable".

**Q8. The difference is 0.032 vs 0.035. Does that matter?**
**"On its own, no. It matters because it's the only dimension besides interpretability
where the two models genuinely differ at the decision."**
- Performance differences don't change who gets help.
- Calibration and fairness are ties.

**Q9. Are the explanations stable?**
**"Yes, for both. Age is the top driver in every refit, and the same 8 features lead each
time."**
- On the course's importance distance ‖φ₁ − φ₂‖₂, XGBoost is even slightly more stable.
- Logistic's advantage is in decisions, not explanations. (Being honest here earns trust.)

**Q10. Did you compute the course's ‖θ₁ − θ₂‖₂?**
**"Yes: 0.70 between bootstrap refits and 0.90 between disjoint halves. But it overstates
instability for our model."**
- With a full one-hot encoding the coefficients aren't identified.
- Two refits with different seeds give *identical* predictions but ‖Δθ‖ = 0.31.
- So we judge on predictions and on the large coefficients. The 22 largest never change
  sign.

**Q11. What causes the instability: data or algorithm?**
**"Mostly data."**
- XGBoost's seed alone moves scores by 0.006; a new data sample moves them by 0.035.
- Logistic's seed changes nothing.
- Software and threads matter for XGBoost too: 80 decisions changed on another machine.

**Q12. Which people are unstable?**
**"Only people near the cut."**
- Within ±2 percentile points of the line: 100% contested.
- Beyond 20 points: 0%.
- Re-arrested people are contested 2.5× as often (17% vs 7%).

## Abstention (slide 18)

**Q13. Why not just refer the uncertain cases to a human?**
**"We tested that. Precision rises from 0.822 to 0.846, but the gender gap widens from
−0.090 to −0.119."**
- 47% of the 118 women selected are at the margin, against 30% of the men.
- Abstention removes more of the few offers women receive.

**Q14. Why −0.090 on slide 18 when the fairness section says −0.096?**
**"Different baseline."**
- The abstention curve decides by majority vote of the 8 refits.
- The fairness section uses the single published model.
- Both are correct for their definitions.

**Q15. Does abstention hurt on race too?**
**"No. Race is balanced at the margin (32% vs 31%), and the race gap even narrows slightly
(−0.013 → −0.006)."**

**Q16. So what should the client do?**
**"Don't abstain by default. If they do, audit the human-review queue by gender."**
- Better still, show caseworkers the refit vote count, so borderline cases are visible
  rather than hidden.

## Recommendation

**Q17. XGBoost is more accurate. Why ship the weaker model?**
**"Its accuracy edge doesn't change who gets help."**
- 85% of the selected people are the same (Jaccard 0.849), and only 254 of 7,807 differ.
- The difference in captured re-arrests has a CI that includes zero.
- Logistic wins on stability, direct interpretability, reproducibility and 9× speed.

**Q18. So logistic is fairer / better calibrated?**
**"No, and we don't claim it. Both are ties."**
- Gender-gap difference: −0.013, CI −0.036 to +0.011.
- Cox calibration slopes: 0.999 vs 1.005.

**Q19. Why not TabICL? It has the best AUC.**
**"Its probabilities are measurably too extreme, it has no native explanation, and it
needs a GPU."**
- Calibration slope 0.913, z = 3.76.
- 42 s per run.
- It *is* the best choice for very small datasets (learning curve at 1,500 rows).

**Q20. What would change your mind?**
**"Three things."**
1. Scores are quoted as probabilities to people.
2. Operating at a scale where about 12 extra captured re-arrests per 1,561 offers matter.
3. A new feature set that widens the non-linear model's margin.
- The challenger in the pilot tells us.

**Q21. What's a shadow pilot?**
**"The model scores everyone, but decisions follow current practice. We compare what it
would have done with what happened, before it allocates anything."**

**Q22. Are the dollar figures savings?**
**"No. The effectiveness is assumed."**
- The useful number is the break-even: support must prevent at least 12% of re-arrests
  among those helped for the programme to pay off with our model, against 15% with the
  current tool.

**Q23. Does the best model depend on capacity?**
**"Yes. Logistic has the highest net value at 5%, 25%, 30% and 35% capacity; XGBoost or
TabICL at the others."**
- The differences are at most about $0.17M.
- The gap to the current tool is $2–3.5M.

**Q24. How often should the model be retrained?**
**"On a fixed schedule, and support already offered should not be withdrawn just because
a retrain moved someone below the cut."**
- Even the most stable model swaps about 13% of the selected people per retrain.

**Q25. Would you change C for stability?**
**"We'd consider it in the pilot."**
- C = 0.05 costs 0.0006 CV AUC and gives 22 fewer coefficients, 12% less drift and +1.3
  points of Jaccard.
- That's modest, and it was measured with training-only CV.

## Traps: things *not* to say

- ❌ "Logistic is more stable than every model" → only more than XGBoost; it ties with TabICL on selection.
- ❌ "Logistic is better on all 28 pairs" for Jaccard → 26 of 28.
- ❌ "23% of people change" → 13%.
- ❌ "Logistic has smaller subgroup gaps" (the app's governance tab still says this) → a tie.
- ❌ "The data has a performance ceiling" → three models converging doesn't prove one.
- ❌ "Stable means fair" → abstention shows the opposite.
