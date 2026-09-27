# 1 · Stability: concepts and maths

> Read this first. It explains every idea and formula behind slides 17–18, with worked
> examples that use our own numbers. The course section is §7 (stability); the page
> references come from `docs/PLAN.md`. Check the wording against the course slides.

---

## 1.1 What "stability" means in the course

**Course definition (§7.1):** if two datasets are drawn from the same population, a
stable method should produce *approximately the same model*.

Why the client should care:

| Why | In our product |
|---|---|
| **Consistency for the individual** | The same person should not get support this quarter and lose it next quarter just because the model was retrained. |
| **Trust in the explanation** | If the drivers change between refits, the explanation shown to a caseworker is not reliable. |
| **Maintenance** | A vendor retrains models regularly, and every retrain changes who is prioritised. |
| **Auditability** | A regulator or an appeals process needs the same answer to the same question. |

**Stability is not the same as performance uncertainty.** A bootstrap interval on AUC
(0.718 to 0.741 for logistic) says how much the *score* might move. It says nothing about
whether the *same people* get selected. The team's first version measured only the
former, and the dev log (`docs/JOURNEY.md`) records the switch to structural stability.

## 1.2 Three levels, from the model to the person

1. **Structural stability, i.e. the model itself.** Do the predictions, rankings,
   coefficients and explanations stay the same? (slide 17)
2. **Decision stability, i.e. one person.** Does *this* person keep the same decision?
   (slide 18)
3. **Performance stability.** Does AUC stay the same? This is the weakest notion, and we
   only mention it.

## 1.3 Where the randomness comes from (course §7.2)

| Source | Example | How we isolate it |
|---|---|---|
| **Training data** | A different sample of people | Bootstrap refits with the model seed fixed; disjoint halves |
| **Algorithm seed** | XGBoost subsamples 82% of rows and 62% of columns per tree | Refit on the same data with different seeds |
| **Hardware / software** | Thread count, package version, CPU vs GPU | Measured separately (thread-sensitivity file) |

The published audit varies **only the data**. The eight resamples are shared by all
three models, and the model seed is fixed at 42. That is what makes the three models
comparable. The extras in `03_*` add the other two sources.

## 1.4 Resampling schemes

**Bootstrap.** Draw n = 18,028 rows *with replacement* from the training set.
- About 63.2% of the people appear at least once (1 − 1/e), and the rest are duplicates.
- Two bootstrap samples share about 40% of people, i.e. (1 − 1/e)² ≈ 0.40.
- It is a mild perturbation: the samples are full-sized and overlap heavily.

**Disjoint halves.** Split the training set into two halves with no one in common.
- This is the literal "two datasets from the same population" definition.
- It is harsher: each model sees only half the data, and the two share zero people.

**Seed-only.** Same data, different random seed. This measures randomness that comes
purely from the algorithm.

## 1.5 The metrics, one by one

For two refits *a* and *b*, each scoring the same 7,807 evaluation people.

### Score drift: mean |Δp|
$$\text{mean }|\Delta p| = \frac{1}{n}\sum_{i=1}^{n} |p_i^{(a)} - p_i^{(b)}|$$
- **Ours:** logistic 0.032, XGBoost 0.035, TabICL 0.035.
- **Say:** "On average a person's predicted risk moves by about 3 percentage points
  between refits."
- **p95 version:** logistic 0.084, which means 1 person in 20 moves by 8 points or more.

### Rank agreement: Spearman ρ
This is the correlation of the two *rankings*, not of the scores themselves.
- **Ours:** 0.975 / 0.971 / 0.972.
- It matters because the product *ranks* people and keeps the top 20%.

### Same people selected? Jaccard
$$J = \frac{|A \cap B|}{|A \cup B|}$$
Here A and B are the two selected top-20% sets, 1,561 people each.

**Trap: Jaccard is not "the share that changed".** With equal-sized sets of size k and
overlap m:
$$J = \frac{m}{2k - m} \;\Rightarrow\; m = \frac{2kJ}{1+J} \;\Rightarrow\; \text{replaced share} = \frac{k-m}{k} = \frac{1-J}{1+J}$$

Toy check: take A = {1,2,3,4,5} and B = {1,2,3,4,6}. Then J = 4/6 = 0.667, and the replaced
share is (1 − 0.667)/(1 + 0.667) = 0.2, which is 1 person out of 5. ✓

| Model | J | Replaced share | People swapped (of 1,561) |
|---|---:|---:|---:|
| Logistic | 0.7725 | 12.8% | ≈ 200 |
| XGBoost | 0.7468 | 14.5% | ≈ 226 |
| TabICLv2 | 0.7758 | 12.6% | ≈ 197 |

**Say:** "A Jaccard of 0.77 means about 13% of the people offered support change when we
retrain, not 23%."

### The course's distance between models (added in `03_*`)
- **Parameter distance**, for models with parameters (logistic):
  $$\|\theta_1 - \theta_2\|_2 = \sqrt{\textstyle\sum_j (\theta_{1j} - \theta_{2j})^2}$$
  We also report it *relative* to the size of the coefficients, since the raw number
  depends on scale.
- **Explanation distance**, which works for any model:
  $$\|\varphi(f_1) - \varphi(f_2)\|_2$$
  where φ is the vector of feature importances. We normalise φ to shares that sum to 1,
  so logistic and XGBoost are on the same scale. The distance is 0 when the explanations
  are identical and at most √2.
- **Why two measures:** XGBoost has no single θ, since it is 1,196 trees. That is why the
  course also compares importance vectors.

### Decision stability for one person
- Refit 8 times and count how many refits select the person (0–8).
- **Unanimous** means 0/8 or 8/8, a decision we can stand behind.
- **Contested** means 1–7 out of 8: the published decision was partly luck.
- **Score range** is max − min of the person's 8 scores (median 0.079 for logistic).

### Selective prediction (abstention)
**Policy:** decide only the unanimous cases and send contested ones to a human. Evaluate
it like any policy:
- **Coverage:** the share of people still decided automatically.
- **Precision among decided:** of those selected, how many were re-arrested.
- **Subgroup gap:** the FNR gap between groups among those decided.

## 1.6 Why we don't give p-values for "logistic is more stable"

From 8 refits there are C(8,2) = 28 pairs, but the pairs share refits: pair (1,2) and
pair (1,3) both use refit 1. So the 28 numbers are **not independent**. Treat "logistic
drifts less on all 28 pairs" as a *consistent tendency*, not an exact significance level.
The report says exactly this.

## 1.7 What stability does not tell you

- **Not temporal drift.** All resamples come from the same 2013–2015 Georgia cohort. They
  cannot show what happens when the population changes over time.
- **Not transfer to other agencies.** Subsamples of one cohort don't validate other states.
- **Not fairness.** A model can be perfectly stable and still unfair, since the same
  people are consistently missed. Slide 18 shows that the *fix* for instability
  (abstention) interacts with fairness.

## 1.8 One-line summaries to memorise

- "Stability asks: if we had drawn a different sample, would we have built the same model,
  and would the same people get help?"
- "Jaccard of 0.77 means 13% of the selected change, not 23%."
- "Stable is not the same as fair: abstention makes decisions more reliable and the
  gender gap wider."
