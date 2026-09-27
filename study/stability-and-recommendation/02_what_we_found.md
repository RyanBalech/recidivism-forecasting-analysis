# 2 · What the team's stability audit found (slides 17–18)

> Every number below was re-read from the raw files on 27 Sep 2026: the pair-level CSVs,
> not just the summaries. The source is given for each block. Section 2.6 lists places
> where the slides, report, notebook or app disagree with the data. Know these before
> the jury does.

---

## 2.1 Protocol: say this precisely

- **8 bootstrap resamples** of the 18,028 training records (numpy seed 7).
- **The same 8 resamples for all three models.** That is what makes them comparable.
  Earlier versions gave each model its own resamples, which confounded the comparison.
- **Model seed fixed at 42**, so only the *data* varies.
- Every refit scores the same **7,807 evaluation people**.
- 8 refits give **28 pairs**, and every pair is compared.
- The SHA-256 hashes of the resamples are in `artifacts/stability_protocol.json`, so
  anyone can check they are identical.

Scripts: [`scripts/stability_structural.py`](../../scripts/stability_structural.py) (the
model) and [`scripts/individual_stability.py`](../../scripts/individual_stability.py) (one
person). The second reproduces the first one's resamples exactly.

## 2.2 Structural stability: slide 17

Source: `artifacts/stability_summary.csv` and `artifacts/stability_pairs.csv`.

| | Logistic | XGBoost | TabICLv2 |
|---|---:|---:|---:|
| Mean \|Δp\| between refits | **0.0322** | 0.0351 | 0.0350 |
| (range over the 28 pairs) | 0.028–0.037 | 0.032–0.038 | 0.032–0.037 |
| p95 \|Δp\| | **0.084** | 0.091 | 0.090 |
| Spearman rank correlation | **0.975** | 0.971 | 0.972 |
| Top-20% Jaccard | 0.7725 | 0.7468 | **0.7758** |
| Replaced share of the selected set | 12.8% | 14.5% | 12.6% |

**Head-to-head over the 28 paired refits** (computed from `stability_pairs.csv`):

| Comparison | Lower drift | Higher Jaccard |
|---|---|---|
| Logistic vs XGBoost | logistic **28 / 28** | logistic **26 / 28** ⚠️ |
| Logistic vs TabICL | logistic 23 / 28 | logistic 14 / 28, a coin flip |
| TabICL vs XGBoost | TabICL 14 / 28 | TabICL 25 / 28 |

**How to read it:**
- Logistic is clearly more stable than XGBoost.
- Against TabICL, logistic drifts less but selects an equally stable set.
- **Say "more stable than XGBoost", not "the most stable of the three".**

### Are the explanations stable too?

Source: `artifacts/stability_contributions.csv` (SHAP importance across refits).

| | Logistic | XGBoost |
|---|---:|---:|
| Rank correlation of importance vectors, mean (min) | 0.86 (0.78) | 0.89 (0.81) |
| Age at release: share of importance (CV across refits) | 0.455 (CV 0.06) | 0.391 (CV 0.06) |

- Age is the top driver in every refit for both models.
- The same 8 features form the top 8 each time.
- The least stable of those is the mental-health/substance-abuse condition (CV ≈ 0.21).
- **Say:** "The story the model tells, i.e. which features drive risk, survives resampling."

## 2.3 Stability for one person: slide 18

Source: `artifacts/individual_stability_summary.csv` and `artifacts/individual_stability.csv`.

**How many of the 8 refits select each person (logistic):**

| Refits selecting | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| People | 5,707 | 237 | 151 | 104 | 122 | 112 | 127 | 149 | **1,098** |

- **5,707** people are never selected and **1,098** are always selected.
- **1,002 (12.8%) are contested.** XGBoost has 1,140 (14.6%).
- Of the people a majority of refits select, **31.7%** (logistic) and 35.8% (XGBoost)
  are contested. That's roughly "a third of those we help are at the margin".
- A person's score moves by a median of **0.079** (logistic) or 0.085 (XGBoost) between
  their highest and lowest refit, and by 0.12 at the 90th percentile.

**Who is contested?**

| Group | Contested (all) | Selected (majority) | Contested among selected |
|---|---:|---:|---:|
| Women | 12.1% | 118 | **47.5%** |
| Men | 12.9% | 1,490 | **30.5%** |
| Black | 13.2% | 969 | 32.2% |
| White | 12.3% | 639 | 31.0% |
| Re-arrested (y = 1) | 17.2% | | |
| Not re-arrested (y = 0) | 7.0% | | |

- **Gender asymmetry:** instability is spread evenly overall (12.1% vs 12.9%), but among
  the *selected*, women are far more often at the margin (47.5% vs 30.5%). Few women are
  selected, and those who are sit just above the cut.
- **Race is balanced** at the margin (32.2% vs 31.0%). This isn't in the deck; use it in Q&A.
- **Re-arrested people are contested 2.5× as often** (17.2% vs 7.0%). The people who most
  need support are also the most uncertain decisions.

### Abstention: the finding on slide 18

Source: `artifacts/abstention_curve.csv`. "Limit" is how many dissenting refits are
tolerated: 0 keeps only unanimous decisions, and 4 decides everyone by majority vote.

**Logistic**

| Limit | Coverage | Precision | Gender FNR gap (M − F) | Race FNR gap (B − W) | Women sent to review |
|---:|---:|---:|---:|---:|---:|
| 4 (decide all) | 100% | 0.822 | −0.090 | −0.013 | 0% |
| 3 | 98.4% | 0.826 | −0.094 | −0.012 | 1.4% |
| 2 | 95.7% | 0.830 | −0.103 | −0.013 | 4.0% |
| 1 | 92.1% | 0.838 | −0.101 | −0.005 | 7.3% |
| **0 (unanimous)** | **87.2%** | **0.846** | **−0.119** | −0.006 | 12.1% |

**XGBoost**: the same pattern. Coverage falls to 85.4%, precision rises 0.828 → 0.848,
and the gender gap goes from −0.096 to −0.123.

**Mechanism in one sentence:** abstention removes contested cases, and 47% of the few
women selected are contested against 30% of men, so women lose a bigger share of their
offers and the gap widens.

**Extra point:** abstention doesn't hurt on race. The race gap even narrows slightly
(−0.013 → −0.006) because race is balanced at the margin. So abstention is not neutral
for gender, but it is for race.

**Consequence for the client:** if they adopt abstention, the human-review queue must be
audited too. Otherwise the disparity just moves out of the model and into the queue.

## 2.4 Selected-set overlap between models

This is used on slide 19, but it's a stability-style number, so learn it here.
Source: `artifacts/selected_set_overlap.csv`.

| Pair | Jaccard | Chosen by only one | Share of cohort |
|---|---:|---:|---:|
| Logistic vs XGBoost | **0.849** | 254 | 3.25% |
| Logistic vs TabICL | 0.840 | 272 | 3.48% |
| XGBoost vs TabICL | 0.851 | 252 | 3.23% |

**Say:** "Switching model changes who gets help less than retraining the same model on
new data does." Between models J ≈ 0.85; between refits of one model J ≈ 0.75–0.78.
This comparison is not in the deck, and it's a strong line for the recommendation.

## 2.5 Performance stability: mention only

Source: `artifacts/stability.json`.
- Bootstrap AUC interval widths are ≈ 0.023 for all three models.
- Shuffling all features collapses the rank correlation to 0, which is a sanity check
  that the models use the features.
- This is performance uncertainty, not structural stability. Say so if asked.

## 2.6 Inconsistencies to know (jury traps)

| Where | Says | The data says | What to say |
|---|---|---|---|
| Report, notes | Logistic's stability is better "on every one of the 28 pairs" for **both** drift and Jaccard | Drift 28/28, **Jaccard 26/28** | "Lower drift on all 28 pairs, a more stable selected set on 26 of 28" |
| Slide 18 | "14% of decisions flip" | 12.8% (logistic), 14.6% (XGBoost) | "About 13% for logistic, 15% for XGBoost; roughly one in seven". **Fixed by the team on 27 Sep: the slide now says ≈13%.** |
| Slide 18 baseline | Gender FNR gap −0.090, precision 0.822 | The published fit has **−0.096** and **0.823** | The abstention baseline uses the *majority vote of 8 refits*, not the published fit. Both are correct for their definitions. |
| Notebook §8 (cell 37) | Jaccard 0.84 / 0.82 / 0.78 | The deck has 0.77 / 0.75 / 0.78 | The notebook compares 3 refits **with the original model** (trained on all the data). The deck compares refits **with each other**. Refit-vs-original is higher because the original is the "centre". Logistic > XGBoost in both. **TabICL differs:** last in the notebook (0.78), but level with logistic between refits (0.776 vs 0.773). The cause isn't isolated. Notebook cell 38 now explains all of this. |
| App, governance tab | Logistic has "smaller subgroup gaps" | Withdrawn: the gender-gap difference is not significant (−0.013, CI −0.036 to +0.011) | Don't repeat it. The text should be fixed before the demo; see the note in the README. |
| Report vs CSV | Captured re-arrest difference CI "−30 to +1.5" | `calibration_paired_tests.csv`: −29 to 0 | Either way it includes 0. Pick one version and use it consistently. |

## 2.7 Where each number lives

| Number | File | Made by |
|---|---|---|
| Drift, Jaccard, Spearman per model | `artifacts/stability_summary.csv` | `scripts/stability_structural.py` |
| The 28 pairs | `artifacts/stability_pairs.csv` | same |
| Explanation stability | `artifacts/stability_contributions.csv` | same |
| Resample hashes | `artifacts/stability_protocol.json` | same |
| Per person | `artifacts/individual_stability.csv` | `scripts/individual_stability.py` |
| Contested shares | `artifacts/individual_stability_summary.csv` | same |
| Abstention | `artifacts/abstention_curve.csv` | same |
| Figures | `artifacts/figures/structural_stability.png`, `individual_stability.png` | both scripts |
| Model vs model overlap | `artifacts/selected_set_overlap.csv` | `scripts/calibration_tests.py` |
