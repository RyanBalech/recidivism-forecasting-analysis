# 3 · Beyond the deck: course-aligned stability extras

> None of this is in the team's deck. It fills the gap `docs/PLAN.md` still marks as open
> (item P1.9: "Stability, aligned with the course §7"). All numbers come from
> [`analysis/stability_course_aligned.py`](analysis/stability_course_aligned.py), which
> runs on CPU in about 2.5 minutes. Results are in [`results/`](results/).
>
> **Sanity checks passed:**
> - The 8 bootstrap resamples are byte-identical to the published protocol (SHA-256 match).
> - Logistic reproduces the published predictions exactly (max difference 4 × 10⁻¹⁵).
> - The bootstrap numbers reproduce slides 17–18.
>
> TabICL is not refitted here, because it needs the GPU.

---

## E1 · Three sources of instability, from mildest to harshest

The course (§7.1) defines stability as "two datasets from the same population → about the
same model", and §7.2 adds that randomness also comes from seeds and software. The deck
tests only one source (bootstrap). Here are all three, on the same scale.

Source: `results/stability_by_regime_summary.csv` and `results/fig_stability_regimes.png`.

| | Same data, new seed | Bootstrap (deck) | **Disjoint halves** (course definition) |
|---|---:|---:|---:|
| Pairs compared | 28 | 28 | 10 |
| People shared between the two training sets | 100% | ≈ 40% | **0%** |
| **Logistic** mean \|Δp\| | 0.0001 | 0.032 | 0.042 |
| XGBoost mean \|Δp\| | 0.006 | 0.035 | 0.046 |
| **Logistic** top-20% Jaccard | 0.999 | 0.772 | 0.702 |
| XGBoost top-20% Jaccard | 0.956 | 0.749 | 0.683 |
| Logistic: selected people swapped | 0.07% (≈ 1) | 12.8% (≈ 200) | 17.5% (≈ 273) |
| XGBoost: selected people swapped | 2.3% (≈ 36) | 14.4% (≈ 224) | 18.8% (≈ 293) |
| Logistic better than XGBoost, drift | 28/28 | 28/28 | 9/10 |
| Logistic better than XGBoost, Jaccard | 28/28 | 26/28 | 10/10 |

**Three things to say:**
1. **Data is the main source of instability, not the algorithm.** XGBoost's own random
   seed moves scores by about a sixth as much as a new data sample does. Logistic has
   essentially no seed randomness, because its solver is deterministic.
2. **Under the course's own definition (disjoint halves), both models lose stability, and
   logistic stays ahead.** It has lower drift on 9 of 10 splits and a more stable selected
   set on 10 of 10.
3. **Half the data costs little accuracy** (evaluation AUC 0.728 logistic, 0.730 XGBoost).
   So instability, not accuracy, is what a smaller agency should worry about.

## E2 · The course's distances, and one honest surprise

### ‖θ₁ − θ₂‖₂ on logistic coefficients

| | Seed-only | Bootstrap | Halves |
|---|---:|---:|---:|
| ‖θ₁ − θ₂‖₂ | 0.31 | 0.70 | 0.90 |
| Relative to ‖θ‖ | 15% | 31% | 43% |
| Same zero / non-zero pattern | 91% | 79% | 71% |
| Same sign (where both non-zero) | 100% | 90% | 89% |

**Key insight: the parameter distance can overstate instability.** Under seed-only, the
predictions are *identical* (|Δp| = 0.0001), yet ‖θ₁ − θ₂‖₂ = 0.31. That's because our
logistic uses a **full one-hot encoding**, which gives both the "Yes" and "No" columns and
every age band alongside an intercept. So several different coefficient vectors produce
exactly the same predictions (the parameters aren't *identified*). L1 then breaks the tie
arbitrarily, depending on the solver's order.

**Say:** "The course's ‖θ₁ − θ₂‖₂ is meaningful only when the parameters are identified.
With a full one-hot encoding they aren't, so we judge stability on predictions,
decisions and the size and sign of the large coefficients."

### ‖φ(f₁) − φ(f₂)‖₂ on importance vectors: XGBoost is *not* worse here

Importance is taken as shares of total SHAP importance, so both models share a 0 to √2
scale.

| | Seed-only | Bootstrap | Halves |
|---|---:|---:|---:|
| Logistic | 0.0002 | 0.056 | 0.067 |
| XGBoost | 0.005 | 0.054 | 0.063 |
| Logistic lower (pairs) | 28/28 | 11/28 | 1/10 |

**Be honest about this.** On explanation stability, XGBoost ties or is slightly *better*.
Logistic's stability advantage is in **predictions and decisions**, not in how stable the
importance ranking is. Both models keep the same top-8 drivers in every refit (see 2.2).

**Say:** "Both models tell the same story every time. Where they differ is who gets
selected, and there logistic is more stable."

## E3 · Does the white-box story survive resampling?

Source: `results/logistic_coefficient_stability.csv` and
`results/fig_coefficient_stability.png` (8 bootstrap refits).

- Of the **108** coefficient columns, 95 are non-zero in the published fit. **52** are
  non-zero in every refit, 55 in some, and 1 in none.
- The **15 largest** coefficients are non-zero in every refit.
- The **22 largest** (|coef| ≥ 0.17) never change sign (21 of 22 non-zero in all 8 refits).
- **30** coefficients change sign at least once, and *all* of them are tiny (|coef| ≤ 0.055
  in the published fit). Several are Yes/No twins of the same field, which is the
  identifiability effect from E2.

**Say:** "Age, gang affiliation, prior felony arrests and parole revocations keep the same
direction and roughly the same size in every refit. What moves are the small coefficients
that L1 switches on and off."

⚠️ **Jury trap if you show this figure:** "Prior felony arrests = 0" has a *positive*
coefficient (+0.41). That's not "no felonies means higher risk". Each level's coefficient
is conditional on every other field, and with a full one-hot encoding there's no fixed
reference level, so single-level coefficients can't be read alone. For magnitudes, use the
probability contrasts in `artifacts/probability_contrasts.csv`. Safer: keep this figure
for Q&A, not the main slide.

## E4 · Stability vs performance over the L1 penalty (course p192)

Source: `results/logistic_stability_vs_C.csv` and `results/fig_stability_vs_C.png`.
Performance is 5-fold CV **inside the training data only**, so no evaluation labels are
used.

| C | CV AUC | Non-zero coefficients | Mean \|Δp\| | Top-20% Jaccard |
|---:|---:|---:|---:|---:|
| 0.005 | 0.702 | 16 | 0.015 | 0.845 |
| 0.01 | 0.718 | 27 | 0.020 | 0.824 |
| 0.02 | 0.728 | 46 | 0.024 | 0.809 |
| **0.05** | **0.7318** | **67** | **0.028** | **0.785** |
| 0.1 | 0.7323 | 80 | 0.031 | 0.778 |
| **0.2154 (chosen)** | **0.7324** | **89** | **0.032** | **0.772** |
| 1 | 0.7323 | 97 | 0.034 | 0.768 |
| 5 | 0.7322 | 100 | 0.034 | 0.767 |

**Reading:**
- A stronger penalty (smaller C) gives a sparser, more stable model, but below C ≈ 0.05
  the AUC falls quickly. This is the course's trade-off, measured.
- The team chose C by AUC alone. **C = 0.05** costs 0.0006 CV AUC (about an eighth of one
  fold's standard deviation of 0.005). In return it gives 22 fewer coefficients, 12% less
  drift and +1.3 points of Jaccard.
- **Be modest:** that's about 13 fewer people swapped per retrain. It's a pilot tuning
  option, not a reason to change the published model the day before the presentation.

**Say:** "We tuned C for accuracy. The course's stability–performance curve shows a
slightly stronger penalty buys a simpler and more stable model for almost no accuracy.
That's how we'd tune it in the pilot."

## E5 · Instability lives only at the margin

Source: `results/contested_profile.csv` (distance from the top-20% cut, in percentiles of
the average score).

| Distance to the cut | People | Contested (logistic) | Contested (XGBoost) |
|---|---:|---:|---:|
| within ±2 pts | 312 | **100%** | 99.7% |
| 2–5 pts | 468 | 92% | 92% |
| 5–10 pts | 782 | 31% | 43% |
| 10–20 pts | 1,561 | 1.3% | 3.8% |
| beyond 20 pts | 4,684 | **0%** | 0.04% |

**Say:** "Nobody far from the cut ever changes decision. Every contested case lies within
about 10 percentile points of the line. So the practical fix is a *review band* around
the cut, not distrust of the whole model."

This also explains the gender finding on slide 18. The few women selected cluster just
above the cut, which is exactly where decisions are contested.

## E6 · Software and hardware as a source of instability (course §7.2)

We re-ran the published configuration on a different machine: macOS ARM with 10 CPUs,
scikit-learn 1.6.1 and XGBoost 2.1.4. The published run used the team's RTX 4050 laptop with 22
logical CPUs, Python 3.13, scikit-learn 1.7.2 and XGBoost 3.0.5 (`artifacts/validation_manifest.json`).

| | Logistic | XGBoost |
|---|---:|---:|
| AUC here vs published | 0.72983 vs 0.72983 | 0.73237 vs 0.73236 |
| Largest change in one person's score | **0.000000** | 0.036 |
| Selection decisions that changed | **0** | **80 people** (Jaccard 0.95) |

The team's deep review found the same effect from the thread count alone (0.035,
`artifacts/deep_review/thread_sensitivity.json`).

**Say:** "Same model, same data, same seed, and 80 people get a different decision because
the software version or CPU count changed. Logistic gives identical decisions anywhere. For
an auditable public-sector tool, where an appeal needs to reproduce the exact decision,
that matters."

This is the cleanest *new* argument for the recommendation, and it's a stability argument.

---

## How to rerun

```bash
python -m pip install -r requirements.txt          # repo environment
python study/stability-and-recommendation/analysis/stability_course_aligned.py
```

It only writes to `study/stability-and-recommendation/results/`, and it never touches
`artifacts/` or the team pipeline. The run info (versions, CPU count, checks) is in
`results/run_info.json`.
