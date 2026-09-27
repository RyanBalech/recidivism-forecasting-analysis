# 3 · Beyond the deck: course-aligned stability extras

> None of this is in the team's deck. It fills the gap `docs/PLAN.md` still marks as open
> (item P1.9: "Stability, aligned with the course §7"), and it follows the professor's
> slides 182–194. Part E comes from
> [`analysis/stability_course_aligned.py`](analysis/stability_course_aligned.py), about
> 2.5 minutes on CPU. **Part C applies the professor's own methods** and comes from
> [`analysis/course_stability_methods.py`](analysis/course_stability_methods.py), about
> 1 minute. Results are in [`results/`](results/).
>
> **Sanity checks passed:**
> - The 8 bootstrap resamples are byte-identical to the published protocol (SHA-256 match).
> - Logistic reproduces the published predictions exactly (max difference 4 × 10⁻¹⁵).
> - The bootstrap numbers reproduce slides 17–18.
>
> TabICL is not refitted here, because it needs the GPU.

---

## E1 · Three sources of instability, from mildest to harshest

Slide 182: "two datasets from the same population → approximately the same model". There
are several ways to draw those two datasets, and the deck uses only one (bootstrap). Here
are three, on the same scale. The seed row also reflects slide 207 on seeds.

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

## E4 · Stability vs performance over the L1 penalty (the same kind of frontier as slide 192)

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

## E6 · Software and hardware as a source of instability (slides 210–215)

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
In the professor's words (slide 215): "two machines loading the same weights can disagree…
the real divide is a controlled environment versus an unobservable one". And (slide 212):
most last-digit differences are harmless; "the problem arises when the two leading
candidates are nearly tied". For us, "nearly tied" means people near the top-20% cut (E5).

## Part C · The professor's own methods, applied to recidivism

> This is the "we saw it on the slide, then implemented it from scratch on a new dataset"
> section. Script: [`analysis/course_stability_methods.py`](analysis/course_stability_methods.py).
> Results: `results/course_*.csv`, `fig_slide186_frontier.png`, `fig_penalty_stability.png`.

### C1 · Retraining on more data: a 50% sample vs the full data (slides 185, 187–188, 191)

The lecture compares a tree trained on a random 50% of the data with one trained on all
of it. We did the same with our two production models, on 5 random halves (D₁ = 50%,
D₂ = all 18,028 records, so n₂ > n₁).

| | Logistic | XGBoost |
|---|---:|---:|
| Evaluation AUC: 50% model → full model | 0.728 → 0.730 | 0.730 → 0.732 |
| Mean \|Δp\| between the two | **0.023** | 0.025 |
| Top-20% Jaccard between the two | **0.830** (9.3% of selected swap) | 0.813 (10.3%) |
| Logistic more stable, per half | Jaccard 4/5 (1 exact tie) · \|Δp\| 5/5 | |
| ‖θ̂₁ − θ̂₂‖₂ (relative) | 0.61 (29%) | n/a (1,196 trees) |
| ‖φ(f₁) − φ(f₂)‖₂ (slide 193) | 0.045 | **0.040** |

**Say:** "As on slide 187, we trained on half the data and then on all of it. The accuracy
barely moves, but about 9% of the people selected change with logistic and 10% with
XGBoost. The explanation vector is equally stable for both."

### C2 · Stability-constrained re-estimation: slide 186 on our logistic model

**The professor's method:**
$$\hat\theta_2 = \arg\min_\theta \; \underbrace{\textstyle\sum_{i \in D_2} \text{NLL}_i(\theta)}_{\text{fit the new data}} \;+\; \underbrace{\lambda\,\|\theta - \hat\theta_1\|_2^2}_{\text{don't move far from the old model}} \qquad \lambda \text{ by cross-validation}$$

**Our implementation:**
- The logistic negative log-likelihood replaces the slide's squared error, because our
  target is binary.
- D₁ and D₂ are **separate** datasets with n₂ > n₁ (40% / 60% of training), as on slide
  185, so the cross-validation inside D₂ never sees the data behind θ̂₁.
- The old model's feature pipeline is reused, so θ̂₁ and θ̂₂ are in the same coordinates.
- A light ridge term (γ = 1, the same for both) keeps the one-hot coefficients identified.
- λ is chosen by 5-fold CV log loss inside D₂.
- Everything is repeated on 5 random draws.

**The λ path, averaged over 5 draws** (`course_slide186_lambda_path.csv`):

| λ | CV AUC (inside D₂) | ‖θ̂₂ − θ̂₁‖₂ | Jaccard with old model | Selected people swapped | Evaluation AUC |
|---:|---:|---:|---:|---:|---:|
| 0 (naive retrain) | 0.7309 | 0.93 | 0.685 | 18.7% | 0.7277 |
| 30 | 0.7326 | 0.63 | 0.728 | 15.7% | 0.7287 |
| **100 (chosen by CV in 4 of 5 draws)** | **0.7333** | **0.44** | **0.785** | **12.1%** | **0.7291** |
| 300 (chosen in 1 of 5) | 0.7330 | 0.25 | 0.853 | 7.9% | 0.7287 |
| 1,000 | 0.7318 | 0.11 | 0.926 | 3.8% | 0.7274 |
| 100,000 (keep the old model) | 0.7302 | 0.00 | 0.999 | 0.0% | 0.7257 |

**Reading it the way slide 192 does** (`fig_slide186_frontier.png`):
- From λ = 0 to λ ≈ 100, **both** losses fall. The naive retrain is *dominated*: it is
  less stable and no more accurate.
- Beyond λ ≈ 100 you are on the frontier, where more stability costs a little accuracy.
- At the CV-chosen λ, in **all 5 draws**, coefficient distance falls by about 53%,
  Jaccard rises by about 10 points, and CV AUC and evaluation AUC do not fall.
- **Slide 191 comparison.** The professor's example paid 4.6% of predictive power for
  +38% stability. Ours pays nothing at λ = 100. At λ = 300, where retrains swap only 8%
  of the selected set, the cost is still negligible (−0.0004 evaluation AUC vs λ = 100).

**Honest caveat: how does this compare with just pooling D₁ and D₂?**
- If the old data can be reused, retraining on D₁ ∪ D₂ gives evaluation AUC 0.7295,
  Jaccard with the old model 0.794, and ‖Δθ‖ 0.59. That's about the same as the anchor at
  λ = 100.
- So the anchor's accuracy gain over the naive retrain mostly comes from carrying D₁'s
  information forward.
- What the anchor adds is a **dial**. At λ = 300 it is clearly more stable than pooling
  (0.853 vs 0.794) for 0.0008 AUC.
- It also works when the old records **can't** be reused, for example because retention
  rules require deletion (the "data privacy" axis on slide 16) or because the agency wants
  to fit the newest cohort.

**Say:**
> "Slide 186 shows how to retrain a model while penalising the distance to the previous
> one, with λ chosen by cross-validation. We implemented it for our logistic regression.
> Without the constraint, a retrain changes almost 19% of the people offered support.
> With λ chosen by cross-validation it changes 12%, the coefficients move half as much,
> and accuracy doesn't drop. That's how we'd run the client's quarterly retrain."

**Why this matters for the recommendation:** slide 186 needs a parameter vector θ. It
applies directly to logistic regression. XGBoost would need the importance-based version
(slide 194), which requires a custom training objective. So the white-box model is also
the one we can *keep* stable in production.

### C3 · Which penalty stabilises the coefficients? (guest lecture, slides 43–44)

The guest lecture says standard logistic regression is "unstable under collinearity",
**Lasso (L1) gives sparsity, Ridge (L2) gives stability, and Elastic net gives both**, with
the penalty chosen by cross-validation. We refitted each on the published 8 bootstrap
resamples, choosing C by 5-fold CV (`course_penalty_stability.csv`).

| | L1 (published) | L2 (ridge) | Elastic net (α = 0.5) |
|---|---:|---:|---:|
| C chosen by CV | 0.2154 | 0.1 | 0.1 |
| CV AUC | 0.7324 | 0.7323 | 0.7324 |
| Non-zero coefficients (of 108) | 89 | 108 | **84** |
| Relative ‖Δθ‖ between refits | 0.315 | **0.284** | 0.304 |
| Same sign, where both non-zero | 90% | 83% | **95%** |
| Top-20% Jaccard across refits | 0.772 | 0.768 | **0.773** |

**Reading:**
- The guest lecture's claim holds, modestly.
  - L2 reduces coefficient drift by about 10%, but keeps every coefficient, including
    tiny noisy ones whose sign flips (83% sign agreement).
  - Elastic net is the sparsest *and* has the most sign-stable coefficients, at the same
    AUC.
- **Who is selected barely changes** across penalties (Jaccard 0.768–0.773). The penalty
  shapes the *story the coefficients tell*, not the decisions.

**Say:** "As the guest lecture predicted, ridge moves the coefficients less and elastic net
keeps them sparse and sign-stable. The people selected are the same either way. For the
pilot, elastic net with the slide-186 anchor would be our tuning."

---

## How to rerun

```bash
python -m pip install -r requirements.txt          # repo environment
python study/stability-and-recommendation/analysis/stability_course_aligned.py   # Part E, ~2.5 min
python study/stability-and-recommendation/analysis/course_stability_methods.py    # Part C, ~1 min
```

It only writes to `study/stability-and-recommendation/results/`, and it never touches
`artifacts/` or the team pipeline. The run info (versions, CPU count, checks) is in
`results/run_info.json`.
