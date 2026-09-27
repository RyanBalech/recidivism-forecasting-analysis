# 6 · Slide plan and speaking script (P6)

> **Time budget.** The deck runs about 14:25 of the 15:00 limit, and P6 has 2:30 for
> slides 17–19. There is only about 35 s of buffer, so **any extra slide needs the team's
> agreement** or time taken from elsewhere. Two options are below.
>
> **How the deck is built.** `reports/ISAF_Recidivism_Presentation.pptx` is *generated* by
> `scripts/build_slides.py`. If you edit the .pptx by hand and someone later reruns the
> script, your edits are lost. Either make the changes in `build_slides.py` or agree that
> nobody regenerates the deck.
>
> "Slide N" in quotes below means the **professor's** slide number. Using his own
> references shows you learned it in class and applied it.

---

## Option A: one added slide, about 3:15 in total (needs +45 s from the team)

### Slide 17 · Would a different sample give the same model? (50 s)
- **Figure:** `results/fig_stability_regimes.png`. It replaces the current number cards.
- **On the slide:**
  - The quote: *"Two datasets from the same population should induce approximately the
    same model"* (course, slide 182).
  - "Seed → bootstrap → disjoint halves: logistic is more stable than XGBoost under all
    three."
  - "Jaccard 0.77 means 13% of the selected change, not 23%."
- **Say:**
  > "The course defines stability as: two datasets from the same population should give
  > approximately the same model. We drew those two datasets three ways, from mildest to
  > harshest. A new random seed barely moves logistic, while it swaps about 2% of XGBoost's
  > selected people. Bootstrap refits, the same eight for every model, move a person's
  > risk by about 3 points, and about 13% of the people offered support change. With two
  > completely separate halves of the data, it's about 18%. Under all three, logistic is
  > more stable than XGBoost. TabICL matches logistic on who is selected."

### Slide 18 · Stability for one person: abstaining is not fairness-neutral (50 s)
- **Figure:** `artifacts/figures/individual_stability.png`, as now.
- **Fix on the slide:** "14% of decisions flip" → **"13% (logistic) / 15% (XGBoost) of
  decisions are contested"**.
- **Add one line:** "Every contested case sits within 10 percentile points of the cut."
- **Say:**
  > "Now one person. Across the eight refits, about one decision in seven is contested,
  > and all of them sit close to the cut. As the course puts it for LLMs, small numerical
  > differences only matter when two candidates are nearly tied, and here that means near
  > the line. The natural fix is to refer contested cases to a human. That raises
  > precision from 0.822 to 0.846, but the gender gap widens from −0.090 to −0.119,
  > because 47% of the few women we select are at the margin, against 30% of men.
  > Abstaining is not fairness-neutral."

### NEW slide 18b · Slide 186 on our data: retraining without reshuffling (35 s)
- **Figure:** `results/fig_slide186_frontier.png`, or just its left panel.
- **On the slide:**
  - The formula: θ̂₂ = argmin L(θ; D₂) + λ‖θ − θ̂₁‖², λ by cross-validation (course,
    slide 186).
  - Naive retrain: 19% of the selected change. With the constraint: 12%. The coefficients
    move half as much, and accuracy doesn't drop (5 of 5 draws).
  - Footer: "Same model, another machine: logistic changes 0 decisions, XGBoost 80."
- **Say:**
  > "The course shows how to retrain while penalising the distance to the previous model,
  > with λ chosen by cross-validation. We implemented it for our logistic regression.
  > A naive retrain changes almost one in five of the people offered support. With the
  > constraint it's one in eight, the coefficients move half as much, and the accuracy
  > doesn't drop. It needs a parameter vector, so it works for logistic and not directly
  > for XGBoost. And on reproducibility: the same XGBoost on another machine changed 80
  > decisions, where logistic changed none."

### Slide 19 · Pilot logistic; XGBoost as challenger (60 s)
- Keep the trade-off matrix. Optionally show it next to the course's trade-off picture
  (predictive performance vs interpretability, with stability, fairness, frugality and
  privacy, course slide 16).
- **Rewrite these bullets:**
  - "More stable than XGBoost on all 28 refit pairs" → **"Lower drift on all 28 refit
    pairs; identical decisions on any machine; retrainable under the slide-186
    constraint"**
  - "Fairness and calibration: tied — not reasons" → **"Fairness: equivalent within ±5
    pts (TOST); calibration: no detectable difference, so not reasons"**
- **Say:**
  > "Reading across the course's dimensions. Performance: XGBoost's edge is real but
  > small, and the two models select 85% of the same people. The difference in re-arrests
  > captured includes zero. Fairness doesn't separate them: the gap difference is
  > equivalent within ±5 points. Calibration shows no detectable difference. What
  > separates them is stability, reproducibility, direct interpretability and frugality,
  > and all four favour logistic. So we recommend a shadow pilot of logistic regression,
  > with XGBoost as the challenger. It reverses if the client quotes scores as
  > probabilities, or works at a scale where a dozen extra captured re-arrests matter. And
  > no model allocates real support until the pilot shows the programme helps: it must
  > prevent at least 12% of re-arrests to pay for itself."

## Option B: no extra core time (fits the current 2:30)

- **Slide 17:** swap the number cards for `fig_stability_regimes.png` plus the slide-182
  quote (40 s).
- **Slide 18:** fix "14%" as above (50 s).
- **Slide 19:** rewrite the two bullets as above (60 s).
- **Appendix A8** "The course's stability constraint (slide 186) on our data": the
  frontier figure plus the λ table from 03 C2.
- **Appendix A9** "Stability extras": disjoint halves, 0 vs 80 decisions across machines,
  L1/L2/elastic net (`fig_penalty_stability.png`), coefficient stability.

**My suggestion:** Option A. Slide 18b is the one slide that shows the jury you took a
method from the lecture and implemented it on a new problem. If the team can't give you
45 s, use Option B and bring up A8 in Q&A.

## Figures you can use

| Figure | File | Best for |
|---|---|---|
| Three regimes | `results/fig_stability_regimes.png` | Slide 17 |
| Slide 186 frontier | `results/fig_slide186_frontier.png` | Slide 18b or A8 |
| Stability vs C | `results/fig_stability_vs_C.png` | Backup (same idea, on the L1 penalty) |
| L1 / L2 / elastic net | `results/fig_penalty_stability.png` | A9 (guest lecture, slides 43–44) |
| Coefficient stability | `results/fig_coefficient_stability.png` | A9 only (see the trap in 03, E3) |
| Per person + abstention | `artifacts/figures/individual_stability.png` | Slide 18 |
| Trade-off matrix | `artifacts/figures/tradeoff_matrix.png` | Slide 19 |

## Hand-over lines

- **From P5 (fairness) into you:** "…and that gender gap comes back in the next section,
  because stability and fairness interact."
- **From you into the demo (P1):** "Every trade-off I've described can be tested live in
  the app. Here's one person, their scores from all three models, and how many of the
  eight refits select them."
