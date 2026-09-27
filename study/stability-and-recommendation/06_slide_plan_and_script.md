# 6 · Slide plan and speaking script (P6)

> **Time budget.** The deck runs about 14:25 of the 15:00 limit, and P6 has 2:30 for
> slides 17–19. There is only about 35 s of buffer, so **any extra slide needs the team's
> agreement** or time taken from elsewhere. Two options are below.
>
> **How the deck is built.** `reports/ISAF_Recidivism_Presentation.pptx` is *generated* by
> `scripts/build_slides.py`. If you edit the .pptx by hand and someone later reruns the
> script, your edits are lost. Either make the changes in `build_slides.py` or agree that
> nobody regenerates the deck.

---

## Option A: two added slides, about 3:15 in total (needs +45 s from the team)

### Slide 17 · Would a different sample give the same model? (50 s)
- **Figure:** `results/fig_stability_regimes.png`. It *replaces* the current number cards.
- **On the slide:**
  - "Three sources of randomness (course §7): seed → bootstrap → disjoint halves"
  - "Logistic is more stable than XGBoost under all three"
  - "Jaccard 0.77 means 13% of the selected change, not 23%"
- **Say:**
  > "The course defines stability as: two samples from the same population should give
  > about the same model. We tested three sources of randomness, from mildest to
  > harshest. Changing only the random seed barely moves logistic, and it swaps about 2%
  > of XGBoost's selected people. Bootstrap refits, the same eight for every model, move
  > a person's risk by about 3 points on average, and about 13% of the people offered
  > support change. With two completely separate halves of the data, it's about 18%.
  > Under all three, logistic is more stable than XGBoost. TabICL matches logistic on the
  > selected set."

### Slide 18 · Stability for one person: abstaining is not fairness-neutral (50 s)
- **Figure:** `artifacts/figures/individual_stability.png`, as now.
- **Fix on the slide:** "14% of decisions flip" → **"13% (logistic) / 15% (XGBoost) of
  decisions are contested"**.
- **Add one line:** "Every contested case sits within 10 percentile points of the cut."
- **Say:**
  > "Now one person. Across the eight refits, about one decision in seven is contested,
  > and all of them sit close to the cut. Nobody far from the line ever changes. The
  > natural fix is to refer contested cases to a human. That raises precision from 0.822
  > to 0.846. But the gender gap widens from −0.090 to −0.119, because 47% of the few
  > women we select are at the margin, against 30% of men. Abstaining is not
  > fairness-neutral. If the client does it, the review queue itself must be audited."

### NEW slide 18b · Stability in practice: reproducibility and tuning (35 s)
- **Left panel.** Title: "Same model, same data, different machine". Content:
  - Logistic: **0** decisions change.
  - XGBoost: **80** decisions change (Jaccard 0.95).
  - Source: E6.
- **Right panel.** `results/fig_stability_vs_C.png`, captioned "A stronger L1 penalty buys
  stability for almost no accuracy (course p192)".
- **Say:**
  > "Two practical points. First, reproducibility. We re-ran the same XGBoost with the same
  > data and seed on another machine, and 80 people got a different decision. Logistic
  > was identical to the last digit. If a supervisee appeals, we must be able to
  > reproduce their exact decision. Second, the course's stability–performance
  > trade-off. We tuned the penalty for accuracy, but a slightly stronger penalty gives a
  > simpler, more stable model for 0.0006 AUC. That's how we'd tune it in the pilot."

### Slide 19 · Pilot logistic; XGBoost as challenger (60 s)
- Keep the trade-off matrix.
- Replace "More stable than XGBoost on all 28 refit pairs" with **"Lower drift on all 28
  refit pairs; identical decisions on any machine"**.
- Optional last bullet: "Break-even: support must prevent ≥ 12% of re-arrests (vs 15%
  with the current tool)".
- **Say:**
  > "Reading across the four dimensions. Performance is a tie in practice: XGBoost's
  > edge is real but small, and the two models select 85% of the same people. The
  > difference in re-arrests captured includes zero. Calibration and fairness are ties
  > too, so they are *not* our reasons. What separates the models is stability,
  > reproducibility, direct interpretability and cost, and all four favour logistic. So
  > we recommend a shadow pilot of logistic regression, with XGBoost as the challenger.
  > It reverses if the client quotes scores as probabilities, or works at a scale where
  > a dozen extra captured re-arrests matter. And no model should allocate real support
  > until the pilot shows the programme actually helps people. Our break-even says it
  > must prevent at least 12% of re-arrests to pay for itself."

## Option B: no extra core time (fits the current 2:30)

- **Slide 17:** swap the number cards for `fig_stability_regimes.png` and keep 40 s. Say
  only the bootstrap row, plus "and the ordering holds under the course's disjoint halves".
- **Slide 18:** fix "14%" as above (50 s).
- **Slide 19:** add "identical decisions on any machine" (60 s).
- **Appendix A8** "Stability extras": the C sweep, coefficient stability and the
  contested-by-distance table, for Q&A only.
- **Appendix A9** "Reproducibility": 0 vs 80 decisions.

**My suggestion:** ask the team for Option A. If they refuse, Option B still gets
everything in front of the jury through Q&A.

## Figures you can use

| Figure | File | Best for |
|---|---|---|
| Three regimes | `results/fig_stability_regimes.png` | Slide 17 |
| Stability vs C | `results/fig_stability_vs_C.png` | Slide 18b or appendix |
| Coefficient stability | `results/fig_coefficient_stability.png` | Appendix only (see the trap in E3) |
| Refit stability (original) | `artifacts/figures/structural_stability.png` | Backup |
| Per person + abstention | `artifacts/figures/individual_stability.png` | Slide 18 |
| Trade-off matrix | `artifacts/figures/tradeoff_matrix.png` | Slide 19 (has an age row; the team agreed to keep it) |

## Hand-over lines

- **From P5 (fairness) into you:** "…and that gender gap comes back in the next section,
  because stability and fairness interact."
- **From you into the demo (P1):** "Every trade-off I've described can be tested live in
  the app. Here's one person, their scores from all three models, and how many of the
  eight refits select them."
