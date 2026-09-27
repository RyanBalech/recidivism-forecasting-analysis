# 10 · Your 3-minute part: slide by slide, with the script

> **Deck to paste in:** [`slides/P6_Stability_Recommendation.pptx`](slides/P6_Stability_Recommendation.pptx)
> contains 4 talk slides (17, 18, 18b, 19) plus 2 backups (A12, A13). The same script is in
> each slide's speaker notes. Built by [`slides/build_p6_slides.py`](slides/build_p6_slides.py),
> which uses the team deck's exact design and reads every number from the CSVs. **The team's
> deck was not modified.**
>
> Script length: 431 words ≈ **2:52** at 150 words per minute, leaving ~8 s for pauses
> within 3:00.

---

## A. Check: does the team's deck match what you'll say?

I compared the merged deck (`reports/ISAF_Recidivism_Presentation.pptx`, main `f6a16de`,
32 slides) with this study pack and the script below.

| # | Where | Team deck says | Status | What to do |
|---|---|---|---|---|
| 1 | **Timing** | P6 = **2:30** (17: 40 s, 18: 50 s, 19: 60 s); the whole talk is 14:25 | ⚠️ **Mismatch** | 3:00 needs **+30 s** from the team, bringing the total to 14:55 of 15:00. If the team also adds the no-gang "re-check" slide (+40 s), the talk runs to 15:35, **over the limit**. Agree this with the team before tomorrow. |
| 2 | Slide 17 numbers | 0.032 / 0.035 / 0.035; Jaccard 0.77 / 0.75 / 0.78; "13–15%" | ✅ Match | — |
| 3 | Slide 17 card | "28 / 28 refit pairs where logistic is more stable than XGBoost" | ⚠️ Imprecise | True for **score drift**. For selection (Jaccard) it's **26/28**. My slide 17 says both. |
| 4 | Slide 17 content | Only bootstrap, with the course definition only in the notes | ➕ Missing | My slide 17 adds the course quote (slide 182) and the seed / bootstrap / halves test. |
| 5 | Slide 18 numbers | ≈13%; 32% at the margin; 0.822 → 0.846; −0.090 → −0.119; 47% vs 30% | ✅ Match | The old "14%" error is fixed in the redesign. |
| 6 | Slide 18 wording | Notes: "one decision in **eight**" | ✅ Correct | 12.8% ≈ 1 in 8. **Say "one in eight"**, not "one in seven" (older notes and my earlier docs said seven). |
| 7 | Slide 18 content | No link to gang affiliation | ➕ Missing | The asymmetry disappears without gang (33% vs 33%). It's one line on my slide 18, with the detail in A13. |
| 8 | Slide 18b | Doesn't exist | ➕ New | The course's slide-186 method on our data, plus reproducibility (0 vs 80 decisions). |
| 9 | Slide 19 wording | "**Calibration and fairness are tied** — not a reason to choose" | ⚠️ **Conflicts with the course** | Professor's slide 263: failing to reject ≠ fair. My slide says "Fairness: **equivalent within ±5 pts** (TOST); calibration: no detectable difference". |
| 10 | Slide 19 numbers | Jaccard 0.850, 254 differ, CI includes 0, ~9× faster, second arm (A11) | ✅ Match | — |
| 11 | Slide 19 notes | "performance is a tie" | ⚠️ Loose | Say "XGBoost's edge is real but small" (it's statistically significant: +0.0025 AUC). |
| 12 | Trade-off matrix figure | Includes the age row | ✅ OK | The outline still has an old "remove the age row" to-do, but the team decided on 26 Sep to keep it. |
| 13 | Appendix labels | A1–A11 used | ✅ OK | Your backups are **A12** and **A13**. |
| 14 | Notebook behind slides 17–18 | Evidence only in cell 38, **on your branch** | ⚠️ Not on main | Merge `ayush-stability` (or its notebook fix) before submission. |

**Bottom line:** every number you'll say matches the team's artifacts. The mismatches are
**timing** (#1), **one wording conflict with the course** (#9), and **content your version
adds** (#4, #7, #8). None contradicts the team's results.

## B. Your slides and script

Timing tags follow the deck's convention: `[P6 · start time · duration]`. P6 starts at 11:10
after P5's slide 16 ("…Over to [P6]").

### Slide 17 · "Would a different sample give the same model?" (45 s, 11:10–11:55)

**On the slide**
- Course definition card (slide 182)
- Table: drift 0.032 / 0.035 / 0.035 and Jaccard 0.77 / 0.75 / 0.78
- Big stat: **13%** of offers change per refit, not 23%
- Figure: seed → bootstrap → disjoint halves
- Footer: 28/28 · 26/28 · 10/10

**Script**
> The course defines stability simply: two datasets from the same population should give
> approximately the same model. We tested that three ways, from mildest to harshest.
> Changing only the random seed barely moves logistic regression; XGBoost already swaps about
> 2% of the people it selects. With eight bootstrap refits — the same eight for all three
> models — a person's risk moves by about three points, and about 13% of the people offered
> support change. Not 23%: Jaccard is not the share that changed. With two completely
> separate halves of the data, it's about 18%. In every test, logistic is more stable than
> XGBoost, and TabICL matches logistic on who gets selected.

**Point at:** the middle panel of the figure (Jaccard bars), left to right.

### Slide 18 · "Stability for one person: abstaining is not fairness-neutral" (45 s, 11:55–12:40)

**On the slide**
- Four stats: ≈13% contested (32% of those selected) · 0.822 → 0.846 · −0.090 → −0.119 ·
  47% vs 30%
- Figure: per-person stability
- Footer: all contested cases sit near the cut, and without gang it's 33% vs 33%

**Script**
> Now one person. Across the eight refits, about one decision in eight is contested — some
> refits select the person, others don't — and all of them sit close to the cut. The natural
> fix is to send contested cases to a human. That raises precision from 0.822 to 0.846. But
> the gender gap widens, from minus 0.090 to minus 0.119, because the few women we select sit
> at the margin far more often than men: 47% against 30%. So abstaining is not
> fairness-neutral — and that asymmetry comes from gang affiliation: without it, women and
> men are equally often at the margin.

**Point at:** the red stats (third and fourth), then the right panel of the figure.

### Slide 18b (new) · "Retraining without reshuffling: the course's stability constraint" (35 s, 12:40–13:15)

**On the slide**
- Formula card (course slide 186)
- Three stats: **19% → 12%** reshuffled · **−53%** coefficient movement · **No AUC loss**
  (5 of 5 draws)
- Frontier figure
- Reproducibility card: logistic 0 vs XGBoost 80 decisions changed on another machine

**Script**
> The course also shows how to fix this. When you retrain on new data, penalise the distance
> to the previous model, with lambda chosen by cross-validation. We implemented it for our
> logistic regression. A naive retrain reshuffles 19% of the people offered support; with the
> constraint, 12%. The coefficients move half as much, and accuracy doesn't drop. It needs a
> parameter vector, so it works for logistic, not XGBoost. And reproducibility: the same
> XGBoost on another machine changed 80 decisions; logistic changed none.

**Point at:** the left panel of the figure. The line first goes *down-left*: more stable
*and* better, so the naive retrain is dominated.

### Slide 19 · "Pilot logistic regression; run XGBoost as the challenger" (55 s, 13:15–14:10)

**On the slide**
- Trade-off matrix (left)
- Recommendation box: same people (J 0.850, 254 differ, CI includes 0) · more stable
  (28/28), same decisions on any machine, interpretable, ~9× faster · fairness equivalent
  within ±5 pts, calibration no detectable difference · second arm without gang (A11)
- "What would reverse it" card and limits

**Script**
> So, across the four dimensions. Performance: XGBoost's edge is real but small — the two
> models select 85% of the same people, and the difference in re-arrests captured includes
> zero. Fairness doesn't separate them: the gap difference is equivalent within five points.
> Calibration shows no detectable difference. What does separate them is stability,
> reproducibility, direct interpretability and cost — and all four favour logistic. So we
> recommend a shadow pilot of L1 logistic regression, with XGBoost as the challenger, plus a
> second arm without gang affiliation, where logistic is still the more stable model. The
> choice reverses if scores are quoted to people as probabilities, or at a scale where a
> dozen extra captured re-arrests matter. And no model allocates real support until the
> pilot shows the programme helps. Over to the demo.

### Backups: don't present, keep for Q&A
- **A12 · The course's stability toolkit:** the seed / bootstrap / halves / 50%-vs-full /
  importance distance / machine table, the honest reading (XGBoost ties on explanation
  stability), and the L1 / L2 / elastic-net check. Use for Q&A 9, 10, 11, 27 and 30 in 05.
- **A13 · No-gang arm, slides 17–18 recomputed:** use for Q33–Q35, and if the team makes
  no-gang primary.

## C. If the team makes the no-gang model primary (scenario B)

Don't change your slides. Say the extra lines in [09](09_no_gang_pilot_arm.md) §9.5 on
slides 17–18, and change the slide-19 recommendation sentence to "L1 logistic regression
**without gang affiliation** for the shadow pilot; XGBoost without gang as challenger".
**Timing warning:** the team's re-check slide (+40 s) plus your 3:00 goes over 15:00. Then
cut 18b to a single sentence on slide 19, or move it to the appendix.

## D. How to put these slides into the team deck (manually)

1. Open **both** files in PowerPoint:
   - `study/stability-and-recommendation/slides/P6_Stability_Recommendation.pptx`
   - `reports/ISAF_Recidivism_Presentation.pptx`
2. In the P6 file, select slides **17, 18, 18b, 19** in the thumbnail pane and copy them.
3. In the team deck, click below slide 16, paste, and choose **Keep Source Formatting**.
   Speaker notes come with the slides.
4. Delete the team deck's old slides 17, 18 and 19 (now right after your pasted ones). Check
   that the demo slide follows your slide 19.
5. Copy **A12** and **A13** and paste them after A11.
6. Slide-number footers are text boxes: 18b's footer reads "18b", and the following slides
   keep their numbers. Adjust by hand if the team renumbers.
7. ⚠️ If anyone reruns `scripts/build_slides.py`, the deck is regenerated and **manual
   pastes are lost**. Tell the team, or ask me to add these slides to `build_slides.py`
   instead.

**Preview caveat:** there's no PowerPoint on this Mac. I checked the layout with a preview
renderer that uses Arial, which is wider than Calibri, so anything that fits in the preview
fits in PowerPoint. Still click through the six slides once in PowerPoint before presenting.

## E. Rebuild

```bash
python study/stability-and-recommendation/slides/build_p6_slides.py
```
