# 9 · The no-gang pilot arm: what changed on 27 Sep, and your slides 17–19 in both scenarios

> **What happened.** On 27 Sep the team merged PR #10 (deck redesign, including Ziqi's PR #9).
> The recommendation on slide 19 now adds a **second pilot arm without gang affiliation**,
> with evidence in the new appendix slide **A11**. Slide 16's last line was changed to match.
> **Whether the no-gang model becomes the *primary* model is a pending team decision**
> (`reports/presentation_outline.md`, section "Open decision"). This file prepares you for
> both outcomes.
>
> Sources:
> - The team's `scripts/gang_variant_eval.py`, which writes `artifacts/gang_variant_eval.csv`
>   and `gang_variant_comparison.png`.
> - The stability and abstention numbers for the no-gang arm come from
>   [`analysis/no_gang_stability.py`](analysis/no_gang_stability.py), which writes
>   `results/no_gang_*.csv`, using the exact published protocol.

---

## 9.1 Why the team considered dropping gang affiliation (one-minute recap)

- `Gang_Affiliated` is **never recorded for women** (Cramér's V = 1.0 with gender on the raw
  field). It reflects record-keeping, not behaviour.
- The course's FPDP (slides 247–260) flags it as the variable behind the gender FNR gap.
- Nested 5-fold out-of-fold: dropping it closes ≈ 70% of the gap for ≈ 0.01 AUC (slide 16).
- The team then re-checked the fix on **all four dimensions** (A11). That's the course's
  detect → explain → mitigate, plus a re-check.

## 9.2 The team's re-check, with vs without gang (evaluation cohort)

Source: `artifacts/gang_variant_eval.csv`.

| | Logistic with | Logistic **without** | XGBoost with | XGBoost **without** |
|---|---:|---:|---:|---:|
| ROC AUC | 0.730 | **0.715** | 0.732 | 0.719 |
| Brier | 0.2054 | 0.2095 | 0.2044 | 0.2084 |
| Calibration slope | 0.999 | 0.978 | 1.005 | 0.981 |
| Re-arrests captured in top 20% | 1,285 | 1,258 | 1,294 | 1,257 |
| Net value at 20% | $5.045M | $4.775M | $5.135M | $4.765M |
| Gender FNR gap (M − F) | −0.096 | **+0.000** | −0.114 | −0.012 |
| Race FNR gap (B − W) | −0.005 | **+0.026** (CI 0.000 to 0.054) | −0.018 | +0.018 |
| Age FNR gap (<33 − 33+) | −0.245 | −0.216 | −0.228 | −0.197 |
| Same people as the with-gang model (Jaccard) | | **0.61** | | 0.60 |

**Reading:**
- The gender gap closes.
- The race gap flips sign but stays within ±5 points (TOST).
- AUC falls 0.014, and 27 fewer re-arrests are captured.
- It still beats the historical score by far: $4.8M vs $2.7M.
- Break-even effectiveness ≈ 12.4% (no gang) vs 14.8% (historical score).
- About a quarter of support places go to different people (Jaccard 0.61). That's
  intended: the gender selection gap falls from 0.095 to 0.039.

## 9.3 Your slides 17–18 for the no-gang model (new: published protocol)

The team's script refits XGBoost with a *different seed per refit*, while slide 17's audit
fixes the seed at 42. So we recomputed everything with the slide-17 protocol: the same 8
resamples (seed 7) and model seed 42. The **with-gang logistic rows reproduce the published
slide 17–18 numbers exactly** (the script checks this).

### Slide 17 (structural), no-gang
Source: `results/no_gang_stability_summary.csv` and `no_gang_stability_pair_wins.csv`.

| | Logistic with | Logistic **without** | XGBoost with | XGBoost **without** |
|---|---:|---:|---:|---:|
| Mean \|Δp\| between refits | 0.0322 | **0.0319** | 0.0349 | 0.0349 |
| Top-20% Jaccard between refits | 0.7725 | **0.7536** | 0.7488 | 0.7227 |
| Selected people replaced per refit | 12.8% | 14.1% | 14.4% | 16.1% |
| Logistic more stable than XGBoost (pairs) | drift 28/28 · J 26/28 | **drift 28/28 · J 27/28** | | |

- The team's own numbers (0.032 / 0.75 and 0.035 / 0.72) are confirmed. The seed
  difference changes XGBoost's no-gang Jaccard by only 0.002 (0.7209 vs 0.7227).
- Dropping gang **slightly lowers** selection stability for both models (−0.02 to −0.03
  Jaccard). Scores drift just as much.
- **Logistic beats XGBoost even more consistently** without gang (27/28 pairs). Your
  slide-19 argument survives.

### Slide 18 (one person + abstention), no-gang
Sources: `results/no_gang_stability_summary.csv`, `no_gang_contested_by_gender.csv` and
`no_gang_abstention_curve.csv`. This was **never computed by the team**.

| Logistic | With gang (slide 18 today) | **Without gang** |
|---|---:|---:|
| Contested decisions | 12.8% | 13.9% |
| Of those selected, contested | 31.7% | 33.0% |
| Women selected (majority of refits) | 118 | **162** |
| Selected women at the margin | **47%** | **33%** |
| Selected men at the margin | 30% | 33% |
| Abstention: precision | 0.822 → 0.846 | 0.808 → 0.824 |
| Abstention: gender FNR gap (M − F) | **−0.090 → −0.119** (widens) | **+0.004 → −0.002** (≈ 0, unchanged) |
| Abstention: race FNR gap (B − W) | −0.013 → −0.006 | +0.018 → +0.016 |

**This is the most important new finding for your part:**
- Slide 18's result, "abstention is not fairness-neutral: women sit at the margin 47% vs
  30%", **is caused by gang affiliation**.
- Without the field, more women are selected (162 vs 118). Those selected are no more
  often at the margin than men (33% vs 33%), and abstention leaves the gender gap at
  about 0.
- The FNR gap figures are point estimates on a few hundred women with no confidence
  interval here, so say "about zero", not "equal".

XGBoost without gang: 16.4% contested; margin 36% (women) vs 42% (men); abstention moves
the gap from −0.010 to +0.016.

## 9.4 Scenario A: the deck as merged (with gang primary, no-gang as second arm)

**Slides 17–18: no change.** Slide 18 already says ≈13%, and the team fixed the old "14%".

**Slide 19: one added sentence (≈ 5 s).**
> "And the pilot runs a second arm without gang affiliation, the field never recorded for
> women. Appendix A11 shows it on all four dimensions. Logistic is still the more stable
> model there."

**Q&A line on slide 18 (use it if asked "so what do you do about abstention?"):**
> "The margin asymmetry comes from gang affiliation. In the no-gang arm, the selected women
> are no more often at the margin than men, and abstention leaves the gender gap at about
> zero. That's one more reason the second arm exists."

## 9.5 Scenario B: the team makes the no-gang model primary

The outline says slides 1–15 and 17–18 stay as they are (they describe the baseline where
the problem was found), a new "re-check" slide goes between 18 and 19, and slide 19 changes.

**Slide 17 (one extra line, spoken):**
> "Without gang affiliation, the picture is the same: logistic drifts less than XGBoost on
> all 28 pairs and keeps a more stable selected set on 27 of 28."

**Slide 18 (one extra line, spoken). It closes the loop with fairness:**
> "The margin asymmetry you see here comes from gang affiliation. In the no-gang model,
> selected women are at the margin as often as men, 33% each, and abstention no longer
> widens the gender gap."

**New re-check slide (≈ 40 s, owned by P5 or P6; agree with the team):** use the table in
9.2 plus the stability row from 9.3.

**Slide 19, rewritten:**
> "L1 logistic regression **without gang affiliation** for the shadow pilot; XGBoost without
> gang as the challenger. We accept 0.014 AUC and 27 fewer captured re-arrests to close the
> gender gap. Stability barely moves, and logistic is still the more stable model."

## 9.6 Q&A for both scenarios

**"You drop a predictive variable. Isn't that throwing away accuracy?"**
"0.014 AUC and 27 of 1,285 captured re-arrests. In return, the gender FNR gap goes from
−0.096 to about zero. It still beats the historical score by $2M in the scenario. And
the field measures record-keeping for women, not behaviour."

**"Does dropping it make the model less stable?"**
"Slightly. Selection Jaccard goes from 0.77 to 0.75 for logistic, with the same score
drift. Logistic still beats XGBoost on 28/28 pairs for drift and 27/28 for selection."

**"Does your abstention finding still hold?"**
"Not for gender. The margin asymmetry came from gang affiliation. Without it, 33% of
selected women and 33% of selected men are contested, and abstention leaves the gender gap
at about zero. That's a nice consistency check between our fairness and stability
sections."

**"Why did you recompute the team's stability numbers?"**
"The re-check script varied XGBoost's seed per refit, while slide 17 fixes it. Under the
slide-17 protocol the numbers barely change (XGBoost Jaccard 0.721 vs 0.723), so the team's
conclusion holds."

**"The race gap flips sign. Is that a problem?"**
"It moves from −0.005 to +0.026, still within the ±5-point tolerance (TOST). Its CI just
touches zero (0.000 to 0.054), so we monitor it in the pilot."

**"Why not keep gang and fill it differently?"** The team's answer is in
`reports/presentation_notes.md`, under "Couldn't you keep gang affiliation and just fill it
differently?". Neither alternative beats dropping it, and a "not recorded" category *is* the
gender marker.

## 9.7 Rerun

```bash
python scripts/gang_variant_eval.py                                            # team's re-check (writes artifacts/)
python study/stability-and-recommendation/analysis/no_gang_stability.py        # slides 17–18 for no-gang (~1–2 min)
```
