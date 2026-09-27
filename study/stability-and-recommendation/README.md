# Study pack: Stability and Recommendation (P6)

**Presenter:** Ayush · **Deck slides:** 17–19 (and optionally an added 18b) · **Talk time:**
2:30 in the current plan · **Presentation:** Mon 28 Sep 2026

This folder has everything needed to present and defend the stability section and the final
recommendation:
- the team's results, verified against the raw files;
- the professor's stability methods (course slides 182–194) **implemented from scratch on
  the recidivism data**;
- a slide-by-slide map to the course deck, a Q&A bank, and a slide plan with a speaking
  script.

It doesn't change the team pipeline, `artifacts/` or the deck.

---

## Study order for tonight (≈ 3 hours)

| # | File | Time | What you get |
|---|---|---|---|
| 1 | [01_stability_concepts_and_maths.md](01_stability_concepts_and_maths.md) | 30 min | The course definition, bootstrap vs halves, every formula (Jaccard → replaced share, ‖θ₁ − θ₂‖₂, ‖φ₁ − φ₂‖₂), worked examples |
| 2 | [02_what_we_found.md](02_what_we_found.md) | 30 min | The deck's numbers, re-verified; who is contested; abstention; **7 inconsistencies to know** |
| 3 | [03_beyond_the_deck_course_aligned_extras.md](03_beyond_the_deck_course_aligned_extras.md) | 45 min | **New results.** Part C: the professor's methods on our data (50% vs full, **slide-186 stability constraint**, L1/L2/elastic net). Part E: seed / bootstrap / halves, the course's distances, stability vs C, instability only near the cut, 0 vs 80 decisions across machines |
| 4 | [04_recommendation.md](04_recommendation.md) | 30 min | The five-step argument, the trade-off matrix row by row, reversal conditions, break-even economics, deployment gates |
| 5 | [05_qa_bank.md](05_qa_bank.md) | 30 min | 25 questions with short answers, plus the traps list |
| 6 | [06_slide_plan_and_script.md](06_slide_plan_and_script.md) | 15 min | Option A (+1 slide) or B (appendix), word-for-word script, timings |
| 7 | [07_code_walkthrough.md](07_code_walkthrough.md) | 15 min | Which script, notebook cell and app tab does what, with line numbers |
| 9 | [09_no_gang_pilot_arm.md](09_no_gang_pilot_arm.md) | 20 min | **New on 27 Sep:** the team added a no-gang second pilot arm (slide 19, A11). Slides 17–19 for both scenarios (second arm / primary), plus the **no-gang abstention result** |
| 10 | [10_p6_slides_and_script_3min.md](10_p6_slides_and_script_3min.md) | 20 min | **Your final 3:00 talk:** check of the team deck vs your script (14 items), slide-by-slide content, word-for-word script, how to paste [`slides/P6_Stability_Recommendation.pptx`](slides/P6_Stability_Recommendation.pptx) into the deck |
| 8 | [08_professor_slides_map.md](08_professor_slides_map.md) | 20 min | **The professor's slides → your section**: definitions, formulas and phrases with slide numbers (§7.1, §7.2, guest lecture, §8.3) |

**Short on time?** Read 08, then the cheat sheet below, then 09, then 03 part C, then 05.
**Rehearsing?** Use 10 and the speaker notes in `slides/P6_Stability_Recommendation.pptx`.

## Cheat sheet: memorise these

**Stability (8 bootstrap refits; the same resamples for all three models; model seed fixed)**
- Score drift, mean |Δp|: **0.032** logistic / 0.035 XGBoost / 0.035 TabICL.
- Top-20% Jaccard: 0.77 / 0.75 / 0.78. That means **≈ 13% of the selected change**, not
  23%, because (1 − J)/(1 + J).
- Logistic vs XGBoost: lower drift on **28/28** pairs, higher Jaccard on **26/28**.
  Against TabICL it's a tie on the selected set.
- Contested decisions: **13%** logistic / 15% XGBoost. About a third of the selected are
  contested.
- Abstention (unanimous decisions only): coverage 87%, precision **0.822 → 0.846**, gender
  FNR gap **−0.090 → −0.119**.
  - 47% of the 118 women selected are at the margin, against 30% of the 1,490 men.
  - Race is balanced at the margin (32% vs 31%).

**The professor's methods on our data (03, part C)**
- **Slide 186** (retrain on D₂, penalise ‖θ − θ̂₁‖², λ by CV):
  - A naive retrain swaps **19%** of the selected people; with the constraint, **12%**.
  - ‖Δθ‖ is halved, and CV AUC doesn't fall (0.731 → 0.733). This holds in 5/5 draws.
  - Caveat: pooling old and new data is similar; the anchor is a dial and works without
    keeping old records.
- **Slides 187/191** (50% sample vs full data): 9% (logistic) vs 10% (XGBoost) of the
  selected people change.
- **Slides 43–44:** ridge reduces coefficient drift by 10%, and elastic net is the sparsest
  and most sign-stable. Who is selected doesn't change (Jaccard 0.77 for all three).

**Other extras (03, part E)**
- Seed-only: logistic Jaccard 0.999, XGBoost 0.956. Disjoint halves: 0.70 vs 0.68, and
  logistic is ahead on 10/10 splits.
- Explanation distance ‖φ₁ − φ₂‖₂: XGBoost ties or is slightly *better*. Logistic's
  advantage is in decisions.
- ‖θ₁ − θ₂‖₂ overstates instability: with full one-hot encoding the parameters aren't
  identified.
- The 22 largest coefficients never change sign.
- Stability vs C: C = 0.05 costs 0.0006 CV AUC for +1.3 points of Jaccard and 22 fewer
  coefficients.
- All contested cases sit within ≈ 10 percentile points of the cut, and no one beyond 20
  points ever changes.
- **On another machine: 0 decisions change for logistic, 80 for XGBoost.**

**No-gang pilot arm (09; added to the deck on 27 Sep; primary or not is a team decision)**
- Without gang affiliation: AUC 0.730 → 0.715, gender FNR gap −0.096 → **0.000**, race
  +0.026 (still within ±5 pts), and 27 fewer re-arrests captured.
- Stability under the slide-17 protocol: logistic 0.032 / J 0.754, XGBoost 0.035 / J 0.723.
  Logistic is more stable on 28/28 (drift) and 27/28 (Jaccard) pairs.
- **Slide 18 changes:** without gang, 33% of selected women and 33% of selected men are at
  the margin (47% vs 30% with gang). Abstention leaves the gender gap at about 0
  (+0.004 → −0.002), so the "not fairness-neutral" result comes from gang affiliation.

**Recommendation**
- **Shadow pilot of L1 logistic, with XGBoost as challenger.**
- XGBoost: +0.0025 AUC [0.0006, 0.0044], but the two share 85% of the selected people
  (J 0.849; only 254 differ) and the captured-re-arrests CI includes 0.
- Logistic: more stable, reproducible, directly interpretable, 9× faster.
- Fairness: **equivalent within ±5 points** (TOST; the CI [−0.036, +0.011] sits inside ±0.05).
  Calibration: no detectable difference. So neither is a reason. Don't say "equal because
  not significant" (slide 263).
- It reverses if scores are quoted as probabilities, at large scale, or with a new feature
  set.
- Break-even: support must prevent **≥ 12%** of re-arrests (15% with the current tool).

## Folder map

```
study/stability-and-recommendation/
├── README.md                        ← you are here
├── 01_stability_concepts_and_maths.md
├── 02_what_we_found.md
├── 03_beyond_the_deck_course_aligned_extras.md
├── 04_recommendation.md
├── 05_qa_bank.md
├── 06_slide_plan_and_script.md
├── 07_code_walkthrough.md
├── 08_professor_slides_map.md       ← course slides → your section
├── 09_no_gang_pilot_arm.md
├── 10_p6_slides_and_script_3min.md  ← final 3:00 slides + script + deck check
├── slides/
│   ├── build_p6_slides.py           ← builds the separate P6 deck (team deck untouched)
│   └── P6_Stability_Recommendation.pptx  ← slides 17, 18, 18b, 19 + backups A12, A13
├── analysis/
│   ├── stability_course_aligned.py  ← part E extras (CPU, ~2.5 min)
│   ├── course_stability_methods.py  ← part C: slides 186/187/43–44 on our data (CPU, ~1 min)
│   └── no_gang_stability.py         ← slides 17–18 for the no-gang arm, published protocol (CPU, ~1–2 min)
└── results/
    ├── stability_by_regime_summary.csv     seed / bootstrap / halves × model
    ├── stability_pairs_by_regime.csv       every pair, every distance
    ├── logistic_coefficient_stability.csv  per coefficient across 8 refits
    ├── logistic_stability_vs_C.csv         L1-penalty stability/accuracy frontier
    ├── course_retrain_50pct_vs_full.csv    slides 187/191 replication
    ├── course_slide186_lambda_path.csv     slide 186: every λ, every draw
    ├── course_slide186_chosen.csv          slide 186: CV-chosen λ vs naive retrain
    ├── course_penalty_stability.csv        slides 43–44: L1 vs L2 vs elastic net
    ├── no_gang_stability_summary.csv       no-gang vs with-gang: drift, Jaccard, contested, abstention
    ├── no_gang_stability_pair_wins.csv     logistic vs XGBoost per refit pair, both variants
    ├── no_gang_abstention_curve.csv        slide 18's abstention curve for both variants
    ├── no_gang_contested_by_gender.csv     who sits at the margin, by gender and variant
    ├── contested_profile.csv               who is contested, and how close to the cut
    ├── run_info.json                       versions, CPU count, reproduction checks
    ├── fig_stability_regimes.png
    ├── fig_stability_vs_C.png
    ├── fig_coefficient_stability.png
    ├── fig_slide186_frontier.png           ← our version of slide 192
    └── fig_penalty_stability.png
```

## Source material in the repo

| Kind | Files |
|---|---|
| Scripts | `scripts/stability_structural.py`, `scripts/individual_stability.py`, `scripts/tradeoff_matrix.py`, `scripts/calibration_tests.py`, `scripts/incumbent_benchmark.py`, `scripts/learning_curve.py` |
| Artifacts | `artifacts/stability_*.csv/json`, `artifacts/individual_stability*.csv`, `artifacts/abstention_curve.csv`, `artifacts/selected_set_overlap.csv`, `artifacts/tradeoff_matrix.md`, `artifacts/incumbent_*.csv` |
| Figures | `artifacts/figures/structural_stability.png`, `individual_stability.png`, `tradeoff_matrix.png` |
| Notebook | `Recidivism_Project_Submission.ipynb`: cells 36–37 (§8 code), **38 (stability reading, which reconciles with the deck)**, 44–45 (trade-off table), 46 (recommendation). The old appendix was removed on `main` on 27 Sep. |
| App | `app.py` Stability tab (lines 272–318), Governance tab (340–351) |
| Reports | `reports/technical_report.md` (Stability; Recommendation), `reports/presentation_notes.md` (Q&A), `reports/presentation_outline.md` (slides 17–19) |
| Course deck | `Slides ISAF 2026_2027.pdf` (your Downloads; not committed, since it's course material): §7.1 slides 182–194, §7.2 slides 195–223, guest lecture 41–52, §8.3 slides 263–277 |

## Still to do

- [x] **Checked against the course slides** (27 Sep): definitions, notation and slide numbers
      now follow the professor's deck; see 08. Corrections made: "p192" is the
      Bertsimas–Digalakis frontier, not a C sweep; §7.2 is about LLM randomness.
- [x] **Notebook stability evidence restored** (27 Sep, this branch). `main`'s cleanup
      removed the appendix that backed slides 17–18. The notebook now has cell 38, "Reading
      the stability results", generated from the artifacts, and all executed outputs are
      untouched.
- [ ] **After this branch is merged:** someone reruns `python scripts/end_to_end_audit.py`
      (the GPU box), so `artifacts/end_to_end/audit.json` records the new notebook hash.
- [ ] **Agree Option A or B with the team** (see 06), because of the time budget.
- [ ] **Ask the team to fix the app's governance tab.** It says "smaller subgroup gaps".
- [ ] **Fix "14%" on slide 18** and **"all 28 pairs" for Jaccard** in the report and notes.
      *Update 27 Sep: the redesigned deck's slide 18 now says ≈13% (fixed by the team). Slide
      17 says "28/28 pairs more stable", which is correct for drift. The "all 28" wording for
      Jaccard remains in the report and notes.*
- [ ] **Team decision: is the no-gang model primary or a second arm?** Scripts for both are
      in 09. Slide 19 still says "Calibration and fairness are tied"; suggest the slide-263
      wording from 06.
- [ ] Rehearse 05 out loud, especially Q7, Q10, Q13, Q14, Q17, Q18 and Q26–Q29.

*Environment for the extras: Python 3.9.6, scikit-learn 1.6.1 and XGBoost 2.1.4 on macOS
ARM (10 CPUs). The pinned repo environment is scikit-learn 1.7.2 and XGBoost 3.0.5.
Logistic reproduces the published predictions exactly. XGBoost matches on AUC, and the
individual scores differ by up to 0.036, which is itself result E6.*
