# Improvement journey — where we started, what we tried, where we landed

> Historical development log, superseded where noted by the September research review.
> Use current generated artifacts for numerical results. This log documents repeated
> evaluation-set inspection, so it cannot support an untouched-holdout claim. The later
> review corrected threshold/calibration reasoning, Jaccard interpretation, and unsupported
> claims about a data ceiling or present-day agency tools; see reports/research_review.md.

## GPU ensemble and merge hardening (23 September 2026)

Team work was merged without discarding the methodology audit. The shared TabICLv2 configuration now uses 16 ensemble members on CUDA, and all three model families receive eight bootstrap stability refits. The incoming ensemble sweep originally read the reused evaluation labels; it was moved to a fixed stratified development slice of the training partition before the final full-training fit. A systematic leakage audit and regression test also guard the gender-aligned `Gang_Affiliated` missingness fix. These changes improve compute depth and consistency without relabeling the evaluation set as untouched.

This is the process log for the jury: the starting point of each dimension, every change we tried
(including the ones we **rejected**), the measured result, and why. Every number here is reproducible
from `scripts/` and `artifacts/`; git history has the commit-level trail. Figure:
`artifacts/figures/improvement_journey.png`.

Starting point = the repository at commit `3b84187` (a working end-to-end analysis). Landing point =
current `main`.

---

## Framing / client
| | |
|---|---|
| **Started** | Generic "hypothetical agency"; project framed for the HEC course. |
| **Tried** | Named a concrete client: a software vendor selling risk tools to US community-supervision agencies. |
| **Landed** | Client + product framing throughout; the learning curve maps directly to the vendor's "which model for which agency size" decision. |
| **Why** | The brief requires a client you present to; a vendor with many differently-sized agencies makes the learning-curve finding a business decision, not a curiosity. |

## Predictive accuracy (XGBoost)
| Step | Held-out AUC | Brier | ECE | Verdict |
|---|---:|---:|---:|---|
| Repo baseline (one-hot, hand-set params) | 0.7299 | 0.2054 | 0.0118 | start |
| Ordinal-encode the "N or more" count columns | 0.7314 | 0.2048 | — | kept (order was being thrown away) |
| Deeper/more trees (max_depth 5, 1200 est.) | 0.7269 | — | — | **rejected — overfit, worse** |
| Mid tuning (depth 4) | 0.7308 | — | — | superseded |
| **5-fold CV random search (60 draws) + ordinal** | **0.7326** | **0.2044** | **0.0109** | **kept (final)** |

- **Landed:** XGBoost 0.7326 AUC, best calibration of the three. CV best was 0.7343; held-out 0.7326.
- **Key decision:** we then **stopped optimizing accuracy.** All three models (and the NIJ challenge
  historical runs) sit at similar AUC values. This is not proof of a data ceiling; NIJ's annual
  challenge scores concern a different target and are not directly comparable.
- Logistic stayed 0.7295; TabICLv2 0.7338. The tuning closed the XGBoost–TabICL gap from 0.0037 to
  0.0012 but did **not** overtake it — we say so plainly.

## Economic performance
| | |
|---|---|
| **Started** | Single capacity scenario with assumed costs → one dollar figure. |
| **Tried** | (1) Benchmarked against the **incumbent** tool already in the data (`Supervision_Risk_Score_First`) and a random baseline; (2) added a sensitivity sweep over capacity (5–50%) and effectiveness. |
| **Landed** | Incumbent 0.60 AUC / $2.75M vs models ~0.73 / ~$5.1–5.3M. Models dominate the incumbent at **every** capacity, so the conclusion is assumption-robust. |
| **Why** | The client's real question is "better than what we run today?", not "which of your three?". This became the economic centerpiece. |

## Interpretability
| | |
|---|---|
| **Started** | Global permutation importance only; TabICL got a 10-feature manual permutation. No individual explanations. |
| **Added** | SHAP global + **individual waterfalls** (logistic, XGBoost); **LIME** on the same person (different mechanism, consistent story); **XPER** (Pérignon's method — attributes AUC, not predictions); depth-3 **global surrogate** (test fidelity R²=0.61); **PDP/ICE** for all three models. |
| **Landed** | All three models covered on global + local interpretability; the brief's "explain individual predictions" is met. |
| **Learned & disclosed** | TabICLv2 has **no native attribution path**; KernelSHAP over 7,807 rows is impractical on CPU. We treat this as a deployment cost, not an oversight. |

## Stability
| | |
|---|---|
| **Started** | Bootstrap CI widths + a feature-shuffle test — these measure *performance uncertainty*, not structural stability. |
| **Replaced with** | **Structural stability**: refit each model on bootstrap resamples of the training data, then measure distance between refits (mean \|Δp\|), decision overlap (top-20% Jaccard), and drift in feature contributions (SHAP rank correlation). |
| **Landed** | Historical (pre-fix, CPU, 4 TabICL refits): TabICLv2 drifted most (0.045) with 73% overlap. **Current run (GPU, 8 refits for every model):** mean \|Δp\| 0.032 / 0.035 / 0.036 and top-20% Jaccard 0.77 / 0.75 / 0.77 (logistic / XGBoost / TabICLv2) — the three models are close. Jaccard ≈ 0.76 means about 14% of the selected set is replaced between refits, not 1 person in 4. |
| **Disclosed** | Resamples are not shared across models, and data and seed vary together; separating sampling from estimator randomness is still open (PLAN.md P1.9). |

## Fairness — the deepest iteration
| Step | What changed | Result |
|---|---|---|
| Start | Gaps audited at threshold **0.5** only | XGB race 0.042 / gender 0.108; TabICL gender **0.272** |
| Fix 1 | Audit at the **deployed top-20%** operating point (what the product ships) | gaps shrink 2–4× (TabICL gender 0.272 → **0.072**) |
| Fix 2 | **Bootstrap 95% CIs** on every gap (inference test) | e.g. logistic race gap at top-20% is **not** significant |
| Fix 3 | **Impossibility result, split by attribute** | race base rates ≈equal (0.582/0.564) → gap **fixable**; gender differ 13.7 pts (0.591/0.454) → theorem **binds** |
| Fix 4 | **Mitigation frontier**, first race-only | group thresholds drive race gap → ~0 keeping ~all events |
| Fix 5 | **Extended exploratory threshold search to gender** | Historical gap 0.072 → 0.002 on the same labels used to optimize; unchanged probabilities retain unchanged calibration |
| Fix 6 | **Race A/B + twin test** | adding race changes AUC by at most about 0.001; twins differ by up to about 4.6 pts. (The earlier "TabICL proxy leakage" reading predated the gang-missingness fix and no longer holds.) |
| Fix 7 | **FNR primary + TOST equivalence + course test table + age audit** | race gaps equivalent within ±5 pts for all models; gender FNR gap −0.10 to −0.12; age FNR gap −0.23 to −0.25 |
| Fix 8 | **FPDP + mitigation** (course §8.2) | gender-gap candidates: gang affiliation (never recorded for women) and age; dropping gang removes the equal-opportunity rejection at about −0.014 AUC; statistical parity still rejected |

- **Landed:** fairness is the centerpiece, audited at the right operating point, with inference tests,
  equivalence tests, FPDP diagnosis of the gender gap, a per-attribute theorem story, mitigation, and a documented legal caveat
  (group-specific policy needs independent validation and policy/legal review — shown as an analytic device,
  not a shipping option).
- Figure: `artifacts/figures/improvement_journey.png` (left panel) shows the gender-gap path.

## Deliverables / narrative
| | |
|---|---|
| **Started** | Report/slides/notebook described the pre-incumbent, pre-mitigation project. |
| **Landed** | Report, 15-slide deck, executed notebook, and presentation notes all rewritten to the current results, plus the 3×4 **trade-off matrix** the brief requires and this journey log. |

---

### How we kept track
- **Git history** on `main` — each step is a commit with a message.
- **`PLAN.md`** — the plan and its two external-review revisions.
- **`artifacts/`** — every number is a CSV/JSON; every figure regenerates from `scripts/`.
- **This file** — the human-readable narrative of the path, for the presentation and Q&A.
