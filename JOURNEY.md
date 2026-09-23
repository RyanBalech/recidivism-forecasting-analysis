# Improvement journey — where we started, what we tried, where we landed

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

## Data leakage (found & fixed)
| | |
|---|---|
| **Started** | Original `tabicl_frames` passed raw NaN to TabICLv2, which encodes NaN as its own category. |
| **Found** | A teammate flagged a possible leak; a scan confirmed `Gang_Affiliated` is missing for **exactly the 2,217 women and no men**, so the missingness perfectly encoded the excluded Gender. It inflated TabICL's gender FPR gap (0.27 at 0.5) and lent it ~0.001 AUC. |
| **Fixed** | Mode-fill categorical NaN before TabICL (matching the other pipelines), a regression test, and a full `scripts/leakage_audit.py` (also a pytest) that scans every feature's missingness vs race and gender, checks for post-scoring features, and verifies the split is disjoint. |
| **Landed** | Audit passes: no protected-aligned NaN reaches TabICL, no post-scoring feature, train/test disjoint. After the fix, TabICL's gender gap dropped to 0.041 and it lost its illusory AUC edge — the three models became a genuine tie. |
| **Lesson** | Attribute exclusion is not enough; representation-level leaks (missingness, encodings) can smuggle a protected attribute back in. This is a headline trustworthy-AI finding, not a footnote. |

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
  winners) sit at ~0.73–0.74 — the data's signal ceiling. Chasing more is not gradeable and not the
  point of the course. Proven, not asserted (three model classes + a CV search all converge).
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
| **Landed** | All three comparably stable (drift ~0.034–0.036, decision overlap ~76–77%, 8 refits each on GPU). ~1 person in 4 changes priority status across refits — scores need governance. |
| **GPU note** | Early runs used 2 TabICL estimators / 4 refits on CPU, which made TabICL look least stable. On GPU (RTX 4060) we use 16 estimators / 8 refits, and the gap disappears. |

## Fairness — the deepest iteration
| Step | What changed | Result |
|---|---|---|
| Start | Gaps audited at threshold **0.5** only | XGB gender 0.108; TabICL gender **0.106** (0.27 while the leak was live) |
| Fix 1 | Audit at the **deployed top-20%** operating point (what the product ships) | gaps shrink ~2–3× (TabICL gender 0.106 → **0.041**) |
| Fix 2 | **Bootstrap 95% CIs** on every gap (inference test) | e.g. logistic race gap at top-20% is **not** significant |
| Fix 3 | **Impossibility result, split by attribute** | race base rates ≈equal (0.582/0.564) → gap **fixable**; gender differ 13.7 pts (0.591/0.454) → theorem **binds** |
| Fix 4 | **Mitigation frontier**, first race-only | group thresholds drive race gap → ~0 keeping ~all events |
| Fix 5 | **Extended mitigation to gender** (reviewer caught race-only) | gender gap → **~0.00** for all models, keeps ~1,285/1,295 events — **but decalibrates women** (base-rate gap 0.137) |
| Fix 6 | **Race A/B + twin test** | adding race changes AUC ≤0.0007 but makes twins differ up to 4.8 pts; removing it is free + fairer, though TabICL shows proxy leakage |

- **Landed:** fairness is the centerpiece, audited at the right operating point, with inference tests,
  a per-attribute theorem story, mitigation for both attributes, and a documented legal caveat
  (per-group thresholds = disparate treatment, *Ricci v. DeStefano* — shown as an analytic device,
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
