# 15-minute presentation notes

Timing target ~13 min talk + buffer. Every member must be able to defend any slide.

## 1 — Title / thesis (0:45)
We are a consultancy for a vendor that sells risk-assessment tools to US community-supervision
agencies. The headline: our models don't just work, they clearly beat the tool agencies use today.
The talk judges three models on performance, interpretability, stability, fairness.

## 2 — Client and decision (0:45)
The score ranks people for *voluntary* re-entry support at supervision start. Never sanctions or
surveillance. Outcome is a recorded arrest — institutionally mediated — so we predict the recorded
outcome, not inherent propensity.

## 3 — Data design (1:00)
18,028 / 7,807 official NIJ split. Post-release variables excluded (leakage — they accrue after the
score). Race, gender, PUMA excluded from inputs, kept for audit. Balanced outcome (57.8%), no
resampling.

## 4 — Three models (0:45)
Logistic (white-box), XGBoost (ordinal counts + 5-fold CV tuning), TabICLv2 (foundation model, no
training, no native explanation). Same fields, same test cohort.

## 5 — Incumbent benchmark — the hook (1:30)
`Supervision_Risk_Score_First` is Georgia's existing 1–10 tool. Alone it scores **0.60 AUC** — barely
better than a coin flip at ranking. Our models hit **0.73** and roughly double net value ($2.75M →
$5.1–5.3M). That is the client's real question, answered. The choice among our three is secondary.

## 6 — Predictive performance (1:00)
Among the three: near-tie. TabICL best AUC/Brier by a hair; XGBoost best calibration. Bootstrap 95%
CIs overlap — don't oversell the ranking.

## 7 — Learning curve (1:15)
Sizes 1.5k/5k/10k/full. TabICL leads on small data (small county); XGBoost catches up as data grows
(large state). Gap closes to +0.004, no crossover. This is *why* model choice depends on agency size.

## 8 — Economic performance (0:45)
20% capacity, editable $5k/$50k/20% scenario. Models beat incumbent at every capacity 5–50%
(sensitivity sweep). Scenario, not causal savings.

## 9 — Interpretability (1:15)
SHAP global + individual waterfall (logistic, XGBoost); LIME agrees via a different mechanism; XPER
(Pérignon's method) attributes AUC — age at release is the top driver. Depth-3 surrogate R²=0.61.
**TabICL has no native explanation path** — a real deployment cost, only PDP/ICE cover it.

## 10 — Stability (1:00)
Refits on resampled data (8 each). All three comparably stable (~0.035 drift). But ~1 person in 4
changes priority status across refits, so scores need governance. No time split → temporal
stability is a deployment gate.

## 11 — Fairness at the deployed point (1:15)
Audit at top-20% (what we ship), not 0.5 — gaps are ~2–3× smaller there (TabICL gender 0.106→0.041).
Bootstrap CIs on every gap. XGBoost has the smallest gender gap; TabICL the largest.

## 12 — Impossibility result (1:15)
Our data shows both sides. Race base rates near-equal (0.582 vs 0.564) → gap is fixable, group
thresholds drive it to ~0. Gender differs 13.7 pts (0.591 vs 0.454) → theorem binds, equalizing
gender FPR **decalibrates women**. We surface the trade-off, we don't hide it.

## 13 — Trade-off matrix (1:00)
The required slide. Performance near-tie → decision driven by interpretability, fairness, cost.
Walk the green/amber/red columns.

## 14 — Recommendation (1:00)
Deploy XGBoost (best calibration + net value, smallest gender gap, explainable, ~10× faster). Logistic
challenger. TabICL only for very small agencies. Benefit-only, prospective pilot, appeal route,
quarterly audits, stop rules.

## 15 — App + close (0:45)
Demo: score one person across all 3 models + SHAP; fairness at deployed point; incumbent + sensitivity.
Close on the rule: deploy only if benefit is shown without unacceptable subgroup harm.

---

# Likely Q&A

**Did you actually beat the foundation model?** No — at full data it's a statistical tie (0.7328 vs 0.7326).
We don't claim to. Our finding is the crossover *shape* and that both crush the 0.60 incumbent. The
recommendation rests on trust dimensions, where XGBoost wins.

**Your impossibility claim — does it hold for gender?** No, and that's the point. Race base rates are
near-equal so the gap is fixable; gender base rates differ 13.7 pts so the theorem binds. Same slide,
both sides.

**You excluded race, then used race-specific thresholds — explain.** We didn't ship those. Group
thresholds are an *analytic device* to trace the frontier. A per-race threshold is disparate treatment
(Ricci v. DeStefano) and contradicts excluding race. The deployment option is the group-blind single
threshold; we show what a race-blind alternative costs.

**Why audit at top-20% not 0.5?** The product allocates the top 20% by risk, so 0.5 describes an
operating point we never deploy. Auditing at 0.5 overstates gaps ~2–3×.

**Is TabICL less stable?** No — with 16 estimators on GPU we run 8 refits for all three; they are
comparably stable (~0.035 drift, ~76-77% decision overlap). The earlier "TabICL least stable" reading
was an artifact of running it light (2 estimators, 4 refits) before we had the GPU.

**Why exclude the dynamic variables?** They accrue after the score is made — leakage, and they encode
supervision intensity that's downstream of the outcome.

**Is the economic number credible?** It's a scenario, not a savings forecast — every assumption is
editable, and the ranking vs incumbent holds across the whole sensitivity sweep. Real impact needs a
prospective randomized/quasi-experimental study.

**What would make you stop the model?** Calibration drift, growing subgroup gaps at the deployed point,
data-quality failure, evidence of adverse use, or no net benefit in the pilot.
