# 8 · The professor's slides, and where each idea shows up in your part

> Source: `Slides ISAF 2026_2027.pdf`, the course deck by Prof. Pérignon (315 PDF pages).
> **Slide numbers below are the numbers printed on the slides**, which is what the
> professor will recognise. The PDF page is usually slide + 34 in this part of the deck
> (for example, slide 182 is PDF page 216).
>
> The aim: you can say "as on slide N, we…" and show that you learned the concept in class
> and then implemented it from scratch on a new dataset.

---

## 8.1 Where the course frames the whole project

| Slide | What the professor shows | How it connects to you |
|---|---|---|
| **12** | "Life-changing algorithms", which condition access to credit, work, education, love and **freedom**, illustrated by a recidivism risk score (low risk 3 vs high risk 10) | Recidivism is the professor's own opening example. Our project is the *support* version of it: the score decides who gets help, not who is detained. |
| **16** | The trade-off picture: **Predictive performance** on the x-axis, **Interpretability** on the y-axis, and **Stability, Fairness, Frugality, Data privacy** as further axes | This is the frame for slide 19. Our recommendation is a position on this picture: we give up a little predictive performance for more interpretability, stability and frugality. |
| **24–25** | "The algorithm rejected you. Sorry, I do not have any additional information" is not acceptable; explanation is required by law (GDPR, French public-health code) | Why a supervisee must be able to get a reproducible explanation and an appeal, which favours logistic. |
| **26** | Whom to explain to: developers, model checkers, management, regulators, clients | The shadow pilot's audiences: the agency (management), an auditor (model checker), the supervisee (client). |
| **61** | "There is no universally best model." Logistic regression suits cases where effects are roughly linear and **interpretability and inference on coefficients are important** | The textbook justification for choosing logistic when accuracy is a near-tie. |

## 8.2 Stability: §7.1, slides 182–194 (your core material)

| Slide | Professor's content | What we did on recidivism |
|---|---|---|
| **182** | **Definition:** "If we obtain two datasets from the same population (same underlying probability distribution), then the ML algorithm should induce approximately the same model from both datasets." (Turney, 1995) | This is the opening sentence of your slide 17. Our three tests (seed, bootstrap, disjoint halves) are three ways of drawing "two datasets from the same population". |
| **183–184** | *Why it matters:* a doctor gives data to an analyst, who returns a tree. More data arrive, the analyst returns a *different* tree, and the doctor asks "?!" | The same story for a supervision agency: after the quarterly retrain, about 13% of the people offered support change. The caseworker's "?!" is our slide 18. |
| **185** | Distance between two regression models: **d(f̂₁, f̂₂) = ‖θ̂₁ − θ̂₂‖₂**, estimated on D₁ (n₁) and D₂ (n₂ > n₁) | We compute it for logistic (E2, C1). With a full one-hot encoding it overstates instability because of collinearity; see slide 43 below. |
| **186** | **Stability-constrained re-estimation:** θ̂₂ = argmin ‖y − Xθ‖² + λ‖θ − θ̂₁‖², with **λ chosen by cross-validation** | **Implemented on our data (C2):** retrain on D₂ while penalising distance to the old model. |
| **187–188** | Best tree on a 50% sample vs on the full data: the structure changes | **Reproduced (C1):** logistic and XGBoost on a 50% sample vs all the data. |
| **189–190** | Distance between two tree paths, and between two trees (an optimal matching of paths) | Defined for *single* trees. XGBoost has 1,196 trees, so we use the importance distance (slide 193) instead. Say this if asked. |
| **191** | Bertsimas & Digalakis (2023): stable trees give **−4.6% predictive power for +38% stability**, averaged over 6 healthcare cases | Our version of that sentence comes from C2 (see 03, part C). |
| **192** | The frontier: **stability loss (L2 norm) vs predictive-power loss (MSE)**, in- and out-of-sample | **Our frontier:** `results/fig_slide186_frontier.png` (λ path) and `fig_stability_vs_C.png` (the L1 penalty path) |
| **193** | Distance using feature importances: **d(f₁, f₂) = ‖φ(f₁) − φ(f₂)‖₂** | Computed for logistic and XGBoost (E2). XGBoost ties or is slightly better here. Be honest about it. |
| **194** | Stability constraint on the importance vector: f̂₂ = argmin loss + λ‖φ(f) − φ(f̂₁)‖² | The XGBoost counterpart of slide 186. Not implemented, because it needs a custom training objective. Mention it as the next step for the challenger model. |

## 8.3 Randomness: §7.2, slides 195–223 ("Randomness in large language models")

This section comes from the professor's own 2026 paper (Coqueret, Llull, Oswald,
**Pérignon**, Scheuch, Vilhuber). It's about LLMs, but its mechanisms apply directly to
our foundation model (TabICL is a transformer) and to XGBoost.

| Slide | Professor's point | Our evidence |
|---|---|---|
| **202** | "Data generated by an LLM should be treated as **draws from a distribution**, not as fixed measurements" | The same applies to a risk score. It's one draw from the refit distribution; that's why the app shows "selected by k of 8 refits". |
| **207** | Seeds: "report the seed, but do not rely on it" | Our seeds (bootstrap 7, model 42) are recorded in `stability_protocol.json`. Seed-only refits move XGBoost's selection by about 2% (E1). |
| **208** | Silent model updates: a checkpoint can change behind the same name, so record the identifier | We pin the TabICL checkpoint by name (`tabicl-classifier-v2-20260212.ckpt`) and record its SHA-256 in the manifests. |
| **210** | **Floating-point arithmetic is not associative:** (1 + 10²⁰) − 10²⁰ = 0, but 1 + (10²⁰ − 10²⁰) = 1. The order of operations changes the answer. | XGBoost sums histogram gradients in a thread-dependent order, so a different CPU count changes scores by up to 0.035 (`deep_review/thread_sensitivity.json`). |
| **211–212** | What changes the order: **server load / batch size**, **hardware**, model splitting. "Most changes in the last digits are harmless. **The problem arises when the two leading candidates are nearly tied.**" | TabICL's scores change with the query batch size (up to 0.00095, `deep_review/tabicl_inference_checks.json`). On another machine, XGBoost changes **80** decisions and logistic **0** (E6). And the harm only appears **at the cut**: contested cases all lie within about 10 percentile points of it (E5), which is the professor's "nearly tied". |
| **215** | "Local is not automatically safe… two machines loading the same weights can disagree. **The real divide is a controlled environment versus an unobservable one.**" | Exactly E6. Logistic is reproducible across environments; XGBoost is not. |
| **222** | Reporting standard: model name and version, weights hash, parameters, raw outputs, a variability report, cost/runtime | Our manifests: `artifacts/validation_manifest.json` (versions, hashes), `stability_protocol.json` (seeds, resample hashes), `results/run_info.json`, runtime in `model_metrics.csv`. |
| **223** | Recommendations: remove deliberate randomness, never call it deterministic, promise only what the venue supports | For the client: pin versions, fix threads, archive models, and report a variability band per person. |

## 8.4 The guest lecture (AdaLogit, slides 41–52): support for picking logistic

| Slide | Point | Use |
|---|---|---|
| **42** | Black boxes are hard to audit, and post-hoc explainers (SHAP, LIME) can be **unfaithful, unstable and contradictory** | Why "XGBoost + SHAP" is weaker than logistic's own coefficients. It links to the team's LIME-fidelity and explanation-disagreement results. |
| **43** | Standard logistic regression: coefficients are always **dense**, and **unstable under collinearity** ("large estimate differences for small changes in data") | Explains our ‖θ₁ − θ₂‖₂ finding: one-hot levels are perfectly collinear with the intercept. |
| **44** | **Lasso (L1) → sparsity; Ridge (L2) → stability; Elastic net → both**, with λ chosen by cross-validation | **Tested (C3):** L1 (published) vs L2 vs elastic net on our 8 resamples. |
| **48** | AdaLogit's oracle properties → "feature selection stability, estimation stability" | Names the two stability notions we report: which coefficients are non-zero (selection) and how much they move (estimation). |
| **52** | Radar chart over 5 dimensions (AUC, balanced accuracy, sensitivity = 1 − FNR, calibration, sparsity): sparse logistic ≈ **TabPFN** (a tabular foundation model) on AUC, and better on calibration and sparsity | In the professor's own course material, a well-regularised logistic regression matches a foundation model on AUC. That's our result too (0.730 vs 0.733), and TabICL is the worse calibrated (slope 0.913). |

## 8.5 Fairness equivalence: §8.3, slides 263–277 (wording for slide 19)

| Slide | Point | Consequence for your wording |
|---|---|---|
| **263** | "Failing to reject the null hypothesis means we have not found evidence that the model is unfair. It **does not certify** that the model is fair." | Don't say "logistic and XGBoost are *equally* fair because the difference isn't significant". |
| **264–265** | The risk to control is **accepting a truly unfair model** | Same logic for comparing two models. |
| **269, 271–272** | Bioequivalence (FDA): tolerance δ, H₀: \|θ\| ≥ δ vs H₁: \|θ\| < δ, tested with **TOST** (Schuirmann, 1987) | **Say instead:** "The difference in gender FNR gap between logistic and XGBoost has a 95% CI of [−0.036, +0.011], entirely inside our ±5-point tolerance, so the two models are *equivalent within ±5 points* (TOST)." That is the professor's test, applied to the model comparison. |

## 8.6 Phrases in the professor's vocabulary

- "Two datasets from the same population should induce approximately the same model"
  (slide 182).
- "Distance between parameter estimates, ‖θ̂₁ − θ̂₂‖₂" (185) and "feature-importance
  distance" (193).
- "Stability constraint, with λ chosen by cross-validation" (186).
- "A small loss in predictive power for a significant gain in stability" (191).
- "Treat the score as a draw from a distribution, not a fixed measurement" (202).
- "The problem arises when the two leading candidates are nearly tied" (212).
- "A controlled environment versus an unobservable one" (215).
- "Failing to reject does not certify fairness; we test equivalence with TOST" (263, 272).
