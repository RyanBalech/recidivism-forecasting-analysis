# 15-minute presentation notes

## Slide 1 — A useful score must earn trust (0:45)

Open with the decision: the client has limited support capacity at the start of supervision. We compare three model families, but the recommendation must balance accuracy, explanations, stability, and fairness.

## Slide 2 — The decision boundary (1:00)

The score ranks people for voluntary re-entry services. It cannot justify sanctions or greater surveillance. The outcome is arrest, which is institutionally mediated, so the model predicts the recorded outcome rather than a person's inherent propensity.

## Slide 3 — Data and leakage guardrail (1:15)

Describe 25,835 Georgia releases, the official 70/30 split, and the three-year target. Emphasize why post-release violations, drug tests, employment, and programs are excluded. Race, gender, and geography stay out of scoring and remain available for auditing.

## Slide 4 — Three deliberately different models (1:00)

Logistic regression is the white-box anchor. XGBoost learns nonlinear interactions. TabICLv2 brings pretrained tabular knowledge with in-context inference. All see the same eligible information and untouched test cohort.

## Slide 5 — Predictive performance is close (1:15)

TabICLv2 wins Brier and AUC. The absolute gain over XGBoost is 0.0015 Brier and 0.0036 AUC. Bootstrap intervals overlap, so do not call this a decisive statistical victory. XGBoost calibrates best and runs in less than half the time.

## Slide 6 — Economic value depends on assumptions (1:00)

At 20% capacity, every model captures about 29% of observed events. Walk through the editable $5k cost, $50k event cost, and 20% effectiveness scenario. State clearly that observational predictions do not estimate intervention effects.

## Slide 7 — Drivers are consistent (1:00)

Age at release, gang affiliation, prior felony arrests, and prison tenure recur across models. Permutation importance is held-out and model-agnostic. It describes reliance, not causal levers.

## Slide 8 — Fairness changes the ranking (1:30)

XGBoost has the smallest race FPR gap. TabICLv2's gender FPR gap is much larger at the 0.5 threshold. Explain that threshold choice and base rates matter. Protected-attribute exclusion reduces direct use but does not remove proxy or label bias.

## Slide 9 — Stability evidence and missing evidence (1:00)

Bootstrap interval widths are similar. Input stress tests show material feature reliance. The dataset has no suitable time split, so temporal transportability is unknown and must be tested prospectively.

## Slide 10 — Recommendation (1:30)

Recommend XGBoost for a controlled pilot, with logistic regression as challenger. The small performance sacrifice versus TabICLv2 buys calibration, speed, simpler operations, and smaller observed fairness gaps. Require benefit-only use and human accountability.

## Slide 11 — Application workflow (1:00)

Demo model comparison, subgroup audits, editable economics, held-out cohort inspection, and the individual sandbox. Point out that the app foregrounds governance limits rather than hiding them.

## Slide 12 — Production roadmap (0:45)

Shadow test, validate prospectively, estimate intervention effects, launch narrowly, and monitor quarterly. End with the rule: deployment depends on demonstrated benefit without unacceptable subgroup harm.

# Likely Q&A

**Why not use TabICLv2 when it has the best Brier score?**  
The improvement is small and uncertain, while its threshold fairness, calibration, runtime, and operational complexity are worse in this run. Trustworthy selection is multi-objective.

**Why exclude race and gender?**  
For this support-allocation prototype, we chose not to use protected status directly. We still audit by both attributes. This does not guarantee fairness, so proxy, label, and outcome audits remain necessary.

**Why exclude the dynamic variables?**  
They occur after the score is supposed to be made. They can encode supervision intensity and may be downstream of recidivism, creating leakage and an unusable baseline model.

**Why use arrest as the target?**  
It is the challenge outcome and operationally observable, but it is imperfect. Our model card explicitly treats it as an institutionally mediated proxy and restricts use to beneficial support.

**Is the economic estimate credible?**  
It is a scenario, not a forecast of savings. The app exposes each assumption. A prospective randomized or quasi-experimental study is required to estimate actual intervention value.

**Would equal thresholds be fair?**  
Not necessarily. Equal thresholds can yield unequal error rates when group distributions differ. Threshold policy must reflect the benefit-only use case, legal review, and explicit fairness objectives.

**What would make you stop the model?**  
Calibration drift, material subgroup-gap growth, degraded data quality, evidence of adverse use, or failure to show net benefit in the prospective pilot.
