| Dimension / metric | Logistic | XGBoost | TabICLv2 |
|---|---|---|---|
| **PERFORMANCE** | | | |
| ROC AUC (evaluation) | 0.730 | 0.732 | 0.733 |
| Brier loss | 0.205 | 0.204 | 0.204 |
| Net value @20% ($M) | 5.04 | 5.17 | 5.12 |
| **INTERPRETABILITY** | | | |
| Local explanation | coefficients | SHAP + surrogate | none native |
| Global surrogate fidelity | exact (linear) | tree R²=0.61 | PDP/ICE only |
| **STABILITY** | | | |
| Score drift across refits | 0.032 | 0.035 | 0.036 |
| Top-20% overlap | 77% | 75% | 77% |
| **FAIRNESS (top-20%)** | | | |
| Race FNR gap (B − W) | -0.005 | -0.017 | -0.023 |
| Gender FNR gap (M − F) | -0.096 | -0.112 | -0.124 |
| Age FNR gap (<33 − 33+) | -0.245 | -0.228 | -0.237 |
| **COST** | | | |
| Train+predict (s) | 1.0 | 8.7 | 41.8 |
| Auditability | high | medium | low |