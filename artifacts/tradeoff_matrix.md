| Dimension / metric | Logistic | XGBoost | TabICLv2 |
|---|---|---|---|
| **PERFORMANCE** | | | |
| ROC AUC (held-out) | 0.729 | 0.733 | 0.733 |
| Brier / calibration | 0.206 | 0.204 | 0.204 |
| Net value @20% ($M) | 5.01 | 5.26 | 5.12 |
| **INTERPRETABILITY** | | | |
| Local explanation | coefficients | SHAP + surrogate | none native |
| Global surrogate fidelity | exact (linear) | tree R²=0.61 | PDP/ICE only |
| **STABILITY** | | | |
| Score drift across refits | 0.034 | 0.035 | 0.036 |
| Top-20% overlap | 77% | 76% | 77% |
| **FAIRNESS (top-20%)** | | | |
| Race FPR gap | 0.017 | 0.020 | 0.024 |
| Gender FPR gap | 0.040 | 0.036 | 0.041 |
| **COST** | | | |
| Train+predict (s) | 0.8 | 2.4 | 23.1 |
| Auditability | high | medium | low |