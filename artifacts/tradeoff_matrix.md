| Dimension / metric | Logistic | XGBoost | TabICLv2 |
|---|---|---|---|
| **PERFORMANCE** | | | |
| ROC AUC (held-out) | 0.729 | 0.733 | 0.734 |
| Brier / calibration | 0.206 | 0.204 | 0.204 |
| Net value @20% ($M) | 5.01 | 5.26 | 5.14 |
| **INTERPRETABILITY** | | | |
| Local explanation | coefficients | SHAP + surrogate | none native |
| Global surrogate fidelity | exact (linear) | tree R²=0.61 | PDP/ICE only |
| **STABILITY** | | | |
| Score drift across refits | 0.034 | 0.035 | 0.045 |
| Top-20% overlap | 77% | 76% | 73% |
| **FAIRNESS (top-20%)** | | | |
| Race FPR gap | 0.017 | 0.020 | 0.019 |
| Gender FPR gap | 0.040 | 0.036 | 0.072 |
| **COST** | | | |
| Train+predict (s) | 1.2 | 2.9 | 103.7 |
| Auditability | high | medium | low |