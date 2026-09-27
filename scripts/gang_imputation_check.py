"""Exploratory check: can Gang_Affiliated be kept by filling it differently instead of dropping it?

Gang_Affiliated is missing for every woman and no man. The shipped pipelines fill the missing value
with the training mode ("No"). Two alternatives are compared on the evaluation cohort:

- neutralize: every person gets the same value (course p262 Panel B) — read from
  fairness_mitigation.csv, alongside dropping the field and refitting (Panel A);
- men's average: women get the share of men with a recorded affiliation (≈0.176) instead of 0.

Answer: neutralizing costs as much AUC as dropping, because the field then separates no one. The
men's-average fill looks good for logistic regression but creates a value that only women have —
a new gender marker. Trees can isolate it, and XGBoost's gender FNR gap roughly doubles.
Exploratory and evaluated on the evaluation cohort only; not used to select any model.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR
from recidivism.data import load_official_split
from recidivism.metrics import capacity_selection
from recidivism.modeling import logistic_model, xgboost_model

warnings.filterwarnings("ignore")
CAPACITY = 0.20


def main() -> None:
    split = load_official_split()
    gender, y = split.audit_test.Gender.to_numpy(), split.y_test.to_numpy()
    recorded = split.X_train.Gang_Affiliated.dropna()
    men_share = float((recorded == "Yes").mean())

    def encode(frame: pd.DataFrame, fill: float) -> pd.DataFrame:
        out = frame.copy()
        out["Gang_Affiliated"] = out.Gang_Affiliated.map({"Yes": 1.0, "No": 0.0}).fillna(fill)
        return out

    def fnr_gap(p) -> float:
        sel, pos = capacity_selection(p, CAPACITY), y == 1
        return float((1 - sel[pos & (gender == "M")].mean()) - (1 - sel[pos & (gender == "F")].mean()))

    rows = []
    for name, builder in [("logistic", logistic_model), ("xgboost", xgboost_model)]:
        for variant, fill in [("women = No (shipped rule)", 0.0), (f"women = men's average ({men_share:.3f})", men_share)]:
            train, test = encode(split.X_train, fill), encode(split.X_test, fill)
            p = builder(train).fit(train, split.y_train).predict_proba(test)[:, 1]
            rows.append({"model": name, "variant": variant, "auc": roc_auc_score(y, p),
                         "gender_fnr_gap_m_minus_f": fnr_gap(p), "women_mean_score": float(p[gender == "F"].mean()),
                         "women_observed_rate": float(y[gender == "F"].mean())})

    mitigation = pd.read_csv(ARTIFACT_DIR / "fairness_mitigation.csv")
    course = mitigation[(mitigation.attribute == "Gender") & mitigation.feature.isin(["(none)", "Gang_Affiliated"])]
    for _, r in course.iterrows():
        rows.append({"model": r.model, "variant": f"course {r.panel}", "auc": r.auc,
                     "gender_fnr_gap_m_minus_f": r.get("fnr_gap", float("nan"))})

    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACT_DIR / "gang_imputation_check.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
