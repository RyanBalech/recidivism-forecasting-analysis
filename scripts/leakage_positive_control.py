"""Deliberately ineligible post-release features, ONLY to illustrate timing leakage.

This model is never saved or selected. It must not be used for release-time
prediction. A separate file prevents confusion with eligible accuracy candidates.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.config import DATA_PATH, EXCLUDED_FROM_MODEL, FEATURE_COLUMNS
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics
from recidivism.modeling import xgboost_model


def main():
    s = load_official_split()
    raw = pd.read_csv(DATA_PATH).set_index("ID").loc[s.audit_train.ID].reset_index()
    a, b = train_test_split(range(len(raw)), test_size=.2, stratify=s.y_train, random_state=20260924)
    later = [c for c in raw if c not in ["ID", "Training_Sample"]
             and c not in EXCLUDED_FROM_MODEL and not c.startswith("Recidivism_")]
    rows = []
    for label, fields in [("eligible_baseline", FEATURE_COLUMNS), ("INELIGIBLE_post_release_control", later)]:
        X = raw[fields]
        model = xgboost_model(X.iloc[a], n_jobs=4).fit(X.iloc[a], s.y_train.iloc[a])
        p = model.predict_proba(X.iloc[b])[:, 1]
        row = {"condition": label, "features": len(fields), **classification_metrics(s.y_train.iloc[b], p)}
        rows.append(row)
        print(label, row["roc_auc"], row["brier"], flush=True)
    out = ROOT / "artifacts/deep_review"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "INELIGIBLE_timing_positive_control.csv", index=False)


if __name__ == "__main__":
    main()
