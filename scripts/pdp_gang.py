"""Partial dependence for gang affiliation, and PDP contrasts for all three models.

interpretability.py draws PDP/ICE for age, prior felony arrests and the Georgia score. This adds
gang affiliation (No vs Yes) on exactly the same 200 evaluation people (same RNG draw), for all
three models, and writes the two contrasts quoted on slide 9 as PDP differences in probability
points: age 23-27 -> 48 or older (from pdp_values.csv) and gang No -> Yes (computed here).

A PDP difference between two values is the course's average marginal effect for a categorical
field (p62): everyone is set to value a, then to value b, and the predictions are averaged.
TabICLv2 needs one in-context fit and 400 predictions (about 15 minutes on CPU).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, MODEL_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.modeling import tabicl_frames, tabicl_model

warnings.filterwarnings("ignore")


def main() -> None:
    split = load_official_split()
    rng = np.random.default_rng(RANDOM_SEED)
    # Same draw as interpretability.py: first 200 of a 1,000-row evaluation sample.
    sample = split.X_test.iloc[rng.choice(len(split.X_test), 1000, replace=False)].reset_index(drop=True)
    ice_rows = sample.iloc[:200].reset_index(drop=True)

    batch = pd.concat([ice_rows.assign(Gang_Affiliated=v) for v in ["No", "Yes"]], ignore_index=True)
    predictors = {m: joblib.load(MODEL_DIR / f"{m}.joblib").predict_proba for m in ["logistic", "xgboost"]}
    tab = tabicl_model(RANDOM_SEED)
    tab.fit(tabicl_frames(split.X_train, split.X_train)[0], split.y_train.to_numpy())
    predictors["tabicl"] = lambda X: tab.predict_proba(tabicl_frames(split.X_train, X)[1])

    rows = []
    for name, predict in predictors.items():
        p = predict(batch)[:, 1].reshape(2, len(ice_rows))
        for value, curve in zip(["No", "Yes"], p):
            rows.append({"model": name, "feature": "Gang_Affiliated", "value": value, "pdp": float(curve.mean())})
        print(f"{name}: gang No {p[0].mean():.3f} -> Yes {p[1].mean():.3f}", flush=True)
    gang = pd.DataFrame(rows)
    gang.to_csv(ARTIFACT_DIR / "pdp_gang.csv", index=False)

    pdp = pd.concat([pd.read_csv(ARTIFACT_DIR / "pdp_values.csv"), gang], ignore_index=True)
    at = pdp.set_index(["model", "feature", "value"]).pdp
    contrasts = []
    for model in ["logistic", "xgboost", "tabicl"]:
        contrasts.append({"model": model, "contrast": "Age 23-27 -> 48 or older",
                          "pdp_difference": at[(model, "Age_at_Release", "48 or older")] - at[(model, "Age_at_Release", "23-27")]})
        contrasts.append({"model": model, "contrast": "Gang No -> Yes",
                          "pdp_difference": at[(model, "Gang_Affiliated", "Yes")] - at[(model, "Gang_Affiliated", "No")]})
    out = pd.DataFrame(contrasts)
    out.to_csv(ARTIFACT_DIR / "pdp_contrasts.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
