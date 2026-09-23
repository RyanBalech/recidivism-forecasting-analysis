"""Race A/B test requested by the instructor.

Model A: trained WITH race as an input.  Model B: trained WITHOUT race (production setup).
1. Counterfactual twin test: same person, only race flipped Black <-> White.
   A fair model gives the same probability to both twins.
2. A/B comparison: accuracy and race fairness gaps (top-20% operating point) for A vs B.
Race-blind (B) guarantees identical twins by construction, but not the absence of proxies,
so we also report group gaps for B.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics
from recidivism.modeling import logistic_model, tabicl_frames, xgboost_model

CAPACITY = 0.20


def fit_predict(name, X_train, y_train, X_eval_list):
    if name == "tabicl":
        from tabicl import TabICLClassifier
        m = TabICLClassifier(n_estimators=16, random_state=RANDOM_SEED, n_jobs=-1)
        m.fit(tabicl_frames(X_train, X_train)[0], y_train.to_numpy())
        return [m.predict_proba(tabicl_frames(X_train, X)[1])[:, 1] for X in X_eval_list]
    else:
        m = logistic_model(X_train) if name == "logistic" else xgboost_model(X_train)
        m.fit(X_train, y_train)
    return [m.predict_proba(X)[:, 1] for X in X_eval_list]


def race_gaps(p, y, race):
    """Gaps between Black and White at the deployed top-20% threshold."""
    sel = p >= np.quantile(p, 1 - CAPACITY)
    out = {}
    for g in ["BLACK", "WHITE"]:
        m = race == g
        yy, ss = y[m], sel[m]
        out[g] = {"selection": ss.mean(),
                  "fpr": (ss & (yy == 0)).sum() / max(1, (yy == 0).sum()),
                  "tpr": (ss & (yy == 1)).sum() / max(1, (yy == 1).sum()),
                  "mean_p": p[m].mean()}
    return {f"{k}_gap_B_minus_W": out["BLACK"][k] - out["WHITE"][k] for k in out["BLACK"]}


def main(models=("logistic", "xgboost", "tabicl")):
    s = load_official_split()
    y = s.y_test.to_numpy()
    race = s.audit_test["Race"].to_numpy()

    Xa_train = s.X_train.assign(Race=s.audit_train["Race"].to_numpy())
    Xa_test = s.X_test.assign(Race=race)
    twins_black = s.X_test.assign(Race="BLACK")
    twins_white = s.X_test.assign(Race="WHITE")

    rows = []
    for name in models:
        # Model A: with race
        pA, pA_black, pA_white = fit_predict(name, Xa_train, s.y_train, [Xa_test, twins_black, twins_white])
        # Model B: without race (production)
        (pB,) = fit_predict(name, s.X_train, s.y_train, [s.X_test])
        diff = pA_black - pA_white
        for variant, p, twin_abs, twin_mean in [
            ("A_with_race", pA, np.abs(diff).mean(), diff.mean()),
            ("B_without_race", pB, 0.0, 0.0),
        ]:
            r = {"model": name, "variant": variant,
                 "roc_auc": classification_metrics(y, p)["roc_auc"],
                 "twin_mean_abs_diff": twin_abs, "twin_mean_diff_B_minus_W": twin_mean,
                 "twin_max_abs_diff": np.abs(diff).max() if variant.startswith("A") else 0.0}
            r.update(race_gaps(p, y, race))
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACT_DIR / "race_ab_test.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
