"""Training-partition-only nested accuracy experiment; never publishes new models.

Three outer folds evaluate a prespecified search over XGBoost encoding and
regularization; three inner folds select by Brier loss (AUC breaks ties).
Historical defaults were previously tuned on these people, so the outer scores
are an internal robustness check, not a new untouched test cohort.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.config import ARTIFACT_DIR
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics
from recidivism.modeling import (
    XGB_PARAMS, logistic_model, preprocessor, tabicl_frames, tabicl_model, xgboost_model,
)

OUT = ARTIFACT_DIR / "deep_review"
SEED = 20260923


def candidate_grid():
    # Small, fixed search: encoding, interaction depth, shrinkage, regularization.
    grid = [{"encoding": "ordinal", "params": XGB_PARAMS.copy()}]
    for encoding in ["ordinal", "onehot"]:
        for depth, trees, rate, child, reg in [
            (1, 1600, .04, 10, 10), (2, 1200, .025, 10, 10),
            (3, 900, .025, 20, 15), (4, 700, .02, 30, 20),
            (2, 1800, .015, 30, 20), (3, 1000, .02, 50, 30),
        ]:
            grid.append({"encoding": encoding, "params": {
                "max_depth": depth, "n_estimators": trees, "learning_rate": rate,
                "min_child_weight": child, "reg_lambda": reg, "reg_alpha": .2,
                "subsample": .85, "colsample_bytree": .9, "gamma": 0.,
            }})
    return grid


def build(frame, candidate):
    params = {**candidate["params"], "n_jobs": 4}
    if candidate["encoding"] == "ordinal":
        return xgboost_model(frame, **params)
    return Pipeline([
        ("prepare", preprocessor(frame)),
        ("model", XGBClassifier(**params, objective="binary:logistic",
                                eval_metric="logloss", tree_method="hist", random_state=42)),
    ])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = load_official_split()
    X, y = split.X_train, split.y_train
    grid = candidate_grid()
    # Record the protocol before obtaining outer validation scores.
    protocol = {"seed": SEED, "outer_folds": 3, "inner_folds": 3,
                "selection": "minimum mean Brier; AUC tie break", "xgboost_n_jobs": 4,
                "candidates": grid, "official_evaluation_used": False,
                "limitations": "Historical defaults already tuned on this training partition",
                "blend": "fixed equal weight selected XGBoost and TabICL"}
    (OUT / "accuracy_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    fold_rows, searches, predictions = [], [], []
    outer = StratifiedKFold(3, shuffle=True, random_state=SEED)
    for fold, (it, iv) in enumerate(outer.split(X, y), 1):
        a, b, ya, yb = X.iloc[it], X.iloc[iv], y.iloc[it], y.iloc[iv]
        inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED + fold).split(a, ya))
        ranking, oof_scores = [], []
        for k, candidate in enumerate(grid):
            start = time.perf_counter()
            oof = np.zeros(len(a))
            losses, aucs = [], []
            for jt, jv in inner:
                m = build(a.iloc[jt], candidate).fit(a.iloc[jt], ya.iloc[jt])
                oof[jv] = m.predict_proba(a.iloc[jv])[:, 1]
                met = classification_metrics(ya.iloc[jv], oof[jv])
                losses.append(met["brier"]); aucs.append(met["roc_auc"])
            row = {"outer_fold": fold, "candidate": k, "encoding": candidate["encoding"],
                   "inner_brier": np.mean(losses), "inner_auc": np.mean(aucs),
                   "seconds": time.perf_counter() - start}
            searches.append(row); ranking.append((row["inner_brier"], -row["inner_auc"], k))
            oof_scores.append(oof)
            pd.DataFrame(searches).to_csv(OUT / "accuracy_inner_search.csv", index=False)
            print(f"outer {fold}/3 candidate {k+1}/{len(grid)} Brier={row['inner_brier']:.6f}", flush=True)
        chosen = min(ranking)[2]
        current = build(a, grid[0]).fit(a, ya)
        selected = build(a, grid[chosen]).fit(a, ya)
        pred = {"xgboost_current": current.predict_proba(b)[:, 1],
                "xgboost_selected": selected.predict_proba(b)[:, 1],
                "logistic": logistic_model(a).fit(a, ya).predict_proba(b)[:, 1]}
        # Calibrator only sees predictions of models that excluded each inner row.
        calibrator = LogisticRegression(C=1., solver="lbfgs")
        calibrator.fit(logit(np.clip(oof_scores[chosen], 1e-6, 1-1e-6)).reshape(-1, 1), ya)
        pred["xgboost_calibrated"] = calibrator.predict_proba(
            logit(np.clip(pred["xgboost_selected"], 1e-6, 1-1e-6)).reshape(-1, 1))[:, 1]
        at, bt = tabicl_frames(a.reset_index(drop=True), b.reset_index(drop=True))
        tfm = tabicl_model(42).fit(at, ya.to_numpy())
        pred["tabicl"] = tfm.predict_proba(bt)[:, 1]
        pred["blend_equal"] = (pred["tabicl"] + pred["xgboost_selected"]) / 2
        for name, p in pred.items():
            row = {"fold": fold, "model": name, "selected_candidate": chosen,
                   **classification_metrics(yb, p)}
            fold_rows.append(row)
            print(f"OUTER {fold} {name}: AUC={row['roc_auc']:.6f} Brier={row['brier']:.6f}", flush=True)
        predictions.append(pd.DataFrame({"ID": split.audit_train.ID.iloc[iv].to_numpy(),
                                         "fold": fold, "actual": yb.to_numpy(), **pred}))
        pd.DataFrame(fold_rows).to_csv(OUT / "accuracy_outer_folds.csv", index=False)
        pd.concat(predictions).to_csv(OUT / "accuracy_oof.csv", index=False)
    table = pd.DataFrame(fold_rows).groupby("model")[["roc_auc", "brier", "log_loss", "accuracy"]].mean()
    table.to_csv(OUT / "accuracy_summary.csv")
    print(table.sort_values("brier").to_string(), flush=True)


if __name__ == "__main__":
    main()
