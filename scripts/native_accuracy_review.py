"""Exploratory follow-up: native categorical trees on the same outer folds.

Reported separately because this follow-up was added after reviewing the initial
nested experiment. No official evaluation labels are used.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics
from recidivism.modeling import XGB_PARAMS, ordinal_encode
from accuracy_review import OUT, SEED


def native_frames(train, test, catboost=False):
    """Learn categories/modes on fit rows; unknown levels are missing at prediction."""
    a, b = ordinal_encode(train), ordinal_encode(test)
    categorical = a.select_dtypes(exclude="number").columns.tolist()
    for c in categorical:
        modes = a[c].mode()
        mode = modes.iloc[0] if len(modes) else "__EMPTY__"
        a[c], b[c] = a[c].fillna(mode), b[c].fillna(mode)
        if catboost:
            a[c], b[c] = a[c].astype(str), b[c].astype(str)
        else:
            levels = sorted(a[c].unique())
            a[c] = pd.Categorical(a[c], categories=levels)
            b[c] = pd.Categorical(b[c], categories=levels)
    return a, b, categorical


def fit_predict(a, ya, b, params):
    at, bt, _ = native_frames(a, b)
    model = XGBClassifier(**params, tree_method="hist", enable_categorical=True,
                          max_cat_to_onehot=1, device="cuda", n_jobs=4,
                          objective="binary:logistic", eval_metric="logloss", random_state=42)
    model.fit(at, ya)
    return model.predict_proba(bt)[:, 1]


def main():
    s = load_official_split(); X, y = s.X_train, s.y_train
    grid = [XGB_PARAMS, {**XGB_PARAMS, "max_depth": 3, "min_child_weight": 20,
                        "reg_lambda": 15, "n_estimators": 900, "learning_rate": .02},
            {**XGB_PARAMS, "max_depth": 2, "min_child_weight": 30,
             "reg_lambda": 20, "n_estimators": 1600, "learning_rate": .015}]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "native_protocol.json").write_text(json.dumps({"grid": grid,
        "device": "cuda", "follow_up_after_initial_search": True,
        "catboost": "fixed 784 iterations, depth 5, learning rate .0148, L2 6.1721",
        "selection": "minimum mean inner Brier, 3 folds", "seed": SEED}, indent=2), encoding="utf-8")
    rows, searches, predictions = [], [], []
    for fold, (it, iv) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y), 1):
        a, b, ya, yb = X.iloc[it], X.iloc[iv], y.iloc[it], y.iloc[iv]
        rank = []
        for k, params in enumerate(grid):
            losses = []
            for jt, jv in StratifiedKFold(3, shuffle=True, random_state=SEED+fold).split(a, ya):
                p = fit_predict(a.iloc[jt], ya.iloc[jt], a.iloc[jv], params)
                losses.append(classification_metrics(ya.iloc[jv], p)["brier"])
            rank.append((np.mean(losses), k))
            searches.append({"fold": fold, "candidate": k, "inner_brier": np.mean(losses)})
            print(f"Native XGB outer {fold} candidate {k}: Brier={np.mean(losses):.6f}", flush=True)
        chosen = min(rank)[1]
        pred = {"xgboost_native": fit_predict(a, ya, b, grid[chosen])}
        at, bt, cats = native_frames(a, b, catboost=True)
        model = CatBoostClassifier(iterations=784, depth=5, learning_rate=.0148, l2_leaf_reg=6.1721,
            random_seed=42, verbose=False, thread_count=4, allow_writing_files=False, cat_features=cats)
        pred["catboost_native"] = model.fit(at, ya).predict_proba(bt)[:, 1]
        for name, p in pred.items():
            row = {"fold": fold, "model": name, "selected_candidate": chosen,
                   **classification_metrics(yb, p)}
            rows.append(row)
            print(f"OUTER {fold} {name}: AUC={row['roc_auc']:.6f} Brier={row['brier']:.6f}", flush=True)
        predictions.append(pd.DataFrame({"ID": s.audit_train.ID.iloc[iv].to_numpy(),
                                         "fold": fold, "actual": yb.to_numpy(), **pred}))
        pd.DataFrame(rows).to_csv(OUT / "native_outer_folds.csv", index=False)
        pd.DataFrame(searches).to_csv(OUT / "native_inner_search.csv", index=False)
        pd.concat(predictions).to_csv(OUT / "native_oof.csv", index=False)
    print(pd.DataFrame(rows).groupby("model")[["roc_auc", "brier"]].mean())


if __name__ == "__main__":
    main()
