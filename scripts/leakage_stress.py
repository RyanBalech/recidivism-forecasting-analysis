"""Reproducible negative controls and feature-timing sensitivity, train rows only.

Chance-level shuffled-label scores cannot rule out temporal leakage: a future
feature still loses its association when the training target is shuffled.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics
from recidivism.modeling import tabicl_frames, tabicl_model, xgboost_model

OUT = ROOT / "artifacts/deep_review"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = load_official_split()
    a, b, ya, yb = train_test_split(split.X_train, split.y_train, test_size=.2,
                                    stratify=split.y_train, random_state=20260924)
    conditions = [("baseline", None, [])]
    conditions += [(f"shuffled_labels_{seed}", seed, []) for seed in [11, 22, 33]]
    conditions += [("remove_uncertain_timing", None,
                    ["Gang_Affiliated", "Supervision_Risk_Score_First", "Supervision_Level_First"])]
    rows, batch_checks = [], {}
    for condition, seed, drops in conditions:
        train, dev = a.drop(columns=drops), b.drop(columns=drops)
        labels = ya.to_numpy() if seed is None else np.random.default_rng(seed).permutation(ya.to_numpy())
        for name in ["xgboost", "tabicl"]:
            if name == "xgboost":
                model = xgboost_model(train).fit(train, labels)
                p = model.predict_proba(dev)[:, 1]
            else:
                tr, dv = tabicl_frames(train.reset_index(drop=True), dev.reset_index(drop=True))
                model = tabicl_model(42).fit(tr, labels)
                p = model.predict_proba(dv)[:, 1]
                if condition == "baseline":
                    # Same cached model and rows, different query composition.
                    small = model.predict_proba(dv.head(8))[:, 1]
                    single = np.array([model.predict_proba(dv.iloc[[i]])[0, 1] for i in range(8)])
                    batch_checks = {"device": str(model.device_), "checkpoint": model.checkpoint_version,
                        "small_vs_full_max_abs": float(np.max(np.abs(small-p[:8]))),
                        "single_vs_small_max_abs": float(np.max(np.abs(single-small))),
                        "note": "Numerical differences possible; no labels passed at predict time"}
            row = {"condition": condition, "model": name, **classification_metrics(yb, p)}
            rows.append(row)
            print(f"{condition} {name}: AUC={row['roc_auc']:.6f} Brier={row['brier']:.6f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "leakage_stress.csv", index=False)
    (OUT / "tabicl_inference_checks.json").write_text(json.dumps(batch_checks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
