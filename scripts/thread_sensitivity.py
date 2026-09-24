"""Measure thread-dependent training variation; never select threads by score."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics
from recidivism.modeling import xgboost_model


def main():
    s = load_official_split()
    a, b, ya, yb = train_test_split(s.X_train, s.y_train, test_size=.2,
                                    stratify=s.y_train, random_state=20260924)
    predictions, rows = {}, []
    for threads in [1, 4, -1]:
        for repeat in [0, 1]:
            model = xgboost_model(a, n_jobs=threads).fit(a, ya)
            p = model.predict_proba(b)[:, 1]
            predictions[threads, repeat] = p
            rows.append({"n_jobs": threads, "repeat": repeat, **classification_metrics(yb, p)})
    result = {"logical_cpus": os.cpu_count(), "xgboost_version": xgboost.__version__,
        "max_abs_diff_4_vs_all": float(np.max(np.abs(predictions[4, 0]-predictions[-1, 0]))),
        "max_abs_repeat_difference": {str(n): float(np.max(np.abs(predictions[n, 0]-predictions[n, 1])))
                                       for n in [1, 4, -1]},
        "interpretation": "Runtime setting sensitivity, not target leakage. Do not optimize threads using evaluation scores."}
    out = ROOT / "artifacts/deep_review"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "thread_sensitivity.csv", index=False)
    (out / "thread_sensitivity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["n_jobs", "repeat", "roc_auc", "brier"]].to_string(index=False))
    print(result)


if __name__ == "__main__":
    main()
