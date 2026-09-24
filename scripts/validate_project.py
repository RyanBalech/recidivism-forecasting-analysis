"""Reproducible evidence audit: provenance, baselines, paired inference and CV.

Existing model predictions are preserved. CV evaluates fixed configurations;
it is not nested validation of the earlier hyperparameter/feature search.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.config import ARTIFACT_DIR, DATA_DIR, DATA_PATH, MODEL_DIR
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics, fairness_table
from recidivism.modeling import logistic_model, xgboost_model


def main():
    split = load_official_split()
    pred = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    expected = split.audit_test.assign(actual=split.y_test)
    pd.testing.assert_frame_equal(pred[expected.columns], expected, check_dtype=False)
    raw = pd.read_csv(DATA_PATH)
    training = pd.read_csv(DATA_DIR / "nij-challenge2021_training_dataset.csv")
    test = pd.read_csv(DATA_DIR / "nij-challenge2021_test_dataset_1.csv")
    assert set(training.ID) == set(split.audit_train.ID)
    assert set(test.ID) == set(split.audit_test.ID)
    for year in [2, 3]:
        released = pd.read_csv(ROOT / f"nij-challenge2021_test_dataset_{year}.csv")
        eligible = raw.Training_Sample.eq(0)
        for earlier in range(1, year):
            eligible &= raw[f"Recidivism_Arrest_Year{earlier}"].eq("No")
        assert set(released.ID) == set(raw.loc[eligible, "ID"])
    year_cols = [f"Recidivism_Arrest_Year{i}" for i in [1, 2, 3]]
    annual = raw[year_cols].apply(lambda s: s.map({"Yes": 1, "No": 0})).astype(int)
    assert (annual.sum(axis=1) <= 1).all()
    assert np.array_equal(annual.max(axis=1), raw.Recidivism_Within_3years.map({"Yes": 1, "No": 0}))

    y = pred.actual.to_numpy()
    names = [c.removeprefix("p_") for c in pred if c.startswith("p_")]
    scores = {m: pred[f"p_{m}"].to_numpy() for m in names}
    scores["training_prevalence"] = np.full(len(y), split.y_train.mean())
    incumbent = logistic_model(split.X_train[["Supervision_Risk_Score_First"]])
    incumbent.fit(split.X_train, split.y_train)
    scores["calibrated_incumbent"] = incumbent.predict_proba(split.X_test)[:, 1]
    rows, groups = [], []
    baseline_loss = brier_score_loss(y, scores["training_prevalence"])
    for name, p in scores.items():
        row = {"model": name, **classification_metrics(y, p)}
        row["brier_skill_vs_prevalence"] = 1 - row["brier"] / baseline_loss
        rows.append(row)
        for attribute in ["Race", "Gender"]:
            groups.append(fairness_table(y, p, pred[attribute], attribute).assign(model=name))
        groups.append(fairness_table(y, p, pred.Race + " / " + pred.Gender,
                                     "Race x Gender").assign(model=name))
        groups.append(fairness_table(y, p, split.X_test.Age_at_Release,
                                     "Age at release").assign(model=name))
    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "validation_baselines.csv", index=False)
    pd.concat(groups).to_csv(ARTIFACT_DIR / "intersectional_audit.csv", index=False)

    # The same resampled people for both models preserve their error correlation.
    rng = np.random.default_rng(42)
    comparisons = []
    for a, b in itertools.combinations(names, 2):
        draws = []
        for _ in range(1000):
            idx = rng.integers(len(y), size=len(y))
            draws.append([roc_auc_score(y[idx], scores[a][idx]) - roc_auc_score(y[idx], scores[b][idx]),
                          np.mean((y[idx] - scores[a][idx]) ** 2 - (y[idx] - scores[b][idx]) ** 2)])
        point = [roc_auc_score(y, scores[a]) - roc_auc_score(y, scores[b]),
                 np.mean((y - scores[a]) ** 2 - (y - scores[b]) ** 2)]
        for j, metric in enumerate(["roc_auc", "brier"]):
            lo, hi = np.quantile(np.asarray(draws)[:, j], [0.025, 0.975])
            comparisons.append(dict(model_a=a, model_b=b, metric=metric,
                                    difference_a_minus_b=point[j], ci_low=lo, ci_high=hi,
                                    repeats=1000))
    pd.DataFrame(comparisons).to_csv(ARTIFACT_DIR / "paired_comparisons.csv", index=False)

    cv_rows = []
    for name, model in [("logistic", logistic_model(split.X_train)),
                        ("xgboost", xgboost_model(split.X_train, n_jobs=1))]:
        result = cross_validate(model, split.X_train, split.y_train,
                                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring={"auc": "roc_auc", "brier": "neg_brier_score"}, n_jobs=2)
        for fold in range(5):
            cv_rows.append(dict(model=name, fold=fold + 1, roc_auc=result["test_auc"][fold],
                                brier=-result["test_brier"][fold]))
    pd.DataFrame(cv_rows).to_csv(ARTIFACT_DIR / "validation_cv.csv", index=False)
    reproduction = {}
    for name in ["logistic", "xgboost"]:
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        actual = model.predict_proba(split.X_test)[:, 1]
        delta = float(np.max(np.abs(actual - scores[name])))
        reproduction[name] = delta
        assert delta < 1e-6, f"{name} saved model and predictions disagree: {delta}"
    manifest = {
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scikit-learn", "xgboost", "tabicl", "torch", "shap", "streamlit",
                      "lime", "XPER", "scipy", "joblib", "python-pptx", "nbformat", "nbclient"]},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [DATA_PATH, ARTIFACT_DIR / "test_predictions.csv", *MODEL_DIR.glob("*.joblib")]},
        "saved_prediction_max_absolute_error": reproduction,
        "official_ids_verified": True, "annual_target_consistency_verified": True,
        "cv_scope": ("Fixed conventional configurations; not nested search evaluation. "
                     "TabICL ensemble sensitivity uses one training-only development split, not CV"),
        "holdout_status": "Repeatedly inspected during historical development; exploratory comparisons",
    }
    (ARTIFACT_DIR / "validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["model", "roc_auc", "brier", "brier_skill_vs_prevalence"]].to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
