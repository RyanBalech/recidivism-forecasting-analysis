"""Category-aware LIME with reported fidelity, over predetermined cases and seeds.

The earlier implementation explained the *transformed* one-hot space. Perturbing
one-hot columns independently creates impossible rows (two categories at once, or
none), so the local surrogate fitted a region the model never sees and scored
R2 ~= 0.25. Here LIME works on the raw feature space with `categorical_features`
declared, so a perturbation replaces a category with another real category
(https://github.com/marcotcr/lime).

Fidelity (the local surrogate R2) is reported next to every explanation rather
than left implicit: an explanation that does not approximate the model locally is
not evidence about the model. Course reference: fidelity vs interpretability,
printed slides 110-116.

Cases are fixed by rule before looking at any explanation: the highest, median and
lowest predicted risk in the evaluation cohort. Each case is explained under
several seeds so the reported weights carry a visible sampling spread.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, MODEL_DIR, RANDOM_SEED, pretty
from recidivism.data import load_official_split

SEEDS = [0, 1, 2]
NUM_FEATURES = 10
NUM_SAMPLES = 5_000
MODELS = ["logistic", "xgboost"]


def reference_space(X_train: pd.DataFrame, X_eval: pd.DataFrame):
    """Encode the raw frame for LIME: numerics median-filled, categoricals as codes.

    The two pipelines impute with exactly these statistics, so filling here leaves
    the model's own imputers as no-ops and keeps the explained space identical to
    the space the model scores.
    """
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]

    medians = X_train[numeric].median()
    modes = {c: X_train[c].mode(dropna=True).iloc[0] for c in categorical}
    levels = {c: sorted(map(str, X_train[c].fillna(modes[c]).unique())) for c in categorical}

    def encode(frame: pd.DataFrame) -> np.ndarray:
        out = pd.DataFrame(index=frame.index)
        for col in X_train.columns:
            if col in numeric:
                out[col] = frame[col].fillna(medians[col]).astype(float)
            else:
                lookup = {v: i for i, v in enumerate(levels[col])}
                filled = frame[col].fillna(modes[col]).astype(str)
                # An unseen category falls back to the training mode's code.
                out[col] = filled.map(lookup).fillna(lookup[str(modes[col])]).astype(float)
        return out[X_train.columns].to_numpy(dtype=float)

    def decode(arr: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(arr, columns=list(X_train.columns))
        for col in X_train.columns:
            if col in numeric:
                frame[col] = frame[col].astype(X_train[col].dtype)
            else:
                codes = np.clip(np.rint(frame[col].to_numpy()), 0, len(levels[col]) - 1).astype(int)
                frame[col] = [levels[col][c] for c in codes]
        return frame

    cat_idx = [list(X_train.columns).index(c) for c in categorical]
    cat_names = {list(X_train.columns).index(c): levels[c] for c in categorical}
    return encode(X_train), encode(X_eval), decode, cat_idx, cat_names


def main() -> None:
    from lime.lime_tabular import LimeTabularExplainer

    split = load_official_split()
    X_train, X_test = split.X_train, split.X_test
    train_arr, test_arr, decode, cat_idx, cat_names = reference_space(X_train, X_test)
    names = list(X_train.columns)

    models = {m: joblib.load(MODEL_DIR / f"{m}.joblib") for m in MODELS}

    # Cases fixed by rule, before any explanation is inspected.
    anchor = models["xgboost"].predict_proba(X_test)[:, 1]
    order = np.argsort(anchor)
    cases = {
        "highest_risk": int(order[-1]),
        "median_risk": int(order[len(order) // 2]),
        "lowest_risk": int(order[0]),
    }

    rows, fidelity = [], []
    for model_name, pipe in models.items():
        def predict(arr: np.ndarray, _pipe=pipe) -> np.ndarray:
            return _pipe.predict_proba(decode(arr))

        for case_name, idx in cases.items():
            for seed in SEEDS:
                explainer = LimeTabularExplainer(
                    train_arr,
                    feature_names=names,
                    class_names=["no", "yes"],
                    categorical_features=cat_idx,
                    categorical_names=cat_names,
                    discretize_continuous=True,
                    random_state=seed,
                )
                exp = explainer.explain_instance(
                    test_arr[idx], predict, num_features=NUM_FEATURES, num_samples=NUM_SAMPLES
                )
                for condition, weight in exp.as_list():
                    rows.append({"model": model_name, "case": case_name, "seed": seed,
                                 "condition": condition, "weight": weight})
                fidelity.append({
                    "model": model_name, "case": case_name, "seed": seed,
                    "local_r2": float(exp.score),
                    "local_intercept": float(exp.intercept[1]),
                    "model_probability": float(pipe.predict_proba(X_test.iloc[[idx]])[:, 1][0]),
                    "surrogate_prediction": float(exp.local_pred[0]),
                    "record_id": int(split.audit_test.iloc[idx]["ID"]),
                })

    weights = pd.DataFrame(rows)
    scores = pd.DataFrame(fidelity)
    weights.to_csv(ARTIFACT_DIR / "lime_weights.csv", index=False)
    scores.to_csv(ARTIFACT_DIR / "lime_fidelity.csv", index=False)

    # Stability of the explanation across seeds: does the same condition keep the same sign?
    spread = (weights.groupby(["model", "case", "condition"])
              .agg(mean_weight=("weight", "mean"), sd_weight=("weight", "std"), seeds=("weight", "size"))
              .reset_index())
    spread["sign_stable"] = spread.mean_weight.abs() > spread.sd_weight.fillna(0)
    spread.to_csv(ARTIFACT_DIR / "lime_seed_spread.csv", index=False)

    _figure(weights, scores)

    summary = scores.groupby("model").local_r2.agg(["mean", "min", "max"])
    print("LIME local fidelity (R2) by model:")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nsign-stable conditions: {int(spread.sign_stable.sum())} / {len(spread)}")


def _figure(weights: pd.DataFrame, scores: pd.DataFrame) -> None:
    cases = ["highest_risk", "median_risk", "lowest_risk"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, case in zip(axes, cases):
        part = weights[(weights.model == "xgboost") & (weights.case == case)]
        agg = (part.groupby("condition").weight.agg(["mean", "std"])
               .sort_values("mean").tail(10))
        colors = ["#C1121F" if v > 0 else "#2A9D8F" for v in agg["mean"]]
        ax.barh([pretty(c) for c in agg.index], agg["mean"], xerr=agg["std"].fillna(0), color=colors, capsize=3)
        r2 = scores[(scores.model == "xgboost") & (scores.case == case)].local_r2
        ax.set_title(f"{case.replace('_', ' ')}\nlocal fidelity R2 = {r2.mean():.2f}", fontsize=12)
        ax.set_xlabel("LIME weight (mean +/- sd over seeds)")
        ax.tick_params(labelsize=9)
    fig.suptitle("Category-aware LIME (XGBoost): explanation and its fidelity", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lime_individual.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "--figure-only" in sys.argv:  # redraw from saved results, no LIME rerun
        _figure(pd.read_csv(ARTIFACT_DIR / "lime_weights.csv"), pd.read_csv(ARTIFACT_DIR / "lime_fidelity.csv"))
    else:
        main()
