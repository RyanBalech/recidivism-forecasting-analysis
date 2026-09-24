"""Average marginal effects and categorical probability contrasts.

A logistic coefficient is a log-odds change per *transformed* unit, which is not
the quantity a client can act on. Two readings that are (course reference:
printed slides 33-38):

1. **Average marginal effect (AME)** for numeric inputs: the mean change in
   predicted probability per one raw unit, averaged over the evaluation cohort.
   Computed by finite difference on the raw feature so the whole pipeline -
   imputation, scaling, encoding - is included.

2. **Average probability contrast** for categorical inputs: set the feature to a
   level for everyone, then to the reference level for everyone, and average the
   difference in predicted probability. This is valid because each counterfactual
   row holds exactly one real category, unlike perturbing one-hot columns
   independently.

Both are computed for the white-box model and for XGBoost, so the same quantity
can be compared across a linear and a nonlinear model.
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

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, MODEL_DIR
from recidivism.data import load_official_split

MODELS = ["logistic", "xgboost"]
LABELS = {"logistic": "Logistic regression", "xgboost": "XGBoost"}


def average_marginal_effects(pipe, X: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Mean dP/dx per raw unit, by central finite difference."""
    rows = []
    base = pipe.predict_proba(X)[:, 1]
    for col in numeric:
        values = X[col].dropna()
        if values.empty or values.nunique() < 2:
            continue
        # Step scaled to the feature's own spread, floored so integer-like counts move.
        step = max(float(values.std()) * 0.01, 1e-3)
        up, down = X.copy(), X.copy()
        up[col] = X[col] + step
        down[col] = X[col] - step
        effect = (pipe.predict_proba(up)[:, 1] - pipe.predict_proba(down)[:, 1]) / (2 * step)
        rows.append({
            "feature": col,
            "average_marginal_effect": float(np.mean(effect)),
            "sd_marginal_effect": float(np.std(effect)),
            "share_positive": float(np.mean(effect > 0)),
            "baseline_mean_probability": float(base.mean()),
        })
    return pd.DataFrame(rows)


def probability_contrasts(pipe, X: pd.DataFrame, categorical: list[str],
                          references: dict[str, str]) -> pd.DataFrame:
    """Mean P(level) - P(reference level), holding every other feature fixed."""
    rows = []
    for col in categorical:
        levels = [v for v in X[col].dropna().unique()]
        ref = references[col]
        ref_frame = X.copy()
        ref_frame[col] = ref
        p_ref = pipe.predict_proba(ref_frame)[:, 1]
        for level in levels:
            if str(level) == str(ref):
                continue
            alt = X.copy()
            alt[col] = level
            p_alt = pipe.predict_proba(alt)[:, 1]
            diff = p_alt - p_ref
            rows.append({
                "feature": col,
                "level": str(level),
                "reference": str(ref),
                "average_probability_contrast": float(np.mean(diff)),
                "sd_contrast": float(np.std(diff)),
                "n_with_level": int((X[col].astype(str) == str(level)).sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    split = load_official_split()
    X = split.X_test
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    references = {c: split.X_train[c].mode(dropna=True).iloc[0] for c in categorical}

    ame_frames, contrast_frames = [], []
    for name in MODELS:
        pipe = joblib.load(MODEL_DIR / f"{name}.joblib")
        ame = average_marginal_effects(pipe, X, numeric)
        ame.insert(0, "model", name)
        ame_frames.append(ame)
        con = probability_contrasts(pipe, X, categorical, references)
        con.insert(0, "model", name)
        contrast_frames.append(con)

    ame_all = pd.concat(ame_frames, ignore_index=True)
    con_all = pd.concat(contrast_frames, ignore_index=True)
    ame_all.to_csv(ARTIFACT_DIR / "logistic_marginal_effects.csv", index=False)
    con_all.to_csv(ARTIFACT_DIR / "probability_contrasts.csv", index=False)

    _figure(ame_all, con_all)

    top = (ame_all[ame_all.model == "logistic"]
           .reindex(ame_all[ame_all.model == "logistic"]
                    .average_marginal_effect.abs().sort_values(ascending=False).index).head(6))
    print("Largest average marginal effects (logistic, probability per raw unit):")
    print(top[["feature", "average_marginal_effect", "share_positive"]]
          .to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    big = (con_all[con_all.model == "logistic"]
           .reindex(con_all[con_all.model == "logistic"]
                    .average_probability_contrast.abs().sort_values(ascending=False).index).head(6))
    print("\nLargest categorical probability contrasts (logistic):")
    print(big[["feature", "level", "reference", "average_probability_contrast"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


def _figure(ame: pd.DataFrame, con: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(19, 8))

    piv = ame.pivot(index="feature", columns="model", values="average_marginal_effect").dropna()
    piv = piv.reindex(piv.abs().max(axis=1).sort_values().index).tail(12)
    y = np.arange(len(piv))
    axes[0].barh(y - 0.2, piv["logistic"], height=0.4, label=LABELS["logistic"], color="#234E70")
    axes[0].barh(y + 0.2, piv["xgboost"], height=0.4, label=LABELS["xgboost"], color="#FB8500")
    axes[0].set_yticks(y, [f.replace("_", " ") for f in piv.index], fontsize=9)
    axes[0].axvline(0, color="grey", lw=1)
    axes[0].set(title="Average marginal effect\n(probability change per raw unit)", xlabel="dP/dx")
    axes[0].legend(fontsize=9)

    c = con[con.model == "logistic"].copy()
    c["label"] = c.feature.str.replace("_", " ") + " = " + c.level
    c = c.reindex(c.average_probability_contrast.abs().sort_values().index).tail(12)
    colors = ["#C1121F" if v > 0 else "#2A9D8F" for v in c.average_probability_contrast]
    axes[1].barh(c.label, c.average_probability_contrast, color=colors)
    axes[1].axvline(0, color="grey", lw=1)
    axes[1].set(title="Average probability contrast vs reference level\n(logistic)",
                xlabel="Mean change in predicted probability")
    axes[1].tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "marginal_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
