"""Interpretability for all three models.

- SHAP (global mean |SHAP| + one individual waterfall) for logistic and XGBoost.
- Global surrogate: a depth-3 tree fitted to XGBoost's predictions, with fidelity.
- PDP + ICE for all three models (model-agnostic, so TabICL is covered).
- One LIME explanation for the same individual, to compare with SHAP.
TabICL has no native attribution path; KernelSHAP over the full test set is impractical on CPU.
That is reported as a deployment cost, not hidden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, MODEL_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.modeling import ordinal_encode, tabicl_frames

PDP_FEATURES = ["Age_at_Release", "Prior_Arrest_Episodes_Felony", "Supervision_Risk_Score_First"]
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500", "tabicl": "#7B2CBF"}
DISPLAY = {"logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}


def original_feature(name: str, columns: list[str]) -> str:
    """Map a transformed column (one-hot level, missing indicator) back to its source feature."""
    name = name.removeprefix("missingindicator_")
    matches = [c for c in columns if name == c or name.startswith(c + "_")]
    return max(matches, key=len) if matches else name


def shap_for(pipe, X_background: pd.DataFrame, X_explain: pd.DataFrame, kind: str):
    """SHAP values on the final estimator, summed back to the original features."""
    transform = pipe[:-1]
    Xb = transform.transform(X_background)
    Xe = transform.transform(X_explain)
    Xb = Xb.toarray() if hasattr(Xb, "toarray") else Xb
    Xe = Xe.toarray() if hasattr(Xe, "toarray") else Xe
    names = list(pipe[-2].get_feature_names_out())
    if kind == "logistic":
        explainer = shap.LinearExplainer(pipe[-1], Xb)
    else:
        explainer = shap.TreeExplainer(pipe[-1])
    values = explainer.shap_values(Xe)
    base = float(np.ravel(explainer.expected_value)[0])
    cols = list(X_explain.columns)
    grouped = pd.DataFrame(values, columns=names).T.groupby(lambda n: original_feature(n, cols)).sum().T
    return grouped[[c for c in cols if c in grouped.columns]], base


def main() -> None:
    split = load_official_split()
    X_train, X_test = split.X_train, split.X_test
    rng = np.random.default_rng(RANDOM_SEED)
    sample = X_test.iloc[rng.choice(len(X_test), 1000, replace=False)].reset_index(drop=True)
    background = X_train.iloc[rng.choice(len(X_train), 500, replace=False)]
    models = {m: joblib.load(MODEL_DIR / f"{m}.joblib") for m in ["logistic", "xgboost"]}
    summary = {}

    # Individual to explain everywhere: the highest-risk person under XGBoost in the sample.
    p_xgb = models["xgboost"].predict_proba(sample)[:, 1]
    person_idx = int(np.argmax(p_xgb))
    person = sample.iloc[[person_idx]]
    person.to_csv(ARTIFACT_DIR / "explained_individual.csv", index=False)

    # --- SHAP: global + individual, logistic and XGBoost ---
    sns.set_theme(style="whitegrid", context="talk")
    fig_g, axes_g = plt.subplots(1, 2, figsize=(16, 7))
    fig_w, axes_w = plt.subplots(1, 2, figsize=(18, 7))
    shap_rows = []
    for ax_g, ax_w, name in zip(axes_g, axes_w, ["logistic", "xgboost"]):
        values, base = shap_for(models[name], background, sample, name)
        importance = values.abs().mean().sort_values(ascending=False)
        for feat, v in importance.items():
            shap_rows.append({"model": name, "feature": feat, "mean_abs_shap": v})
        top = importance.head(10)[::-1]
        ax_g.barh(top.index.str.replace("_", " "), top.values, color=PALETTE[name])
        ax_g.set(title=f"{DISPLAY[name]}: global SHAP", xlabel="Mean |SHAP| (log-odds)")

        contrib = values.iloc[person_idx].sort_values(key=np.abs, ascending=False).head(10)[::-1]
        colors = ["#C1121F" if v > 0 else "#2A9D8F" for v in contrib.values]
        ax_w.barh(contrib.index.str.replace("_", " "), contrib.values, color=colors)
        ax_w.axvline(0, color="grey", lw=1)
        logit = base + values.iloc[person_idx].sum()
        ax_w.set(title=f"{DISPLAY[name]}: why this person? p={1 / (1 + np.exp(-logit)):.2f}",
                 xlabel="SHAP contribution (log-odds); red raises risk")
        summary[f"{name}_individual_probability"] = float(models[name].predict_proba(person)[:, 1][0])
    for fig, fname in [(fig_g, "shap_global.png"), (fig_w, "shap_individual.png")]:
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / fname, dpi=180, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(shap_rows).to_csv(ARTIFACT_DIR / "shap_importance.csv", index=False)

    # --- Global surrogate: shallow tree mimicking XGBoost ---
    Xs_train = pd.get_dummies(ordinal_encode(X_train), dummy_na=False).fillna(-1)
    Xs_test = pd.get_dummies(ordinal_encode(X_test), dummy_na=False).reindex(columns=Xs_train.columns, fill_value=0).fillna(-1)
    target_train = models["xgboost"].predict_proba(X_train)[:, 1]
    target_test = models["xgboost"].predict_proba(X_test)[:, 1]
    surrogate = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=RANDOM_SEED)
    surrogate.fit(Xs_train, target_train)
    fidelity = r2_score(target_test, surrogate.predict(Xs_test))
    summary["surrogate_depth"] = 3
    summary["surrogate_fidelity_r2_test"] = float(fidelity)
    (ARTIFACT_DIR / "surrogate_tree.txt").write_text(export_text(surrogate, feature_names=list(Xs_train.columns)), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(surrogate, feature_names=[c.replace("_", " ") for c in Xs_train.columns], filled=True,
              rounded=True, fontsize=10, impurity=False, precision=2, ax=ax)
    ax.set_title(f"Global surrogate of XGBoost (depth 3, test fidelity R² = {fidelity:.2f})")
    fig.savefig(FIGURE_DIR / "global_surrogate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- LIME: same individual, XGBoost, on the model's preprocessed feature space ---
    try:
        from lime.lime_tabular import LimeTabularExplainer
        pipe = models["xgboost"]
        transform = pipe[:-1]
        dense = lambda a: a.toarray() if hasattr(a, "toarray") else a
        train_arr = dense(transform.transform(X_train))
        names = list(pipe[-2].get_feature_names_out())
        explainer = LimeTabularExplainer(train_arr, feature_names=names, class_names=["no", "yes"],
                                         discretize_continuous=True, random_state=RANDOM_SEED)
        exp = explainer.explain_instance(dense(transform.transform(person))[0], pipe[-1].predict_proba,
                                         num_features=10, num_samples=3000)
        lime_rows = pd.DataFrame(exp.as_list(), columns=["condition", "weight"])
        lime_rows.to_csv(ARTIFACT_DIR / "lime_individual.csv", index=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        lr = lime_rows[::-1]
        ax.barh(lr.condition, lr.weight, color=["#C1121F" if w > 0 else "#2A9D8F" for w in lr.weight])
        ax.set(title="LIME (XGBoost), same individual: local linear surrogate", xlabel="Weight")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "lime_individual.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        summary["lime"] = "ok (features shown in scaled, preprocessed space)"
    except Exception as exc:  # LIME is illustrative; never block the pipeline on it
        summary["lime"] = f"failed: {exc!r}"

    # --- PDP + ICE for all three models ---
    from tabicl import TabICLClassifier
    tab = TabICLClassifier(n_estimators=16, random_state=RANDOM_SEED, n_jobs=-1)
    tab.fit(*tabicl_frames(X_train, X_train)[:1], split.y_train.to_numpy())
    predictors = {"logistic": models["logistic"].predict_proba, "xgboost": models["xgboost"].predict_proba,
                  "tabicl": lambda X: tab.predict_proba(tabicl_frames(X_train, X)[1])}
    ice_rows = sample.iloc[:200].reset_index(drop=True)
    pdp_records = []
    fig, axes = plt.subplots(len(PDP_FEATURES), 3, figsize=(20, 5.2 * len(PDP_FEATURES)), squeeze=False)
    for r, feat in enumerate(PDP_FEATURES):
        values = X_train[feat].dropna().unique()
        if feat == "Age_at_Release":
            order = {v: k for k, v in enumerate(["18-22", "23-27", "28-32", "33-37", "38-42", "43-47", "48 or older"])}
            values = sorted(values, key=order.get)
        elif pd.api.types.is_numeric_dtype(X_train[feat]):
            values = sorted(values)
        else:
            values = sorted(values, key=lambda s: int(str(s).split()[0]))
        for c, (name, predict) in enumerate(predictors.items()):
            batch = pd.concat([ice_rows.assign(**{feat: v}) for v in values], ignore_index=True)
            batch[feat] = batch[feat].astype(X_train[feat].dtype)
            probs = predict(batch)[:, 1].reshape(len(values), len(ice_rows))
            ax = axes[r][c]
            ax.plot(range(len(values)), probs, color=PALETTE[name], alpha=0.07, lw=1)
            ax.plot(range(len(values)), probs.mean(axis=1), color="black", lw=3, label="PDP (average)")
            ax.set_xticks(range(len(values)), [str(v) for v in values], rotation=40, fontsize=9)
            ax.set(title=f"{DISPLAY[name]}: {feat.replace('_', ' ')}", ylabel="Predicted risk", ylim=(0, 1))
            for v, mean in zip(values, probs.mean(axis=1)):
                pdp_records.append({"model": name, "feature": feat, "value": str(v), "pdp": mean})
    axes[0][0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pdp_ice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(pdp_records).to_csv(ARTIFACT_DIR / "pdp_values.csv", index=False)

    summary["tabicl_native_attribution"] = "none: explained only through model-agnostic PDP/ICE and permutation importance"
    (ARTIFACT_DIR / "interpretability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
