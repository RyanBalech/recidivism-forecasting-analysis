"""Local interpretability for one person on the support cut-off, each model with its own tool (slide 10).

The person is taken from the 200 evaluation people used for PDP/ICE (pdp_ice_slide.py): the one whose
predicted risk is closest to the top-20% cut-off in all three models. Being on the cut-off is what
makes a local explanation matter: a small change decides whether the person is offered support.

- Logistic: coefficient x (value - training average), summed per original field. For a linear model
  this is exactly its SHAP value (LinearExplainer), so it is read from the model itself.
- XGBoost: TreeSHAP, exact and fast for trees.
- TabICLv2: no coefficients and no fast SHAP, so this person's own ICE curves (age, gang, prior
  felony arrests), read from artifacts/ice_age_gang.csv, against the cut-off.

    python scripts/local_person.py                 # also runs TabICL LIME if it is missing (~1 h on CPU)
    python scripts/local_person.py --tabicl-lime   # TabICL LIME only, replacing earlier TabICL rows
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from interpretability import shap_for
from pdp_ice_slide import FEATURES, PALETTE
from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, MODEL_DIR, RANDOM_SEED, pretty
from recidivism.data import load_official_split

warnings.filterwarnings("ignore")
MODELS = ["logistic", "xgboost", "tabicl"]
DISPLAY = {"logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
CAPACITY = 0.20
TOP = 8


def pick_person(people: pd.DataFrame, ice: pd.DataFrame, cut: dict) -> tuple[int, dict]:
    """Index (among the 200) of the person closest to the cut-off in all three models, and their risks."""
    age = ice[ice.feature == "Age_at_Release"]
    own = {m: np.array([age[(age.model == m) & (age.person == i) & (age.value == people.Age_at_Release.iloc[i])].p.iloc[0]
                        for i in range(len(people))]) for m in MODELS}
    distance = sum(np.abs(own[m] - cut[m]) for m in MODELS)
    i = int(np.argmin(distance))
    return i, {m: float(own[m][i]) for m in MODELS}


def contributions_figure(model: str, contrib: pd.Series, p: float, cut: float) -> None:
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:TOP])[::-1]
    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.barh([pretty(f) for f in top.index], top.values, color=["#C1121F" if v > 0 else "#2A9D8F" for v in top.values])
    ax.axvline(0, color="grey", lw=1)
    for y, v in enumerate(top.values):
        ax.text(v + (0.015 if v >= 0 else -0.015), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=11)
    lo, hi = min(top.min(), 0), max(top.max(), 0)
    ax.set_xlim(lo - 0.25 * (hi - lo) - 0.05, hi + 0.25 * (hi - lo) + 0.05)
    status = "offered support" if p >= cut else "not offered"
    ax.set_title(f"{DISPLAY[model]}: risk {p:.3f}, cut-off {cut:.3f}\n→ {status}", fontsize=13)
    ax.set_xlabel("Contribution (log-odds): red raises risk, green lowers it", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"local_{model}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def tabicl_figure(ice: pd.DataFrame, person: int, own: pd.Series, p: float, cut: float) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7, 8.6), sharey=True)
    for ax, (feature, (values, label, _, _)) in zip(axes, FEATURES.items()):
        curve = ice[(ice.model == "tabicl") & (ice.feature == feature) & (ice.person == person)].set_index("value").p[values]
        x = np.arange(len(values))
        ax.axhspan(cut, 1, color="#2A9D8F", alpha=0.10)
        ax.axhline(cut, color="#2A9D8F", lw=1.5, ls="--")
        ax.plot(x, curve.to_numpy(), color=PALETTE["tabicl"], lw=2.5, marker="o", ms=5)
        at = values.index(str(own[feature]))
        ax.plot(at, curve.iloc[at], "o", ms=13, mfc="none", mec="black", mew=2)
        ax.set_title(f"{label}: this person = {own[feature]}", fontsize=12)
        ax.set_xticks(x, [v.replace(" or more", "+").replace(" or older", "+") for v in values], fontsize=10)
        lo = max(0, min(curve.min(), cut) - 0.12)
        ax.set_ylim(lo, min(1, max(curve.max(), cut) + 0.08))
        ax.set_ylabel("Predicted risk", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].text(0.99, 0.96, f"above {cut:.3f} = offered support", transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=10, color="#1F7A70")
    fig.suptitle(f"TabICLv2: this person's ICE (risk {p:.3f}). Circle = actual value", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "local_tabicl.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def lime_person(split, one: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    """LIME on the same person (logistic, XGBoost; 3 seeds), each condition checked against SHAP's sign.

    Same category-aware LIME as lime_local_fidelity.py. TabICLv2 runs separately (lime_tabicl), with
    fewer perturbed rows, because each of its predictions is expensive.
    """
    from lime.lime_tabular import LimeTabularExplainer

    from lime_local_fidelity import NUM_FEATURES, NUM_SAMPLES, SEEDS, reference_space

    train_arr, one_arr, decode, cat_idx, cat_names = reference_space(split.X_train, one)
    names = list(split.X_train.columns)
    rows = []
    for model in ["logistic", "xgboost"]:
        pipe = joblib.load(MODEL_DIR / f"{model}.joblib")
        shap_sign = contributions[contributions.model == model].set_index("feature").contribution
        for seed in SEEDS:
            explainer = LimeTabularExplainer(train_arr, feature_names=names, class_names=["no", "yes"],
                                             categorical_features=cat_idx, categorical_names=cat_names,
                                             discretize_continuous=True, random_state=seed)
            exp = explainer.explain_instance(one_arr[0], lambda a: pipe.predict_proba(decode(a)),
                                             num_features=NUM_FEATURES, num_samples=NUM_SAMPLES)
            for condition, weight in exp.as_list():
                feature = max((f for f in names if condition.startswith(f) or f" {f} " in f" {condition} "), key=len)
                rows.append({"model": model, "seed": seed, "condition": condition, "feature": feature,
                             "weight": weight, "shap": float(shap_sign[feature]), "local_r2": float(exp.score)})
    out = pd.DataFrame(rows)
    out["same_sign"] = np.sign(out.weight) == np.sign(out.shap)
    path = ARTIFACT_DIR / "local_person_lime.csv"
    kept = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=out.columns)
    pd.concat([out, kept[kept.model == "tabicl"]], ignore_index=True).to_csv(path, index=False)  # keep TabICL rows
    return out


def lime_tabicl(split, one: pd.DataFrame, samples: int = 1_000) -> None:
    """LIME for TabICLv2 on the same person, appended to local_person_lime.csv.

    TabICLv2 predicts ~1-2 rows a second on CPU, so each run uses 1,000 perturbed rows instead of
    5,000 (3 seeds: about an hour). There is no SHAP to compare with; the sign check is left empty.
    """
    from lime.lime_tabular import LimeTabularExplainer

    from lime_local_fidelity import NUM_FEATURES, SEEDS, reference_space
    from recidivism.modeling import tabicl_frames, tabicl_model

    train_arr, one_arr, decode, cat_idx, cat_names = reference_space(split.X_train, one)
    names = list(split.X_train.columns)
    tab = tabicl_model(RANDOM_SEED)
    tab.fit(tabicl_frames(split.X_train, split.X_train)[0], split.y_train.to_numpy())
    rows = []
    for seed in SEEDS:
        explainer = LimeTabularExplainer(train_arr, feature_names=names, class_names=["no", "yes"],
                                         categorical_features=cat_idx, categorical_names=cat_names,
                                         discretize_continuous=True, random_state=seed)
        exp = explainer.explain_instance(one_arr[0], lambda a: tab.predict_proba(tabicl_frames(split.X_train, decode(a))[1]),
                                         num_features=NUM_FEATURES, num_samples=samples)
        for condition, weight in exp.as_list():
            feature = max((f for f in names if condition.startswith(f) or f" {f} " in f" {condition} "), key=len)
            rows.append({"model": "tabicl", "seed": seed, "condition": condition, "feature": feature,
                         "weight": weight, "shap": np.nan, "local_r2": float(exp.score), "same_sign": np.nan})
        print(f"tabicl LIME seed {seed} done", flush=True)
    path = ARTIFACT_DIR / "local_person_lime.csv"
    old = pd.read_csv(path)
    pd.concat([old[old.model != "tabicl"], pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def main() -> None:
    if "--tabicl-lime" in sys.argv:
        split = load_official_split()
        rng = np.random.default_rng(RANDOM_SEED)
        sample = split.X_test.iloc[rng.choice(len(split.X_test), 1000, replace=False)].reset_index(drop=True)
        person = int(pd.read_csv(ARTIFACT_DIR / "local_person.csv").person.iloc[0])
        lime_tabicl(split, sample.iloc[[person]].reset_index(drop=True))
        return

    split = load_official_split()
    rng = np.random.default_rng(RANDOM_SEED)
    sample = split.X_test.iloc[rng.choice(len(split.X_test), 1000, replace=False)].reset_index(drop=True)
    people = sample.iloc[:200].reset_index(drop=True)
    ice = pd.read_csv(ARTIFACT_DIR / "ice_age_gang.csv", dtype={"value": str})
    preds = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    cut = {m: float(np.quantile(preds[f"p_{m}"], 1 - CAPACITY)) for m in MODELS}

    i, risk = pick_person(people, ice, cut)
    one = people.iloc[[i]]
    rows = []
    for model in ["logistic", "xgboost"]:
        pipe = joblib.load(MODEL_DIR / f"{model}.joblib")
        risk[model] = float(pipe.predict_proba(one)[0, 1])
        values, base = shap_for(pipe, split.X_train, one, model)
        contrib = values.iloc[0]
        contributions_figure(model, contrib, risk[model], cut[model])
        rows += [{"model": model, "feature": f, "value": one[f].iloc[0], "contribution": float(v), "base": base}
                 for f, v in contrib.items()]
    tabicl_figure(ice, i, one.iloc[0], risk["tabicl"], cut["tabicl"])

    contributions = pd.DataFrame(rows)
    contributions.to_csv(ARTIFACT_DIR / "local_person_contributions.csv", index=False)
    lime = lime_person(split, one, contributions)
    agg = lime.groupby(["model", "condition"]).agg(weight=("weight", "mean"), shap=("shap", "first"),
                                                   same_sign=("same_sign", "all"), seeds=("seed", "size"))
    print(agg.sort_values("weight", key=abs, ascending=False).round(3).to_string())
    print(lime.groupby("model").local_r2.mean().round(3).to_string())
    if not (pd.read_csv(ARTIFACT_DIR / "local_person_lime.csv").model == "tabicl").any():
        lime_tabicl(split, one.reset_index(drop=True))
    gang = ice[(ice.person == i) & (ice.feature == "Gang_Affiliated")].set_index(["model", "value"]).p
    summary = pd.DataFrame([{"model": m, "person": i, "risk": risk[m], "cutoff": cut[m], "offered": risk[m] >= cut[m],
                             "risk_if_gang_no": gang[(m, "No")]} for m in MODELS])
    summary.to_csv(ARTIFACT_DIR / "local_person.csv", index=False)
    print(one.T.iloc[:, 0].to_string())
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
