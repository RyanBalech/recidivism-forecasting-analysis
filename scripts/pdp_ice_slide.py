"""PDP + ICE for age, gang affiliation and prior felony arrests, all three models (global interpretability).

Same 200 evaluation people as interpretability.py (first 200 of a 1,000-row draw with the project
seed). Each person is set to every age band, gang No / Yes and every felony count; each prediction
is one point on that person's ICE curve, and the average over people is the PDP. Per-person values
are saved, and a rerun only predicts the (model, feature) pairs missing from the CSV, so the figure
can be redrawn without predicting again:

    python scripts/pdp_ice_slide.py                 # predict (TabICL: ~1 hour on CPU) and draw
    python scripts/pdp_ice_slide.py --figure-only   # redraw from artifacts/ice_age_gang.csv

Logistic and XGBoost use the saved models; TabICLv2 is refitted on the training set with the
shipped configuration (in-context models are not saved).
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

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, MODEL_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.modeling import tabicl_frames, tabicl_model

warnings.filterwarnings("ignore")
MODELS = ["logistic", "xgboost", "tabicl"]
DISPLAY = {"logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500", "tabicl": "#7B2CBF"}
AGES = ["18-22", "23-27", "28-32", "33-37", "38-42", "43-47", "48 or older"]
GANG = ["No", "Yes"]
FELONY = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10 or more"]
FEATURES = {"Age_at_Release": (AGES, "Age at release", ("23-27", "48 or older"), "23–27 → 48+"),
            "Gang_Affiliated": (GANG, "Gang affiliated", ("No", "Yes"), "No → Yes"),
            "Prior_Arrest_Episodes_Felony": (FELONY, "Prior felony arrests", ("1", "10 or more"), "1 → 10+")}
OUT = ARTIFACT_DIR / "ice_age_gang.csv"


def compute() -> pd.DataFrame:
    done = pd.read_csv(OUT, dtype={"value": str}) if OUT.exists() else pd.DataFrame(columns=["model", "feature"])
    have = set(zip(done.model, done.feature))
    todo = {m: [f for f in FEATURES if (m, f) not in have] for m in MODELS}
    if not any(todo.values()):
        return done

    split = load_official_split()
    rng = np.random.default_rng(RANDOM_SEED)
    sample = split.X_test.iloc[rng.choice(len(split.X_test), 1000, replace=False)].reset_index(drop=True)
    people = sample.iloc[:200].reset_index(drop=True)

    predictors = {m: joblib.load(MODEL_DIR / f"{m}.joblib").predict_proba for m in ["logistic", "xgboost"]}
    if todo["tabicl"]:
        tab = tabicl_model(RANDOM_SEED)
        tab.fit(tabicl_frames(split.X_train, split.X_train)[0], split.y_train.to_numpy())
        predictors["tabicl"] = lambda X: tab.predict_proba(tabicl_frames(split.X_train, X)[1])

    rows = []
    for model in MODELS:
        if not todo[model]:
            continue
        settings = [(f, v) for f in todo[model] for v in FEATURES[f][0]]
        batch = pd.concat([people.assign(**{f: v}) for f, v in settings], ignore_index=True)
        p = predictors[model](batch)[:, 1].reshape(len(settings), len(people))
        for (feature, value), preds in zip(settings, p):
            rows += [{"model": model, "feature": feature, "value": value, "person": i, "p": float(v)}
                     for i, v in enumerate(preds)]
        print(f"{model}: {', '.join(todo[model])} done", flush=True)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(OUT, index=False)
    return out


def figure(ice: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5), sharey=True)
    for c, model in enumerate(MODELS):
        for r, (feature, (values, label, (first, last), what)) in enumerate(FEATURES.items()):
            ax = axes[r][c]
            part = ice[(ice.model == model) & (ice.feature == feature)]
            wide = part.pivot(index="person", columns="value", values="p")[values]
            x = np.arange(len(values))
            for _, curve in wide.iterrows():
                ax.plot(x, curve.to_numpy(), color=PALETTE[model], alpha=0.07, lw=1)
            pdp = wide.mean()
            ax.plot(x, pdp.to_numpy(), color="black", lw=3, marker="o", ms=5, label="PDP (average)")
            delta = (pdp[last] - pdp[first]) * 100
            ax.set_title(f"{DISPLAY[model]} · {label}\nPDP {what}: {delta:+.0f} pts", fontsize=12)
            ax.set_xticks(x, [v.replace(" or more", "+").replace(" or older", "+") for v in values],
                          rotation=35 if feature == "Age_at_Release" else 0, fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("Predicted re-arrest risk", fontsize=11)
    axes[0][0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Global view, same tool for all three models: PDP (black) and ICE (one faint line per person, 200 people)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pdp_ice_age_gang_felony.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_tabicl(ice: pd.DataFrame) -> None:
    """Slide 9, TabICLv2 column: no coefficients, no fast SHAP, so read it from outside with PDP/ICE."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 8.6), sharey=True)
    for ax, (feature, (values, label, (first, last), what)) in zip(axes, FEATURES.items()):
        part = ice[(ice.model == "tabicl") & (ice.feature == feature)]
        wide = part.pivot(index="person", columns="value", values="p")[values]
        x = np.arange(len(values))
        for _, curve in wide.iterrows():
            ax.plot(x, curve.to_numpy(), color=PALETTE["tabicl"], alpha=0.08, lw=1)
        pdp = wide.mean()
        ax.plot(x, pdp.to_numpy(), color="black", lw=3, marker="o", ms=6, label="PDP (average)")
        ax.plot([], [], color=PALETTE["tabicl"], lw=1.5, alpha=0.6, label="ICE (one person)")
        ax.set_title(f"{label}: PDP {what} {(pdp[last] - pdp[first]) * 100:+.0f} pts", fontsize=13)
        ax.set_xticks(x, [v.replace(" or more", "+").replace(" or older", "+") for v in values], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Predicted re-arrest risk", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=10, loc="upper right")
    fig.suptitle("TabICLv2: PDP (black) + ICE (200 people)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pdp_ice_tabicl.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ice = pd.read_csv(OUT, dtype={"value": str}) if "--figure-only" in sys.argv else compute()
    figure(ice)
    figure_tabicl(ice)
    print("Saved artifacts/figures/pdp_ice_age_gang_felony.png, pdp_ice_tabicl.png")


if __name__ == "__main__":
    main()
