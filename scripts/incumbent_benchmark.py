"""Incumbent benchmark: is any model better than the tool agencies already use?

Georgia's existing actuarial tool is `Supervision_Risk_Score_First` (1-10). We compare
discrimination and economic value of that incumbent score, a random-allocation baseline,
and the three models, then run a sensitivity sweep so the business case does not rest on a
single set of cost assumptions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics, economic_value

MODELS = ["logistic", "xgboost", "tabicl"]
DISPLAY = {"random": "Random allocation", "incumbent": "Incumbent score",
           "logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
PALETTE = {"random": "#9AA0A6", "incumbent": "#C1121F", "logistic": "#234E70",
           "xgboost": "#FB8500", "tabicl": "#7B2CBF"}


def build_scores() -> tuple[pd.Series, dict[str, np.ndarray]]:
    """Aligned held-out labels and each ranker's risk score in [0, 1]-ish order."""
    split = load_official_split()
    preds = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    y = split.y_test.to_numpy()
    assert (preds["actual"].to_numpy() == y).all(), "prediction file is out of sync with the split"
    assert np.array_equal(preds.ID, split.audit_test.ID), "prediction IDs are out of order"

    incumbent = split.X_test["Supervision_Risk_Score_First"]
    incumbent = incumbent.fillna(split.X_train["Supervision_Risk_Score_First"].median()).to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)

    scores = {"random": rng.random(len(y)), "incumbent": incumbent}
    for m in MODELS:
        scores[m] = preds[f"p_{m}"].to_numpy()
    return pd.Series(y, name="actual"), scores


def discrimination(y: np.ndarray, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        m = classification_metrics(y, np.clip(s / max(s.max(), 1e-9), 0, 1))
        rows.append({"ranker": name, "roc_auc": m["roc_auc"], "average_precision": m["average_precision"]})
    return pd.DataFrame(rows)


def economics(y: np.ndarray, scores: dict[str, np.ndarray], capacity: float = 0.20) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        e = economic_value(y, s, capacity=capacity)
        rows.append({"ranker": name, **e})
    return pd.DataFrame(rows)


def sweep_capacity(y, scores, grid) -> pd.DataFrame:
    rows = []
    for cap in grid:
        for name, s in scores.items():
            e = economic_value(y, s, capacity=cap)
            rows.append({"ranker": name, "capacity": cap,
                         "net_value": e["assumed_net_value"], "recall": e["recall_at_capacity"]})
    return pd.DataFrame(rows)


def sweep_effectiveness(y, scores, grid, capacity=0.20) -> pd.DataFrame:
    rows = []
    for eff in grid:
        for name, s in scores.items():
            e = economic_value(y, s, capacity=capacity, effectiveness=eff)
            rows.append({"ranker": name, "effectiveness": eff, "net_value": e["assumed_net_value"]})
    return pd.DataFrame(rows)


def main() -> None:
    y, scores = build_scores()
    disc = discrimination(y.to_numpy(), scores)
    econ = economics(y.to_numpy(), scores)
    cap_grid = np.round(np.arange(0.05, 0.51, 0.05), 2)
    eff_grid = np.round(np.arange(0.10, 0.41, 0.05), 2)
    cap = sweep_capacity(y.to_numpy(), scores, cap_grid)
    eff = sweep_effectiveness(y.to_numpy(), scores, eff_grid)

    disc.to_csv(ARTIFACT_DIR / "incumbent_discrimination.csv", index=False)
    econ.to_csv(ARTIFACT_DIR / "incumbent_economics.csv", index=False)
    cap.to_csv(ARTIFACT_DIR / "incumbent_capacity_sweep.csv", index=False)
    eff.to_csv(ARTIFACT_DIR / "incumbent_effectiveness_sweep.csv", index=False)

    print("Discrimination (AUC / AP):")
    print(disc.round(4).to_string(index=False))
    print("\nEconomic value at 20% capacity:")
    print(econ[["ranker", "captured_events", "recall_at_capacity", "assumed_net_value"]].round(3).to_string(index=False))

    order = ["random", "incumbent", "logistic", "xgboost", "tabicl"]
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

    d = disc.set_index("ranker").loc[order].reset_index()
    x = np.arange(len(d))
    axes[0].bar(x, d.roc_auc, color=[PALETTE[r] for r in d.ranker])
    axes[0].axhline(0.5, ls="--", color="grey", lw=1)
    axes[0].set(title="Discrimination (ROC AUC)", ylabel="AUC", ylim=(0.5, 0.76))
    axes[0].set_xticks(x, [DISPLAY[r] for r in d.ranker], rotation=25, ha="right", fontsize=10)

    e = econ.set_index("ranker").loc[order].reset_index()
    x = np.arange(len(e))
    axes[1].bar(x, e.assumed_net_value / 1e6, color=[PALETTE[r] for r in e.ranker])
    axes[1].set(title="Net value at 20% capacity", ylabel="Assumed net value ($M)")
    axes[1].set_xticks(x, [DISPLAY[r] for r in e.ranker], rotation=25, ha="right", fontsize=10)

    for name in order:
        g = cap[cap.ranker.eq(name)]
        axes[2].plot(g.capacity, g.net_value / 1e6, marker="o", lw=2.3, color=PALETTE[name], label=DISPLAY[name])
    axes[2].set(title="Net value vs service capacity", xlabel="Capacity (share served)", ylabel="Net value ($M)")
    axes[2].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "incumbent_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved figure artifacts/figures/incumbent_benchmark.png")


if __name__ == "__main__":
    main()
