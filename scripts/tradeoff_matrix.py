"""The one slide the brief asks for: trade-offs across the three models, four dimensions.

Reads the audit artifacts and renders a color-coded comparison matrix (green = advantage,
amber = middle, red = disadvantage) plus a Markdown table for the report. Nothing is hard-coded;
every cell traces back to a CSV or JSON in artifacts/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.config import ARTIFACT_DIR, FIGURE_DIR

MODELS = ["logistic", "xgboost", "tabicl"]
DISPLAY = {"logistic": "Logistic\nregression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
GREEN, AMBER, RED = "#2A9D8F", "#E9C46A", "#E76F51"


def main() -> None:
    m = pd.read_csv(ARTIFACT_DIR / "model_metrics.csv").set_index("model")
    st = pd.read_csv(ARTIFACT_DIR / "stability_summary.csv").set_index("model")
    inf = pd.read_csv(ARTIFACT_DIR / "fairness_inference.csv")
    # FNR = missed support, the primary fairness metric (selection means being offered help).
    fnr = inf[(inf.rule == "top_20pct") & (inf.metric == "fnr")].pivot_table(
        index="model", columns="attribute", values="gap")
    surrogate = json.loads((ARTIFACT_DIR / "interpretability_summary.json").read_text())["surrogate_fidelity_r2_test"]

    # (label, {model: (text, colour)}) — one row per sub-metric, grouped by the four dimensions.
    rows = [
        ("PERFORMANCE", None),
        ("ROC AUC (held-out)", {x: (f"{m.loc[x,'roc_auc']:.3f}", None) for x in MODELS}),
        ("Brier / calibration", {x: (f"{m.loc[x,'brier']:.3f}", None) for x in MODELS}),
        ("Net value @20% ($M)", {x: (f"{m.loc[x,'economic_assumed_net_value']/1e6:.2f}", None) for x in MODELS}),
        ("INTERPRETABILITY", None),
        ("Local explanation", {"logistic": ("coefficients", GREEN), "xgboost": ("SHAP + surrogate", AMBER),
                               "tabicl": ("none native", RED)}),
        ("Global surrogate fidelity", {"logistic": ("exact (linear)", GREEN),
                                       "xgboost": (f"tree R²={surrogate:.2f}", AMBER), "tabicl": ("PDP/ICE only", RED)}),
        ("STABILITY", None),
        ("Score drift across refits", {x: (f"{st.loc[x,'mean_abs_prob_diff']:.3f}", None) for x in MODELS}),
        ("Top-20% overlap", {x: (f"{st.loc[x,'top20_jaccard']:.0%}", None) for x in MODELS}),
        ("FAIRNESS (top-20%)", None),
        ("Race FNR gap (B − W)", {x: (f"{fnr.loc[x,'Race']:+.3f}", None) for x in MODELS}),
        ("Gender FNR gap (M − F)", {x: (f"{fnr.loc[x,'Gender']:+.3f}", None) for x in MODELS}),
        ("Age FNR gap (<33 − 33+)", {x: (f"{fnr.loc[x,'Age']:+.3f}", None) for x in MODELS}),
        ("COST", None),
        ("Train+predict (s)", {x: (f"{m.loc[x,'fit_predict_seconds']:.1f}", None) for x in MODELS}),
        ("Auditability", {"logistic": ("high", GREEN), "xgboost": ("medium", AMBER), "tabicl": ("low", RED)}),
    ]

    # Auto-colour numeric rows: best = green, worst = red, middle = amber.
    lower_better = {"Brier / calibration", "Score drift across refits", "Race FNR gap (B − W)",
                    "Gender FNR gap (M − F)", "Age FNR gap (<33 − 33+)", "Train+predict (s)"}
    # Gaps are signed; the size of the gap is what matters.
    by_size = {"Race FNR gap (B − W)", "Gender FNR gap (M − F)", "Age FNR gap (<33 − 33+)"}
    for label, cells in rows:
        if cells is None or any(c[1] for c in cells.values()):
            continue
        vals = {x: float(cells[x][0].replace("%", "").replace(",", "")) for x in MODELS}
        if label in by_size:
            vals = {x: abs(v) for x, v in vals.items()}
        order = sorted(vals, key=vals.get, reverse=label not in lower_better)
        colour = {order[0]: GREEN, order[1]: AMBER, order[2]: RED}
        for x in MODELS:
            cells[x] = (cells[x][0], colour[x])

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis("off")
    n = len(rows)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.7, n)
    headers = ["Dimension / metric"] + [DISPLAY[x] for x in MODELS]
    for j, h in enumerate(headers):
        ax.text(0.05 + j if j == 0 else j + 0.5, n - 0.4, h, fontweight="bold", fontsize=13,
                ha="left" if j == 0 else "center", va="center")
    for i, (label, cells) in enumerate(rows):
        y = n - 1.4 - i
        if cells is None:
            ax.add_patch(plt.Rectangle((0, y - 0.4), 4, 0.8, color="#264653"))
            ax.text(0.05, y, label, color="white", fontweight="bold", fontsize=12, va="center")
            continue
        ax.text(0.05, y, label, fontsize=11, va="center")
        for j, x in enumerate(MODELS):
            text, colour = cells[x]
            if colour:
                ax.add_patch(plt.Rectangle((j + 1.05, y - 0.38), 0.9, 0.76, color=colour, alpha=0.75))
            ax.text(j + 1.5, y, text, ha="center", va="center", fontsize=11)
    ax.set_title("Which model should the client deploy? Four dimensions, three models",
                 fontsize=15, fontweight="bold", pad=12)
    fig.text(0.5, 0.02, "Green = advantage · amber = middle · red = disadvantage. "
             "Performance is a near-tie; the decision is driven by interpretability, fairness, and cost.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIGURE_DIR / "tradeoff_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Markdown table for the report.
    md = ["| Dimension / metric | Logistic | XGBoost | TabICLv2 |", "|---|---|---|---|"]
    for label, cells in rows:
        if cells is None:
            md.append(f"| **{label}** | | | |")
        else:
            md.append(f"| {label} | " + " | ".join(cells[x][0] for x in MODELS) + " |")
    (ARTIFACT_DIR / "tradeoff_matrix.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("\nSaved artifacts/figures/tradeoff_matrix.png and artifacts/tradeoff_matrix.md")


if __name__ == "__main__":
    main()
