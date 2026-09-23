"""Visualize the process: where each dimension started and where it landed.

Two panels: (1) the fairness gender-FPR-gap journey (0.5 audit -> deployed-point audit ->
group-threshold mitigation), the clearest 'we improved this' story; (2) the XGBoost accuracy
journey (baseline -> ordinal -> CV-tuned, with the rejected deeper-tree attempt marked).
Numbers are measured in this project; see JOURNEY.md for the full log and sources.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.config import ARTIFACT_DIR, FIGURE_DIR

TEAL, ORANGE, RED, GREY, NAVY = "#2A9D8F", "#FB8500", "#E76F51", "#828FA0", "#0A1F44"


def main() -> None:
    inf = pd.read_csv(ARTIFACT_DIR / "fairness_inference.csv")
    front = pd.read_csv(ARTIFACT_DIR / "fairness_frontier.csv")

    def gender_fpr(rule):
        r = inf[(inf.rule == rule) & (inf.attribute == "Gender") & (inf.metric == "fpr")]
        return r.set_index("model").gap

    at05, at20 = gender_fpr("threshold_0.5"), gender_fpr("top_20pct")
    mit = front[(front.attribute == "Gender") & (front.method == "group_thresholds_equal_fpr")].set_index("model").fpr_gap.abs()
    models = ["logistic", "xgboost", "tabicl"]
    labels = {"logistic": "Logistic", "xgboost": "XGBoost", "tabicl": "TabICLv2"}

    fig, (axf, axa) = plt.subplots(1, 2, figsize=(17, 6.2))

    stages = ["Audit @0.5\n(naive)", "Audit @ top-20%\n(deployed point)", "Group-threshold\nmitigation"]
    x = range(len(stages))
    for model, colour in zip(models, [NAVY, ORANGE, "#7B2CBF"]):
        axf.plot(x, [at05[model], at20[model], mit[model]], marker="o", ms=11, lw=3,
                 color=colour, label=labels[model])
    axf.set_xticks(list(x), stages, fontsize=11)
    axf.set_ylabel("Gender FPR gap (M − F)")
    axf.set_title("Fairness journey: gender false-positive gap", fontweight="bold")
    axf.axhline(0, color="grey", lw=1)
    axf.legend(fontsize=10)
    axf.annotate("Same model — the naive 0.5\naudit overstated the gap 2–4×",
                 xy=(1, at20["tabicl"]), xytext=(0.35, 0.20), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=GREY))

    stages_a = ["Repo baseline\n(one-hot, hand-set)", "Rejected:\ndeeper trees",
                "+ ordinal\nencoding", "+ 5-fold CV\nsearch (kept)"]
    auc = [0.7299, 0.7269, 0.7314, 0.7326]
    colours = [GREY, RED, ORANGE, TEAL]
    bars = axa.bar(stages_a, auc, color=colours)
    axa.set_ylim(0.724, 0.734)
    axa.set_ylabel("XGBoost held-out ROC AUC")
    axa.set_title("Accuracy journey: XGBoost (near the data ceiling)", fontweight="bold")
    axa.axhline(0.7338, ls="--", color="#7B2CBF", lw=1.5)
    axa.text(3.4, 0.7339, "TabICL 0.734", color="#7B2CBF", fontsize=9, ha="right")
    for bar, v in zip(bars, auc):
        axa.text(bar.get_x() + bar.get_width() / 2, v + 0.0002, f"{v:.4f}", ha="center", fontsize=10)
    axa.tick_params(axis="x", labelsize=10)

    fig.suptitle("Where we started and where we landed (measured)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURE_DIR / "improvement_journey.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved artifacts/figures/improvement_journey.png")
    print("gender FPR gap @0.5:", at05.round(3).to_dict())
    print("gender FPR gap @top20:", at20.round(3).to_dict())
    print("gender FPR gap mitigated:", mit.round(3).to_dict())


if __name__ == "__main__":
    main()
