"""Visualize the process: where each dimension started and where it landed.

Two panels: (1) the fairness gender-FNR-gap journey (0.5 audit -> deployed-point audit ->
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

    # FNR (missed support) is the primary fairness metric: selection means being offered help.
    def gender_fnr(rule):
        r = inf[(inf.rule == rule) & (inf.attribute == "Gender") & (inf.metric == "fnr")]
        return r.set_index("model").gap

    at05, at20 = gender_fnr("threshold_0.5"), gender_fnr("top_20pct")
    mit = front[(front.attribute == "Gender") & (front.method == "group_thresholds_equal_fnr")].set_index("model").fnr_gap
    models = ["logistic", "xgboost", "tabicl"]
    labels = {"logistic": "Logistic", "xgboost": "XGBoost", "tabicl": "TabICLv2"}

    fig, (axf, axa) = plt.subplots(1, 2, figsize=(17, 6.2))

    stages = ["Audit @0.5\n(reference)", "Audit @ top-20%\n(deployed point)", "Group thresholds\n(in-sample)"]
    x = range(len(stages))
    for model, colour in zip(models, [NAVY, ORANGE, "#7B2CBF"]):
        axf.plot(x, [at05[model], at20[model], mit[model]], marker="o", ms=11, lw=3,
                 color=colour, label=labels[model])
    axf.set_xticks(list(x), stages, fontsize=11)
    axf.set_ylabel("Gender FNR gap (M − F); negative = women missed more")
    axf.set_title("Fairness journey: gender missed-support (FNR) gap", fontweight="bold")
    axf.axhline(0, color="grey", lw=1)
    axf.legend(fontsize=10)
    axf.annotate("Same models, different operating point:\nthe gap depends on where you audit",
                 xy=(1, at20["xgboost"]), xytext=(0.3, min(at05.min(), at20.min()) * 0.5), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=GREY))

    stages_a = ["Repo baseline\n(one-hot, hand-set)", "Rejected:\ndeeper trees",
                "+ ordinal\nencoding", "+ 5-fold CV\nsearch (kept)"]
    auc = [0.7299, 0.7269, 0.7314, 0.7326]
    colours = [GREY, RED, ORANGE, TEAL]
    bars = axa.bar(stages_a, auc, color=colours)
    axa.set_ylim(0.724, 0.734)
    axa.set_ylabel("XGBoost held-out ROC AUC")
    axa.set_title("Historical XGBoost development (reused evaluation set)", fontweight="bold")
    tab_auc = float(pd.read_csv(ARTIFACT_DIR / "model_metrics.csv").set_index("model").loc["tabicl", "roc_auc"])
    axa.axhline(tab_auc, ls="--", color="#7B2CBF", lw=1.5)
    axa.text(3.4, tab_auc + 0.0001, f"TabICL {tab_auc:.4f}", color="#7B2CBF", fontsize=9, ha="right")
    for bar, v in zip(bars, auc):
        axa.text(bar.get_x() + bar.get_width() / 2, v + 0.0002, f"{v:.4f}", ha="center", fontsize=10)
    axa.tick_params(axis="x", labelsize=10)

    fig.suptitle("Historical experiments and operating-point comparisons", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURE_DIR / "improvement_journey.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved artifacts/figures/improvement_journey.png")
    print("gender FNR gap @0.5:", at05.round(3).to_dict())
    print("gender FNR gap @top20:", at20.round(3).to_dict())
    print("gender FNR gap mitigated:", mit.round(3).to_dict())


if __name__ == "__main__":
    main()
