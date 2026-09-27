"""Figures made for the deck that no analysis script produces on its own.

1. eda_overview.png — training data only: re-arrest rate by race x gender, and missingness by
   gender (Gang_Affiliated is missing for every woman, which is the leak the talk hands to P2).
2. fairness_race_gaps.png / fairness_gender_gaps.png — one attribute each, at the deployed
   top-20% rule: selection, FNR and FPR gaps with 95% bootstrap CIs and the ±5-point TOST band.
   Reads artifacts/fairness_inference.csv, so it never disagrees with fairness_audit.py.
3. performance_benchmark.png — ROC curves plus the AUC scale used in reviews of US recidivism
   tools, with the random, historical-score and model AUCs placed on it.
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

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR
from recidivism.data import load_official_split

MODELS = ["logistic", "xgboost", "tabicl"]
DISPLAY = {"logistic": "Logistic", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500", "tabicl": "#7B2CBF"}
METRICS = {"selection_rate": "Selection rate", "fnr": "FNR (missed support)", "fpr": "FPR"}
DELTA = 0.05


def eda_overview() -> None:
    split = load_official_split()
    audit = split.audit_train.assign(y=split.y_train.to_numpy())
    rates = audit.groupby(["Race", "Gender"]).y.agg(["mean", "size"]).reset_index()
    missing = split.X_train.isna().groupby(audit["Gender"].to_numpy()).mean().T
    missing = missing[(missing > 0).any(axis=1)].rename(index=lambda c: c.replace("_", " "))

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_r, ax_m) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(2)
    for j, (gender, colour) in enumerate([("M", "#234E70"), ("F", "#E76F51")]):
        part = rates[rates.Gender == gender].set_index("Race").loc[["BLACK", "WHITE"]]
        bars = ax_r.bar(x + (j - 0.5) * 0.38, part["mean"], 0.36, color=colour, label="Men" if gender == "M" else "Women")
        for bar, (m, n) in zip(bars, zip(part["mean"], part["size"])):
            ax_r.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.2f}\nn={n:,}", ha="center", fontsize=11)
    ax_r.axhline(audit.y.mean(), color="grey", ls="--", lw=1.5)
    ax_r.text(1.55, audit.y.mean() + 0.01, f"overall {audit.y.mean():.1%}", ha="right", fontsize=11, color="grey")
    ax_r.set_xticks(x, ["Black", "White"])
    ax_r.set(ylim=(0, 0.8), ylabel="Re-arrested within 3 years", title="Re-arrest rate by race × gender")
    ax_r.legend(fontsize=11, loc="lower right")

    missing[["M", "F"]].plot.barh(ax=ax_m, color=["#234E70", "#E76F51"], width=0.7)
    for i, (m, f) in enumerate(zip(missing["M"], missing["F"])):
        ax_m.text(max(m, f) + 0.02, i, f"men {m:.0%} · women {f:.0%}", va="center", fontsize=11)
    ax_m.set(xlim=(0, 1.35), xlabel="Share missing", title="Missing values by gender")
    ax_m.legend(["Men", "Women"], fontsize=11, loc="upper right")
    fig.suptitle("Training data only (18,028 people)", fontsize=13, color="grey")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def attribute_gaps(attribute: str, label: str, out: str) -> None:
    inf = pd.read_csv(ARTIFACT_DIR / "fairness_inference.csv")
    part = inf[(inf.rule == "top_20pct") & (inf.attribute == attribute) & inf.metric.isin(METRICS)]
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(METRICS))
    for j, model in enumerate(MODELS):
        r = part[part.model == model].set_index("metric").loc[list(METRICS)]
        ax.errorbar(x + (j - 1) * 0.22, r.gap, yerr=[r.gap - r.ci_low, r.ci_high - r.gap], fmt="o", ms=10,
                    capsize=6, color=PALETTE[model], label=DISPLAY[model])
    ax.axhspan(-DELTA, DELTA, color="#2A9D8F", alpha=0.10, label=f"±{DELTA:.0%} TOST tolerance")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(x, list(METRICS.values()))
    ax.set(ylabel=f"Gap ({label})", title=f"{attribute} gaps at the deployed top-20% rule, 95% bootstrap CI")
    ax.legend(fontsize=11, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fpdp_focus(attribute: str, features: list[str], out: str) -> None:
    """Readable FPDP for a slide: three features instead of the full 29-panel grid."""
    from recidivism.config import pretty

    fp = pd.read_csv(ARTIFACT_DIR / "fpdp_values.csv")
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, len(features), figsize=(5.2 * len(features), 5), sharey=True)
    for ax, feat in zip(axes, features):
        for model in ["logistic", "xgboost"]:
            part = fp[(fp.model == model) & (fp.attribute == attribute) & (fp.feature == feat)]
            ax.plot(range(len(part)), part.p_equal_opportunity, marker="o", lw=2.5, color=PALETTE[model],
                    label=DISPLAY[model])
            ax.set_xticks(range(len(part)), part.value.astype(str), rotation=40, ha="right", fontsize=10)
        ax.axhline(0.05, color="#C1121F", lw=2)
        ax.set_title(pretty(feat), fontsize=14)
        ax.set_ylim(-0.03, 1.0)
    axes[0].set_ylabel("Equal-opportunity p-value")
    axes[0].legend(fontsize=11, loc="upper left")
    axes[-1].text(0.98, 0.07, "p = 0.05", transform=axes[-1].transAxes, ha="right", color="#C1121F", fontsize=11)
    fig.suptitle(f"FPDP ({attribute.lower()}): fix one feature for everyone, re-test fairness. Above the red line = candidate variable",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def performance_benchmark() -> None:
    """Slide 7: ROC curves for the three models, and where their AUC sits on the scale that
    US reviews of recidivism tools use (Desmarais & Singh 2013, anchored to Rice & Harris 2005)."""
    from sklearn.metrics import roc_auc_score, roc_curve

    preds = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    incumbent = pd.read_csv(ARTIFACT_DIR / "incumbent_discrimination.csv").set_index("ranker").roc_auc
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_roc, ax_scale) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})

    aucs = {}
    for m in MODELS:
        aucs[m] = roc_auc_score(preds.actual, preds[f"p_{m}"])
        fpr, tpr, _ = roc_curve(preds.actual, preds[f"p_{m}"])
        ax_roc.plot(fpr, tpr, color=PALETTE[m], lw=2.5, label=f"{DISPLAY[m]} ({aucs[m]:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC: the three curves overlap")
    ax_roc.legend(fontsize=12, loc="lower right")

    bands = [(0.45, 0.55, "Poor", "#E5E7EB"), (0.55, 0.64, "Fair", "#FDE68A"),
             (0.64, 0.71, "Good", "#BBF7D0"), (0.71, 0.80, "Excellent", "#4ADE80")]
    for lo, hi, name, colour in bands:
        ax_scale.axhspan(lo, hi, color=colour, alpha=0.8, lw=0)
        ax_scale.text(0.04, (lo + hi) / 2, name, va="center", fontsize=14, fontweight="bold", color="#374151")
    markers = [(incumbent["random"], f"Random allocation  {incumbent['random']:.2f}", "#4B5563"),
               (incumbent["incumbent"], f"Historical Georgia score  {incumbent['incumbent']:.2f}", "#C1121F"),
               (np.mean(list(aucs.values())), f"Our 3 models  {min(aucs.values()):.3f}–{max(aucs.values()):.3f}", "#0A1F44")]
    for auc, name, colour in markers:
        ax_scale.plot(0.5, auc, "o", ms=14, color=colour)
        ax_scale.text(0.56, auc, name, va="center", fontsize=13, color=colour, fontweight="bold")
    ax_scale.set(xlim=(0, 1.35), ylim=(0.45, 0.80), xticks=[], ylabel="ROC AUC",
                 title="How good is 0.73? US recidivism-tool scale")
    ax_scale.set_xlabel("Bands: Desmarais & Singh (2013), after Rice & Harris (2005)", fontsize=11, color="#4B5563")
    ax_scale.grid(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "performance_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def shap_summary() -> None:
    """SHAP summary (beeswarm) for logistic and XGBoost on the full evaluation cohort (course p159-160).

    Exact explainers (linear / tree), so all 7,807 people are used. One-hot levels are summed back to
    their raw field. Colour = the raw field's value: meaningful for ordered fields (age band, counts,
    risk score) and yes/no fields; arbitrary for unordered categories (offence type, education).
    """
    import joblib
    import shap

    from interpretability import shap_for
    from recidivism.config import MODEL_DIR, pretty
    from recidivism.modeling import ordinal_encode

    split = load_official_split()
    X = split.X_test
    colour = ordinal_encode(X)
    for col in colour.columns:
        if not pd.api.types.is_numeric_dtype(colour[col]):
            cats = sorted(colour[col].dropna().astype(str).unique())
            colour[col] = colour[col].map({c: i for i, c in enumerate(cats)})
        colour[col] = colour[col].astype(float).fillna(colour[col].astype(float).median())

    rows = []
    fig = plt.figure(figsize=(17, 8.6))
    for i, model in enumerate(["logistic", "xgboost"], start=1):
        pipe = joblib.load(MODEL_DIR / f"{model}.joblib")
        values, _ = shap_for(pipe, split.X_train, X, model)
        values = values[list(X.columns)]
        for feat, v in values.abs().mean().items():
            rows.append({"model": model, "feature": feat, "mean_abs_shap": v, "n_people": len(X)})
        plt.subplot(1, 2, i)
        shap.summary_plot(values.to_numpy(), colour.to_numpy(), feature_names=[pretty(c) for c in X.columns],
                          max_display=8, show=False, plot_size=None, color_bar=(i == 2))
        plt.title(f"{DISPLAY[model]}: SHAP summary, all {len(X):,} people", fontsize=15)
        plt.xlabel("SHAP value (log-odds): right = raises risk", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "shap_importance_full.csv", index=False)


def interpretability_matrix() -> None:
    """Which interpretability method was applied to which model, grouped as in the course syllabus
    (global, local, explaining performance), with where each result is shown."""
    DONE, PART, NONE, NATIVE = "#2A9D8F", "#E9C46A", "#D9D9D9", "#1D6F63"
    rows = [
        ("NATIVELY INTERPRETABLE (white box)", None),
        ("Coefficients / odds ratios", [("native", NATIVE, "notebook"), ("—", NONE, ""), ("—", NONE, "")]),
        ("Average marginal effects", [("✓", DONE, "notebook"), ("—", NONE, ""), ("—", NONE, "")]),
        ("AdaLogit / PLTR (white-box variants)", [("✓ tie", DONE, "AUC 0.7296 / 0.7292"), ("—", NONE, ""), ("—", NONE, "")]),
        ("GLOBAL — what drives the model", None),
        ("Impurity (gain) importance", [("—", NONE, "trees only"), ("✓", DONE, "artifact"), ("—", NONE, "")]),
        ("SHAP summary", [("✓", DONE, "slide 9"), ("✓", DONE, "slide 9"), ("✗ too slow", NONE, "no exact explainer")]),
        ("PDP (average curve)", [("✓", DONE, "slides 9-10"), ("✓", DONE, "slides 9-10"), ("✓", DONE, "slides 9-10")]),
        ("Global surrogate tree", [("not needed", NONE, "already linear"), ("✓ R² 0.61", PART, "slide 10 · A4"), ("—", NONE, "")]),
        ("LOCAL — one person", None),
        ("ICE (one curve per person)", [("✓", DONE, "slide 10"), ("✓", DONE, "slide 10"), ("✓", DONE, "slide 10")]),
        ("SHAP (one person)", [("✓", DONE, "slide 11"), ("✓", DONE, "slide 11"), ("✗ too slow", NONE, "")]),
        ("LIME + fidelity check", [("✓", DONE, "slide 11 · A8"), ("✓", DONE, "slide 11 · A8"), ("—", NONE, "")]),
        ("What-if: change one field", [("✓", DONE, "app"), ("✓", DONE, "app"), ("✓", DONE, "app")]),
        ("PERFORMANCE — what drives the AUC", None),
        ("Permutation importance", [("✓", DONE, "slide 12 · A9"), ("✓", DONE, "slide 12 · A9"), ("partial", PART, "10 of 29 fields")]),
        ("XPER", [("✓", DONE, "slide 12"), ("✓", DONE, "slide 12"), ("✗ too slow", NONE, "")]),
        ("Method agreement (SHAP·PI·XPER)", [("✓", DONE, "slide 12 · A3"), ("✓", DONE, "slide 12 · A3"), ("—", NONE, "")]),
    ]
    fig, ax = plt.subplots(figsize=(14, 10.2))
    ax.axis("off")
    n = len(rows)
    ax.set_xlim(0, 4.3)
    ax.set_ylim(-0.4, n + 0.9)
    for j, h in enumerate(["Logistic regression", "XGBoost", "TabICLv2"]):
        ax.text(1.75 + j * 0.9, n + 0.3, h, ha="center", va="center", fontsize=13, fontweight="bold")
    for i, (label, cells) in enumerate(rows):
        y = n - 0.5 - i
        if cells is None:
            ax.add_patch(plt.Rectangle((0, y - 0.4), 4.3, 0.8, color="#264653"))
            ax.text(0.05, y, label, color="white", fontweight="bold", fontsize=11.5, va="center")
            continue
        ax.text(0.05, y, label, fontsize=11.5, va="center")
        for j, (mark, colour, where) in enumerate(cells):
            x = 1.33 + j * 0.9
            ax.add_patch(plt.Rectangle((x, y - 0.38), 0.84, 0.76, color=colour, alpha=0.85))
            ax.text(x + 0.42, y + (0.12 if where else 0), mark, ha="center", va="center", fontsize=11.5,
                    fontweight="bold", color="white" if colour in (DONE, NATIVE) else "#333333")
            if where:
                ax.text(x + 0.42, y - 0.17, where, ha="center", va="center", fontsize=8.5,
                        color="white" if colour in (DONE, NATIVE) else "#333333")
    ax.set_title("Interpretability methods applied, by model (course grouping)",
                 fontsize=14, fontweight="bold", pad=6)
    fig.text(0.5, 0.01, "Logistic explains itself; XGBoost needs post-hoc tools; TabICLv2 can only be probed from outside "
             "(PDP/ICE, what-if) — exact SHAP, LIME and XPER need too many foundation-model predictions.",
             ha="center", fontsize=10, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIGURE_DIR / "interpretability_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def permutation_figure() -> None:
    """Permutation importance for the three models (course: explaining performance), readable labels."""
    from recidivism.config import pretty

    pi = pd.read_csv(ARTIFACT_DIR / "permutation_importance.csv")
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, model in zip(axes, MODELS):
        part = pi[pi.model == model].sort_values("importance", ascending=False).head(8)[::-1]
        ax.barh(part.feature.map(pretty), part.importance, color=PALETTE[model])
        n = int((pi.model == model).sum())
        ax.set_title(f"{DISPLAY[model]}" + ("" if n == 29 else f" ({n} of 29 fields measured)"), fontsize=15)
        ax.set_xlabel("Increase in Brier loss when shuffled", fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
    fig.suptitle("Permutation importance: how much worse the model gets when one field is shuffled", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "permutation_importance_readable.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def impurity_importance() -> None:
    """XGBoost impurity (total gain) importance per raw field, normalised to sum to 1 (course p112)."""
    import joblib

    from interpretability import original_feature
    from recidivism.config import FEATURE_COLUMNS, MODEL_DIR

    pipe = joblib.load(MODEL_DIR / "xgboost.joblib")
    names = list(pipe[-2].get_feature_names_out())
    total = pipe[-1].get_booster().get_score(importance_type="total_gain")
    gain = pd.Series({original_feature(names[int(k[1:])], FEATURE_COLUMNS): 0.0 for k in total})
    for k, v in total.items():
        gain[original_feature(names[int(k[1:])], FEATURE_COLUMNS)] += v
    (gain / gain.sum()).sort_values(ascending=False).rename("normalized_total_gain").rename_axis("feature") \
        .reset_index().to_csv(ARTIFACT_DIR / "xgboost_impurity_importance.csv", index=False)


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    impurity_importance()
    permutation_figure()
    interpretability_matrix()
    shap_summary()
    eda_overview()
    performance_benchmark()
    attribute_gaps("Race", "Black minus White", "fairness_race_gaps.png")
    attribute_gaps("Gender", "men minus women", "fairness_gender_gaps.png")
    fpdp_focus("Gender", ["Gang_Affiliated", "Age_at_Release", "Prior_Arrest_Episodes_Felony"], "fpdp_gender_focus.png")
    fpdp_focus("Age", ["Age_at_Release", "Supervision_Risk_Score_First", "Prior_Arrest_Episodes_Felony"], "fpdp_age_focus.png")
    print("Saved eda_overview, performance_benchmark, fairness_race_gaps, fairness_gender_gaps, fpdp_gender_focus, fpdp_age_focus")


if __name__ == "__main__":
    main()
