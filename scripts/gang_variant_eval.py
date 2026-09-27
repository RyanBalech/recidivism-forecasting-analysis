"""Re-evaluate the recommended models without Gang_Affiliated on all four dimensions.

The fairness audit located the gender gap in Gang_Affiliated (never recorded for women) and the
nested check showed that dropping it closes most of the gap out of fold. Before recommending that
change, this script checks what dropping it does to everything else, for logistic regression and
XGBoost (the recommended model and its challenger), side by side with the shipped configuration:

1. Performance: AUC, Brier, calibration (ECE, Cox slope), paired AUC difference, allocation value.
2. Interpretability: SHAP global importance — which features take gang affiliation's place.
3. Stability: the same eight bootstrap resamples as stability_structural.py (seed 7).
4. Fairness: race, gender and age gaps at the top-20% rule, with bootstrap CIs, TOST and the
   course test table (incl. sufficiency), reusing fairness_audit.py.

Both variants are refitted here in the same environment, so they are compared like with like.
TabICL is not re-evaluated: it is neither the recommendation nor the challenger.
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fairness_audit as fa
from interpretability import shap_for
from recidivism.config import ARTIFACT_DIR, FEATURE_COLUMNS, FIGURE_DIR, RANDOM_SEED, pretty
from recidivism.data import load_official_split
from recidivism.metrics import capacity_selection, classification_metrics, economic_value
from recidivism.modeling import logistic_model, xgboost_model

warnings.filterwarnings("ignore")
NO_GANG = [c for c in FEATURE_COLUMNS if c != "Gang_Affiliated"]
VARIANTS = {"with_gang": FEATURE_COLUMNS, "no_gang": NO_GANG}
BUILDERS = {"logistic": lambda X, seed: logistic_model(X), "xgboost": lambda X, seed: xgboost_model(X, random_state=seed)}
REFITS = 8
CAPACITY = 0.20


def cox_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Calibration slope: regress the outcome on logit(p). 1 = well calibrated, <1 = too extreme."""
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return float(LogisticRegression(penalty=None, max_iter=1000).fit(logit, y).coef_[0, 0])


def main() -> None:
    split = load_official_split()
    y = split.y_test.to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    background = split.X_train.iloc[rng.choice(len(split.X_train), 500, replace=False)]
    explain = split.X_test.iloc[rng.choice(len(split.X_test), 1000, replace=False)]

    # ---- fit both variants of both models on the full training set
    preds, rows, shap_rows = {}, [], []
    for model, build in BUILDERS.items():
        for variant, cols in VARIANTS.items():
            key = f"{model}__{variant}"
            fitted = build(split.X_train[cols], RANDOM_SEED).fit(split.X_train[cols], split.y_train)
            p = fitted.predict_proba(split.X_test[cols])[:, 1]
            preds[key] = p
            m = classification_metrics(y, p)
            value = economic_value(y, p, capacity=CAPACITY)
            rows.append({"model": model, "variant": variant, "roc_auc": m["roc_auc"], "brier": m["brier"],
                         "ece_10": m["ece_10"], "calibration_slope": cox_slope(y, p),
                         "captured_events": value["captured_events"], "precision_top20": value["precision_at_capacity"],
                         "net_value": value["assumed_net_value"]})
            values, _ = shap_for(fitted, background[cols], explain[cols], model)
            for rank, (feat, v) in enumerate(values.abs().mean().sort_values(ascending=False).items(), 1):
                shap_rows.append({"model": model, "variant": variant, "rank": rank, "feature": feat, "mean_abs_shap": v})
            print(f"fitted {key}: AUC {m['roc_auc']:.4f}", flush=True)
    perf = pd.DataFrame(rows)

    # ---- paired bootstrap: AUC(no gang) - AUC(with gang), same people
    boot = np.random.default_rng(RANDOM_SEED)
    draws = [boot.integers(0, len(y), len(y)) for _ in range(1000)]
    for model in BUILDERS:
        a, b = preds[f"{model}__no_gang"], preds[f"{model}__with_gang"]
        diffs = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in draws]
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        perf.loc[perf.model == model, "auc_diff_no_minus_with_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"
        sel_a, sel_b = capacity_selection(a, CAPACITY), capacity_selection(b, CAPACITY)
        perf.loc[perf.model == model, "selected_set_jaccard_vs_with"] = (sel_a & sel_b).sum() / (sel_a | sel_b).sum()

    # ---- stability: same eight bootstrap resamples as stability_structural.py
    srng = np.random.default_rng(7)
    samples = [srng.integers(0, len(split.X_train), len(split.X_train)) for _ in range(REFITS)]
    k = int(round(len(y) * CAPACITY))
    stab = []
    for model, build in BUILDERS.items():
        for variant, cols in VARIANTS.items():
            refit_preds = []
            for seed, idx in enumerate(samples):
                X, yy = split.X_train.iloc[idx][cols].reset_index(drop=True), split.y_train.iloc[idx].reset_index(drop=True)
                refit_preds.append(build(X, seed).fit(X, yy).predict_proba(split.X_test[cols])[:, 1])
            pairs = []
            for i, j in itertools.combinations(range(REFITS), 2):
                ti, tj = set(np.argsort(-refit_preds[i])[:k]), set(np.argsort(-refit_preds[j])[:k])
                pairs.append({"drift": np.mean(np.abs(refit_preds[i] - refit_preds[j])), "jaccard": len(ti & tj) / len(ti | tj)})
            pairs = pd.DataFrame(pairs)
            stab.append({"model": model, "variant": variant, "mean_abs_prob_diff": pairs.drift.mean(), "top20_jaccard": pairs.jaccard.mean()})
            print(f"stability {model}/{variant}: drift {pairs.drift.mean():.4f}", flush=True)
    perf = perf.merge(pd.DataFrame(stab), on=["model", "variant"])

    # ---- fairness: reuse the audit on a prediction frame with four "models"
    frame = split.audit_test.copy()
    frame["actual"] = y
    for key, p in preds.items():
        frame[f"p_{key}"] = p
    frame = fa.with_audit_columns(frame)
    fa.MODELS = list(preds)
    inf = fa.audit(frame)
    tests = fa.course_tests(frame)
    top = inf[(inf.rule == "top_20pct") & inf.metric.isin(["fnr", "selection_rate"])]
    for _, r in top.iterrows():
        model, variant = r.model.split("__")
        mask = (perf.model == model) & (perf.variant == variant)
        perf.loc[mask, f"{r.attribute.lower()}_{r.metric}_gap"] = r.gap
        perf.loc[mask, f"{r.attribute.lower()}_{r.metric}_ci"] = f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
        perf.loc[mask, f"{r.attribute.lower()}_{r.metric}_tost_equivalent"] = r.equivalent_within_delta
    for _, r in tests.iterrows():
        model, variant = r.model.split("__")
        mask = (perf.model == model) & (perf.variant == variant)
        perf.loc[mask, f"{r.attribute.lower()}_sufficiency_p"] = r.sufficiency
        perf.loc[mask, f"{r.attribute.lower()}_equal_opportunity_p"] = r.equal_opportunity
    women = frame.Gender.eq("F").to_numpy()
    for key, p in preds.items():
        model, variant = key.split("__")
        perf.loc[(perf.model == model) & (perf.variant == variant), "women_mean_score"] = p[women].mean()
    perf["women_observed_rate"] = y[women].mean()

    perf.to_csv(ARTIFACT_DIR / "gang_variant_eval.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(ARTIFACT_DIR / "gang_variant_shap.csv", index=False)
    comparison_figure(perf, pd.DataFrame(shap_rows))
    show = ["model", "variant", "roc_auc", "brier", "ece_10", "calibration_slope", "captured_events",
            "mean_abs_prob_diff", "top20_jaccard", "gender_fnr_gap", "race_fnr_gap", "age_fnr_gap",
            "race_fnr_tost_equivalent", "gender_sufficiency_p", "women_mean_score"]
    print(perf[show].round(4).to_string(index=False))
    print(perf[["model", "auc_diff_no_minus_with_ci", "selected_set_jaccard_vs_with", "gender_fnr_ci", "race_fnr_ci", "age_fnr_ci"]].to_string(index=False))


def comparison_figure(perf: pd.DataFrame, shap: pd.DataFrame) -> None:
    """A table image for the appendix slide: with vs without gang affiliation, four dimensions."""
    order = [("logistic", "with_gang"), ("logistic", "no_gang"), ("xgboost", "with_gang"), ("xgboost", "no_gang")]
    header = ["Logistic\nwith gang", "Logistic\nno gang", "XGBoost\nwith gang", "XGBoost\nno gang"]
    r = {o: perf[(perf.model == o[0]) & (perf.variant == o[1])].iloc[0] for o in order}
    top = {o: "\n".join(pretty(f) for f in shap[(shap.model == o[0]) & (shap.variant == o[1])].nsmallest(2, "rank").feature) for o in order}
    lines = [
        ("PERFORMANCE", None),
        ("ROC AUC", lambda x: f"{x.roc_auc:.3f}"),
        ("Brier ↓", lambda x: f"{x.brier:.4f}"),
        ("Re-arrests captured in top 20%", lambda x: f"{int(x.captured_events):,}"),
        ("INTERPRETABILITY", None),
        ("Top-2 SHAP drivers", "top"),
        ("STABILITY (8 refits)", None),
        ("Score drift, mean |Δp| ↓", lambda x: f"{x.mean_abs_prob_diff:.4f}"),
        ("Top-20% Jaccard ↑", lambda x: f"{x.top20_jaccard:.3f}"),
        ("FAIRNESS (FNR gap, top 20%)", None),
        ("Race (B − W)", lambda x: f"{x.race_fnr_gap:+.3f}"),
        ("Gender (M − F)", lambda x: f"{x.gender_fnr_gap:+.3f}"),
        ("Age (<33 − 33+)", lambda x: f"{x.age_fnr_gap:+.3f}"),
    ]
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.axis("off")
    n = len(lines)
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-0.3, n + 0.8)
    for j, h in enumerate(header):
        ax.text(1.95 + j * 0.82, n + 0.2, h, ha="center", va="center", fontsize=12, fontweight="bold")
    for i, (label, fn) in enumerate(lines):
        yy = n - 0.6 - i
        if fn is None:
            ax.add_patch(plt.Rectangle((0, yy - 0.38), 5.2, 0.76, color="#264653"))
            ax.text(0.05, yy, label, color="white", fontweight="bold", fontsize=11, va="center")
            continue
        ax.text(0.05, yy, label, fontsize=11, va="center")
        for j, o in enumerate(order):
            text = top[o] if fn == "top" else fn(r[o])
            ax.add_patch(plt.Rectangle((1.56 + j * 0.82, yy - 0.36), 0.78, 0.72,
                                       color="#E9F5F3" if o[1] == "no_gang" else "#F4F4F4"))
            ax.text(1.95 + j * 0.82, yy, text, ha="center", va="center", fontsize=9.5 if fn == "top" else 11)
    ax.set_title("Dropping gang affiliation, re-evaluated on all four dimensions (evaluation cohort)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "gang_variant_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "--figure-only" in sys.argv:
        comparison_figure(pd.read_csv(ARTIFACT_DIR / "gang_variant_eval.csv"), pd.read_csv(ARTIFACT_DIR / "gang_variant_shap.csv"))
    else:
        main()
