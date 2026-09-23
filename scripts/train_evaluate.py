"""Train all three models once and generate every quantitative audit artifact.

Run from the project root with:
    python scripts/train_evaluate.py

The script intentionally keeps the official test labels out of fitting. It uses
them only after prediction to compare performance, fairness, and stability.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_curve

# Add src/ to Python's import path when this file is run as a standalone script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, MODEL_DIR, RANDOM_SEED, pretty
from recidivism.data import load_official_split
from recidivism.metrics import (
    bootstrap_intervals,
    calibration_points,
    classification_metrics,
    economic_value,
    fairness_gaps,
    fairness_table,
)
from recidivism.modeling import logistic_model, tabicl_frames, xgboost_model


DISPLAY_NAMES = {
    "logistic": "Logistic regression",
    "xgboost": "XGBoost",
    "tabicl": "TabICLv2",
}


def save_json(value, path: Path) -> None:
    """Write readable, deterministic metadata and interval files."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fit_models(include_tabicl: bool = True) -> tuple[dict, dict[str, np.ndarray], dict]:
    """Fit the models and return held-out probabilities, never hard labels."""
    split = load_official_split()
    models, predictions, timings = {}, {}, {}

    # Logistic regression and XGBoost share a scikit-learn preprocessing API,
    # so they can be trained, predicted, and saved in the same loop.
    for name, model in [
        ("logistic", logistic_model(split.X_train)),
        ("xgboost", xgboost_model(split.X_train)),
    ]:
        start = time.perf_counter()
        model.fit(split.X_train, split.y_train)
        # Column 1 is P(recidivism=1), the quantity needed for Brier score,
        # ranking, calibration plots, and resource-allocation scenarios.
        predictions[name] = model.predict_proba(split.X_test)[:, 1]
        timings[name] = time.perf_counter() - start
        models[name] = model
        joblib.dump(model, MODEL_DIR / f"{name}.joblib", compress=3)

    if include_tabicl:
        # Imported here so a quick --skip-tabicl CPU run does not load PyTorch
        # or require the foundation-model checkpoint.
        from tabicl import TabICLClassifier

        train_num, test_num = tabicl_frames(split.X_train, split.X_test)
        model = TabICLClassifier(
            # Two ensemble views balance quality with the available 6 GB GPU.
            # The representation cache speeds up repeated permutation audits.
            n_estimators=16,
            batch_size=1,
            kv_cache="repr",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=True,
        )
        start = time.perf_counter()
        model.fit(train_num, split.y_train.to_numpy())
        predictions["tabicl"] = model.predict_proba(test_num)[:, 1]
        timings["tabicl"] = time.perf_counter() - start
        models["tabicl"] = model

    return models, predictions, {"split": split, "timings": timings}


def audit(models: dict, predictions: dict[str, np.ndarray], context: dict) -> None:
    """Compare models on the four course dimensions and save the results."""
    split, timings = context["split"], context["timings"]
    pred_frame = split.audit_test.copy()
    pred_frame["actual"] = split.y_test
    metrics_rows, fairness_frames, gap_rows, importance_rows = [], [], [], []
    intervals, stability = {}, {}

    # Explanation calculations are repeated many times. A fixed 1,000-person
    # subsample keeps them practical while remaining identical across models.
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(len(split.X_test), size=min(1_000, len(split.X_test)), replace=False)
    X_sample = split.X_test.iloc[sample_idx].reset_index(drop=True)
    y_sample = split.y_test.iloc[sample_idx].reset_index(drop=True)
    # This deliberately destroys all feature/row relationships. The change in
    # predictions is a stress test, not a realistic deployment simulation.
    shuffled = X_sample.copy()
    for col in shuffled.columns:
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())

    for name, p in predictions.items():
        # Keep one row per held-out person so reviewers can compare all models
        # against the same observed outcome and demographic audit attributes.
        pred_frame[f"p_{name}"] = p
        row = classification_metrics(split.y_test, p)
        row.update({"model": name, "fit_predict_seconds": timings[name]})
        row.update({f"economic_{k}": v for k, v in economic_value(split.y_test, p).items()})
        metrics_rows.append(row)
        intervals[name] = bootstrap_intervals(split.y_test, p, repeats=400)

        # Protected attributes were excluded from model inputs; they re-enter
        # only here, after predictions exist, to expose group differences.
        for attribute in ["Gender", "Race"]:
            table = fairness_table(split.y_test, p, split.audit_test[attribute], attribute)
            table.insert(0, "model", name)
            fairness_frames.append(table)
            gap_rows.append({"model": name, "attribute": attribute, **fairness_gaps(table)})

        if name != "tabicl":
            # Permutation importance asks: how much worse is Brier performance
            # when one raw field is shuffled and all other fields stay intact?
            result = permutation_importance(
                models[name], X_sample, y_sample, scoring="neg_brier_score",
                n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1,
            )
            for feature, mean, std in zip(X_sample.columns, result.importances_mean, result.importances_std):
                importance_rows.append({"model": name, "feature": feature, "importance": mean, "std": std})
            p_original = models[name].predict_proba(X_sample)[:, 1]
            p_perturbed = models[name].predict_proba(shuffled)[:, 1]
        else:
            train_num, sample_num = tabicl_frames(split.X_train, X_sample)
            _, shuffled_num = tabicl_frames(split.X_train, shuffled)
            p_original = models[name].predict_proba(sample_num)[:, 1]
            p_perturbed = models[name].predict_proba(shuffled_num)[:, 1]

            # Foundation-model inference is more expensive, so we audit ten
            # prespecified fields instead of repeating inference for all 29.
            candidate_features = [
                "Supervision_Risk_Score_First", "Age_at_Release", "Gang_Affiliated",
                "Prior_Arrest_Episodes_Felony", "Prior_Arrest_Episodes_Misd",
                "Prior_Arrest_Episodes_Violent", "Prior_Arrest_Episodes_Drug",
                "Education_Level", "Prison_Years", "Supervision_Level_First",
            ]
            baseline = brier_score_loss(y_sample, p_original)
            for feature in candidate_features:
                permuted = X_sample.copy()
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                _, permuted_num = tabicl_frames(split.X_train, permuted)
                p_perm = models[name].predict_proba(permuted_num)[:, 1]
                importance_rows.append({
                    "model": name,
                    "feature": feature,
                    "importance": brier_score_loss(y_sample, p_perm) - baseline,
                    "std": np.nan,
                })

        # Bootstrap widths describe sampling uncertainty. The shuffled-feature
        # measures describe sensitivity when the input signal is destroyed.
        stability[name] = {
            "all_features_shuffled_mae": float(np.mean(np.abs(p_original - p_perturbed))),
            "all_features_shuffled_rank_correlation": float(pd.Series(p_original).corr(pd.Series(p_perturbed), method="spearman")),
            "bootstrap_brier_interval_width": intervals[name]["brier"]["high"] - intervals[name]["brier"]["low"],
            "bootstrap_auc_interval_width": intervals[name]["roc_auc"]["high"] - intervals[name]["roc_auc"]["low"],
        }

    metrics = pd.DataFrame(metrics_rows).sort_values("brier")
    fairness = pd.concat(fairness_frames, ignore_index=True)
    gaps = pd.DataFrame(gap_rows)
    importance = pd.DataFrame(importance_rows).sort_values(["model", "importance"], ascending=[True, False])

    # These files are the single source of truth for the notebook, slides, app,
    # and written report. Regenerating here keeps every deliverable consistent.
    pred_frame.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
    metrics.to_csv(ARTIFACT_DIR / "model_metrics.csv", index=False)
    fairness.to_csv(ARTIFACT_DIR / "fairness_by_group.csv", index=False)
    gaps.to_csv(ARTIFACT_DIR / "fairness_gaps.csv", index=False)
    importance.to_csv(ARTIFACT_DIR / "permutation_importance.csv", index=False)
    save_json(intervals, ARTIFACT_DIR / "bootstrap_intervals.json")
    save_json(stability, ARTIFACT_DIR / "stability.json")
    save_json({
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": RANDOM_SEED,
        "train_rows": len(split.X_train),
        "test_rows": len(split.X_test),
        "features": list(split.X_train.columns),
        "protected_attributes_used_for_audit_only": ["Gender", "Race"],
        "tabicl_estimators": 16 if "tabicl" in predictions else 0,
    }, ARTIFACT_DIR / "run_manifest.json")
    make_figures(split.y_test, predictions, metrics, fairness, importance)


def make_figures(y: pd.Series, predictions: dict[str, np.ndarray], metrics: pd.DataFrame,
                 fairness: pd.DataFrame, importance: pd.DataFrame) -> None:
    """Create presentation-ready plots directly from saved audit results."""
    sns.set_theme(style="whitegrid", context="talk")
    palette = {"logistic": "#234E70", "xgboost": "#FB8500", "tabicl": "#7B2CBF"}

    # Left: ranking quality (ROC). Right: whether probabilities mean what they say.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, p in predictions.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = classification_metrics(y, p)["roc_auc"]
        axes[0].plot(fpr, tpr, lw=2.5, color=palette[name], label=f"{DISPLAY_NAMES[name]} ({auc:.3f})")
        cal = calibration_points(y, p)
        axes[1].plot(cal.predicted, cal.observed, marker="o", lw=2.5, color=palette[name], label=DISPLAY_NAMES[name])
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set(title="Discrimination", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[1].set(title="Calibration", xlabel="Mean predicted risk", ylabel="Observed rate")
    for ax in axes: ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "performance_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plot_data = metrics.melt(id_vars="model", value_vars=["brier", "roc_auc", "average_precision"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot_data, x="metric", y="value", hue="model", palette=palette, ax=ax)
    ax.set(title="Held-out model comparison", xlabel="", ylabel="Score")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # FPR matters here because a false high-risk flag could misallocate a scarce
    # service or cause harm if the tool were used outside its intended purpose.
    race = fairness[fairness.attribute.eq("Race")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(data=race, x="group", y="fpr", hue="model", palette=palette, ax=axes[0])
    sns.barplot(data=race, x="group", y="brier", hue="model", palette=palette, ax=axes[1])
    axes[0].set(title="False-positive rate by race", xlabel="", ylabel="FPR")
    axes[1].set(title="Brier score by race", xlabel="", ylabel="Brier (lower is better)")
    axes[0].legend(title="")
    axes[1].get_legend().remove()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "race_fairness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    top = importance.sort_values("importance", ascending=False).groupby("model").head(8)
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 6), squeeze=False)
    for ax, name in zip(axes[0], predictions):
        part = top[top.model.eq(name)].sort_values("importance")
        ax.barh(part.feature.map(pretty), part.importance, color=palette[name])
        ax.set_title(DISPLAY_NAMES[name])
        ax.set_xlabel("Increase in Brier loss when shuffled")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Command-line entry point: create folders, train, audit, and print results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tabicl", action="store_true", help="Train only the two conventional models.")
    args = parser.parse_args()
    for directory in [ARTIFACT_DIR, MODEL_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    models, predictions, context = fit_models(include_tabicl=not args.skip_tabicl)
    audit(models, predictions, context)
    print(pd.read_csv(ARTIFACT_DIR / "model_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
