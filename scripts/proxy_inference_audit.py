"""How much of a protected attribute can be recovered from the features we kept?

Every version of this project says that excluding race, gender and geography
removes the direct input but not the proxies. That is asserted everywhere and
measured nowhere. The test is simple: try to predict the protected attribute
*from the model's own feature set*. The AUC of that attempt is the amount of
protected information still available to any model trained on these columns.

    AUC 0.50  the attribute is genuinely unavailable
    AUC 1.00  the attribute is fully reconstructable, exclusion is cosmetic

This also puts a number on the leak the team found. `Gang_Affiliated` is missing
for every woman and no man, so before the mode-fill a model that encoded
missingness as a category could read gender directly. Three feature sets are
compared so the size of that channel is visible rather than argued:

    shipped           the 29 eligible fields as the models receive them
    with_missingness  the same fields plus explicit missing-value indicators
    without_gang      the shipped set minus Gang_Affiliated (the FPDP mitigation)

The top-ranked features of each attacker are the proxies, named rather than
guessed at.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.modeling import logistic_model, xgboost_model

warnings.filterwarnings("ignore")

ATTRIBUTES = {"Gender": "F", "Race": "BLACK"}
FOLDS = 5
PALETTE = {"Gender": "#C1121F", "Race": "#234E70"}


def feature_variants(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Three views of the same eligible columns."""
    with_missing = X.copy()
    for column in X.columns:
        if X[column].isna().any():
            with_missing[f"{column}__is_missing"] = X[column].isna().astype(int)
    return {
        "shipped": X,
        "with_missingness": with_missing,
        "without_gang": X.drop(columns=["Gang_Affiliated"], errors="ignore"),
    }


def attack(X: pd.DataFrame, target: np.ndarray) -> tuple[float, float]:
    """Cross-validated AUC of predicting the protected attribute from the features."""
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for train_idx, test_idx in cv.split(X, target):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        model = xgboost_model(Xtr, n_estimators=200).fit(Xtr, target[train_idx])
        scores.append(roc_auc_score(target[test_idx], model.predict_proba(Xte)[:, 1]))
    return float(np.mean(scores)), float(np.std(scores))


def proxies(X: pd.DataFrame, target: np.ndarray, top: int = 8) -> pd.DataFrame:
    """Which columns carry the attribute? Permutation importance of the attacker."""
    model = xgboost_model(X, n_estimators=200).fit(X, target)
    sample = X.sample(min(1500, len(X)), random_state=RANDOM_SEED)
    result = permutation_importance(model, sample, target[sample.index], scoring="roc_auc",
                                    n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns, "importance": result.importances_mean})
            .sort_values("importance", ascending=False).head(top).reset_index(drop=True))


def main() -> None:
    split = load_official_split()
    X = pd.concat([split.X_train, split.X_test], ignore_index=True)
    audit = pd.concat([split.audit_train, split.audit_test], ignore_index=True)
    variants = feature_variants(X)

    rows, proxy_rows = [], []
    for attribute, positive in ATTRIBUTES.items():
        target = (audit[attribute].to_numpy() == positive).astype(int)
        for variant, frame in variants.items():
            mean, sd = attack(frame, target)
            rows.append({
                "attribute": attribute, "positive_class": positive, "feature_set": variant,
                "n_features": frame.shape[1],
                "recovery_auc": mean, "recovery_auc_sd": sd,
                # 0 = no better than a coin flip, 1 = fully reconstructable.
                "recoverable_share": max(0.0, (mean - 0.5) / 0.5),
                "base_rate": float(target.mean()),
            })
            print(f"{attribute:7} {variant:17} recovery AUC = {mean:.4f} (sd {sd:.4f})", flush=True)
        top = proxies(variants["shipped"], target)
        top.insert(0, "attribute", attribute)
        proxy_rows.append(top)

    recovery = pd.DataFrame(rows)
    proxy = pd.concat(proxy_rows, ignore_index=True)
    recovery.to_csv(ARTIFACT_DIR / "proxy_recovery.csv", index=False)
    proxy.to_csv(ARTIFACT_DIR / "proxy_features.csv", index=False)

    _figure(recovery, proxy)

    pd.set_option("display.width", 200)
    print("\nHow much protected information survives exclusion:")
    print(recovery[["attribute", "feature_set", "recovery_auc", "recoverable_share"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nStrongest proxies in the shipped feature set:")
    print(proxy.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _figure(recovery: pd.DataFrame, proxy: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    order = ["shipped", "with_missingness", "without_gang"]
    width = 0.35
    x = np.arange(len(order))
    for i, attribute in enumerate(ATTRIBUTES):
        part = recovery[recovery.attribute == attribute].set_index("feature_set").loc[order]
        axes[0].bar(x + (i - 0.5) * width, part.recovery_auc, width,
                    yerr=part.recovery_auc_sd, capsize=4,
                    color=PALETTE[attribute], label=attribute)
    axes[0].axhline(0.5, color="grey", ls="--", lw=1.5)
    axes[0].text(len(order) - 0.55, 0.512, "0.5 = attribute not recoverable", fontsize=9, color="grey")
    axes[0].set_xticks(x, [o.replace("_", " ") for o in order])
    axes[0].set(title="Can the protected attribute be predicted\nfrom the features we kept?",
                ylabel="cross-validated AUC of the attacker", ylim=(0.45, 1.02))
    axes[0].legend()

    for i, attribute in enumerate(ATTRIBUTES):
        part = proxy[proxy.attribute == attribute].sort_values("importance")
        offset = i * (len(part) + 1)
        axes[1].barh(np.arange(len(part)) + offset,
                     part.importance, color=PALETTE[attribute], label=attribute)
        for j, name in enumerate(part.feature):
            axes[1].text(0.0005, j + offset, name.replace("_", " "), va="center", fontsize=8)
    axes[1].set(title="Strongest proxies for each attribute\n(permutation importance of the attacker)",
                xlabel="drop in attacker AUC when shuffled")
    axes[1].set_yticks([])
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "proxy_recovery.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
