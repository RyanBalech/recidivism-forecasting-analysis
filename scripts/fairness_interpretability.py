"""Fairness interpretability and mitigation, following the course (§8.2, pp. 247-262).

Which features *generate* the unfairness found by fairness_audit.py, and what does removing them cost?

1. Fairness Partial Dependence Plot (FPDP, p248-257). For feature X_A and value c, set X_A = c for
   every test person, re-select the top 20%, and recompute the fairness test p-value. The test is
   equal opportunity (chi-squared on selection x group among people who were re-arrested), our
   primary metric. X_A is a **candidate variable** if some value c pushes the p-value above 0.05.
2. X/D vs X/Y dependence (p259-260): Cramer's V of each feature with the protected attribute D and
   with the outcome Y. High X/D with low X/Y is a proxy that buys little accuracy. This is also
   the proxy-variable analysis (are gang affiliation or prior arrests proxies for race?).
3. Mitigation (p261-262): for each candidate, Panel A drops it and re-estimates the model; Panel B
   keeps the model and neutralizes the feature at its best FPDP value. Report fairness p-values and AUC.

Attributes: Gender and Age, where the fairness null is rejected for every model. Race is included
for the X/D scatter (proxy question) and for FPDP only where a race test rejects. TabICL is left out
of FPDP and re-estimation: each FPDP point is a full in-context prediction pass (~a minute on CPU),
which we state as a cost of the foundation model.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fairness_audit import AGE_ORDER, CAPACITY, _chi2_p, with_audit_columns
from recidivism.config import ARTIFACT_DIR, FEATURE_COLUMNS, FIGURE_DIR, MODEL_DIR, pretty
from recidivism.data import load_official_split

warnings.filterwarnings("ignore")
MODELS = ["logistic", "xgboost"]
DISPLAY = {"logistic": "Logistic regression", "xgboost": "XGBoost"}
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500"}
# Protected (D=1) group listed second, as in fairness_audit.py.
ATTRIBUTES = {"Gender": ("M", "F"), "Age": ("Under 33", "33 or older"), "Race": ("BLACK", "WHITE")}
MAX_VALUES = 12


def top_k(p: np.ndarray) -> np.ndarray:
    """Deployed rule: the top 20% by score. Ties broken by order so capacity is exact."""
    k = int(round(len(p) * CAPACITY))
    sel = np.zeros(len(p), bool)
    sel[np.argsort(-p, kind="stable")[:k]] = True
    return sel


def fairness_p(y: np.ndarray, sel: np.ndarray, d: np.ndarray) -> dict[str, float]:
    """Equal opportunity (primary) and statistical parity p-values, course chi-squared tests."""
    pos = y == 1
    return {"p_equal_opportunity": _chi2_p(pd.crosstab(d[pos], sel[pos]).to_numpy())[1],
            "p_statistical_parity": _chi2_p(pd.crosstab(d, sel).to_numpy())[1]}


def grid(values: pd.Series) -> list:
    """Values at which to evaluate the FPDP: every level, or quantiles for long numeric ranges."""
    uniq = values.dropna().unique()
    if len(uniq) <= MAX_VALUES:
        if values.name == "Age_at_Release":
            return [v for v in AGE_ORDER if v in set(uniq)]
        try:
            return sorted(uniq, key=lambda v: float(str(v).split()[0]))
        except ValueError:
            return sorted(uniq, key=str)
    return sorted(set(np.quantile(values.dropna(), np.linspace(0, 1, MAX_VALUES)).round(2)))


def cramers_v(x: pd.Series, z: np.ndarray) -> float:
    """Dependence between a feature (binned if numeric) and a binary variable, in [0, 1]."""
    x = x.copy()
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > 10:
        x = pd.qcut(x, 5, duplicates="drop")
    table = pd.crosstab(x.astype(str), z).to_numpy()
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))


def fpdp(models: dict, X: pd.DataFrame, y: np.ndarray, groups: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        for feat in FEATURE_COLUMNS:
            for c in grid(X[feat]):
                Xc = X.copy()
                Xc[feat] = pd.Series([c] * len(X), index=X.index).astype(X[feat].dtype)
                p = model.predict_proba(Xc)[:, 1]
                sel = top_k(p)
                for attr, d in groups.items():
                    rows.append({"model": name, "attribute": attr, "feature": feat, "value": str(c),
                                 "auc": roc_auc_score(y, p), **fairness_p(y, sel, d)})
    return pd.DataFrame(rows)


def dependence(X: pd.DataFrame, y: np.ndarray, groups: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for feat in FEATURE_COLUMNS:
        row = {"feature": feat, "x_y_dependence": cramers_v(X[feat], y)}
        for attr, d in groups.items():
            row[f"x_d_dependence_{attr}"] = cramers_v(X[feat], d)
        rows.append(row)
    return pd.DataFrame(rows)


def mitigation(models: dict, split, y: np.ndarray, groups: dict[str, np.ndarray],
               fp: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Panel A: drop the candidate and re-estimate. Panel B: neutralize it in the fitted model."""
    from recidivism.modeling import logistic_model, xgboost_model

    rows = []
    for name, model in models.items():
        p = model.predict_proba(split.X_test)[:, 1]
        for attr, d in groups.items():
            rows.append({"model": name, "attribute": attr, "panel": "baseline", "feature": "(none)",
                         "value": "", "auc": roc_auc_score(y, p), **fairness_p(y, top_k(p), d)})
        for attr in groups:
            cands = candidates[(candidates.model == name) & (candidates.attribute == attr)]
            for feat in cands.feature:
                d = groups[attr]
                # Panel A: the feature is removed from the inputs and the model is refitted.
                keep = [c for c in FEATURE_COLUMNS if c != feat]
                builder = logistic_model if name == "logistic" else xgboost_model
                refit = builder(split.X_train[keep]).fit(split.X_train[keep], split.y_train)
                pa = refit.predict_proba(split.X_test[keep])[:, 1]
                rows.append({"model": name, "attribute": attr, "panel": "A: drop + re-estimate", "feature": feat,
                             "value": "", "auc": roc_auc_score(y, pa), **fairness_p(y, top_k(pa), d)})
                # Panel B: best neutral value from the FPDP, model unchanged.
                best = fp[(fp.model == name) & (fp.attribute == attr) & (fp.feature == feat)] \
                    .sort_values("p_equal_opportunity", ascending=False).iloc[0]
                rows.append({"model": name, "attribute": attr, "panel": "B: neutralize, no re-estimation",
                             "feature": feat, "value": best.value, "auc": best.auc,
                             "p_equal_opportunity": best.p_equal_opportunity,
                             "p_statistical_parity": best.p_statistical_parity})
    return pd.DataFrame(rows)


def figures(fp: pd.DataFrame, dep: pd.DataFrame, candidates: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    for attr in ["Gender", "Age"]:
        feats = FEATURE_COLUMNS
        ncol = 6
        nrow = int(np.ceil(len(feats) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
        for ax, feat in zip(axes.flat, feats):
            for name in MODELS:
                part = fp[(fp.model == name) & (fp.attribute == attr) & (fp.feature == feat)]
                ax.plot(range(len(part)), part.p_equal_opportunity, marker="o", ms=3, color=PALETTE[name],
                        label=DISPLAY[name])
            ax.axhline(0.05, color="red", lw=1.5)
            is_cand = feat in set(candidates[candidates.attribute == attr].feature)
            ax.set_title(pretty(feat)[:38], fontsize=9, color="#C1121F" if is_cand else "black",
                         fontweight="bold" if is_cand else "normal")
            ax.set_xticks([])
            ax.set_ylim(-0.02, max(0.1, float(fp[(fp.attribute == attr) & (fp.feature == feat)]
                                              .p_equal_opportunity.max()) * 1.1))
        for ax in list(axes.flat)[len(feats):]:
            ax.axis("off")
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"FPDP, {attr}: equal-opportunity p-value when each feature is fixed to one value "
                     f"(red line = 0.05; red titles = candidate variables)", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"fpdp_{attr.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, attr in zip(axes, ["Race", "Gender", "Age"]):
        col = f"x_d_dependence_{attr}"
        cand = set(candidates[candidates.attribute == attr].feature)
        colors = ["#C1121F" if f in cand else "#457B9D" for f in dep.feature]
        ax.scatter(dep.x_y_dependence, dep[col], c=colors, s=60)
        for _, r in dep.iterrows():
            if r.feature in cand or r[col] > dep[col].quantile(0.85) or r.x_y_dependence > dep.x_y_dependence.quantile(0.85):
                ax.annotate(pretty(r.feature)[:30], (r.x_y_dependence, r[col]), fontsize=8,
                            xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel="X/Y dependence (Cramér's V with re-arrest)", ylabel=f"X/D dependence (Cramér's V with {attr})",
               title=f"{attr}: which features carry the protected attribute?")
    fig.suptitle("Top-left = proxy that adds little accuracy; red = candidate variable from the FPDP", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fairness_dependence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    split = load_official_split()
    audit = with_audit_columns(pd.read_csv(ARTIFACT_DIR / "test_predictions.csv"))
    y = audit["actual"].to_numpy()
    groups = {attr: (audit[attr] == b).to_numpy() for attr, (a, b) in ATTRIBUTES.items()}
    models = {m: joblib.load(MODEL_DIR / f"{m}.joblib") for m in MODELS}

    # Only explain a rejected null: keep attributes whose baseline equal-opportunity test rejects.
    base = {(m, attr): fairness_p(y, top_k(models[m].predict_proba(split.X_test)[:, 1]), d)["p_equal_opportunity"]
            for m in MODELS for attr, d in groups.items()}
    fp = fpdp(models, split.X_test, y, groups)
    fp["baseline_rejected"] = [base[(m, a)] < 0.05 for m, a in zip(fp.model, fp.attribute)]
    dep = dependence(split.X_test, y, groups)

    best = fp.groupby(["model", "attribute", "feature"]).agg(
        max_p=("p_equal_opportunity", "max"), rejected=("baseline_rejected", "first")).reset_index()
    candidates = best[best.rejected & (best.max_p > 0.05)].sort_values(["attribute", "model", "max_p"],
                                                                       ascending=[True, True, False])
    mit = mitigation(models, split, y, groups, fp, candidates)

    fp.to_csv(ARTIFACT_DIR / "fpdp_values.csv", index=False)
    dep.to_csv(ARTIFACT_DIR / "fairness_dependence.csv", index=False)
    candidates.to_csv(ARTIFACT_DIR / "fairness_candidates.csv", index=False)
    mit.to_csv(ARTIFACT_DIR / "fairness_mitigation.csv", index=False)
    figures(fp, dep, candidates)

    print("Baseline equal-opportunity p-values:", {f"{m}/{a}": round(v, 4) for (m, a), v in base.items()})
    print("\nCandidate variables:\n", candidates.round(4).to_string(index=False))
    print("\nMitigation:\n", mit.round(4).to_string(index=False))
    print("\nStrongest X/D dependence per attribute:")
    for attr in ATTRIBUTES:
        col = f"x_d_dependence_{attr}"
        print(attr, dep.nlargest(5, col)[["feature", col, "x_y_dependence"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
