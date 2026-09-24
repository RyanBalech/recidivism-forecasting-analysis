"""Nested validation of the fairness mitigation.

The headline mitigation result (drop `Gang_Affiliated`, re-estimate, equal-opportunity
p goes from 0.000 to ~0.99) was produced by selecting the candidate variable *and*
measuring the improvement on the same evaluation cohort. That is in-sample evidence:
it shows the mitigation can be fitted, not that it generalises.

This script separates selection from evaluation (scikit-learn cross-validation
guidance, https://scikit-learn.org/stable/modules/cross_validation.html):

    for each outer fold of the TRAINING partition only
        run the course FPDP candidate search on the outer-training part
        pick the candidate variable from that part alone
        fit baseline and mitigated models on the outer-training part
        measure accuracy AND disparity on the held-out outer fold

The evaluation cohort is never touched here. Reported per fold: AUC change, the
equal-opportunity p-value, and the FNR gap, plus how often the same variable is
selected (selection stability).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FEATURE_COLUMNS, RANDOM_SEED
from recidivism.data import load_official_split
from recidivism.metrics import capacity_selection
from recidivism.modeling import logistic_model, xgboost_model

warnings.filterwarnings("ignore")

CAPACITY = 0.20
MAX_VALUES = 12
AGE_ORDER = ["18-22", "23-27", "28-32", "33-37", "38-42", "43-47", "48 or older"]
BUILDERS = {"logistic": logistic_model, "xgboost": xgboost_model}


def _chi2_p(table: np.ndarray) -> float:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    return float(stats.chi2_contingency(table, correction=False)[1])


def equal_opportunity_p(y: np.ndarray, sel: np.ndarray, d: np.ndarray) -> float:
    pos = y == 1
    return _chi2_p(pd.crosstab(d[pos], sel[pos]).to_numpy())


def fnr_gap(y: np.ndarray, sel: np.ndarray, d: np.ndarray) -> float:
    """FNR(reference) - FNR(protected): negative means the protected group misses more support."""
    out = {}
    for flag in (0, 1):
        m = (d == flag) & (y == 1)
        out[flag] = float(1 - sel[m].mean()) if m.any() else float("nan")
    return out[0] - out[1]


def grid(values: pd.Series) -> list:
    uniq = values.dropna().unique()
    if len(uniq) <= MAX_VALUES:
        if values.name == "Age_at_Release":
            return [v for v in AGE_ORDER if v in set(uniq)]
        try:
            return sorted(uniq, key=lambda v: float(str(v).split()[0]))
        except ValueError:
            return sorted(uniq, key=str)
    return sorted(set(np.quantile(values.dropna(), np.linspace(0, 1, MAX_VALUES)).round(2)))


def select_candidate(model, X: pd.DataFrame, y: np.ndarray, d: np.ndarray,
                     features: list[str]) -> tuple[str | None, float]:
    """Course FPDP rule, run on training data only: the variable whose best forced
    value lifts the equal-opportunity p-value furthest above 0.05."""
    best_feature, best_p = None, 0.05
    for feat in features:
        for c in grid(X[feat]):
            Xc = X.copy()
            Xc[feat] = pd.Series([c] * len(X), index=X.index).astype(X[feat].dtype)
            sel = capacity_selection(model.predict_proba(Xc)[:, 1], CAPACITY)
            p = equal_opportunity_p(y, sel, d)
            if np.isfinite(p) and p > best_p:
                best_feature, best_p = feat, p
    return best_feature, best_p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", nargs="*", default=list(BUILDERS))
    args = parser.parse_args()

    split = load_official_split()
    X = split.X_train.reset_index(drop=True)
    y = split.y_train.to_numpy()
    gender = split.audit_train["Gender"].to_numpy()
    d = (gender == "F").astype(int)  # protected group = women

    # Stratify on outcome and protected group together so every fold holds both.
    strata = pd.Series(y).astype(str) + "_" + pd.Series(d).astype(str)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_SEED)

    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, strata), start=1):
        X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        y_tr, y_te = y[tr], y[te]
        d_tr, d_te = d[tr], d[te]

        for name in args.models:
            build = BUILDERS[name]
            base = build(X_tr).fit(X_tr, y_tr)

            # --- selection: training part only ---
            feature, sel_p = select_candidate(base, X_tr, y_tr, d_tr, FEATURE_COLUMNS)

            p_base = base.predict_proba(X_te)[:, 1]
            sel_base = capacity_selection(p_base, CAPACITY)
            row = {
                "fold": fold, "model": name,
                "selected_feature": feature or "(none found)",
                "selection_p_in_training": sel_p,
                "baseline_auc": roc_auc_score(y_te, p_base),
                "baseline_p_equal_opportunity": equal_opportunity_p(y_te, sel_base, d_te),
                "baseline_fnr_gap": fnr_gap(y_te, sel_base, d_te),
            }

            if feature is not None:
                kept = [c for c in X_tr.columns if c != feature]
                mit = build(X_tr[kept]).fit(X_tr[kept], y_tr)
                p_mit = mit.predict_proba(X_te[kept])[:, 1]
                sel_mit = capacity_selection(p_mit, CAPACITY)
                row.update({
                    "mitigated_auc": roc_auc_score(y_te, p_mit),
                    "mitigated_p_equal_opportunity": equal_opportunity_p(y_te, sel_mit, d_te),
                    "mitigated_fnr_gap": fnr_gap(y_te, sel_mit, d_te),
                })
            rows.append(row)
            print(f"fold {fold} {name:9} selected={row['selected_feature']:22} "
                  f"auc {row['baseline_auc']:.4f} -> {row.get('mitigated_auc', float('nan')):.4f} | "
                  f"eo p {row['baseline_p_equal_opportunity']:.3f} -> "
                  f"{row.get('mitigated_p_equal_opportunity', float('nan')):.3f}")

    result = pd.DataFrame(rows)
    result["auc_change"] = result.mitigated_auc - result.baseline_auc
    result["fnr_gap_change"] = result.mitigated_fnr_gap.abs() - result.baseline_fnr_gap.abs()
    result.to_csv(ARTIFACT_DIR / "mitigation_nested.csv", index=False)

    summary = (result.groupby("model")
               .agg(folds=("fold", "size"),
                    modal_selection=("selected_feature", lambda s: s.mode().iloc[0]),
                    selection_agreement=("selected_feature", lambda s: s.eq(s.mode().iloc[0]).mean()),
                    baseline_auc=("baseline_auc", "mean"),
                    mitigated_auc=("mitigated_auc", "mean"),
                    auc_change=("auc_change", "mean"),
                    baseline_eo_p=("baseline_p_equal_opportunity", "mean"),
                    mitigated_eo_p=("mitigated_p_equal_opportunity", "mean"),
                    baseline_fnr_gap=("baseline_fnr_gap", "mean"),
                    mitigated_fnr_gap=("mitigated_fnr_gap", "mean"),
                    folds_eo_not_rejected=("mitigated_p_equal_opportunity",
                                           lambda s: int((s > 0.05).sum())))
               .reset_index())
    summary.to_csv(ARTIFACT_DIR / "mitigation_nested_summary.csv", index=False)

    print("\nOut-of-fold summary (selection and assessment separated):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
