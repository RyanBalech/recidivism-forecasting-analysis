"""Two white-box models from the course, compared with the shipped logistic regression and XGBoost.

- AdaLogit (guest lecture, pp. 69-81): adaptive L1 logistic regression. A pilot ridge fit gives
  weights w_j = |beta_pilot_j|^-gamma; the weighted L1 problem is solved by rescaling each column by
  1/w_j and fitting an ordinary L1 logit, then mapping the coefficients back. Important features are
  penalised less, noise features more.
- PLTR (Dumitrescu et al. 2022, pp. 95-103): penalised logistic tree regression. Step 1 learns
  threshold rules from short trees: one depth-1 tree per feature (univariate rule) and one depth-2
  tree per feature pair (bivariate rule from the second split). Step 2 fits an adaptive-L1 logit on
  the original features plus those binary rules. It stays a logistic regression, so every selected
  rule has a readable coefficient.

Selection uses 5-fold CV on the training set only; the evaluation cohort is scored once at the end.
PLTR's rules are extracted from the full training set before CV, so its CV score is slightly
optimistic; its evaluation-cohort score is not affected (rules never see evaluation rows).
Exploratory: not in the recommendation.
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FEATURE_COLUMNS, RANDOM_SEED, pretty
from recidivism.data import load_official_split
from recidivism.modeling import logistic_model, ordinal_encode, preprocessor, xgboost_model

warnings.filterwarnings("ignore")
CV = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
C_GRID = np.logspace(-3, 0.5, 8)
GAMMAS = [0.5, 1.0, 2.0]


# ---------------------------------------------------------------- AdaLogit
def adaptive_scales(X: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """Column scales |beta_pilot|^gamma from a ridge pilot fit (course eq. 1)."""
    pilot = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    return np.maximum(np.abs(pilot.coef_[0]), 1e-6) ** gamma


def fit_adaptive(X: np.ndarray, y: np.ndarray, gamma: float, C: float):
    scale = adaptive_scales(X, y, gamma)
    model = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000).fit(X * scale, y)
    return model, scale


def predict_adaptive(model, scale, X):
    return model.predict_proba(X * scale)[:, 1]


def cv_adaptive(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Best (gamma, C) by 5-fold CV AUC on the training set."""
    best = (None, None, -1.0)
    for gamma in GAMMAS:
        for C in C_GRID:
            scores = []
            for tr, va in CV.split(X, y):
                model, scale = fit_adaptive(X[tr], y[tr], gamma, C)
                scores.append(roc_auc_score(y[va], predict_adaptive(model, scale, X[va])))
            if np.mean(scores) > best[2]:
                best = (gamma, C, float(np.mean(scores)))
    return best


# ---------------------------------------------------------------- PLTR rules
def numeric_raw(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """One numeric column per raw feature: ordinal where ordered, category codes otherwise."""
    out = ordinal_encode(frame)
    ref = ordinal_encode(reference)
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            cats = sorted(ref[col].dropna().astype(str).unique())
            out[col] = out[col].astype(str).map({c: i for i, c in enumerate(cats)}).astype(float)
        out[col] = out[col].fillna(ref[col].median() if pd.api.types.is_numeric_dtype(ref[col]) else 0.0)
    return out


def learn_rules(Xn: pd.DataFrame, y: np.ndarray) -> list[tuple]:
    """Univariate rules from depth-1 trees; bivariate rules from the second split of depth-2 trees."""
    rules = []
    for col in Xn.columns:
        t = DecisionTreeClassifier(max_depth=1, min_samples_leaf=200, random_state=RANDOM_SEED).fit(Xn[[col]], y)
        if t.tree_.feature[0] >= 0:
            rules.append(("uni", col, t.tree_.threshold[0]))
    for a, b in itertools.combinations(Xn.columns, 2):
        t = DecisionTreeClassifier(max_depth=2, min_samples_leaf=200, random_state=RANDOM_SEED).fit(Xn[[a, b]], y)
        tr = t.tree_
        if tr.feature[0] < 0:
            continue
        root_col, root_thr = [a, b][tr.feature[0]], tr.threshold[0]
        for side, child in [("le", tr.children_left[0]), ("gt", tr.children_right[0])]:
            if tr.feature[child] >= 0:
                rules.append(("bi", root_col, root_thr, side, [a, b][tr.feature[child]], tr.threshold[child]))
    return rules


def apply_rules(Xn: pd.DataFrame, rules: list[tuple]) -> np.ndarray:
    cols = []
    for r in rules:
        if r[0] == "uni":
            cols.append(Xn[r[1]].to_numpy() <= r[2])
        else:
            _, c1, t1, side, c2, t2 = r
            first = Xn[c1].to_numpy() <= t1 if side == "le" else Xn[c1].to_numpy() > t1
            cols.append(first & (Xn[c2].to_numpy() <= t2))
    return np.column_stack(cols).astype(float)


def describe(rule: tuple) -> str:
    if rule[0] == "uni":
        return f"{pretty(rule[1])} ≤ {rule[2]:.1f}"
    _, c1, t1, side, c2, t2 = rule
    return f"{pretty(c1)} {'≤' if side == 'le' else '>'} {t1:.1f} AND {pretty(c2)} ≤ {t2:.1f}"


def main() -> None:
    split = load_official_split()
    ytr, yte = split.y_train.to_numpy(), split.y_test.to_numpy()
    rows = []

    def record(name, cv_auc, p, n_terms, seconds, note=""):
        rows.append({"model": name, "cv_roc_auc": cv_auc, "test_roc_auc": roc_auc_score(yte, p),
                     "test_brier": brier_score_loss(yte, p), "nonzero_terms": n_terms, "seconds": seconds, "note": note})
        print(f"{name:32s} cv {cv_auc:.4f}  test {rows[-1]['test_roc_auc']:.4f}  terms {n_terms}", flush=True)

    # References: shipped configurations, refitted here.
    for name, builder in [("logistic (shipped L1)", logistic_model), ("xgboost (shipped)", xgboost_model)]:
        start = time.perf_counter()
        pipe = builder(split.X_train).fit(split.X_train, split.y_train)
        p = pipe.predict_proba(split.X_test)[:, 1]
        n = int((np.abs(pipe[-1].coef_[0]) > 1e-8).sum()) if name.startswith("logistic") else -1
        record(name, float("nan"), p, n, time.perf_counter() - start, "reference; CV AUC in logistic_tuning.json / ml_model_comparison.csv")

    # Shared standardized design matrix (one-hot for categoricals), as in the shipped logistic.
    prep = preprocessor(split.X_train).fit(split.X_train)
    dense = lambda a: a.toarray() if hasattr(a, "toarray") else a
    Xtr, Xte = dense(prep.transform(split.X_train)), dense(prep.transform(split.X_test))

    # AdaLogit
    start = time.perf_counter()
    gamma, C, cv_auc = cv_adaptive(Xtr, ytr)
    model, scale = fit_adaptive(Xtr, ytr, gamma, C)
    record("AdaLogit (adaptive L1)", cv_auc, predict_adaptive(model, scale, Xte),
           int((np.abs(model.coef_[0]) > 1e-8).sum()), time.perf_counter() - start, f"gamma={gamma}, C={C:.4g}")

    # PLTR
    start = time.perf_counter()
    ntr, nte = numeric_raw(split.X_train, split.X_train), numeric_raw(split.X_test, split.X_train)
    rules = learn_rules(ntr, ytr)
    Rtr, Rte = apply_rules(ntr, rules), apply_rules(nte, rules)
    keep = Rtr.std(axis=0) > 0
    Rtr, Rte, rules = Rtr[:, keep], Rte[:, keep], [r for r, k in zip(rules, keep) if k]
    mu, sd = Rtr.mean(axis=0), Rtr.std(axis=0)
    Ztr = np.hstack([Xtr, (Rtr - mu) / sd])
    Zte = np.hstack([Xte, (Rte - mu) / sd])
    gamma, C, cv_auc = cv_adaptive(Ztr, ytr)
    model, scale = fit_adaptive(Ztr, ytr, gamma, C)
    coef = model.coef_[0] * scale
    n_rules_kept = int((np.abs(coef[Xtr.shape[1]:]) > 1e-8).sum())
    record("PLTR (trees → adaptive-L1 logit)", cv_auc, predict_adaptive(model, scale, Zte),
           int((np.abs(coef) > 1e-8).sum()), time.perf_counter() - start,
           f"{len(rules)} candidate rules, {n_rules_kept} kept; gamma={gamma}, C={C:.4g}")
    rule_coef = pd.DataFrame({"rule": [describe(r) for r in rules], "coefficient": coef[Xtr.shape[1]:]})
    rule_coef = rule_coef[rule_coef.coefficient.abs() > 1e-8].sort_values("coefficient", key=abs, ascending=False)
    rule_coef.to_csv(ARTIFACT_DIR / "pltr_rules.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACT_DIR / "course_whitebox_models.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("\nTop PLTR rules:\n", rule_coef.head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
