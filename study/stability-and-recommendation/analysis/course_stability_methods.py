"""The professor's own stability methods (course slides 182–194), applied to recidivism.

`stability_course_aligned.py` measures stability. This script reproduces the three
things the lecture *does* with it, on our data and our L1 logistic model:

1. **Retraining on more data (slides 185, 187, 191).** Fit on a random 50% of the
   training records (D₁), then on all of them (D₂ ⊃ D₁, n₂ > n₁), which is the lecture's
   "50% sample vs full data" comparison. It is repeated on five random halves, for
   logistic and XGBoost.
2. **Stability-constrained re-estimation (slide 186).**
       θ̂₂ = argmin_θ  L(θ; D₂) + λ ‖θ − θ̂₁‖²₂ ,   λ chosen by cross-validation,
   with the logistic negative log-likelihood as L. D₁ and D₂ are *separate* datasets with
   n₂ > n₁ (40% / 60% of the training records), as on slide 185, so the cross-validation
   inside D₂ never sees the data behind θ̂₁. A light ridge term γ‖θ‖² (the same for θ̂₁
   and θ̂₂) keeps the full one-hot parameterisation identified. The output is the
   predictive-loss vs stability-loss frontier of slide 192.
3. **Which penalty stabilises the coefficients (guest lecture, slides 43–44).** L1
   (the published model), L2 and elastic net, refitted on the published eight bootstrap
   resamples. The lecture's claim is that L1 gives sparsity, L2 gives stability under
   collinearity, and elastic net gives both.

CPU only, a few minutes. Writes only to ../results/.

    python study/stability-and-recommendation/analysis/course_stability_methods.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from recidivism.data import load_official_split
from recidivism.metrics import capacity_selection
from recidivism.modeling import LOGIT_PARAMS, logistic_model, preprocessor, xgboost_model
from stability_course_aligned import (MODEL_SEED, PALETTE, bootstrap_samples, coefficients,
                                      fit, fitted_summary, pair_distances)

CAPACITY = 0.20
GAMMA = 1.0                      # light ridge for identifiability (≈ sklearn L2 with C = 0.5)
LAMBDAS = [0, 10, 30, 100, 300, 1_000, 3_000, 10_000, 100_000]
N_SPLITS = 5


def _dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m, dtype=float)


def jaccard_top(p_a: np.ndarray, p_b: np.ndarray) -> float:
    a, b = capacity_selection(p_a, CAPACITY), capacity_selection(p_b, CAPACITY)
    return float((a & b).sum() / (a | b).sum())


# ---------------------------------------------------------------- 1. retraining on more data

def retrain_on_more_data(split, X_imp) -> pd.DataFrame:
    """Slides 187/191: a model on a random 50% of the data vs the model on all of it."""
    rows = []
    for kind in ["logistic", "xgboost"]:
        full = fitted_summary(kind, fit(kind, split.X_train, split.y_train), split.X_train, split, X_imp)
        for r in range(N_SPLITS):
            idx, _ = train_test_split(np.arange(len(split.X_train)), train_size=0.5,
                                      stratify=split.y_train, random_state=100 + r)
            X, y = split.X_train.iloc[idx].reset_index(drop=True), split.y_train.iloc[idx].reset_index(drop=True)
            half = fitted_summary(kind, fit(kind, X, y), X, split, X_imp)
            rows.append({"model": kind, "split": r, "auc_half": roc_auc_score(split.y_test, half["p"]),
                         "auc_full": roc_auc_score(split.y_test, full["p"]), **pair_distances(half, full)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. slide 186

def _anchored_logit(X: np.ndarray, y: np.ndarray, anchor: np.ndarray | None, lam: float,
                    start: np.ndarray | None = None) -> np.ndarray:
    """argmin  Σ NLL + γ‖w‖² + λ‖w − w₁‖²  (intercept unpenalised). Returns [w, b]."""
    p = X.shape[1]
    w1 = np.zeros(p) if anchor is None else anchor[:p]

    def objective(theta):
        w, b = theta[:p], theta[p]
        z = X @ w + b
        nll = np.sum(np.logaddexp(0.0, z) - y * z)
        resid = expit(z) - y
        diff = w - w1
        value = nll + GAMMA * w @ w + lam * diff @ diff
        grad_w = X.T @ resid + 2 * GAMMA * w + 2 * lam * diff
        return value, np.append(grad_w, resid.sum())

    x0 = np.zeros(p + 1) if start is None else start
    res = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": 5_000, "gtol": 1e-6})
    return res.x


def _predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return expit(X @ theta[:-1] + theta[-1])


def stability_constrained(split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slide 186 on two separate datasets D₁ (40%) and D₂ (60%), repeated on five draws."""
    per_split, cv_rows = [], []
    for r in range(N_SPLITS):
        d1, d2 = train_test_split(np.arange(len(split.X_train)), train_size=0.4,
                                  stratify=split.y_train, random_state=200 + r)
        # The incumbent's feature pipeline is reused for the retrain, so θ̂₁ and θ̂₂ live in
        # exactly the same coordinates and ‖θ̂₂ − θ̂₁‖ is well defined.
        prep = preprocessor(split.X_train.iloc[d1]).fit(split.X_train.iloc[d1])
        X1 = _dense(prep.transform(split.X_train.iloc[d1]))
        X2 = _dense(prep.transform(split.X_train.iloc[d2]))
        Xe = _dense(prep.transform(split.X_test))
        y1, y2 = split.y_train.to_numpy()[d1].astype(float), split.y_train.to_numpy()[d2].astype(float)
        theta1 = _anchored_logit(X1, y1, None, 0.0)
        p1 = _predict(theta1, Xe)

        folds = list(StratifiedKFold(5, shuffle=True, random_state=MODEL_SEED).split(X2, y2))
        for lam in LAMBDAS:
            cv_auc, cv_ll = [], []
            for tr, va in folds:
                th = _anchored_logit(X2[tr], y2[tr], theta1, lam, start=theta1)
                pv = _predict(th, X2[va])
                cv_auc.append(roc_auc_score(y2[va], pv))
                cv_ll.append(log_loss(y2[va], pv))
            theta2 = _anchored_logit(X2, y2, theta1, lam, start=theta1)
            p2 = _predict(theta2, Xe)
            cv_rows.append({"split": r, "lambda": lam,
                            "cv_auc": float(np.mean(cv_auc)), "cv_log_loss": float(np.mean(cv_ll)),
                            "coef_distance_l2": float(np.linalg.norm(theta2[:-1] - theta1[:-1])),
                            "coef_distance_relative": float(np.linalg.norm(theta2[:-1] - theta1[:-1])
                                                            / np.linalg.norm(theta1[:-1])),
                            "top20_jaccard_vs_old": jaccard_top(p1, p2),
                            "mean_abs_prob_diff_vs_old": float(np.mean(np.abs(p2 - p1))),
                            "eval_auc_new": float(roc_auc_score(split.y_test, p2)),
                            "eval_auc_old": float(roc_auc_score(split.y_test, p1))})
        print(f"  slide-186 split {r + 1}/{N_SPLITS} done", flush=True)

    cv = pd.DataFrame(cv_rows)
    for r, part in cv.groupby("split"):
        best = part.loc[part.cv_log_loss.idxmin()]
        naive = part[part["lambda"] == 0].iloc[0]
        per_split.append({"split": r, "lambda_cv": best["lambda"],
                          "cv_log_loss_naive": naive.cv_log_loss, "cv_log_loss_cv": best.cv_log_loss,
                          "cv_auc_naive": naive.cv_auc, "cv_auc_cv": best.cv_auc,
                          "coef_distance_naive": naive.coef_distance_l2, "coef_distance_cv": best.coef_distance_l2,
                          "jaccard_naive": naive.top20_jaccard_vs_old, "jaccard_cv": best.top20_jaccard_vs_old,
                          "eval_auc_naive": naive.eval_auc_new, "eval_auc_cv": best.eval_auc_new})
    return cv, pd.DataFrame(per_split)


# ---------------------------------------------------------------- 3. slides 43–44

def _penalty_pipeline(name: str, X: pd.DataFrame, C: float) -> Pipeline:
    if name in ("L1 (published)", "L2 (ridge)"):
        return logistic_model(X, penalty="l1" if name.startswith("L1") else "l2", C=C)
    return Pipeline([("prepare", preprocessor(X)),
                     ("model", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=C,
                                                  max_iter=10_000, tol=1e-4, random_state=MODEL_SEED))])


def penalty_comparison(split) -> pd.DataFrame:
    """L1 vs L2 vs elastic net: CV AUC for C, then coefficient stability over the 8 resamples."""
    folds = list(StratifiedKFold(5, shuffle=True, random_state=MODEL_SEED).split(split.X_train, split.y_train))
    samples = bootstrap_samples(len(split.X_train))
    rows = []
    for name, grid in [("L1 (published)", [LOGIT_PARAMS["C"]]), ("L2 (ridge)", [0.005, 0.02, 0.1, 0.5]),
                       ("Elastic net (α = 0.5)", [0.02, 0.1, 0.2154])]:
        scores = {}
        for C in grid:
            auc = []
            for tr, va in folds:
                m = _penalty_pipeline(name, split.X_train.iloc[tr], C).fit(split.X_train.iloc[tr], split.y_train.iloc[tr])
                auc.append(roc_auc_score(split.y_train.iloc[va], m.predict_proba(split.X_train.iloc[va])[:, 1]))
            scores[C] = float(np.mean(auc))
        C = max(scores, key=scores.get)
        fits = []
        for idx in samples:
            X, y = split.X_train.iloc[idx].reset_index(drop=True), split.y_train.iloc[idx].reset_index(drop=True)
            m = _penalty_pipeline(name, X, C).fit(X, y)
            fits.append({"p": m.predict_proba(split.X_test)[:, 1], "coef": coefficients(m)})
        pairs = []
        for a, b in itertools.combinations(fits, 2):
            ca, cb = a["coef"].align(b["coef"], fill_value=0.0)
            both = (ca != 0) & (cb != 0)
            pairs.append({"coef_l2": np.linalg.norm(ca - cb),
                          "coef_l2_relative": np.linalg.norm(ca - cb) / ((np.linalg.norm(ca) + np.linalg.norm(cb)) / 2),
                          "sign_agreement": (np.sign(ca[both]) == np.sign(cb[both])).mean(),
                          "zero_pattern_agreement": ((ca == 0) == (cb == 0)).mean(),
                          "jaccard": jaccard_top(a["p"], b["p"]),
                          "dp": np.mean(np.abs(a["p"] - b["p"]))})
        pairs = pd.DataFrame(pairs)
        rows.append({"penalty": name, "C_by_cv": C, "cv_auc": scores[C],
                     "n_nonzero_coefs": float(np.mean([(f["coef"] != 0).sum() for f in fits])),
                     "n_coefs": int(len(fits[0]["coef"])),
                     **{c: float(pairs[c].mean()) for c in pairs.columns}})
        print(f"  {name}: C={C}, CV AUC {scores[C]:.4f}, relative ‖Δθ‖ {pairs.coef_l2_relative.mean():.3f}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def figure_frontier(cv: pd.DataFrame) -> None:
    """Our version of slide 192: stability loss against predictive loss, one point per λ."""
    agg = cv.groupby("lambda").mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    ax.plot(agg.cv_log_loss, agg.coef_distance_l2, "o-", color=PALETTE["logistic"], lw=2)
    for _, r in agg.iterrows():
        ax.annotate(f"λ={r['lambda']:g}", (r.cv_log_loss, r.coef_distance_l2), xytext=(6, 2),
                    textcoords="offset points", fontsize=8, color="grey")
    ax.set(xlabel="Predictive loss: CV log loss inside D₂ (lower = better)",
           ylabel="Stability loss: ‖θ̂₂ − θ̂₁‖₂ (lower = stabler)",
           title="Our slide 192: the stability / accuracy frontier")
    ax = axes[1]
    ax.plot(agg["lambda"].replace(0, 1), agg.top20_jaccard_vs_old, "s-", color="#2A9D8F", lw=2)
    ax.set_xscale("log")
    ax.set(xlabel="λ (0 plotted at 1)", ylabel="top-20% Jaccard, old model vs retrained",
           title="Same people selected after the retrain?")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Slide 186 on recidivism: retrain on D₂ while penalising distance to the old model θ̂₁ (5 draws)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_slide186_frontier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_penalties(pen: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = ["#234E70", "#2A9D8F", "#E9C46A"]
    for ax, col, title in [(axes[0], "coef_l2_relative", "Coefficient drift  ‖Δθ‖ / ‖θ‖  (lower = stabler)"),
                           (axes[1], "n_nonzero_coefs", "Non-zero coefficients (sparsity)"),
                           (axes[2], "jaccard", "Top-20% Jaccard across refits")]:
        bars = ax.bar(pen.penalty, pen[col], color=colors)
        ax.bar_label(bars, fmt="%.3f" if col != "n_nonzero_coefs" else "%.0f", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Guest lecture, slides 43–44: penalties change the coefficients, not who is selected", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_penalty_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    split = load_official_split()
    X_imp = split.X_test.sample(1000, random_state=0)

    print("1. retraining on more data (slides 185/187/191)")
    retrain = retrain_on_more_data(split, X_imp)
    retrain.to_csv(OUT / "course_retrain_50pct_vs_full.csv", index=False)

    print("2. stability-constrained re-estimation (slide 186)")
    cv, chosen = stability_constrained(split)
    cv.to_csv(OUT / "course_slide186_lambda_path.csv", index=False)
    chosen.to_csv(OUT / "course_slide186_chosen.csv", index=False)
    figure_frontier(cv)

    print("3. L1 vs L2 vs elastic net (slides 43–44)")
    pen = penalty_comparison(split)
    pen.to_csv(OUT / "course_penalty_stability.csv", index=False)
    figure_penalties(pen)

    pd.set_option("display.width", 220)
    print(retrain.groupby("model")[["auc_half", "auc_full", "mean_abs_prob_diff", "top20_jaccard",
                                    "importance_l2", "coef_l2", "coef_l2_relative"]].mean().round(4).to_string())
    print(cv.groupby("lambda").mean(numeric_only=True).drop(columns="split").round(4).to_string())
    print(chosen.round(4).to_string(index=False))
    print(pen.round(4).to_string(index=False))
    info = {"runtime_seconds": round(time.perf_counter() - t0, 1), "gamma_ridge": GAMMA, "lambdas": LAMBDAS}
    (OUT / "course_methods_run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"done in {info['runtime_seconds']}s")


if __name__ == "__main__":
    main()
