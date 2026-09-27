"""Course-aligned stability extras for the P6 study pack (course §7).

The team's stability audit (scripts/stability_structural.py, scripts/individual_stability.py)
measures how much predictions and selected sets move between bootstrap refits. The course
defines stability more literally, and this script adds the pieces docs/PLAN.md item P1.9
left open:

1. **Distance between models, the course's way.** ‖θ₁ − θ₂‖₂ on the logistic coefficients,
   and ‖φ(f₁) − φ(f₂)‖₂ on normalised feature-importance vectors for both models.
2. **Two disjoint halves of the training set** ("two datasets from the same population"),
   next to the bootstrap refits the deck already reports.
3. **Seed-only versus data-only randomness** (course §7.2): same data with different model
   seeds, against different data with the same seed.
4. **Stability versus performance over the L1 penalty C** for logistic (course p192),
   with performance measured by training-only cross-validation.
5. **Which logistic coefficients survive resampling**: how often L1 keeps each one non-zero,
   and whether its sign ever flips.
6. **Where the contested decisions are**, re-read from the saved per-person file.

TabICLv2 is not refitted here: it needs a GPU. Its bootstrap numbers stay those of the
published audit. Everything runs on CPU in a few minutes and writes only to this folder.

    python study/stability-and-recommendation/analysis/stability_course_aligned.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "results"
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FEATURE_COLUMNS, pretty
from recidivism.data import load_official_split
from recidivism.metrics import capacity_selection
from recidivism.modeling import LOGIT_PARAMS, logistic_model, xgboost_model

CAPACITY = 0.20
N_BOOT = 8
BOOTSTRAP_SEED = 7   # identical to stability_structural.py, so the resamples are the published ones
MODEL_SEED = 42
N_HALVES = 10
C_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, LOGIT_PARAMS["C"], 0.5, 1.0, 5.0]
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500", "tabicl": "#7B2CBF"}
DISPLAY = {"logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}


# ---------------------------------------------------------------- model internals

def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=float)


def _raw_feature(column: str) -> str:
    """Map a transformed column (one-hot level, missing flag) back to its raw NIJ field."""
    column = column.removeprefix("missingindicator_")
    for raw in sorted(FEATURE_COLUMNS, key=len, reverse=True):
        if column == raw or column.startswith(raw + "_"):
            return raw
    return column


def _transform(pipe, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = list(pipe.named_steps["prepare"].get_feature_names_out())
    return _dense(pipe[:-1].transform(X)), names


def coefficients(pipe) -> pd.Series:
    names = pipe.named_steps["prepare"].get_feature_names_out()
    return pd.Series(pipe.named_steps["model"].coef_[0], index=names)


def importance(pipe, kind: str, X_eval: pd.DataFrame, X_fit: pd.DataFrame) -> pd.Series:
    """Mean |contribution| per raw feature, i.e. SHAP importance aggregated to the 29 fields.

    Logistic: exact linear SHAP, coef × (x − training mean) in the transformed space.
    XGBoost: exact TreeSHAP from XGBoost's own ``pred_contribs`` (no shap package needed).
    Contributions of one-hot levels are summed per person before taking |·|, as in the deck.
    """
    Xt, names = _transform(pipe, X_eval)
    if kind == "logistic":
        background = _transform(pipe, X_fit)[0].mean(axis=0)
        contrib = (Xt - background) * pipe.named_steps["model"].coef_[0]
    else:
        booster = pipe.named_steps["model"].get_booster()
        contrib = booster.predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]
    frame = pd.DataFrame(contrib, columns=names)
    per_raw = frame.T.groupby([_raw_feature(c) for c in names]).sum().T
    return per_raw.abs().mean().reindex(FEATURE_COLUMNS, fill_value=0.0)


def fit(kind: str, X: pd.DataFrame, y: pd.Series, seed: int = MODEL_SEED, **overrides):
    if kind == "logistic":
        # logistic_model fixes random_state=42 internally; override it on the pipeline.
        return logistic_model(X, **overrides).set_params(model__random_state=seed).fit(X, y)
    return xgboost_model(X, random_state=seed, **overrides).fit(X, y)


# ---------------------------------------------------------------- distances

def pair_distances(a: dict, b: dict) -> dict:
    """Every distance the course and the deck use, for one pair of fitted models."""
    top_a = capacity_selection(a["p"], CAPACITY)
    top_b = capacity_selection(b["p"], CAPACITY)
    ia, ib = a["imp"] / a["imp"].sum(), b["imp"] / b["imp"].sum()
    row = {
        "mean_abs_prob_diff": float(np.mean(np.abs(a["p"] - b["p"]))),
        "p95_abs_prob_diff": float(np.quantile(np.abs(a["p"] - b["p"]), 0.95)),
        "rank_correlation": float(spearmanr(a["p"], b["p"]).statistic),
        "top20_jaccard": float((top_a & top_b).sum() / (top_a | top_b).sum()),
        # Course: ‖φ(f₁) − φ(f₂)‖₂ on importance vectors. Normalised to shares of total
        # importance so that logistic and XGBoost are on the same 0–√2 scale.
        "importance_l2": float(np.linalg.norm(ia - ib)),
        "importance_rank_correlation": float(spearmanr(ia, ib).statistic),
    }
    if "coef" in a:
        ca, cb = a["coef"].align(b["coef"], fill_value=0.0)
        both = (ca != 0) & (cb != 0)
        row.update({
            # Course: ‖θ₁ − θ₂‖₂ on the parameter vectors.
            "coef_l2": float(np.linalg.norm(ca - cb)),
            "coef_l2_relative": float(np.linalg.norm(ca - cb) / ((np.linalg.norm(ca) + np.linalg.norm(cb)) / 2)),
            "coef_zero_pattern_agreement": float(((ca == 0) == (cb == 0)).mean()),
            "coef_sign_agreement": float((np.sign(ca[both]) == np.sign(cb[both])).mean()) if both.any() else np.nan,
        })
    return row


def fitted_summary(kind: str, pipe, X_fit, split, X_imp) -> dict:
    out = {"p": pipe.predict_proba(split.X_test)[:, 1], "imp": importance(pipe, kind, X_imp, X_fit)}
    if kind == "logistic":
        out["coef"] = coefficients(pipe)
    return out


def all_pairs(regime: str, kind: str, fits: list[dict], pairs=None) -> list[dict]:
    pairs = pairs or list(itertools.combinations(range(len(fits)), 2))
    return [{"regime": regime, "model": kind, "fit_i": i, "fit_j": j, **pair_distances(fits[i], fits[j])}
            for i, j in pairs]


# ---------------------------------------------------------------- experiments

def bootstrap_samples(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return [rng.integers(0, n, n) for _ in range(N_BOOT)]


def run_regimes(split, X_imp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Seed-only, bootstrap (data-only) and disjoint halves, for logistic and XGBoost."""
    rows, coef_rows = [], []
    samples = bootstrap_samples(len(split.X_train))
    published = json.loads((ARTIFACT_DIR / "stability_protocol.json").read_text())["sample_sha256"]
    hashes = [hashlib.sha256(s.astype("<i8").tobytes()).hexdigest() for s in samples]
    check = {"bootstrap_samples_match_published": hashes == published}
    print(f"bootstrap resamples identical to the published protocol: {check['bootstrap_samples_match_published']}")

    for kind in ["logistic", "xgboost"]:
        t0 = time.perf_counter()
        # 1. Seed-only: full training data, eight different model seeds.
        seed_fits = [fitted_summary(kind, fit(kind, split.X_train, split.y_train, seed=s), split.X_train, split, X_imp)
                     for s in range(N_BOOT)]
        rows += all_pairs("seed_only", kind, seed_fits)

        # 2. Data-only: the published eight bootstrap resamples, model seed fixed.
        boot_fits = []
        for idx in samples:
            X = split.X_train.iloc[idx].reset_index(drop=True)
            y = split.y_train.iloc[idx].reset_index(drop=True)
            boot_fits.append(fitted_summary(kind, fit(kind, X, y), X, split, X_imp))
        rows += all_pairs("bootstrap", kind, boot_fits)
        if kind == "logistic":
            coef_rows = [f["coef"].rename(f"refit_{k}") for k, f in enumerate(boot_fits)]

        # 3. Disjoint halves: two datasets from the same population with no shared person.
        half_fits = []
        for r in range(N_HALVES):
            a_idx, b_idx = train_test_split(np.arange(len(split.X_train)), test_size=0.5,
                                            stratify=split.y_train, random_state=r)
            for idx in (a_idx, b_idx):
                X = split.X_train.iloc[idx].reset_index(drop=True)
                y = split.y_train.iloc[idx].reset_index(drop=True)
                summary = fitted_summary(kind, fit(kind, X, y), X, split, X_imp)
                summary["auc"] = roc_auc_score(split.y_test, summary["p"])
                half_fits.append(summary)
        rows += all_pairs("disjoint_halves", kind, half_fits, pairs=[(2 * r, 2 * r + 1) for r in range(N_HALVES)])
        check[f"{kind}_half_model_mean_eval_auc"] = float(np.mean([f["auc"] for f in half_fits]))
        print(f"{kind}: seed-only, bootstrap and halves done ({time.perf_counter() - t0:.0f}s)", flush=True)

    return pd.DataFrame(rows), pd.concat(coef_rows, axis=1), check


def coefficient_stability(boot_coefs: pd.DataFrame, full_fit) -> pd.DataFrame:
    """Does the white-box story survive resampling? One row per logistic coefficient."""
    full = coefficients(full_fit)
    table = boot_coefs.reindex(boot_coefs.index.union(full.index)).fillna(0.0)
    nonzero = table.ne(0)
    signs = np.sign(table.where(nonzero))
    out = pd.DataFrame({
        "feature": table.index,
        "label": [pretty(c) for c in table.index],
        "coef_full_fit": full.reindex(table.index).fillna(0.0).to_numpy(),
        "coef_mean_refits": table.mean(axis=1).to_numpy(),
        "coef_min_refits": table.min(axis=1).to_numpy(),
        "coef_max_refits": table.max(axis=1).to_numpy(),
        "share_refits_nonzero": nonzero.mean(axis=1).to_numpy(),
        "sign_flips": (signs.nunique(axis=1) > 1).to_numpy(),
    })
    return out.sort_values("coef_full_fit", key=np.abs, ascending=False).reset_index(drop=True)


def c_sweep(split) -> pd.DataFrame:
    """Course p192: how stability and performance move together as the L1 penalty changes.

    Performance is 5-fold CV AUC inside the training partition, so no evaluation label
    chooses anything. Stability uses the published bootstrap resamples.
    """
    samples = bootstrap_samples(len(split.X_train))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=MODEL_SEED).split(split.X_train, split.y_train))
    rows = []
    for C in C_GRID:
        cv = []
        for tr, va in folds:
            m = fit("logistic", split.X_train.iloc[tr], split.y_train.iloc[tr], C=C)
            cv.append(roc_auc_score(split.y_train.iloc[va], m.predict_proba(split.X_train.iloc[va])[:, 1]))
        fits = []
        for idx in samples:
            X = split.X_train.iloc[idx].reset_index(drop=True)
            y = split.y_train.iloc[idx].reset_index(drop=True)
            m = fit("logistic", X, y, C=C)
            fits.append({"p": m.predict_proba(split.X_test)[:, 1], "coef": coefficients(m)})
        pairs = []
        for a, b in itertools.combinations(fits, 2):
            ca, cb = a["coef"].align(b["coef"], fill_value=0.0)
            top_a, top_b = capacity_selection(a["p"], CAPACITY), capacity_selection(b["p"], CAPACITY)
            pairs.append({"dp": np.mean(np.abs(a["p"] - b["p"])),
                          "jaccard": (top_a & top_b).sum() / (top_a | top_b).sum(),
                          "coef_l2": np.linalg.norm(ca - cb),
                          "coef_l2_relative": np.linalg.norm(ca - cb) / ((np.linalg.norm(ca) + np.linalg.norm(cb)) / 2 or 1)})
        pairs = pd.DataFrame(pairs)
        rows.append({"C": C, "chosen": C == LOGIT_PARAMS["C"],
                     "cv_auc_mean": float(np.mean(cv)), "cv_auc_sd": float(np.std(cv, ddof=1)),
                     "n_nonzero_coefs": float(np.mean([(f["coef"] != 0).sum() for f in fits])),
                     "mean_abs_prob_diff": float(pairs.dp.mean()), "top20_jaccard": float(pairs.jaccard.mean()),
                     "coef_l2": float(pairs.coef_l2.mean()), "coef_l2_relative": float(pairs.coef_l2_relative.mean())})
        print(f"  C={C:<7g} CV AUC {np.mean(cv):.4f}  |Δp| {pairs.dp.mean():.4f}  Jaccard {pairs.jaccard.mean():.3f}", flush=True)
    return pd.DataFrame(rows)


def contested_profile() -> pd.DataFrame:
    """Re-read the published per-person file: who sits at the margin, and how close to the cut?"""
    people = pd.read_csv(ARTIFACT_DIR / "individual_stability.csv")
    rows = []
    for model, part in people.groupby("model"):
        contested = ~part.decision_unanimous.astype(bool)
        selected = part.times_selected >= part.n_refits / 2
        for label, mask in [("all", np.ones(len(part), bool)), ("women", part.Gender.eq("F")), ("men", part.Gender.eq("M")),
                            ("Black", part.Race.eq("BLACK")), ("White", part.Race.eq("WHITE")),
                            ("re-arrested", part.actual.eq(1)), ("not re-arrested", part.actual.eq(0))]:
            mask = np.asarray(mask)
            rows.append({"model": model, "group": label, "n": int(mask.sum()),
                         "contested_share": float(contested[mask].mean()),
                         "n_selected_majority": int((selected & mask).sum()),
                         "contested_share_of_selected": float(contested[selected & mask].mean())})
        # Distance to the cut, in percentiles of the mean score: is instability only at the margin?
        pct = part.mean_probability.rank(pct=True)
        band = pd.cut((pct - (1 - CAPACITY)).abs(), [0, 0.02, 0.05, 0.10, 0.20, 1.0], include_lowest=True)
        for interval, grp in contested.groupby(band, observed=True):
            rows.append({"model": model, "group": f"distance to cut {interval}", "n": int(len(grp)),
                         "contested_share": float(grp.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def figure_regimes(pairs: pd.DataFrame) -> None:
    order = ["seed_only", "bootstrap", "disjoint_halves"]
    names = {"seed_only": "Same data,\nnew seed", "bootstrap": "Bootstrap\n(published)", "disjoint_halves": "Disjoint\nhalves"}
    summary = pairs.groupby(["regime", "model"])[["mean_abs_prob_diff", "top20_jaccard", "importance_l2"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, col, title in [(axes[0], "mean_abs_prob_diff", "Score drift  mean |Δp|  (lower = stabler)"),
                           (axes[1], "top20_jaccard", "Same people selected?  top-20% Jaccard"),
                           (axes[2], "importance_l2", "Explanation drift  ‖φ₁ − φ₂‖₂  (lower = stabler)")]:
        x = np.arange(len(order))
        for k, model in enumerate(["logistic", "xgboost"]):
            vals = [summary.loc[(r, model), col] for r in order]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=PALETTE[model], label=DISPLAY[model])
            ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
        ax.set_xticks(x, [names[r] for r in order])
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(frameon=False)
    fig.suptitle("Three sources of instability, from mildest to harshest (course §7)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stability_regimes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_c_sweep(sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep.C, sweep.cv_auc_mean, "o-", color=PALETTE["logistic"], lw=2, label="CV AUC (training only)")
    ax.set_xscale("log")
    ax.set_xlabel("C  (smaller = stronger L1 penalty)")
    ax.set_ylabel("5-fold CV AUC", color=PALETTE["logistic"])
    twin = ax.twinx()
    twin.plot(sweep.C, sweep.top20_jaccard, "s--", color="#2A9D8F", lw=2, label="top-20% Jaccard across refits")
    twin.set_ylabel("Jaccard across bootstrap refits", color="#2A9D8F")
    chosen = sweep[sweep.chosen].iloc[0]
    ax.axvline(chosen.C, color="grey", ls=":", lw=1.5)
    ax.annotate(f"chosen C = {chosen.C:.4g}", (chosen.C, 0.5), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", color="grey")
    ax.set_title("Stability vs performance over the L1 penalty (course p192)")
    lines = ax.get_lines()[:1] + twin.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", bbox_to_anchor=(1, 0.3), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stability_vs_C.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_coefficients(coefs: pd.DataFrame, top: int = 15) -> None:
    part = coefs[coefs.coef_full_fit != 0].head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(part))
    ax.hlines(y, part.coef_min_refits, part.coef_max_refits, color="#8AA1B1", lw=6, alpha=0.6, label="range over 8 refits")
    ax.plot(part.coef_full_fit, y, "o", color=PALETTE["logistic"], label="published fit")
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(y, [f"{l}  ({s:.0%})" for l, s in zip(part.label, part.share_refits_nonzero)], fontsize=9)
    ax.set_xlabel("coefficient (log-odds per transformed unit)   ·   (%) = refits keeping it non-zero")
    ax.set_title("Largest logistic coefficients across bootstrap refits: does the white-box story hold?")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_coefficient_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    split = load_official_split()
    X_imp = split.X_test.sample(1000, random_state=0)   # features only; no evaluation labels used

    # Environment check against the published run (course §7.2: randomness includes software).
    published = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    env = {"python": platform.python_version(), "platform": platform.platform(), "logical_cpus": os.cpu_count(),
           "scikit_learn": sklearn.__version__, "xgboost": xgb.__version__}
    for kind in ["logistic", "xgboost"]:
        p = fit(kind, split.X_train, split.y_train).predict_proba(split.X_test)[:, 1]
        env[f"{kind}_eval_auc_here"] = float(roc_auc_score(split.y_test, p))
        env[f"{kind}_max_abs_diff_vs_published"] = float(np.abs(p - published[f"p_{kind}"].to_numpy()).max())
    print(json.dumps(env, indent=2))

    pairs, boot_coefs, check = run_regimes(split, X_imp)
    pairs.to_csv(OUT / "stability_pairs_by_regime.csv", index=False)
    summary = pairs.drop(columns=["fit_i", "fit_j"]).groupby(["regime", "model"]).agg(["mean", "min", "max"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.insert(2, "n_pairs", pairs.groupby(["regime", "model"]).size().to_numpy())
    summary.to_csv(OUT / "stability_by_regime_summary.csv", index=False)

    coefs = coefficient_stability(boot_coefs, fit("logistic", split.X_train, split.y_train))
    coefs.to_csv(OUT / "logistic_coefficient_stability.csv", index=False)

    print("C sweep (logistic):")
    sweep = c_sweep(split)
    sweep.to_csv(OUT / "logistic_stability_vs_C.csv", index=False)

    contested_profile().to_csv(OUT / "contested_profile.csv", index=False)

    figure_regimes(pairs)
    figure_c_sweep(sweep)
    figure_coefficients(coefs)

    env.update(check)
    env["runtime_seconds"] = round(time.perf_counter() - start, 1)
    (OUT / "run_info.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    cols = ["regime", "model", "n_pairs", "mean_abs_prob_diff_mean", "top20_jaccard_mean",
            "importance_l2_mean", "coef_l2_mean", "coef_l2_relative_mean"]
    print(summary[[c for c in cols if c in summary]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"done in {env['runtime_seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
