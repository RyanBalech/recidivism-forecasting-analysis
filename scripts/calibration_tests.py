"""Calibration and operating-point differences, tested rather than asserted.

Three claims about the model comparison were being carried as point estimates.
Each is checked here:

1. **"XGBoost is best calibrated."** ECE is bin-dependent, and the ranking flips
   with the binning scheme, so the comparison is redone bin-free with the Cox
   calibration regression (outcome on logit, perfect = intercept 0 / slope 1) and
   the Spiegelhalter z-test. Paired bootstraps then ask whether any pair of models
   actually differs.
2. **"XGBoost captures N more re-arrested people."** Recomputed with an interval at
   the deployed top-20% rule, alongside the overlap between the two selected sets -
   the question is not which ranking is better but whether different people are
   offered support.
3. **"Logistic has the smaller gender FNR gap."** Tested as a paired difference of
   absolute gaps instead of compared as two point estimates.

Everything is paired: each bootstrap resample scores every model on the same rows,
so the comparison is not two independent intervals being eyeballed for overlap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, RANDOM_SEED
from recidivism.metrics import capacity_selection, expected_calibration_error

MODELS = ["logistic", "xgboost", "tabicl"]
PAIRS = [("logistic", "xgboost"), ("logistic", "tabicl"), ("xgboost", "tabicl")]
CAPACITY = 0.20
REPEATS = 2_000
EPS = 1e-9


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def cox_calibration(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Regress the outcome on logit(p). Perfect calibration is intercept 0, slope 1.

    Slope below 1 means the scores are too extreme (over-confident); above 1 means
    too timid. Unlike ECE this needs no bins.
    """
    logit = np.log(_clip(p) / (1 - _clip(p)))
    fit = sm.GLM(np.asarray(y, dtype=float), sm.add_constant(logit),
                 family=sm.families.Binomial()).fit()
    intercept, slope = float(fit.params[0]), float(fit.params[1])
    se_i, se_s = float(fit.bse[0]), float(fit.bse[1])
    return {
        "intercept": intercept, "intercept_se": se_i,
        "slope": slope, "slope_se": se_s,
        "intercept_differs_from_0": bool(abs(intercept) > 1.96 * se_i),
        "slope_differs_from_1": bool(abs(slope - 1) > 1.96 * se_s),
    }


def spiegelhalter_z(y: np.ndarray, p: np.ndarray) -> float:
    """Bin-free calibration test; |z| > 1.96 rejects calibration at 5%."""
    y, p = np.asarray(y, dtype=float), _clip(p)
    numerator = np.sum((y - p) * (1 - 2 * p))
    denominator = np.sqrt(np.sum(((1 - 2 * p) ** 2) * p * (1 - p)))
    return float(numerator / denominator)


def ece_equal_frequency(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE over quantile bins, so no bin is empty or nearly empty."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0] -= EPS
    edges[-1] += EPS
    bucket = np.digitize(p, edges[1:-1])
    total = 0.0
    for b in range(bins):
        mask = bucket == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def brier_reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Reliability term of the Murphy decomposition of the Brier score."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0] -= EPS
    edges[-1] += EPS
    bucket = np.digitize(p, edges[1:-1])
    total = 0.0
    for b in range(bins):
        mask = bucket == b
        if mask.any():
            total += mask.sum() * (p[mask].mean() - y[mask].mean()) ** 2
    return float(total / len(y))


def fnr_gap(y: np.ndarray, selected: np.ndarray, groups: np.ndarray,
            reference: str, protected: str) -> float:
    """FNR(reference) - FNR(protected). Negative: the protected group misses more."""
    out = {}
    for name in (reference, protected):
        mask = (groups == name) & (y == 1)
        out[name] = float(1 - selected[mask].mean()) if mask.any() else np.nan
    return out[reference] - out[protected]


def bin_sensitivity(y, P) -> pd.DataFrame:
    rows = []
    for bins in [5, 10, 15, 20, 50]:
        for model in MODELS:
            rows.append({
                "bins": bins, "model": model,
                "ece_equal_width": expected_calibration_error(y, P[model], bins),
                "ece_equal_frequency": ece_equal_frequency(y, P[model], bins),
            })
    frame = pd.DataFrame(rows)
    winners = []
    for bins, part in frame.groupby("bins"):
        winners.append({
            "bins": bins,
            "best_equal_width": part.loc[part.ece_equal_width.idxmin(), "model"],
            "best_equal_frequency": part.loc[part.ece_equal_frequency.idxmin(), "model"],
        })
    return frame, pd.DataFrame(winners)


def main() -> None:
    pred = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    y = pred["actual"].to_numpy()
    gender = pred["Gender"].to_numpy()
    P = {m: _clip(pred[f"p_{m}"].to_numpy()) for m in MODELS}

    # --- point estimates -------------------------------------------------
    rows = []
    for model in MODELS:
        cox = cox_calibration(y, P[model])
        z = spiegelhalter_z(y, P[model])
        rows.append({
            "model": model, **cox, "spiegelhalter_z": z,
            "calibrated_at_5pct": bool(abs(z) < 1.96),
            "brier_reliability": brier_reliability(y, P[model]),
            "ece_10_equal_width": expected_calibration_error(y, P[model]),
            "ece_10_equal_frequency": ece_equal_frequency(y, P[model]),
        })
    calibration = pd.DataFrame(rows)
    calibration.to_csv(ARTIFACT_DIR / "calibration_tests.csv", index=False)

    sensitivity, winners = bin_sensitivity(y, P)
    sensitivity.to_csv(ARTIFACT_DIR / "calibration_bin_sensitivity.csv", index=False)

    # --- selected-set overlap at the deployed rule -----------------------
    selected = {m: capacity_selection(P[m], CAPACITY) for m in MODELS}
    overlap_rows = []
    for a, b in PAIRS:
        both = selected[a] & selected[b]
        either = selected[a] | selected[b]
        overlap_rows.append({
            "model_a": a, "model_b": b,
            "selected_a": int(selected[a].sum()), "selected_b": int(selected[b].sum()),
            "in_both": int(both.sum()),
            "jaccard": float(both.sum() / either.sum()),
            "chosen_by_only_one": int((selected[a] ^ selected[b]).sum()),
            "share_of_cohort_differing": float((selected[a] ^ selected[b]).mean()),
        })
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(ARTIFACT_DIR / "selected_set_overlap.csv", index=False)

    # --- paired bootstrap ------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y)
    metrics = ["abs_slope_error", "brier_reliability", "ece_equal_frequency",
               "captured_events", "abs_gender_fnr_gap"]
    draws = {pair: {metric: [] for metric in metrics} for pair in PAIRS}

    for _ in range(REPEATS):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        yv, gv = y[idx], gender[idx]
        if not ((gv == "F") & (yv == 1)).any() or not ((gv == "M") & (yv == 1)).any():
            continue
        values = {}
        for model in MODELS:
            pv = P[model][idx]
            sv = capacity_selection(pv, CAPACITY)
            values[model] = {
                "abs_slope_error": abs(cox_calibration(yv, pv)["slope"] - 1),
                "brier_reliability": brier_reliability(yv, pv),
                "ece_equal_frequency": ece_equal_frequency(yv, pv),
                "captured_events": float(yv[sv].sum()),
                "abs_gender_fnr_gap": abs(fnr_gap(yv, sv, gv, "M", "F")),
            }
        for a, b in PAIRS:
            for metric in metrics:
                draws[(a, b)][metric].append(values[a][metric] - values[b][metric])

    comparison_rows = []
    for (a, b), by_metric in draws.items():
        for metric, values in by_metric.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            low, high = np.quantile(arr, [0.025, 0.975])
            comparison_rows.append({
                "model_a": a, "model_b": b, "metric": metric,
                "difference_a_minus_b": float(arr.mean()),
                "ci_low": float(low), "ci_high": float(high),
                "significant": bool(low > 0 or high < 0),
                "repeats": int(arr.size),
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(ARTIFACT_DIR / "calibration_paired_tests.csv", index=False)

    # --- report ----------------------------------------------------------
    pd.set_option("display.width", 200)
    print("Bin-free calibration (perfect = intercept 0, slope 1):")
    print(calibration[["model", "intercept", "slope", "slope_differs_from_1",
                       "spiegelhalter_z", "calibrated_at_5pct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nECE 'winner' by binning choice (it is not stable):")
    print(winners.to_string(index=False))
    print("\nSelected-set overlap at the deployed top-20% rule:")
    print(overlap[["model_a", "model_b", "in_both", "jaccard", "chosen_by_only_one"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPaired differences (negative favours model A for gaps and errors):")
    print(comparison[["model_a", "model_b", "metric", "difference_a_minus_b",
                      "ci_low", "ci_high", "significant"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    main()
