"""Metrics for predictive quality, calibration, uncertainty, fairness, and value."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def validated_predictions(y, p):
    """Reject malformed labels/probabilities instead of silently clipping them."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return y.astype(int), p


def capacity_selection(scores, capacity=0.20):
    """Select exactly round(n * capacity); break ties by stable input row order.

    This deterministic tie policy is reproducible, not a fairness guarantee.
    Scores may be uncalibrated ranks, such as the incumbent 1–10 score.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty finite vector")
    if not np.isfinite(capacity) or not 0 <= capacity <= 1:
        raise ValueError("Capacity must be within [0, 1]")
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(-scores, kind="stable")[:round(len(scores) * capacity)]] = True
    return selected


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Weighted average gap between predicted risk and observed event rate.

    A value of zero would mean that, within every probability bucket, a group
    predicted at (for example) 30% risk actually recidivates about 30% of the time.
    """
    # Cut the [0, 1] probability range into equal-width buckets.
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    result = 0.0
    for i in range(bins):
        mask = bucket == i
        if mask.any():
            # Larger buckets receive more weight than buckets with few people.
            result += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(result)


def classification_metrics(y: Iterable[int], p: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    """Calculate probability metrics and threshold-dependent decision metrics."""
    y_arr, p_arr = validated_predictions(y, p)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be within [0, 1]")
    # Only the metrics below selection_rate depend on this 0.5 conversion.
    pred = (p_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_arr, p_arr)) if len(np.unique(y_arr)) == 2 else np.nan,
        "average_precision": float(average_precision_score(y_arr, p_arr)) if y_arr.sum() else np.nan,
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "log_loss": float(log_loss(y_arr, p_arr, labels=[0, 1])),
        "ece_10": expected_calibration_error(y_arr, p_arr),
        "selection_rate": float(pred.mean()),
        "tpr": float(tp / (tp + fn)) if tp + fn else np.nan,
        "fpr": float(fp / (fp + tn)) if fp + tn else np.nan,
        "precision": float(tp / (tp + fp)) if tp + fp else np.nan,
        "accuracy": float((pred == y_arr).mean()),
    }


def bootstrap_intervals(y: Iterable[int], p: Iterable[float], repeats: int = 500, seed: int = 42) -> dict[str, dict[str, float]]:
    """Estimate 95% sampling intervals by repeatedly resampling test rows.

    Predictions stay fixed; the resampling asks how much reported performance
    could move if we observed another cohort drawn from a similar population.
    """
    y_arr, p_arr = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    names = ["roc_auc", "average_precision", "brier"]
    draws = {name: [] for name in names}
    for _ in range(repeats):
        # Sampling with replacement is what makes this a bootstrap sample.
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if np.unique(y_arr[idx]).size < 2:
            continue
        values = classification_metrics(y_arr[idx], p_arr[idx])
        for name in names:
            draws[name].append(values[name])
    return {
        name: {
            "low": float(np.quantile(vals, 0.025)),
            "high": float(np.quantile(vals, 0.975)),
        }
        for name, vals in draws.items()
    }


def fairness_table(
    y: Iterable[int], p: Iterable[float], groups: Iterable[str], attribute: str, threshold: float = 0.5
) -> pd.DataFrame:
    """Calculate the same metrics separately for each demographic group."""
    y, p = validated_predictions(y, p)
    group = pd.Series(np.asarray(groups)).fillna("Missing").astype(str)
    if len(group) != len(y):
        raise ValueError("Groups must align with labels and probabilities")
    frame = pd.DataFrame({"y": y, "p": p, "group": group})
    rows = []
    for group, part in frame.groupby("group", dropna=False):
        row = classification_metrics(part.y, part.p, threshold)
        row.update({"attribute": attribute, "group": group, "n": len(part), "base_rate": float(part.y.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def fairness_gaps(table: pd.DataFrame) -> dict[str, float]:
    """Summarize disparity as the largest group value minus the smallest.

    Gaps are descriptive audit signals. A smaller gap does not, by itself,
    establish that a model or policy is fair.
    """
    return {
        "demographic_parity_gap": float(table.selection_rate.max() - table.selection_rate.min()),
        "equal_opportunity_gap": float(table.tpr.max() - table.tpr.min()),
        "fpr_gap": float(table.fpr.max() - table.fpr.min()),
        "brier_gap": float(table.brier.max() - table.brier.min()),
    }


def economic_value(
    y: Iterable[int], p: Iterable[float], capacity: float = 0.20,
    intervention_cost: float = 5_000, event_cost: float = 50_000, effectiveness: float = 0.20,
) -> dict[str, float]:
    """Rank people under a service-capacity and cost/benefit scenario.

    ``effectiveness`` is supplied by the user; the observational NIJ data does
    not estimate it. Consequently, the dollar output is a scenario rather than
    a claim that deployment will cause savings.
    """
    y_arr, p_arr = np.asarray(y), np.asarray(p, dtype=float)
    if y_arr.ndim != 1 or len(y_arr) != len(p_arr) or not np.isin(y_arr, [0, 1]).all():
        raise ValueError("Labels must be binary and aligned with scores")
    if not all(np.isfinite(v) and v >= 0 for v in [intervention_cost, event_cost]):
        raise ValueError("Costs must be finite and nonnegative")
    if not np.isfinite(effectiveness) or not 0 <= effectiveness <= 1:
        raise ValueError("Effectiveness must be within [0, 1]")
    # Offer support to the highest predicted risks until capacity is exhausted.
    selected = capacity_selection(p_arr, capacity)
    n_selected = int(selected.sum())
    positives = int(y_arr[selected].sum())
    gross = positives * event_cost * effectiveness
    spend = n_selected * intervention_cost
    return {
        "capacity": capacity,
        "selected": n_selected,
        "captured_events": positives,
        "recall_at_capacity": float(positives / max(1, y_arr.sum())),
        "precision_at_capacity": float(positives / n_selected) if n_selected else np.nan,
        "assumed_gross_benefit": float(gross),
        "assumed_program_cost": float(spend),
        "assumed_net_value": float(gross - spend),
    }


def calibration_points(y: Iterable[int], p: Iterable[float], bins: int = 10) -> pd.DataFrame:
    """Return points used to draw the observed-versus-predicted calibration plot."""
    observed, predicted = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"predicted": predicted, "observed": observed})
