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


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    result = 0.0
    for i in range(bins):
        mask = bucket == i
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(result)


def classification_metrics(y: Iterable[int], p: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_arr, p_arr)),
        "average_precision": float(average_precision_score(y_arr, p_arr)),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "log_loss": float(log_loss(y_arr, p_arr)),
        "ece_10": expected_calibration_error(y_arr, p_arr),
        "selection_rate": float(pred.mean()),
        "tpr": float(tp / (tp + fn)) if tp + fn else np.nan,
        "fpr": float(fp / (fp + tn)) if fp + tn else np.nan,
        "precision": float(tp / (tp + fp)) if tp + fp else np.nan,
        "accuracy": float((pred == y_arr).mean()),
    }


def bootstrap_intervals(y: Iterable[int], p: Iterable[float], repeats: int = 500, seed: int = 42) -> dict[str, dict[str, float]]:
    y_arr, p_arr = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    names = ["roc_auc", "average_precision", "brier"]
    draws = {name: [] for name in names}
    for _ in range(repeats):
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
    frame = pd.DataFrame({"y": y, "p": p, "group": pd.Series(groups).fillna("Missing").astype(str)})
    rows = []
    for group, part in frame.groupby("group", dropna=False):
        row = classification_metrics(part.y, part.p, threshold)
        row.update({"attribute": attribute, "group": group, "n": len(part), "base_rate": float(part.y.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def fairness_gaps(table: pd.DataFrame) -> dict[str, float]:
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
    """Scenario analysis; values are assumptions, not causal estimates."""
    y_arr, p_arr = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n_selected = max(1, int(round(len(y_arr) * capacity)))
    selected = np.argsort(-p_arr)[:n_selected]
    positives = int(y_arr[selected].sum())
    gross = positives * event_cost * effectiveness
    spend = n_selected * intervention_cost
    return {
        "capacity": capacity,
        "selected": n_selected,
        "captured_events": positives,
        "recall_at_capacity": float(positives / max(1, y_arr.sum())),
        "precision_at_capacity": float(positives / n_selected),
        "assumed_gross_benefit": float(gross),
        "assumed_program_cost": float(spend),
        "assumed_net_value": float(gross - spend),
    }


def calibration_points(y: Iterable[int], p: Iterable[float], bins: int = 10) -> pd.DataFrame:
    observed, predicted = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"predicted": predicted, "observed": observed})

