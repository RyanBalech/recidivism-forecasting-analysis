"""Check published artifacts and optionally benchmark CPU/CUDA on training folds.

No model is promoted by this audit. --refit-dir checks an isolated all-model run.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recidivism.config import ARTIFACT_DIR
from recidivism.data import load_official_split
from recidivism.metrics import classification_metrics, economic_value
from recidivism.modeling import TABICL_CHECKPOINT, xgboost_model
from deep_leakage_audit import audit as leakage_audit


def verify_metrics(pred, metrics):
    """Recompute every saved metric, excluding runtime, from aligned predictions."""
    if metrics.model.duplicated().any() or set(metrics.model) != {"logistic", "xgboost", "tabicl"}:
        raise ValueError("Expected exactly one metrics row for each of the three models")
    for row in metrics.to_dict("records"):
        p = pred[f"p_{row['model']}"]
        expected = classification_metrics(pred.actual, p)
        expected.update({f"economic_{k}": v for k, v in economic_value(pred.actual, p).items()})
        for key, value in expected.items():
            # Float32 probabilities round-trip through CSV as float64 decimals.
            if key not in row or not np.isclose(row[key], value, atol=1e-7, rtol=1e-10, equal_nan=True):
                raise ValueError(f"Stale metric: {row['model']} / {key}")


def holm_adjust(p):
    """Holm family-wise correction, valid without independence of the tests."""
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def gpu_benchmark(split, out):
    """Same fixed configuration and paired training folds; actual device is checked."""
    if not torch.cuda.is_available():
        return {"status": "unavailable", "reason": "PyTorch cannot access CUDA"}
    rows = []
    folds = StratifiedKFold(3, shuffle=True, random_state=20260924)
    for fold, (it, iv) in enumerate(folds.split(split.X_train, split.y_train), 1):
        a, b = split.X_train.iloc[it], split.X_train.iloc[iv]
        for device in ["cpu", "cuda"]:
            start = time.perf_counter()
            model = xgboost_model(a, n_jobs=4, device=device).fit(a, split.y_train.iloc[it])
            p = model.predict_proba(b)[:, 1]
            elapsed = time.perf_counter() - start
            config = json.loads(model.named_steps["model"].get_booster().save_config())
            actual = config["learner"]["generic_param"]["device"]
            if not actual.startswith(device):
                raise RuntimeError(f"Requested {device}, XGBoost actually used {actual}")
            rows.append({"fold": fold, "requested_device": device, "actual_device": actual,
                         "seconds": elapsed, **classification_metrics(split.y_train.iloc[iv], p)})
            print(f"XGBoost fold {fold}: actual device {actual}, {elapsed:.2f}s", flush=True)
    pd.DataFrame(rows).to_csv(out / "xgboost_cpu_gpu.csv", index=False)
    return {"status": "measured", "scope": "Paired training-only folds, fixed parameters, four CPU threads",
            "means": pd.DataFrame(rows).groupby("requested_device")[["roc_auc", "brier", "seconds"]].mean().to_dict("index")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-check", action="store_true")
    parser.add_argument("--refit-dir", type=Path)
    args = parser.parse_args()
    out = ARTIFACT_DIR / "end_to_end"
    out.mkdir(exist_ok=True)
    audit_path = out / "audit.json"
    previous = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    split = load_official_split()
    report = {"python": platform.python_version(), "leakage": leakage_audit(),
              "gpu_available_now": torch.cuda.is_available(),
              "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    pred = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    pd.testing.assert_frame_equal(pred[split.audit_test.columns], split.audit_test, check_dtype=False)
    np.testing.assert_array_equal(pred.actual, split.y_test)
    verify_metrics(pred, pd.read_csv(ARTIFACT_DIR / "model_metrics.csv"))
    report["saved_model_max_error"] = {}
    for name in ["logistic", "xgboost"]:
        model = joblib.load(ARTIFACT_DIR / "models" / f"{name}.joblib")
        p = model.predict_proba(split.X_test)[:, 1]
        error = float(np.max(np.abs(p - pred[f"p_{name}"])))
        if error > 1e-6:
            raise ValueError(f"Saved {name} disagrees with published predictions: {error}")
        report["saved_model_max_error"][name] = error
    report["published_tabicl_device"] = json.loads((ARTIFACT_DIR / "run_manifest.json").read_text())["tabicl_device"]
    from huggingface_hub import hf_hub_download
    checkpoint = Path(hf_hub_download("jingang/TabICL", TABICL_CHECKPOINT, local_files_only=True))
    report["cached_checkpoint"] = {"name": TABICL_CHECKPOINT,
                                   "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    # Independently recompute the merged fairness gaps at both operating rules.
    from fairness_audit import ATTRIBUTES, gaps, with_audit_columns
    audited = with_audit_columns(pred)
    inf = pd.read_csv(ARTIFACT_DIR / "fairness_inference.csv")
    for row in inf.itertuples():
        a, b = ATTRIBUTES[row.attribute]
        actual = gaps(audited.actual.to_numpy(), audited[f"p_{row.model}"].to_numpy(),
                      audited[row.attribute].to_numpy(), a, b, row.rule)[row.metric]
        if not np.isclose(actual, row.gap, atol=1e-10):
            raise ValueError(f"Stale fairness gap: {row.model}/{row.attribute}/{row.metric}")
    report["fairness_gaps_recomputed"] = len(inf)
    course = pd.read_csv(ARTIFACT_DIR / "fairness_tests.csv")
    adjusted = course.melt(id_vars=["model", "attribute", "protected_group"],
                           var_name="test", value_name="p_raw")
    adjusted["p_holm"] = holm_adjust(adjusted.p_raw)
    adjusted["reject_holm_005"] = adjusted.p_holm < 0.05
    adjusted.to_csv(out / "fairness_tests_holm.csv", index=False)
    report["fairness_multiplicity"] = {"family": "All 54 course difference tests across models and attributes",
        "raw_rejections": int((adjusted.p_raw < .05).sum()),
        "holm_rejections": int(adjusted.reject_holm_005.sum()),
        "scope": "Difference tests only; does not adjust exploratory TOST, FPDP selection or historical development"}
    # Record why the installed kernel approximation does not guarantee efficiency.
    from XPER.compute import EM
    state = random.getstate()
    random.seed(42)
    coalitions = EM.sampled_coalitions(60, len(split.X_train.columns))
    random.setstate(state)
    report["xper_kernel_inspection"] = {
        "sampled_coalition_sizes": sorted(set(map(len, coalitions))),
        "empty_coalition_present": any(len(c) == 0 for c in coalitions),
        "full_coalition_present": any(len(c) == len(split.X_train.columns) for c in coalitions),
        "kernel_source_sha256": hashlib.sha256(inspect.getsource(EM.XPER_choice).encode()).hexdigest(),
        "finding": "Installed kernel branch fits unconstrained statsmodels WLS. No endpoint constraints force contributions to sum to observed AUC; residuals must remain visible.",
    }
    report["training_oof_checks"] = {}
    for filename in ["accuracy_oof.csv", "native_oof.csv"]:
        oof = pd.read_csv(ARTIFACT_DIR / "deep_review" / filename)
        if oof.ID.duplicated().any() or set(oof.ID) != set(split.audit_train.ID):
            raise ValueError(f"OOF rows do not cover training exactly once: {filename}")
        expected = pd.Series(split.y_train.to_numpy(), index=split.audit_train.ID)
        np.testing.assert_array_equal(oof.actual, expected.loc[oof.ID])
        if set(oof.ID) & set(pred.ID):
            raise ValueError("Official evaluation rows entered the training-fold experiment")
        report["training_oof_checks"][filename] = {"rows": len(oof), "folds": int(oof.fold.nunique())}
    if args.refit_dir:
        fresh = pd.read_csv(args.refit_dir / "test_predictions.csv")
        identifiers = [*split.audit_test.columns, "actual"]
        pd.testing.assert_frame_equal(fresh[identifiers], pred[identifiers], check_dtype=False)
        verify_metrics(fresh, pd.read_csv(args.refit_dir / "model_metrics.csv"))
        report["isolated_refit_manifest"] = json.loads((args.refit_dir / "run_manifest.json").read_text())
        report["isolated_refit_max_probability_difference"] = {
            name: float(np.max(np.abs(fresh[f"p_{name}"] - pred[f"p_{name}"])))
            for name in ["logistic", "xgboost", "tabicl"]}
    else:
        for key in ["isolated_refit_manifest", "isolated_refit_max_probability_difference"]:
            if key in previous:
                report[key] = previous[key]
    if args.gpu_check:
        report["xgboost_gpu_benchmark"] = gpu_benchmark(split, out)
    elif "xgboost_gpu_benchmark" in previous:
        report["xgboost_gpu_benchmark"] = previous["xgboost_gpu_benchmark"]
    pairs = pd.read_csv(ARTIFACT_DIR / "stability_pairs.csv")
    stability = pd.read_csv(ARTIFACT_DIR / "stability_summary.csv").set_index("model")
    for name, part in pairs.groupby("model"):
        if len(part) != 28 or part.duplicated(["refit_i", "refit_j"]).any():
            raise ValueError(f"Incomplete paired stability experiment: {name}")
        for metric in ["mean_abs_prob_diff", "rank_correlation", "top20_jaccard"]:
            if not np.isclose(part[metric].mean(), stability.loc[name, metric], atol=5.1e-5):
                raise ValueError(f"Stale stability summary: {name}/{metric}")
    submission_notebook = ROOT / "Recidivism_Project_Submission.ipynb"
    notebook_path = ROOT / "notebooks/recidivism_analysis.ipynb"
    slides_path = ROOT / "reports/ISAF_Recidivism_Presentation.pptx"
    for path in [submission_notebook, notebook_path, slides_path]:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing deliverable: {path}")
    if submission_notebook.read_bytes() != notebook_path.read_bytes():
        raise ValueError("Root submission notebook and canonical notebook are out of sync")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    if any(c["execution_count"] is None or any(o["output_type"] == "error" for o in c["outputs"]) for c in code_cells):
        raise ValueError("Notebook contains unexecuted cells or execution errors")
    from pptx import Presentation
    report["deliverables"] = {"submission_notebook": submission_notebook.name,
                              "executed_code_cells": len(code_cells),
                              "slides": len(Presentation(slides_path).slides)}
    report["checks_passed"] = ["original release and preprocessing boundaries", "prediction ID/label alignment",
                               "all published performance/economic metrics", "saved conventional model predictions",
                               "merged fairness point estimates", "paired stability summaries", "executed notebook and readable slide deck"]
    inputs = [ARTIFACT_DIR / name for name in ["test_predictions.csv", "model_metrics.csv", "fairness_inference.csv"]]
    inputs += list((ARTIFACT_DIR / "models").glob("*.joblib"))
    inputs += [submission_notebook, notebook_path, slides_path,
               ARTIFACT_DIR / "stability_pairs.csv", ARTIFACT_DIR / "stability_summary.csv"]
    report["input_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for folder in [ROOT / "scripts", ROOT / "src/recidivism"]
                               for p in sorted(folder.glob("*.py"))}
    audit_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Passed: " + "; ".join(report["checks_passed"]), flush=True)
    print("Full evidence: artifacts/end_to_end/audit.json", flush=True)


if __name__ == "__main__":
    main()
