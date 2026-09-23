"""Regenerate every artifact and deliverable, in dependency order, with one command.

    python scripts/run_all.py                          # everything (30-60 min on CPU, much less on GPU)
    python scripts/run_all.py --from fairness_audit    # resume from a step after a failure
    python scripts/run_all.py --only fairness_audit fairness_interpretability

Stops at the first failing step and prints how to resume. Works on Windows, macOS and Linux;
no need to set PYTHONPATH.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (step, what it produces). Later steps read the artifacts of earlier ones.
STEPS = [
    ("train_evaluate", "train the 3 models; predictions, metrics, saved models"),
    ("interpretability", "SHAP, LIME, global surrogate, PDP/ICE"),
    ("xper_attribution", "XPER decomposition of AUC"),
    ("stability_structural", "refits on bootstrap resamples"),
    ("fairness_audit", "FNR/selection gaps, CIs, TOST, course tests, age audit, frontier"),
    ("fairness_interpretability", "FPDP, candidate variables, proxy scatter, mitigation"),
    ("incumbent_benchmark", "comparison with Georgia's existing score"),
    ("learning_curve", "performance vs training size"),
    ("race_ab_test", "race as input vs not; counterfactual twins"),
    ("tradeoff_matrix", "3 models x 4 dimensions table"),
    ("improvement_journey", "process figure for JOURNEY.md"),
    ("build_notebook", "executed analysis notebook"),
    ("build_slides", "presentation deck"),
]


def check_environment() -> None:
    """Warn early about the two things that silently break a run."""
    import sklearn

    required = next(line.split("==")[1].split()[0] for line in (ROOT / "requirements.txt").read_text().splitlines()
                    if line.startswith("scikit-learn=="))
    if sklearn.__version__ != required:
        sys.exit(f"scikit-learn {sklearn.__version__} installed, requirements.txt pins {required}. "
                 f"Run: python -m pip install -r requirements.txt")
    try:
        import torch
        device = "CUDA GPU" if torch.cuda.is_available() else "CPU only (TabICL steps will be slow)"
    except ImportError:
        device = "torch not installed"
    print(f"scikit-learn {sklearn.__version__} OK · TabICL device: {device}\n")


def main() -> None:
    names = [s for s, _ in STEPS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=names, help="resume from this step")
    parser.add_argument("--only", nargs="+", choices=names, help="run only these steps")
    args = parser.parse_args()

    steps = STEPS
    if args.start:
        steps = STEPS[names.index(args.start):]
    if args.only:
        steps = [s for s in STEPS if s[0] in args.only]

    check_environment()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    total = time.perf_counter()
    for i, (name, what) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {name}: {what}", flush=True)
        start = time.perf_counter()
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{name}.py")], cwd=ROOT, env=env)
        if result.returncode != 0:
            sys.exit(f"\nFAILED at {name}. Fix it, then resume with:\n"
                     f"    python scripts/run_all.py --from {name}")
        print(f"    done in {time.perf_counter() - start:.0f}s\n", flush=True)
    print(f"All {len(steps)} steps finished in {(time.perf_counter() - total) / 60:.1f} min.")


if __name__ == "__main__":
    main()
