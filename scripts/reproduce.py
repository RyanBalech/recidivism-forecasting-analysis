"""Regenerate dependent deliverables in order, stopping on any failed step.

    python scripts/reproduce.py                                  # everything
    python scripts/reproduce.py --from-step fairness_audit       # resume after a failure
    python scripts/reproduce.py --only fairness_audit fairness_interpretability

Later steps read the artifacts of earlier ones; `--only` assumes those already exist.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = ["leakage_audit", "estimator_sweep", "train_evaluate", "incumbent_benchmark", "fairness_audit",
         "fairness_interpretability", "mitigation_nested", "validate_project", "learning_curve",
         "interpretability", "lime_local_fidelity", "logistic_effects", "explanation_agreement", "calibration_tests",
         "stability_structural", "individual_stability", "proxy_inference_audit",
         "race_ab_test", "xper_attribution",
         "tradeoff_matrix", "improvement_journey", "refresh_readme", "build_notebook",
         "build_standalone_notebook",
         "deck_figures", "build_slides", "build_prevalidation_pdf", "end_to_end_audit"]


def check_environment():
    """Saved .joblib models only load with the scikit-learn version pinned in requirements.txt."""
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
    print(f"scikit-learn {sklearn.__version__} OK · TabICL device: {device}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-step", choices=STEPS, default=STEPS[0])
    parser.add_argument("--only", nargs="+", choices=STEPS, help="run only these steps, in pipeline order")
    args = parser.parse_args()
    steps = [s for s in STEPS if s in args.only] if args.only else STEPS[STEPS.index(args.from_step):]
    check_environment()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for step in steps:
        if step == "build_standalone_notebook":
            # Validate the published artifacts before building the independent
            # notebook. The final audit hashes the rebuilt deliverables.
            subprocess.run([sys.executable, str(ROOT / "scripts/end_to_end_audit.py")],
                           cwd=ROOT, env=env, check=True)
        print(f"Running {step}", flush=True)
        start = time.perf_counter()
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{step}.py")], cwd=ROOT, env=env)
        if result.returncode != 0:
            sys.exit(f"Failed at {step}. Resume with: python scripts/reproduce.py --from-step {step}")
        print(f"  done in {time.perf_counter() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
