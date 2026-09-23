"""Regenerate dependent deliverables in order, stopping on any failed step."""
import subprocess
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    steps = ["train_evaluate", "incumbent_benchmark", "fairness_audit",
             "validate_project", "learning_curve", "interpretability",
             "stability_structural", "race_ab_test", "xper_attribution",
             "tradeoff_matrix", "improvement_journey", "refresh_readme", "build_notebook",
             "build_slides", "build_prevalidation_pdf"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-step", choices=steps, default=steps[0])
    args = parser.parse_args()
    steps = steps[steps.index(args.from_step):]
    for step in steps:
        print(f"Running {step}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / f"{step}.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
