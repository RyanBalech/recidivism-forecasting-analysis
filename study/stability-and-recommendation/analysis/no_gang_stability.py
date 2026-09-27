"""Slides 17–18 for the no-gang pilot arm, under the published stability protocol.

The team's `scripts/gang_variant_eval.py` (merged 27 Sep) re-checks the models without
Gang_Affiliated on all four dimensions. Two gaps matter for the stability section:

1. Its XGBoost refits pass `random_state=seed` (0–7), so data *and* seed vary between
   refits. `stability_structural.py`, the source of slide 17, fixes the model seed at 42
   so that only the data varies. This script recomputes slide 17 for both variants with
   the published protocol: the same eight resamples (seed 7) and model seed 42.
2. The per-person stability and abstention result on slide 18 was not recomputed for the
   no-gang model (reports/presentation_outline.md says so). This script does that too,
   reusing `per_person()` and `abstention_curve()` from `scripts/individual_stability.py`
   unchanged.

As a check, the with-gang logistic rows must reproduce the published artifacts exactly.
CPU only, about 1–2 minutes. Writes only to ../results/.

    python study/stability-and-recommendation/analysis/no_gang_stability.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from individual_stability import abstention_curve, bootstrap_samples, per_person  # noqa: E402
from recidivism.config import ARTIFACT_DIR, FEATURE_COLUMNS  # noqa: E402
from recidivism.data import load_official_split  # noqa: E402
from recidivism.metrics import capacity_selection  # noqa: E402
from recidivism.modeling import logistic_model, xgboost_model  # noqa: E402

CAPACITY = 0.20
MODEL_SEED = 42
VARIANTS = {"with_gang": FEATURE_COLUMNS, "no_gang": [c for c in FEATURE_COLUMNS if c != "Gang_Affiliated"]}


def refit_matrix(model: str, cols: list[str], split, samples) -> np.ndarray:
    rows = []
    for idx in samples:
        X = split.X_train.iloc[idx][cols].reset_index(drop=True)
        y = split.y_train.iloc[idx].reset_index(drop=True)
        pipe = logistic_model(X) if model == "logistic" else xgboost_model(X, random_state=MODEL_SEED)
        rows.append(pipe.fit(X, y).predict_proba(split.X_test[cols])[:, 1])
    return np.vstack(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = load_official_split()
    y = split.y_test.to_numpy()
    audit = split.audit_test
    samples = bootstrap_samples(len(split.X_train))

    summary, curves, pair_rows, gender_rows = [], [], [], []
    for variant, cols in VARIANTS.items():
        for model in ["logistic", "xgboost"]:
            matrix = refit_matrix(model, cols, split, samples)
            for i, j in itertools.combinations(range(len(samples)), 2):
                a, b = capacity_selection(matrix[i], CAPACITY), capacity_selection(matrix[j], CAPACITY)
                pair_rows.append({"variant": variant, "model": model, "refit_i": i, "refit_j": j,
                                  "mean_abs_prob_diff": float(np.mean(np.abs(matrix[i] - matrix[j]))),
                                  "top20_jaccard": float((a & b).sum() / (a | b).sum())})
            person = per_person(matrix)
            person["actual"] = y
            person["Gender"] = audit["Gender"].to_numpy()
            contested = ~person.decision_unanimous
            selected = person.times_selected >= len(samples) / 2
            curve = abstention_curve(person, y, audit, model)
            curve.insert(0, "variant", variant)
            curves.append(curve)
            full = curve.loc[curve.max_contested_votes.idxmax()]
            unanimous = curve.loc[curve.max_contested_votes == 0].iloc[0]
            summary.append({"variant": variant, "model": model,
                            "share_contested": float(contested.mean()),
                            "contested_share_of_selected": float(contested[selected].mean()),
                            "coverage_unanimous": float(unanimous.coverage),
                            "precision_all": float(full.precision_at_capacity),
                            "precision_unanimous": float(unanimous.precision_at_capacity),
                            "gender_fnr_gap_all": float(full.fnr_gap_gender),
                            "gender_fnr_gap_unanimous": float(unanimous.fnr_gap_gender),
                            "race_fnr_gap_all": float(full.fnr_gap_race),
                            "race_fnr_gap_unanimous": float(unanimous.fnr_gap_race)})
            for g in ("F", "M"):
                mask = (person.Gender == g) & selected
                gender_rows.append({"variant": variant, "model": model, "gender": g,
                                    "n_selected_majority": int(mask.sum()),
                                    "contested_share_of_selected": float(contested[mask].mean()),
                                    "contested_share_all": float(contested[person.Gender == g].mean())})
            print(f"{variant:9s} {model:8s} done", flush=True)

    pairs = pd.DataFrame(pair_rows)
    structural = (pairs.groupby(["variant", "model"])[["mean_abs_prob_diff", "top20_jaccard"]].mean()
                  .reset_index())
    wins = []
    for variant, part in pairs.groupby("variant"):
        lg = part[part.model == "logistic"].set_index(["refit_i", "refit_j"])
        xg = part[part.model == "xgboost"].set_index(["refit_i", "refit_j"])
        wins.append({"variant": variant, "n_pairs": len(lg),
                     "logistic_lower_drift": int((lg.mean_abs_prob_diff < xg.mean_abs_prob_diff).sum()),
                     "logistic_higher_jaccard": int((lg.top20_jaccard > xg.top20_jaccard).sum())})
    summary = pd.DataFrame(summary).merge(structural, on=["variant", "model"])

    # Sanity check: the with-gang logistic rows must reproduce the published artifacts.
    published = pd.read_csv(ARTIFACT_DIR / "stability_summary.csv").set_index("model")
    published_person = pd.read_csv(ARTIFACT_DIR / "individual_stability_summary.csv").set_index("model")
    row = summary[(summary.variant == "with_gang") & (summary.model == "logistic")].iloc[0]
    checks = {
        "drift": abs(row.mean_abs_prob_diff - published.loc["logistic", "mean_abs_prob_diff"]) < 5.1e-5,
        "jaccard": abs(row.top20_jaccard - published.loc["logistic", "top20_jaccard"]) < 5.1e-5,
        "contested": abs(row.share_contested - published_person.loc["logistic", "share_contested"]) < 1e-9,
    }
    print("with-gang logistic reproduces the published audit:", checks)
    if not all(checks.values()):
        raise ValueError("Protocol mismatch with the published stability audit")

    summary.to_csv(OUT / "no_gang_stability_summary.csv", index=False)
    pd.DataFrame(wins).to_csv(OUT / "no_gang_stability_pair_wins.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "no_gang_abstention_curve.csv", index=False)
    pd.DataFrame(gender_rows).to_csv(OUT / "no_gang_contested_by_gender.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    print(pd.DataFrame(wins).to_string(index=False))
    print(pd.DataFrame(gender_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
