"""Stability at the level of one person, and what to do about the unstable ones.

`stability_structural.py` answers "does the model stay the same when the training
data wiggles?" and reports cohort summaries: mean |dp| between refits, Jaccard of
the selected sets. Those numbers cannot answer the question a caseworker actually
asks, which is about one person:

    Would *this* person still be offered support if we had drawn a slightly
    different training sample?

The same eight bootstrap resamples are used here (identical rng seed and draw
order as the structural script, so the protocol hashes match), but the per-person
predictions are kept instead of being collapsed into pairwise summaries. For each
person that gives a spread of scores across refits, and a count of how many refits
would have selected them under the deployed top-20% rule.

That count is the useful object. Someone selected by 8/8 refits is a robust
decision; someone selected by 4/8 is a coin flip that happens to have landed one
way in the published run. Abstaining on the contested cases and referring them to
human review is a selective-prediction policy, and it is evaluated here the way any
policy should be: what does it cost in coverage, what does it buy in reliability,
and what does it do to the subgroup gaps.

TabICLv2 is skipped by default because eight refits need the GPU box; pass
--with-tabicl there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, FIGURE_DIR
from recidivism.data import load_official_split
from recidivism.metrics import capacity_selection
from recidivism.modeling import logistic_model, tabicl_frames, xgboost_model

N_REFITS = 8
CAPACITY = 0.20
BOOTSTRAP_SEED = 7   # must match stability_structural.py
MODEL_SEED = 42
PALETTE = {"logistic": "#234E70", "xgboost": "#FB8500", "tabicl": "#7B2CBF"}


def bootstrap_samples(n: int) -> list[np.ndarray]:
    """Reproduce the exact resamples used by the structural stability script."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return [rng.integers(0, n, n) for _ in range(N_REFITS)]


def refit_predictions(name: str, split, samples: list[np.ndarray]) -> np.ndarray:
    """Return an (n_refits x n_test) matrix of held-out probabilities."""
    out = []
    for seed, idx in enumerate(samples):
        X = split.X_train.iloc[idx].reset_index(drop=True)
        y = split.y_train.iloc[idx].reset_index(drop=True)
        if name == "tabicl":
            from recidivism.modeling import tabicl_model
            model = tabicl_model(MODEL_SEED)
            train, test = tabicl_frames(X, split.X_test)
            model.fit(train, y.to_numpy())
            out.append(model.predict_proba(test)[:, 1])
        else:
            build = logistic_model if name == "logistic" else xgboost_model
            model = (build(X) if name == "logistic"
                     else build(X, random_state=MODEL_SEED)).fit(X, y)
            out.append(model.predict_proba(split.X_test)[:, 1])
        print(f"  {name} refit {seed + 1}/{len(samples)}", flush=True)
    return np.vstack(out)


def per_person(matrix: np.ndarray) -> pd.DataFrame:
    """Score spread and decision count across refits, for every evaluation record."""
    selections = np.vstack([capacity_selection(row, CAPACITY) for row in matrix])
    times = selections.sum(axis=0)
    return pd.DataFrame({
        "mean_probability": matrix.mean(axis=0),
        "sd_probability": matrix.std(axis=0, ddof=1),
        "min_probability": matrix.min(axis=0),
        "max_probability": matrix.max(axis=0),
        "probability_range": matrix.max(axis=0) - matrix.min(axis=0),
        "times_selected": times,
        "n_refits": matrix.shape[0],
        # Unanimous either way is a decision we can stand behind; anything else is contested.
        "decision_unanimous": (times == 0) | (times == matrix.shape[0]),
    })


def abstention_curve(person: pd.DataFrame, y: np.ndarray, audit: pd.DataFrame,
                     model: str) -> pd.DataFrame:
    """Abstain on the least unanimous decisions; report what that costs and buys.

    Coverage falls as we abstain on more contested people. The question is whether
    the decisions we keep are better and whether the subgroup gaps narrow, or
    whether abstention simply removes the groups we were already failing.
    """
    n_refits = int(person.n_refits.iloc[0])
    # Distance from unanimity: 0 = unanimous, n/2 = maximal disagreement.
    contested = np.minimum(person.times_selected, n_refits - person.times_selected)
    selected = person.times_selected >= n_refits / 2
    rows = []
    for limit in range(0, n_refits // 2 + 1):
        keep = (contested <= limit).to_numpy()
        if keep.sum() == 0:
            continue
        yk, sk = y[keep], selected.to_numpy()[keep]
        row = {
            "model": model, "max_contested_votes": limit,
            "coverage": float(keep.mean()),
            "n_retained": int(keep.sum()),
            "precision_at_capacity": float(yk[sk].mean()) if sk.any() else np.nan,
            "selection_rate": float(sk.mean()),
        }
        for attribute, (a, b) in {"Gender": ("M", "F"), "Race": ("BLACK", "WHITE")}.items():
            groups = audit[attribute].to_numpy()[keep]
            gaps = {}
            for grp in (a, b):
                mask = (groups == grp) & (yk == 1)
                gaps[grp] = float(1 - sk[mask].mean()) if mask.any() else np.nan
            row[f"fnr_gap_{attribute.lower()}"] = gaps[a] - gaps[b]
            row[f"abstained_share_{b}"] = float(
                ((audit[attribute].to_numpy() == b) & ~keep).sum()
                / max(1, (audit[attribute].to_numpy() == b).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-tabicl", action="store_true", help="needs a CUDA machine")
    args = parser.parse_args()

    split = load_official_split()
    y = split.y_test.to_numpy()
    audit = split.audit_test
    samples = bootstrap_samples(len(split.X_train))
    models = ["logistic", "xgboost"] + (["tabicl"] if args.with_tabicl else [])

    people, curves, summary = [], [], []
    for name in models:
        print(f"{name}: refitting on {N_REFITS} bootstrap resamples")
        matrix = refit_predictions(name, split, samples)
        frame = per_person(matrix)
        frame.insert(0, "model", name)
        frame.insert(1, "ID", audit["ID"].to_numpy())
        frame["actual"] = y
        for attribute in ["Gender", "Race"]:
            frame[attribute] = audit[attribute].to_numpy()
        people.append(frame)
        curves.append(abstention_curve(frame, y, audit, name))

        contested = ~frame.decision_unanimous
        summary.append({
            "model": name,
            "n_refits": N_REFITS,
            "median_probability_range": float(frame.probability_range.median()),
            "p90_probability_range": float(frame.probability_range.quantile(0.9)),
            "share_decision_unanimous": float(frame.decision_unanimous.mean()),
            "share_contested": float(contested.mean()),
            "n_contested": int(contested.sum()),
            # Of the people the published run selects, how many are borderline?
            "contested_share_of_selected": float(
                contested[frame.times_selected >= N_REFITS / 2].mean()),
        })

    people = pd.concat(people, ignore_index=True)
    curves = pd.concat(curves, ignore_index=True)
    summary = pd.DataFrame(summary)
    people.to_csv(ARTIFACT_DIR / "individual_stability.csv", index=False)
    curves.to_csv(ARTIFACT_DIR / "abstention_curve.csv", index=False)
    summary.to_csv(ARTIFACT_DIR / "individual_stability_summary.csv", index=False)

    _figure(people, curves, models)

    pd.set_option("display.width", 200)
    print("\nPer-person decision stability:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nAbstention policy (limit 0 = keep only unanimous decisions):")
    print(curves[["model", "max_contested_votes", "coverage", "precision_at_capacity",
                  "fnr_gap_gender", "fnr_gap_race"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


def _figure(people: pd.DataFrame, curves: pd.DataFrame, models: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for name in models:
        part = people[people.model == name]
        counts = part.times_selected.value_counts().sort_index()
        axes[0].plot(counts.index, counts.values / len(part), marker="o", lw=2,
                     color=PALETTE[name], label=name)
    axes[0].set(title="How often is each person selected across 8 refits?",
                xlabel="refits selecting this person", ylabel="share of cohort")
    axes[0].set_yscale("log")
    axes[0].legend()

    for name in models:
        part = people[people.model == name]
        axes[1].hist(part.probability_range, bins=50, histtype="step", lw=2,
                     color=PALETTE[name], label=name)
    axes[1].set(title="Spread of one person's score across refits",
                xlabel="max − min predicted probability", ylabel="people")
    axes[1].legend()

    for name in models:
        part = curves[curves.model == name]
        axes[2].plot(part.coverage, part.fnr_gap_gender, marker="o", lw=2,
                     color=PALETTE[name], label=f"{name}: gender")
        axes[2].plot(part.coverage, part.fnr_gap_race, marker="s", ls="--", lw=2,
                     color=PALETTE[name], alpha=0.6, label=f"{name}: race")
    axes[2].axhline(0, color="grey", lw=1)
    axes[2].set(title="Abstaining on contested decisions:\ndoes the gap close or just shrink the cohort?",
                xlabel="coverage (share of people still decided)", ylabel="FNR gap")
    axes[2].legend(fontsize=8)

    fig.suptitle("Stability at the level of one person, and a selective-prediction policy", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "individual_stability.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
