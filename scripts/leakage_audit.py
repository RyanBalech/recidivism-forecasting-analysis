"""Systematic leakage audit — reproducible evidence that the pipeline is clean.

Checks four vectors and prints a verdict:
1. Missingness of any model feature aligned with a protected attribute (race or gender).
   This is how the Gang_Affiliated -> TabICL gender leak entered; we assert it is now handled.
2. No model feature comes from after the scoring moment (all are baseline fields).
3. Train and test IDs are disjoint (official split, no contamination).
4. TabICL's mode-fill removes the NaN category that carried the leak.

Exits non-zero if any check fails, so it can run in CI / as a test.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import BASELINE_COLUMNS, EXCLUDED_FROM_MODEL, FEATURE_COLUMNS
from recidivism.data import load_official_split
from recidivism.modeling import tabicl_frames

ALIGN_THRESHOLD = 0.15  # |P(group | feature missing) - P(group)| above this = suspicious


def main() -> None:
    split = load_official_split()
    X, race, gender = split.X_train, split.audit_train["Race"], split.audit_train["Gender"]
    failures = []

    print("1. Missingness vs protected attributes")
    base_f, base_b = gender.eq("F").mean(), race.eq("BLACK").mean()
    for col in X.columns:
        miss = X[col].isna()
        if not miss.any():
            continue
        dg, dr = abs(gender[miss].eq("F").mean() - base_f), abs(race[miss].eq("BLACK").mean() - base_b)
        suspicious = dg > ALIGN_THRESHOLD or dr > ALIGN_THRESHOLD
        flag = "  <== aligned; must be neutralized before TabICL" if suspicious else ""
        print(f"   {col}: n_missing={int(miss.sum())} F_delta={dg:.2f} Black_delta={dr:.2f}{flag}")
        # A suspicious column is only OK if tabicl_frames removes its NaN (the leak vector).
        if suspicious:
            filled_train, filled_test = tabicl_frames(split.X_train, split.X_test)
            if filled_train[col].isna().any() or filled_test[col].isna().any():
                failures.append(f"{col}: protected-aligned missingness still reaches TabICL as NaN")

    print("2. No post-scoring features")
    post = [c for c in FEATURE_COLUMNS if c not in BASELINE_COLUMNS]
    if post:
        failures.append(f"features not in baseline set: {post}")
    print(f"   features outside the supervision-start baseline: {post or 'none'}")
    print(f"   protected/proxy excluded from inputs: {sorted(EXCLUDED_FROM_MODEL)}")

    print("3. Split integrity")
    overlap = set(split.audit_train.ID) & set(split.audit_test.ID)
    if overlap:
        failures.append(f"{len(overlap)} IDs in both train and test")
    print(f"   train/test ID overlap: {len(overlap)} (must be 0)")

    print("4. TabICL NaN-category removal")
    ft, fe = tabicl_frames(split.X_train, split.X_test)
    cat = split.X_train.select_dtypes(exclude="number").columns
    leftover = [c for c in cat if ft[c].isna().any() or fe[c].isna().any()]
    if leftover:
        failures.append(f"categorical NaN still reaches TabICL: {leftover}")
    print(f"   categorical columns still NaN into TabICL: {leftover or 'none'}")

    print()
    if failures:
        print("LEAKAGE AUDIT FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("LEAKAGE AUDIT PASSED: no protected-attribute leak, no post-scoring feature, clean split.")


if __name__ == "__main__":
    main()
