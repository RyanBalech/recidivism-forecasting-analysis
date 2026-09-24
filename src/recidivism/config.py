"""Central configuration for paths, feature eligibility, and reproducibility.

Start here when you want to understand which raw columns enter the models.
Keeping this choice in one file prevents the notebook, training script, and app
from silently using different definitions of the prediction problem.
"""

from pathlib import Path

# ``parents[2]`` moves from src/recidivism/config.py to the project root.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "nij-challenge2021_full_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
FIGURE_DIR = ARTIFACT_DIR / "figures"
REPORT_DIR = ROOT / "reports"

TARGET = "Recidivism_Within_3years"
SPLIT_COLUMN = "Training_Sample"
ID_COLUMN = "ID"
# These columns never enter a model. We keep them separately to measure whether
# error rates or service allocation differ across demographic groups.
PROTECTED_COLUMNS = ["Gender", "Race"]

# Only fields available when supervision starts are eligible. Dynamic supervision
# variables are deliberately excluded because they accrue after the prediction time.
BASELINE_COLUMNS = [
    "Gender", "Race", "Age_at_Release", "Residence_PUMA", "Gang_Affiliated",
    "Supervision_Risk_Score_First", "Supervision_Level_First", "Education_Level",
    "Dependents", "Prison_Offense", "Prison_Years",
    "Prior_Arrest_Episodes_Felony", "Prior_Arrest_Episodes_Misd",
    "Prior_Arrest_Episodes_Violent", "Prior_Arrest_Episodes_Property",
    "Prior_Arrest_Episodes_Drug", "_v1", "Prior_Arrest_Episodes_DVCharges",
    "Prior_Arrest_Episodes_GunCharges", "Prior_Conviction_Episodes_Felony",
    "Prior_Conviction_Episodes_Misd", "Prior_Conviction_Episodes_Viol",
    "Prior_Conviction_Episodes_Prop", "Prior_Conviction_Episodes_Drug",
    "_v2", "_v3", "_v4", "Prior_Revocations_Parole",
    "Prior_Revocations_Probation", "Condition_MH_SA", "Condition_Cog_Ed",
    "Condition_Other",
]

# The downloaded CSV uses four shortened names. The official NIJ codebook says:
# _v1 = prior arrest with a probation/parole-violation charge
# _v2 = prior conviction with a probation/parole-violation charge
# _v3 = prior conviction with a domestic-violence charge
# _v4 = prior conviction with a gun charge

# Protected attributes and geography are retained for audit but excluded from scoring.
# Residence PUMA is excluded because it can be a strong proxy for race.
EXCLUDED_FROM_MODEL = {"Gender", "Race", "Residence_PUMA"}
FEATURE_COLUMNS = [c for c in BASELINE_COLUMNS if c not in EXCLUDED_FROM_MODEL]

# Human-readable names for every figure, table, and app screen. Raw NIJ names
# (especially _v1 to _v4) are unreadable to a client or a jury.
FEATURE_LABELS = {
    "Age_at_Release": "Age at release",
    "Gang_Affiliated": "Gang affiliated",
    "Supervision_Risk_Score_First": "Georgia risk score (1-10)",
    "Supervision_Level_First": "Initial supervision level",
    "Education_Level": "Education level",
    "Dependents": "Dependents",
    "Prison_Offense": "Prison offense type",
    "Prison_Years": "Years in prison",
    "Prior_Arrest_Episodes_Felony": "Prior felony arrests",
    "Prior_Arrest_Episodes_Misd": "Prior misdemeanor arrests",
    "Prior_Arrest_Episodes_Violent": "Prior violent arrests",
    "Prior_Arrest_Episodes_Property": "Prior property arrests",
    "Prior_Arrest_Episodes_Drug": "Prior drug arrests",
    "_v1": "Prior parole/probation-violation arrests",
    "Prior_Arrest_Episodes_DVCharges": "Prior domestic-violence arrests",
    "Prior_Arrest_Episodes_GunCharges": "Prior gun-charge arrests",
    "Prior_Conviction_Episodes_Felony": "Prior felony convictions",
    "Prior_Conviction_Episodes_Misd": "Prior misdemeanor convictions",
    "Prior_Conviction_Episodes_Viol": "Prior violent convictions",
    "Prior_Conviction_Episodes_Prop": "Prior property convictions",
    "Prior_Conviction_Episodes_Drug": "Prior drug convictions",
    "_v2": "Prior parole/probation-violation convictions",
    "_v3": "Prior domestic-violence convictions",
    "_v4": "Prior gun-charge convictions",
    "Prior_Revocations_Parole": "Prior parole revocations",
    "Prior_Revocations_Probation": "Prior probation revocations",
    "Condition_MH_SA": "Condition: mental health / substance abuse",
    "Condition_Cog_Ed": "Condition: cognitive / education",
    "Condition_Other": "Condition: other",
}


def pretty(name: str) -> str:
    """Readable label for a raw feature, a transformed column, or free text.

    Handles one-hot columns ("Prison_Offense_Drug" -> "Prison offense type = Drug"),
    missing indicators, and text that merely contains feature names (LIME rules).
    """
    name = str(name)
    if name.startswith("missingindicator_"):
        return "Missing: " + pretty(name.removeprefix("missingindicator_"))
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for raw in sorted(FEATURE_LABELS, key=len, reverse=True):
        if name.startswith(raw + "_"):
            return f"{FEATURE_LABELS[raw]} = {name[len(raw) + 1:]}"
    for raw in sorted(FEATURE_LABELS, key=len, reverse=True):
        name = name.replace(raw, FEATURE_LABELS[raw])
    return name


# A fixed seed makes sampling, model fitting, and audits repeatable.
RANDOM_SEED = 42
# Threshold metrics such as TPR and FPR need probabilities converted to 0/1.
DECISION_THRESHOLD = 0.50
