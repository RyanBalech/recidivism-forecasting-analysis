from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "nij-challenge2021_full_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
FIGURE_DIR = ARTIFACT_DIR / "figures"
REPORT_DIR = ROOT / "reports"

TARGET = "Recidivism_Within_3years"
SPLIT_COLUMN = "Training_Sample"
ID_COLUMN = "ID"
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

# Protected attributes and geography are retained for audit but excluded from scoring.
# Residence PUMA is excluded because it can be a strong proxy for race.
EXCLUDED_FROM_MODEL = {"Gender", "Race", "Residence_PUMA"}
FEATURE_COLUMNS = [c for c in BASELINE_COLUMNS if c not in EXCLUDED_FROM_MODEL]

RANDOM_SEED = 42
DECISION_THRESHOLD = 0.50

