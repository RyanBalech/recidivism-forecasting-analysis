from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from recidivism.config import ARTIFACT_DIR, DATA_PATH, FEATURE_COLUMNS
from recidivism.data import load_official_split
from recidivism.metrics import economic_value


st.set_page_config(page_title="Trustworthy Recidivism Model Lab", page_icon="⚖️", layout="wide")


@st.cache_data
def load_results():
    return (
        pd.read_csv(ARTIFACT_DIR / "model_metrics.csv"),
        pd.read_csv(ARTIFACT_DIR / "test_predictions.csv"),
        pd.read_csv(ARTIFACT_DIR / "fairness_by_group.csv"),
        pd.read_csv(ARTIFACT_DIR / "permutation_importance.csv"),
    )


@st.cache_resource
def load_deployable_models():
    return {
        "Logistic regression": joblib.load(ARTIFACT_DIR / "models" / "logistic.joblib"),
        "XGBoost": joblib.load(ARTIFACT_DIR / "models" / "xgboost.joblib"),
    }


st.title("Trustworthy Recidivism Model Lab")
st.caption("Three-year risk at supervision start · NIJ Georgia cohort · Decision support prototype")

required = [ARTIFACT_DIR / "model_metrics.csv", ARTIFACT_DIR / "test_predictions.csv"]
if not all(path.exists() for path in required):
    st.error("Run `python scripts/train_evaluate.py` first to create the audit artifacts.")
    st.stop()

metrics, predictions, fairness, importance = load_results()
labels = {"logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
metrics["Model"] = metrics.model.map(labels)

overview, cohort, individual, governance = st.tabs(["Model comparison", "Cohort & economics", "Individual sandbox", "Governance"])

with overview:
    st.subheader("Held-out performance")
    display = metrics[["Model", "roc_auc", "average_precision", "brier", "ece_10", "fit_predict_seconds"]].copy()
    display.columns = ["Model", "ROC AUC", "Average precision", "Brier ↓", "Calibration error ↓", "Runtime (s)"]
    st.dataframe(display.style.format({c: "{:.3f}" for c in display.columns[1:]}), use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(metrics, x="Model", y="brier", color="Model", title="Probability error (lower is better)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        metric = st.selectbox("Audit metric", ["roc_auc", "brier", "fpr", "tpr", "selection_rate"])
        attribute = st.radio("Protected attribute", ["Race", "Gender"], horizontal=True)
        view = fairness[fairness.attribute.eq(attribute)].copy()
        view["Model"] = view.model.map(labels)
        fig = px.bar(view, x="group", y=metric, color="Model", barmode="group", title=f"{metric.replace('_', ' ').title()} by {attribute.lower()}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("What drives predictions?")
    selected_model = st.selectbox("Model explanation", list(labels), format_func=labels.get)
    imp = importance[importance.model.eq(selected_model)].nlargest(12, "importance").sort_values("importance")
    fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Held-out permutation importance")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Importance is the increase in Brier loss after shuffling a feature. It describes association, not causation.")

with cohort:
    st.subheader("Test a resource-allocation scenario")
    c1, c2, c3, c4 = st.columns(4)
    capacity = c1.slider("Share offered support", 0.05, 0.50, 0.20, 0.05)
    intervention_cost = c2.number_input("Cost per person ($)", 0, 100_000, 5_000, 500)
    event_cost = c3.number_input("Cost per event ($)", 0, 500_000, 50_000, 5_000)
    effectiveness = c4.slider("Assumed effectiveness", 0.0, 1.0, 0.20, 0.05)
    rows = []
    for model in labels:
        result = economic_value(
            predictions.actual, predictions[f"p_{model}"], capacity,
            intervention_cost, event_cost, effectiveness,
        )
        rows.append({"Model": labels[model], **result})
    economics = pd.DataFrame(rows)
    st.dataframe(economics[["Model", "selected", "captured_events", "recall_at_capacity", "precision_at_capacity", "assumed_net_value"]].style.format({
        "recall_at_capacity": "{:.1%}", "precision_at_capacity": "{:.1%}", "assumed_net_value": "${:,.0f}"
    }), use_container_width=True, hide_index=True)
    st.info("This is a transparent scenario calculator. Effectiveness and costs are user assumptions; the observational dataset cannot estimate causal program impact.")

    st.subheader("Inspect held-out people")
    model_filter = st.selectbox("Rank by", list(labels), format_func=labels.get, key="rank")
    group_filter = st.multiselect("Race", sorted(predictions.Race.dropna().unique()), default=sorted(predictions.Race.dropna().unique()))
    view = predictions[predictions.Race.isin(group_filter)].nlargest(100, f"p_{model_filter}")
    shown = ["ID", "Gender", "Race", "actual", "p_logistic", "p_xgboost", "p_tabicl"]
    shown = [c for c in shown if c in view]
    st.dataframe(view[shown], use_container_width=True, hide_index=True)

with individual:
    st.subheader("Counterfactual scoring sandbox")
    st.warning("For classroom demonstration only. A score must never trigger punishment or reduced services, and human review cannot repair an invalid deployment context.")
    split = load_official_split(DATA_PATH)
    models = load_deployable_models()
    model_name = st.selectbox("Scoring model", list(models))
    source_row = st.selectbox("Start from held-out record", range(min(250, len(split.X_test))), format_func=lambda i: f"Record {int(split.audit_test.iloc[i].ID)}")
    row = split.X_test.iloc[[source_row]].copy()
    editable = ["Age_at_Release", "Supervision_Risk_Score_First", "Gang_Affiliated", "Education_Level", "Prison_Years"]
    cols = st.columns(len(editable))
    for col, container in zip(editable, cols):
        options = sorted(split.X_train[col].dropna().unique(), key=lambda x: str(x))
        current = row.iloc[0][col]
        index = options.index(current) if current in options else 0
        row.loc[:, col] = container.selectbox(col.replace("_", " "), options, index=index)
    probability = float(models[model_name].predict_proba(row[FEATURE_COLUMNS])[:, 1][0])
    st.metric("Estimated three-year recidivism probability", f"{probability:.1%}")
    st.caption("Race, gender, and geography are not model inputs. The foundation model is available for held-out cohort comparison; it is not loaded into this low-latency sandbox.")

with governance:
    st.subheader("Recommended operating policy")
    st.markdown("""
    - Use scores only to offer beneficial, capacity-limited support; never to increase surveillance, sanctions, or detention.
    - Keep race and gender out of the score and in the monitoring layer. Audit false-positive rates, calibration, and service allocation every quarter.
    - Require documented overrides, an appeal route, data-quality checks, and automatic suspension when drift or subgroup gaps exceed agreed limits.
    - Pilot prospectively before deployment. Re-estimate program benefit with a randomized or strong quasi-experimental design.
    """)
    st.subheader("Known limits")
    st.write("The cohort covers Georgia releases from 2013–2015. The outcome is a new arrest, which reflects both behavior and exposure to policing. External validity, construct validity, and causal value are therefore unproven.")

