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
sys.path.insert(0, str(ROOT / "scripts"))

from recidivism.config import ARTIFACT_DIR, DATA_PATH, FEATURE_COLUMNS, FIGURE_DIR, RANDOM_SEED, pretty
from recidivism.data import load_official_split
from recidivism.metrics import economic_value
from recidivism.modeling import tabicl_frames

st.set_page_config(page_title="Re-entry Support Allocation Lab", page_icon="⚖️", layout="wide")

LABELS = {"logistic": "Logistic regression", "xgboost": "XGBoost", "tabicl": "TabICLv2"}
CAPACITY = 0.20
SCOPE = ("Decision support for allocating re-entry support services only — "
         "not for detention, sentencing, surveillance, or sanctions.")


@st.cache_data
def load_results():
    read = lambda name: pd.read_csv(ARTIFACT_DIR / name) if (ARTIFACT_DIR / name).exists() else None
    return {name: read(f"{name}.csv") for name in [
        "model_metrics", "test_predictions", "fairness_by_group", "fairness_inference",
        "fairness_impossibility", "fairness_frontier", "incumbent_discrimination", "incumbent_economics",
        "race_ab_test", "stability_summary", "learning_curve", "shap_importance", "permutation_importance",
        "validation_baselines", "paired_comparisons", "intersectional_audit",
    ]}


@st.cache_resource
def load_split():
    return load_official_split(DATA_PATH)


@st.cache_resource
def load_models(include_tabicl=False):
    split = load_split()
    models = {name: joblib.load(ARTIFACT_DIR / "models" / f"{name}.joblib") for name in ["logistic", "xgboost"]}
    if not include_tabicl:
        return models
    from recidivism.modeling import tabicl_model
    tab = tabicl_model(RANDOM_SEED)
    tab.fit(tabicl_frames(split.X_train, split.X_train)[0], split.y_train.to_numpy())
    models["tabicl"] = tab
    return models


def predict(models, name, frame):
    if name == "tabicl":
        split = load_split()
        return models[name].predict_proba(tabicl_frames(split.X_train, frame)[1])[:, 1]
    return models[name].predict_proba(frame)[:, 1]


def figure(name: str, caption: str | None = None):
    path = FIGURE_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"Figure `{name}` not generated yet — run the matching script in `scripts/`.")


st.title("Re-entry Support Allocation Lab")
st.caption("Client: risk-assessment software vendor for US community-supervision agencies · NIJ Georgia cohort 2013–2015")
st.info(SCOPE, icon="⚖️")

data = load_results()
if data["model_metrics"] is None or data["test_predictions"] is None:
    st.error("Run `python scripts/train_evaluate.py` first to create the audit artifacts.")
    st.stop()
metrics, predictions = data["model_metrics"], data["test_predictions"]
LABELS = {m: label for m, label in LABELS.items() if f"p_{m}" in predictions}
metrics["Model"] = metrics.model.map(LABELS)
thresholds = {m: float(np.quantile(predictions[f"p_{m}"], 1 - CAPACITY)) for m in LABELS}

tab_ind, tab_cmp, tab_fair, tab_stab, tab_econ, tab_gov = st.tabs([
    "Individual assessment", "Model comparison", "Fairness audit", "Stability", "Economics", "Governance",
])

with tab_ind:
    st.subheader("Score one person with all three models")
    split = load_split()
    include_tabicl = st.checkbox("Include TabICLv2 live inference (slower first load)", value=False,
                                disabled="tabicl" not in LABELS)
    with st.spinner("Loading selected models…"):
        models = load_models(include_tabicl)
    source_row = st.selectbox("Start from a held-out record", range(min(250, len(split.X_test))),
                              format_func=lambda i: f"Record {int(split.audit_test.iloc[i].ID)}")
    row = split.X_test.iloc[[source_row]].copy()
    editable = ["Age_at_Release", "Supervision_Risk_Score_First", "Gang_Affiliated", "Education_Level",
                "Prison_Years", "Prior_Arrest_Episodes_Felony", "Prior_Revocations_Parole"]
    cols = st.columns(4)
    for i, col in enumerate(editable):
        options = [None, *sorted(split.X_train[col].dropna().unique(), key=lambda x: str(x))]
        current = row.iloc[0][col]
        index = options.index(current) if pd.notna(current) and current in options else 0
        value = cols[i % 4].selectbox(pretty(col), options, index=index,
                                     format_func=lambda v: "Missing" if v is None else str(v))
        row.loc[:, col] = np.nan if value is None else value

    scores = {m: float(predict(models, m, row[FEATURE_COLUMNS])[0]) for m in models if m in LABELS}
    cols = st.columns(3)
    for col, (m, p) in zip(cols, scores.items()):
        priority = p >= thresholds[m]
        col.metric(LABELS[m], f"{p:.1%}", "Priority for support (top 20%)" if priority else "Standard support",
                   delta_color="off")
    st.caption("Priority is a comparison to a historical cohort cutoff. Edited records are hypothetical; this is not a live allocation guarantee. Cohort audits allocate exactly round(n × capacity), breaking ties by row order.")

    st.markdown("**Why this score? Local explanation**")
    explain_model = st.radio("Explain with", ["xgboost", "logistic", "tabicl"], format_func=LABELS.get, horizontal=True)
    if explain_model == "tabicl":
        st.info("TabICLv2 has no native attribution method in this project. PDP/ICE describe model responses; "
                "the feature edits above allow local sensitivity exploration. Neither identifies causal effects.")
    else:
        from interpretability import shap_for
        values, _ = shap_for(models[explain_model], split.X_train.sample(300, random_state=0), row[FEATURE_COLUMNS], explain_model)
        contrib = values.iloc[0].sort_values(key=np.abs, ascending=False).head(10)[::-1].reset_index()
        contrib.columns = ["feature", "contribution"]
        contrib["feature"] = contrib.feature.map(pretty)
        contrib["direction"] = np.where(contrib.contribution > 0, "raises risk", "lowers risk")
        fig = px.bar(contrib, x="contribution", y="feature", color="direction", orientation="h",
                     color_discrete_map={"raises risk": "#C1121F", "lowers risk": "#2A9D8F"},
                     title=f"SHAP contributions (log-odds), {LABELS[explain_model]}")
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Race twin test**")
    st.caption("Race, gender and residence geography are not model inputs, so two people identical except for race "
               "always receive the same score (difference = 0 by construction). The A/B test in the fairness tab shows "
               "what happens if race is added as an input.")

with tab_cmp:
    st.subheader("Original evaluation partition (7,807 people)")
    display = metrics[["Model", "roc_auc", "average_precision", "brier", "ece_10", "fit_predict_seconds"]].copy()
    display.columns = ["Model", "ROC AUC", "Average precision", "Brier ↓", "Calibration error ↓", "Runtime (s)"]
    st.dataframe(display.style.format({c: "{:.3f}" for c in display.columns[1:]}), width="stretch", hide_index=True)
    st.caption("Original evaluation partition, repeatedly inspected during development. Results are exploratory; model fitting excludes these records.")
    review_path = ARTIFACT_DIR / "deep_review/combined_summary.csv"
    if review_path.exists():
        with st.expander("Accuracy experiments within the training partition"):
            st.write("Three outer folds compare prediction quality; XGBoost settings are selected on inner folds. Historical configurations already used this cohort. These internal results support research choices, with external validation still needed.")
            st.dataframe(pd.read_csv(review_path).round(5), width="stretch", hide_index=True)
            st.caption("blend_equal averages XGBoost and TabICL at a fixed 50/50 weight. It is a research candidate; the three individual scoring models above remain the published comparison.")
    if data["paired_comparisons"] is not None:
        with st.expander("Paired uncertainty and probability baselines"):
            st.write("Differences are A minus B: positive AUC favors A; negative Brier favors A. Intervals are exploratory and unadjusted for multiple comparisons.")
            st.dataframe(data["paired_comparisons"].round(5), width="stretch", hide_index=True)
            st.dataframe(data["validation_baselines"].round(4), width="stretch", hide_index=True)
    if data["incumbent_discrimination"] is not None:
        st.markdown("**Historical recorded-score benchmark** (`Supervision_Risk_Score_First`, 1–10 score)")
        st.dataframe(data["incumbent_discrimination"].round(3), width="stretch", hide_index=True)
    figure("learning_curve.png", "Learning curve: the foundation model leads on small data; the gap closes as data grows.")
    c1, c2 = st.columns(2)
    with c1:
        figure("shap_global.png", "Global drivers (SHAP)")
    with c2:
        figure("global_surrogate.png", "A depth-3 tree that mimics XGBoost")
    figure("pdp_ice.png", "Partial dependence (black) and individual curves, all three models")

with tab_fair:
    st.subheader("Fairness audit by race and gender")
    st.write("For beneficial support, missed access matters: inspect FNR (1 − TPR) and selection rates first. Arrest is only a proxy for need; these errors do not identify treatment benefit.")
    figure("fairness_support_access.png", "False-negative-rate gaps at the proposed capacity rule and at 0.5; signed group differences.")
    inf = data["fairness_inference"]
    if inf is not None:
        rule = st.radio("Operating point", ["top_20pct", "threshold_0.5"], horizontal=True,
                        format_func={"top_20pct": "Deployed: top 20% by risk", "threshold_0.5": "Threshold 0.5"}.get)
        attr = st.radio("Attribute", ["Race", "Gender"], horizontal=True)
        view = inf[(inf.rule == rule) & (inf.attribute == attr)].copy()
        view["Model"] = view.model.map(LABELS)
        fig = px.scatter(view, x="metric", y="gap", color="Model", error_y=view.ci_high - view.gap,
                         error_y_minus=view.gap - view.ci_low, title=f"{view.comparison.iloc[0]} gaps with 95% bootstrap CI")
        fig.add_hline(y=0, line_color="grey")
        st.plotly_chart(fig, width="stretch")
        st.caption("A gap whose interval crosses 0 is not statistically distinguishable from no gap.")
    if data["fairness_impossibility"] is not None:
        st.markdown("**Base rates and within-group calibration**")
        st.dataframe(data["fairness_impossibility"].round(3), width="stretch", hide_index=True)
    if data["intersectional_audit"] is not None:
        with st.expander("Race × gender intersections (descriptive, threshold 0.5)"):
            view = data["intersectional_audit"].query("attribute == 'Race x Gender'")
            st.dataframe(view.round(4), width="stretch", hide_index=True)
            st.caption("Inspect subgroup sample sizes; small intersections have greater uncertainty. These point estimates do not establish fairness.")
        with st.expander("Age-group audit (descriptive, threshold 0.5)"):
            st.dataframe(data["intersectional_audit"].query("attribute == 'Age at release'").round(4), width="stretch", hide_index=True)
    figure("fairness_frontier.png", "Group thresholds were optimized using these evaluation labels: exploratory illustration only. Thresholds change decisions, not probability calibration.")
    if data["race_ab_test"] is not None:
        st.markdown("**A/B test: model trained with race vs without race**")
        st.dataframe(data["race_ab_test"].round(4), width="stretch", hide_index=True)

with tab_stab:
    st.subheader("Structural stability across refits on resampled training data")
    if data["stability_summary"] is not None:
        st.dataframe(data["stability_summary"].round(4), width="stretch", hide_index=True)
    figure("structural_stability.png")

with tab_econ:
    st.subheader("Resource-allocation scenario")
    c1, c2, c3, c4 = st.columns(4)
    capacity = c1.slider("Share offered support", 0.05, 0.50, CAPACITY, 0.05)
    intervention_cost = c2.number_input("Cost per person ($)", 0, 100_000, 5_000, 500)
    event_cost = c3.number_input("Cost per event ($)", 0, 500_000, 50_000, 5_000)
    effectiveness = c4.slider("Assumed effectiveness", 0.0, 1.0, 0.20, 0.05)
    split = load_split()
    incumbent = split.X_test["Supervision_Risk_Score_First"]
    rankers = {"Incumbent score": incumbent.fillna(split.X_train["Supervision_Risk_Score_First"].median()).to_numpy(),
               **{LABELS[m]: predictions[f"p_{m}"].to_numpy() for m in LABELS}}
    rows = [{"Ranker": name, **economic_value(predictions.actual, s, capacity, intervention_cost, event_cost, effectiveness)}
            for name, s in rankers.items()]
    econ = pd.DataFrame(rows)
    st.dataframe(econ[["Ranker", "selected", "captured_events", "recall_at_capacity", "assumed_net_value"]].style.format({
        "recall_at_capacity": "{:.1%}", "assumed_net_value": "${:,.0f}"}), width="stretch", hide_index=True)
    st.plotly_chart(px.bar(econ, x="Ranker", y="assumed_net_value", color="Ranker", title="Net value under your assumptions"),
                    width="stretch")
    st.info("Transparent scenario calculator. Costs and effectiveness are user assumptions; the observational data cannot estimate causal program impact.")

with tab_gov:
    st.subheader("Recommended operating policy")
    st.markdown("""
    - Use scores only to offer beneficial, capacity-limited support; never to increase surveillance, sanctions, or detention.
    - Keep race and gender out of the score and in the monitoring layer. Audit error rates and calibration by group every quarter, at the deployed operating point.
    - Require documented overrides, an appeal route, data-quality checks, and automatic suspension when drift or subgroup gaps exceed agreed limits.
    - Pilot prospectively before deployment. Re-estimate program benefit with a randomized or strong quasi-experimental design.
    """)
    st.subheader("Known limits")
    st.write("The cohort covers Georgia releases from 2013–2015. The outcome is a new arrest, which reflects both behavior "
             "and exposure to policing. External validity, construct validity, and causal value are therefore unproven.")
