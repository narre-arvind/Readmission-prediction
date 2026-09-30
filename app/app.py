"""Streamlit interface for individual readmission risk scoring."""
from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import add_features

artifact = joblib.load(ROOT / "models/model_bundle.joblib")
model = artifact["models"][artifact["best_model"]]
threshold = artifact.get("decision_threshold", 0.5)

st.set_page_config(page_title="Diabetes readmission risk", page_icon="🏥", layout="wide")
st.title("30-day diabetes readmission risk")
st.caption("Educational project. Not for clinical use.")
with st.form("patient_form"):
    age = st.selectbox("Age", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"] , index=6)
    cols = st.columns(3)
    time_in_hospital = cols[0].number_input("Days in hospital", 1, 14, 4)
    prior_inpatient = cols[1].number_input("Prior inpatient visits", 0, 20, 0)
    prior_emergency = cols[2].number_input("Prior emergency visits", 0, 20, 0)
    prior_outpatient = st.number_input("Prior outpatient visits", 0, 50, 0)
    number_diagnoses = st.number_input("Number of diagnoses", 1, 16, 5)
    num_lab_procedures = st.number_input("Laboratory procedures", 0, 132, 40)
    num_medications = st.number_input("Number of medications", 0, 80, 10)
    discharge_disposition_id = st.number_input("Discharge disposition code", 1, 30, 1)
    a1c = st.selectbox("A1C result", ["None", "Norm", ">7", ">8"])
    medication_change = st.selectbox("Medication change", ["No", "Ch", "Up", "Down"])
    submitted = st.form_submit_button("Estimate risk")
if submitted:
    row = {"age": age, "time_in_hospital": time_in_hospital, "number_inpatient": prior_inpatient, "number_emergency": prior_emergency, "number_outpatient": prior_outpatient, "number_diagnoses": number_diagnoses, "num_lab_procedures": num_lab_procedures, "num_medications": num_medications, "discharge_disposition_id": discharge_disposition_id, "A1Cresult": a1c, "change": medication_change}
    frame = pd.DataFrame([row]); frame = add_features(frame)
    for column, is_numeric in artifact["feature_schema"].items():
        if column not in frame:
            frame[column] = 0 if is_numeric else "Unknown"
    frame = frame[list(artifact["feature_schema"])]
    probability = float(model.predict_proba(frame)[:, 1][0])
    level = "high" if probability >= threshold else "medium" if probability >= threshold * 0.5 else "low"
    transformed = model.named_steps["preprocess"].transform(frame)
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importance = model.named_steps["model"].feature_importances_
    values = transformed.toarray()[0] if hasattr(transformed, "toarray") else transformed[0]
    top_factors = pd.DataFrame({"factor": feature_names, "contribution": abs(values) * importance}).sort_values("contribution", ascending=False).head(5)
    st.metric("Estimated probability", f"{probability:.1%}")
    st.subheader(f"Risk level: {level.upper()}")
    st.subheader("Top contributing factors")
    st.dataframe(top_factors[["factor", "contribution"]], hide_index=True, use_container_width=True)
    st.info("Review discharge planning, follow-up access, and medication reconciliation for higher-risk patients.")
