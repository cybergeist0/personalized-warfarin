import streamlit as st
import pandas as pd
import joblib
import os 

MODEL_PATH = "models/warfarin_model.joblib"

@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run main.py first.")
    return joblib.load(MODEL_PATH)

pipline = load_pipeline()
st.title("Warfarin Dosage")
st.write('Enter the patient data below to predict the therapeutic dose of Warfarin ')

with st.form("patient_form"):
    age = st.slider("Age", 1, 100, 65)
    height = st.number_input("Height (cm)", value = 170)
    weight=  st.number_input("Weight (kg)", value = 70)
    gender = st.selectbox("Gender", ["male", "female"])
    race = st.selectbox("Race (Reported)", ["White", "Asian", "Black or African American", "Unknown"])
    ethnicity = st.selectbox("Ethnicity (Reported)", ["Not Hispanic or Latino", "Hispanic or Latino", "Unknown"])
    inr = st.number_input("INR on Reported Therapeutic Dose of Warfarin", value=2.8)
    target_inr = st.number_input("Target INR", value = 2.5)
    smoker = st.selectbox("Current Smoker", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    amiodarone = st.selectbox("Amiodarone (Cordarone)", ["Yes", "No"])
    submitted = st.form_submit_button("Predict Dose")

if submitted:
    input_data = {
        "Age": age,
        "Height (cm)": height,
        "Weight (kg)": weight,
        "Gender": gender,
        "Race (Reported)": race,
        "Ethnicity (Reported)": ethnicity,
        "Indication for Warfarin Treatment": "Atrial fibrillation",  # fixed for demo
        "Comorbidities": "None",  # fixed
        "Diabetes": diabetes,
        "Congestive Heart Failure and/or Cardiomyopathy": "No",
        "Valve Replacement": "No",
        "Medications": "",
        "Aspirin": "No",
        "Acetaminophen or Paracetamol (Tylenol)": "No",
        "Simvastatin (Zocor)": "No",
        "Atorvastatin (Lipitor)": "No",
        "Fluvastatin (Lescol)": "No",
        "Lovastatin (Mevacor)": "No",
        "Pravastatin (Pravachol)": "No",
        "Rosuvastatin (Crestor)": "No",
        "Amiodarone (Cordarone)": amiodarone,
        "Carbamazepine (Tegretol)": "No",
        "Phenytoin (Dilantin)": "No",
        "Rifampin or Rifampicin": "No",
        "Sulfonamide Antibiotics": "No",
        "Macrolide Antibiotics": "No",
        "Anti-fungal Azoles": "No",
        "Current Smoker": smoker,
        "Target INR": target_inr,
        "INR on Reported Therapeutic Dose of Warfarin": inr
    }

    X_new = pd.DataFrame([input_data])
    pred = pipline.predict(X_new)[0]
    st.success(f"Predicted dosage for patient: {pred:.2f}mg/week ")
    


