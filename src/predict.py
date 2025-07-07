import pandas as pd
import joblib
from typing import Dict

def predict_dose(patient_data: Dict) -> float:
    pipeline = joblib.load("models/warfarin_model.joblib")
    X_new = pd.DataFrame([patient_data])
    pred = pipeline.predict(X_new)[0]
    return float(pred)
