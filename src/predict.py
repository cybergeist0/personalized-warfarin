import os
import pandas as pd
import joblib
from typing import Dict
from src.preprocess import load_and_preprocess
from src.model import train_regression_model


def predict_dose(patient_data:Dict) -> float:
    model_path = "models/warfarin_model.joblib"
    if not os.path.exists(model_path):
        print("Model missing. Training now...")
        X,y,preprocessor = load_and_preprocess("data/warfarin_data.xls")
        model,_,_,_,_ = train_regression_model(X, y)
    else:
        model = joblib.load(model_path)
    X_new = pd.DataFrame([patient_data])
    pred = model.predict(X_new)[0]
    return float(pred)
