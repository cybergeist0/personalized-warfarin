import pandas as pd
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        parts = age_str.split('-')
        return (int(parts[0]) + int(parts[1])) / 2
    try:
        return float(age_str)
    except:
        return None

def load_and_preprocess(path: str) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    df = pd.read_excel(path, sheet_name="Subject Data")
    df = df.dropna(subset=["Therapeutic Dose of Warfarin"])

    features = [
        # Demographics
        "Age", "Height (cm)", "Weight (kg)", "Gender", "Race (Reported)", "Ethnicity (Reported)",
        # Indication & comorbidities
        "Indication for Warfarin Treatment", "Comorbidities", "Diabetes", "Congestive Heart Failure and/or Cardiomyopathy", "Valve Replacement",
        # Concurrent medications
        "Medications", "Aspirin", "Acetaminophen or Paracetamol (Tylenol)", "Simvastatin (Zocor)", "Atorvastatin (Lipitor)",
        "Fluvastatin (Lescol)", "Lovastatin (Mevacor)", "Pravastatin (Pravachol)", "Rosuvastatin (Crestor)", "Amiodarone (Cordarone)",
        "Carbamazepine (Tegretol)", "Phenytoin (Dilantin)", "Rifampin or Rifampicin", "Sulfonamide Antibiotics", "Macrolide Antibiotics", "Anti-fungal Azoles",
        # Lifestyle
        "Current Smoker",
        # Lab & target
        "Target INR", "INR on Reported Therapeutic Dose of Warfarin"
    ]

    required = features + ["Therapeutic Dose of Warfarin"]
    df = df[required]

    df["Age"] = df["Age"].apply(convert_age)

    X = df[features]
    y = df["Therapeutic Dose of Warfarin"]

    numeric_features = ["Age", "Height (cm)", "Weight (kg)"]
    categorical_features = ["Gender", "Race (Reported)", "Amiodarone (Cordarone)", "Diabetes", "Current Smoker"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return X, y, preprocessor
