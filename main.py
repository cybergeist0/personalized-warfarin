from src.preprocess import load_and_preprocess
from src.model import train_regression_model
from src.evaluate import evaluate_regression, plot_feature_importance
from src.predict import predict_dose
import os

def main():
    model_path = "models/warfarin_model.joblib"
    if not os.path.exists(model_path):
        X, y, preprocessor = load_and_preprocess("data/warfarin_data.xls")
        model, X_train, X_test, y_train, y_test = train_regression_model(X, y, preprocessor)
        evaluate_regression(model, X_test, y_test)
        plot_feature_importance(model.named_steps["regressor"], model.named_steps["preprocessor"])
    else:
        print("Model already trained and loaded from disk.")

    sample_patient = {
        "Age": 65,
        "Height (cm)": 170,
        "Weight (kg)": 70,
        "Gender": "Male",
        "Race (Reported)": "White",
        "Ethnicity (Reported)": "Not Hispanic or Latino",
        "Indication for Warfarin Treatment": "Atrial fibrillation",
        "Comorbidities": "None",
        "Diabetes": "No",
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
        "Amiodarone (Cordarone)": "No",
        "Carbamazepine (Tegretol)": "No",
        "Phenytoin (Dilantin)": "No",
        "Rifampin or Rifampicin": "No",
        "Sulfonamide Antibiotics": "No",
        "Macrolide Antibiotics": "No",
        "Anti-fungal Azoles": "No",
        "Current Smoker": "No",
        "Target INR": 2.5,
        "INR on Reported Therapeutic Dose of Warfarin": 2.8
    }

    dose = predict_dose(sample_patient)
    print(f"Predicted Warfarin dose for sample patient: {dose:.2f} mg/week")

if __name__ == "__main__":
    main()
