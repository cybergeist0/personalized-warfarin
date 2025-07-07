from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

def train_regression_model(X, y, preprocessor, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=random_state))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "models/warfarin_model.joblib")
    return pipeline, X_train, X_test, y_train, y_test
