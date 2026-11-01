import os
import joblib

def test_model_file_exists():
    assert os.path.exists("artifacts/diabetes_model.pkl")

def test_model_can_be_loaded():
    model = joblib.load("artifacts/diabetes_model.pkl")
    assert model is not None

def test_model_prediction():
    model = joblib.load("artifacts/diabetes_model.pkl")

    sample = [[
        "Female", 45, 0, 0,
        "never", 25.5, 5.7, 120
    ]]

    import pandas as pd

    columns = [
        "gender", "age", "hypertension",
        "heart_disease", "smoking_history",
        "bmi", "HbA1c_level", "blood_glucose_level"
    ]

    sample_df = pd.DataFrame(sample, columns=columns)

    prediction = model.predict(sample_df)

    assert prediction[0] in [0, 1]
