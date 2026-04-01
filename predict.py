"""
predict.py  —  MLflow 3.x compatible
─────────────────────────────────────
Loads the registered model from MLflow 3.x and predicts on new data.

Usage (CLI):
    python predict.py --input new_customers.csv --output predictions.csv
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from preprocess import engineer_features

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← must match train_and_log.py
# ─────────────────────────────────────────────────────────────────────────────
MLFLOW_URI         = "mlruns"
REGISTERED_NAME    = "ChurnPredictionModel"
DECISION_THRESHOLD = 0.50       # same as train_and_log.py


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# MLflow 3.x: use model_uri saved from log_model return value
# ─────────────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)

try:
    # Read the model_uri saved by train_and_log.py
    with open("mlflow_artifacts/model_uri.txt") as f:
        model_uri = f.read().strip()
    pipeline = mlflow.sklearn.load_model(model_uri)
    print(f"✔  Model loaded from : {model_uri}")

except FileNotFoundError:
    # Fallback: load from registry by name/version
    model_uri = f"models:/{REGISTERED_NAME}/1"
    try:
        pipeline = mlflow.sklearn.load_model(model_uri)
        print(f"✔  Model loaded from registry : {model_uri}")
    except Exception:
        # Last fallback: load directly from pkl
        import joblib
        pipeline = joblib.load("final_pipeline.pkl")
        print("⚠  Loaded from final_pipeline.pkl (MLflow registry not found)")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def predict_single(customer: dict) -> dict:
    """
    Predict churn for one customer.

    Input : dict of raw features (same columns as train.csv, no id/Churn)
    Output: dict with probability, prediction, risk level

    Example input:
        {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35, "TotalCharges": 151.65
        }
    """
    df_raw   = pd.DataFrame([customer])
    df_feats = engineer_features(df_raw)
    proba    = pipeline.predict_proba(df_feats)[0][1]
    pred     = int(proba >= DECISION_THRESHOLD)

    risk = "HIGH" if proba >= 0.70 else "MEDIUM" if proba >= 0.40 else "LOW"

    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction":  pred,
        "churn_label":       "Yes" if pred == 1 else "No",
        "risk_level":        risk,
        "threshold_used":    DECISION_THRESHOLD,
    }


def predict_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Predict churn for multiple customers.

    Input : raw dataframe (same schema as train.csv, no Churn column)
    Output: same dataframe + churn_probability, churn_prediction, risk_level
    """
    df_feats    = engineer_features(df_raw)
    probas      = pipeline.predict_proba(df_feats)[:, 1]
    predictions = (probas >= DECISION_THRESHOLD).astype(int)

    result = df_raw.copy()
    result["churn_probability"] = np.round(probas, 4)
    result["churn_prediction"]  = predictions
    result["churn_label"]       = np.where(predictions == 1, "Yes", "No")
    result["risk_level"]        = np.where(
        probas >= 0.70, "HIGH",
        np.where(probas >= 0.40, "MEDIUM", "LOW")
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,           help="Input CSV path")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    df_input = pd.read_csv(args.input)
    if "Churn" in df_input.columns:
        df_input = df_input.drop(columns=["Churn"])

    results = predict_batch(df_input)
    results.to_csv(args.output, index=False)

    print(f"\n✔  Saved → {args.output}")
    print(f"   Total     : {len(results)}")
    print(f"   Churners  : {results['churn_prediction'].sum()} "
          f"({results['churn_prediction'].mean()*100:.1f}%)")
    print(f"\n   Risk breakdown:\n{results['risk_level'].value_counts().to_string()}")
