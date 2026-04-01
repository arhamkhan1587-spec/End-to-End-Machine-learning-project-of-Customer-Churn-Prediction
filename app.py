"""
app.py  —  FastAPI production REST API
Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
Docs: http://127.0.0.1:8000/docs
"""

import time, warnings
from typing import List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from predict import predict_single, predict_batch, REGISTERED_NAME, DECISION_THRESHOLD

warnings.filterwarnings("ignore")

app = FastAPI(
    title       = "Churn Prediction API",
    description = "CatBoost + LightGBM + XGBoost Voting Ensemble",
    version     = "1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class CustomerFeatures(BaseModel):
    gender:           str   = Field(..., example="Male")
    SeniorCitizen:    int   = Field(..., example=0)
    Partner:          str   = Field(..., example="Yes")
    Dependents:       str   = Field(..., example="No")
    tenure:           float = Field(..., example=12)
    PhoneService:     str   = Field(..., example="Yes")
    MultipleLines:    str   = Field(..., example="No")
    InternetService:  str   = Field(..., example="Fiber optic")
    OnlineSecurity:   str   = Field(..., example="No")
    OnlineBackup:     str   = Field(..., example="No")
    DeviceProtection: str   = Field(..., example="No")
    TechSupport:      str   = Field(..., example="No")
    StreamingTV:      str   = Field(..., example="No")
    StreamingMovies:  str   = Field(..., example="No")
    Contract:         str   = Field(..., example="Month-to-month")
    PaperlessBilling: str   = Field(..., example="Yes")
    PaymentMethod:    str   = Field(..., example="Electronic check")
    MonthlyCharges:   float = Field(..., example=70.35)
    TotalCharges:     float = Field(..., example=151.65)


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


@app.get("/health")
def health():
    return {"status": "healthy", "model": REGISTERED_NAME, "threshold": DECISION_THRESHOLD}


@app.get("/model/info")
def model_info():
    return {
        "model_name": REGISTERED_NAME,
        "model_type": "VotingClassifier (CatBoost + LightGBM + XGBoost)",
        "imbalance":  "SMOTE",
        "threshold":  DECISION_THRESHOLD,
        "version":    "1.0.0",
    }


@app.post("/predict/single")
def predict_single_endpoint(customer: CustomerFeatures):
    try:
        t0     = time.time()
        result = predict_single(customer.dict())
        result["latency_ms"] = round((time.time() - t0) * 1000, 2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch_endpoint(request: BatchRequest):
    try:
        t0        = time.time()
        df_input  = pd.DataFrame([c.dict() for c in request.customers])
        df_result = predict_batch(df_input)
        n_churn   = int(df_result["churn_prediction"].sum())
        return {
            "total":           len(df_result),
            "predicted_churn": n_churn,
            "churn_rate_pct":  round(n_churn / len(df_result) * 100, 2),
            "predictions":     df_result[["churn_probability","churn_prediction",
                                          "churn_label","risk_level"]].to_dict(orient="records"),
            "latency_ms":      round((time.time() - t0) * 1000, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
