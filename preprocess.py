"""
preprocess.py
Reproduces every feature engineering step from your notebook exactly.
Import this in both train_and_log.py and app.py.
"""

import numpy as np
import pandas as pd

NUM_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
]

CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "tenure_group", "is_new_customer", "high_value_customer",
    "total_services", "internet_service_count", "is_fiber_user",
    "no_internet", "has_multiple_lines", "auto_payment",
    "risky_payment", "long_term_contract", "month_to_month",
    "has_family", "senior_citizen_flag", "streaming_bundle",
    "risk_score",
]

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

SERVICES = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

INTERNET_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def get_risk_score(df: pd.DataFrame) -> pd.Series:
    score  = pd.Series(0, index=df.index)
    score += df["Contract"].map({"Month-to-month": 3, "One year": 1, "Two year": 0})
    score += (df["PaymentMethod"] == "Electronic check").astype(int) * 2
    score += np.where(df["tenure"] < 12, 2, 0)
    score += np.where((df["tenure"] >= 12) & (df["tenure"] < 24), 1, 0)
    score += (df["OnlineSecurity"] == "No").astype(int)
    score += (df["TechSupport"] == "No").astype(int)
    score += (df["InternetService"] == "Fiber optic").astype(int)
    score += df["SeniorCitizen"]
    score += (df["Partner"] == "No").astype(int)
    return score


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 60, 100],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"],
    )
    df["is_new_customer"]      = df["tenure"].apply(lambda x: "Yes" if x <= 12 else "No")
    df["avg_monthly_spend"]    = df["TotalCharges"] / (df["tenure"] + 1)
    df["high_value_customer"]  = df["MonthlyCharges"].apply(lambda x: "Yes" if x > 70 else "No")
    df["total_services"]       = df[SERVICES].apply(lambda x: sum(1 for i in x if "Yes" in str(i)), axis=1)
    df["internet_service_count"] = df[INTERNET_SERVICES].apply(lambda x: sum(1 for i in x if i == "Yes"), axis=1)
    df["is_fiber_user"]        = df["InternetService"].apply(lambda x: "Yes" if x == "Fiber optic" else "No")
    df["no_internet"]          = df["InternetService"].apply(lambda x: "Yes" if x == "No" else "No")
    df["has_multiple_lines"]   = df["MultipleLines"].apply(lambda x: "Yes" if x == "Yes" else "No")
    df["auto_payment"]         = df["PaymentMethod"].apply(lambda x: "Yes" if "automatic" in str(x) else "No")
    df["risky_payment"]        = df["PaymentMethod"].apply(lambda x: "Yes" if x == "Electronic check" else "No")
    df["long_term_contract"]   = df["Contract"].apply(lambda x: "Yes" if x in ["One year", "Two year"] else "No")
    df["month_to_month"]       = df["Contract"].apply(lambda x: "Yes" if x == "Month-to-month" else "No")
    df["has_family"]           = df.apply(lambda x: "Yes" if x["Partner"] == "Yes" or x["Dependents"] == "Yes" else "No", axis=1)
    df["senior_citizen_flag"]  = df["SeniorCitizen"].apply(lambda x: "Yes" if x == 1 else "No")
    df["streaming_bundle"]     = df.apply(lambda x: "Yes" if x["StreamingTV"] == "Yes" and x["StreamingMovies"] == "Yes" else "No", axis=1)
    df["risk_score"]           = get_risk_score(df)

    return df[ALL_FEATURES]
