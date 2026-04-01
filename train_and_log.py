"""
train_and_log.py  —  MLflow 3.x compatible
───────────────────────────────────────────
Loads your saved final_pipeline.pkl, evaluates it,
and logs everything to MLflow 3.x correctly.

Run:
    python train_and_log.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

from preprocess import engineer_features, ALL_FEATURES

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit these paths
# ─────────────────────────────────────────────────────────────────────────────
PIPELINE_PATH      = "final_pipeline.pkl"
DATA_PATH          = "train.csv"
MLFLOW_URI         = "mlruns"               # local  OR  "http://127.0.0.1:5000"
EXPERIMENT_NAME    = "churn_prediction"
RUN_NAME           = "voting_cat_lgb_xgb_smote"
REGISTERED_NAME    = "ChurnPredictionModel"
DECISION_THRESHOLD = 0.50                   # change after threshold tuning


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA  &  REPRODUCE EXACT SPLIT FROM YOUR NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_PATH)
if "id" in df.columns:
    df = df.drop(columns=["id"])

df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

y     = df["Churn"]
X_raw = df.drop(columns=["Churn"])
X     = engineer_features(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading pipeline from {PIPELINE_PATH} ...")
pipeline = joblib.load(PIPELINE_PATH)
print("  ✔ Pipeline loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
print("\nEvaluating ...")
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= DECISION_THRESHOLD).astype(int)

auc      = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)
f1_yes   = f1_score(y_test, y_pred, pos_label=1)
f1_no    = f1_score(y_test, y_pred, pos_label=0)
prec_yes = precision_score(y_test, y_pred, pos_label=1)
prec_no  = precision_score(y_test, y_pred, pos_label=0)
rec_yes  = recall_score(y_test, y_pred, pos_label=1)
rec_no   = recall_score(y_test, y_pred, pos_label=0)
cm       = confusion_matrix(y_test, y_pred)
report   = classification_report(y_test, y_pred)

print(f"  ROC-AUC  : {auc:.5f}")
print(f"  Accuracy : {accuracy:.5f}")
print(f"\n{report}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SAVE PLOTS AS ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("mlflow_artifacts", exist_ok=True)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#00e5ff", lw=2, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Churn Model"); plt.legend(); plt.tight_layout()
plt.savefig("mlflow_artifacts/roc_curve.png", dpi=120); plt.close()

# Precision-Recall Curve
prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(rec_c, prec_c, color="#69ff47", lw=2)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve"); plt.tight_layout()
plt.savefig("mlflow_artifacts/pr_curve.png", dpi=120); plt.close()

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred No", "Pred Yes"],
            yticklabels=["True No", "True Yes"])
plt.title(f"Confusion Matrix (threshold={DECISION_THRESHOLD})")
plt.tight_layout()
plt.savefig("mlflow_artifacts/confusion_matrix.png", dpi=120); plt.close()

# Threshold vs F1
thresholds = np.linspace(0.2, 0.9, 200)
f1_list    = [f1_score(y_test, (y_proba >= t).astype(int), pos_label=1)
              for t in thresholds]
plt.figure(figsize=(7, 5))
plt.plot(thresholds, f1_list, color="#ffd740", lw=2)
plt.axvline(DECISION_THRESHOLD, color="red", ls="--",
            label=f"Current = {DECISION_THRESHOLD}")
plt.xlabel("Threshold"); plt.ylabel("F1 (Churn=Yes)")
plt.title("Threshold vs F1"); plt.legend(); plt.tight_layout()
plt.savefig("mlflow_artifacts/threshold_f1.png", dpi=120); plt.close()

# Classification report as text
with open("mlflow_artifacts/classification_report.txt", "w") as f:
    f.write(report)

print("  ✔ Artifacts saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MLFLOW 3.x — LOG EVERYTHING
#
#  KEY MLflow 3.x API changes applied here:
#  ✔  artifact_path  →  name
#  ✔  model_uri from log_model return value  (model_info.model_uri)
#  ✔  register_model uses model_info.model_uri
#  ✘  DO NOT use mlflow.get_artifact_uri("model")  — broken in 3.x
#  ✘  DO NOT use registered_name in log_model     — removed in 3.x
# ─────────────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

signature     = infer_signature(X_test, y_proba)
input_example = X_test.iloc[:3]

with mlflow.start_run(run_name=RUN_NAME) as run:

    # ── Tags ──────────────────────────────────────────────────────────────────
    mlflow.set_tags({
        "model_type": "VotingClassifier (CatBoost + LightGBM + XGBoost)",
        "imbalance":  "SMOTE",
        "framework":  "sklearn + imblearn",
        "dataset":    "playground-series-s6e3",
        "author":     "your_name",              # ← edit
    })

    # ── Params ────────────────────────────────────────────────────────────────
    mlflow.log_params({
        "cat_iterations":     1000,
        "cat_learning_rate":  0.03,
        "cat_depth":          6,
        "cat_l2_leaf_reg":    3,
        "lgb_n_estimators":   1000,
        "lgb_learning_rate":  0.03,
        "lgb_num_leaves":     31,
        "lgb_subsample":      0.8,
        "xgb_n_estimators":   400,
        "xgb_max_depth":      5,
        "xgb_learning_rate":  0.05,
        "xgb_subsample":      0.8,
        "voting":             "soft",
        "weights":            "1,1,1.5",
        "smote_random_state": 42,
        "test_size":          0.2,
        "decision_threshold": DECISION_THRESHOLD,
    })

    # ── Metrics ───────────────────────────────────────────────────────────────
    mlflow.log_metrics({
        "roc_auc":         round(auc, 5),
        "accuracy":        round(accuracy, 5),
        "f1_churn_yes":    round(f1_yes, 5),
        "f1_churn_no":     round(f1_no, 5),
        "precision_yes":   round(prec_yes, 5),
        "precision_no":    round(prec_no, 5),
        "recall_yes":      round(rec_yes, 5),
        "recall_no":       round(rec_no, 5),
        "true_positives":  int(cm[1][1]),
        "false_positives": int(cm[0][1]),
        "true_negatives":  int(cm[0][0]),
        "false_negatives": int(cm[1][0]),
    })

    # ── Artifacts ─────────────────────────────────────────────────────────────
    mlflow.log_artifact("mlflow_artifacts/roc_curve.png",            "plots")
    mlflow.log_artifact("mlflow_artifacts/pr_curve.png",             "plots")
    mlflow.log_artifact("mlflow_artifacts/confusion_matrix.png",     "plots")
    mlflow.log_artifact("mlflow_artifacts/threshold_f1.png",         "plots")
    mlflow.log_artifact("mlflow_artifacts/classification_report.txt")

    # ── Log Model — MLflow 3.x syntax ─────────────────────────────────────────
    model_info = mlflow.sklearn.log_model(
        sk_model         = pipeline,
        name             = "churn_model",   # 'name' replaces 'artifact_path'
        signature        = signature,
        input_example    = input_example,
        pip_requirements = [
            "scikit-learn",
            "imbalanced-learn",
            "xgboost",
            "lightgbm",
            "catboost",
        ],
    )

    # ── Register Model — MLflow 3.x syntax ────────────────────────────────────
    # Use model_info.model_uri from log_model return value
    reg = mlflow.register_model(
        model_uri = model_info.model_uri,
        name      = REGISTERED_NAME,
    )

    run_id  = run.info.run_id
    version = reg.version

print(f"\n{'='*55}")
print(f"  ✔  MLflow run complete.")
print(f"     Run ID        : {run_id}")
print(f"     Model URI     : {model_info.model_uri}")
print(f"     Model ID      : {model_info.model_id}")
print(f"     Registry      : {REGISTERED_NAME}  v{version}")
print(f"     ROC-AUC       : {auc:.5f}")
print(f"\n  View UI → run : mlflow ui")
print(f"            open : http://127.0.0.1:5000")
print(f"{'='*55}")

# Save model_uri for predict.py to use
with open("mlflow_artifacts/model_uri.txt", "w") as f:
    f.write(model_info.model_uri)
print(f"\n  model_uri saved → mlflow_artifacts/model_uri.txt")
