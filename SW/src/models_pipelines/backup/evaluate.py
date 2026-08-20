"""
evaluate.py (gearbox_final_suite / src / models_pipelines)
[원본 복사본 - data/gearbox_fault_diagnosis_program/src/evaluate.py]
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
METRIC_DIR = os.path.join(RESULT_DIR, "metrics")
os.makedirs(METRIC_DIR, exist_ok=True)

MODEL_PATHS = {
    "Random Forest": os.path.join(MODEL_DIR, "random_forest_model.pkl"),
    "XGBoost": os.path.join(MODEL_DIR, "xgboost_model.pkl"),
    "Decision Tree": os.path.join(MODEL_DIR, "decision_tree_model.pkl"),
}

def evaluate_model(model, model_name, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            roc_auc = None

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
    }
    return metrics, y_pred

def save_metrics(all_metrics):
    for model_name, metrics in all_metrics.items():
        fname = model_name.lower().replace(" ", "_") + "_metrics.json"
        fpath = os.path.join(METRIC_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved: {fpath}")

    comparison = []
    for model_name, metrics in all_metrics.items():
        comparison.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "ROC AUC": metrics["roc_auc"],
        })
    comparison_df = pd.DataFrame(comparison)
    comp_path = os.path.join(METRIC_DIR, "model_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    print(f"Saved: {comp_path}")
    print("\nMODEL COMPARISON:")
    print(comparison_df.to_string(index=False))
    return comparison_df
