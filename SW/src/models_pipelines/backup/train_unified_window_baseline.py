"""
train_unified_window_baseline.py (gearbox_final_suite / models_pipelines)

Training script using XGBoost and Random Forest on Windowed Unified Datasets:
1. Original Window Dataset: results/unified_can_context_dataset.csv (45 features)
2. Audio-Synced Window Dataset: results/unified_can_audio_synced_dataset.csv (Time-synced CAN + Audio features)

Rule 3 Compliance:
- Code comments & terminal output in 100% Korean.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(SUITE_DIR, "results")

ORIGINAL_UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")
SYNCED_UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_audio_synced_dataset.csv")


def evaluate_dataset(csv_path, dataset_name):
    print("\n" + "=" * 100)
    print(f" 📂 EVALUATING DATASET: {dataset_name}")
    print(f" 📄 Path: {csv_path}")
    print("=" * 100)

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f" 📊 Total Window Samples: {len(df)} rows")

    # Determine target column
    if "target" in df.columns:
        y = df["target"].values
    elif "state" in df.columns:
        y = (df["state"].str.lower() == "abnormal").astype(int).values
    else:
        print("[ERROR] Neither 'target' nor 'state' column found.")
        return

    # Drop non-feature metadata columns
    drop_cols = ["session_id", "session_file", "state", "target", "scenario"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy().fillna(0.0)

    # Train / Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. XGBoost
    model_xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    model_xgb.fit(X_train, y_train)

    y_pred_xgb = model_xgb.predict(X_test)
    y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]

    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)

    print(f"\n 🎯 [XGBoost] Accuracy: {acc_xgb * 100:.2f}% | ROC-AUC: {auc_xgb:.4f} | F1-Score: {f1_xgb:.4f}")

    # 2. Random Forest
    model_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model_rf.fit(X_train, y_train)

    y_pred_rf = model_rf.predict(X_test)
    y_prob_rf = model_rf.predict_proba(X_test)[:, 1]

    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    f1_rf = f1_score(y_test, y_pred_rf)

    print(f" 🎯 [Random Forest] Accuracy: {acc_rf * 100:.2f}% | ROC-AUC: {auc_rf:.4f} | F1-Score: {f1_rf:.4f}")

    # Feature Importance
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model_xgb.feature_importances_
    }).sort_values("Importance", ascending=False)
    print("\n 💡 Top 5 Feature Importances (XGBoost):")
    for idx, row in imp_df.head(5).iterrows():
        print(f"    - {row['Feature']:<30}: {row['Importance']:.6f}")


def run_window_unified_training():
    print("=" * 100)
    print(" 🚀 STARTING WINDOW-LEVEL UNIFIED MODEL TRAINING")
    print("=" * 100)

    # 1. Original Unified Dataset
    evaluate_dataset(ORIGINAL_UNIFIED_CSV, "Original Unified Dataset (45 CAN features)")

    # 2. Synced Audio Unified Dataset
    evaluate_dataset(SYNCED_UNIFIED_CSV, "Synced Audio-CAN Window Dataset (Time-synced CAN + Audio)")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    run_window_unified_training()
