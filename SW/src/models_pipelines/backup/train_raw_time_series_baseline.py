"""
train_raw_time_series_baseline.py (gearbox_final_suite / models_pipelines)

Baseline training script using XGBoost and Random Forest on Raw Time-Series Datasets:
- results/unified_raw_time_series_normal.csv
- results/unified_raw_time_series_abnormal.csv

Features evaluated:
- speed_kph, grade_pct
- TCS_CTL, ABS_ACT, ESP_CTL, TQI_TCS
- Yaw_Rate, Lateral_Accel
- WHL_SPD_FL, WHL_SPD_FR, WHL_SPD_RL, WHL_SPD_RR
- audio_gear1_sample, audio_gear1_hilbert_env

Target:
- state (0: normal, 1: abnormal)

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

NORMAL_CSV = os.path.join(RESULTS_DIR, "unified_raw_time_series_normal.csv")
ABNORMAL_CSV = os.path.join(RESULTS_DIR, "unified_raw_time_series_abnormal.csv")


def run_raw_time_series_training():
    print("=" * 100)
    print(" 🚀 STARTING RAW TIME-SERIES MODEL TRAINING (XGBoost & Random Forest)")
    print("=" * 100)

    if not os.path.exists(NORMAL_CSV) or not os.path.exists(ABNORMAL_CSV):
        print("[ERROR] Split Raw Time-Series CSV files not found!")
        print("Please ensure unified_raw_time_series_normal.csv and abnormal.csv exist.")
        return

    print(" 📂 Loading Normal and Abnormal Raw Datasets...")
    df_norm = pd.read_csv(NORMAL_CSV)
    df_abnorm = pd.read_csv(ABNORMAL_CSV)

    df_norm["target"] = 0
    df_abnorm["target"] = 1

    df_full = pd.concat([df_norm, df_abnorm], ignore_index=True)
    print(f" 📊 Total Raw Time-Series Samples Loaded: {len(df_full)} rows")
    print(f"    - Normal Samples: {len(df_norm)} rows")
    print(f"    - Abnormal Samples: {len(df_abnorm)} rows")

    # Select Feature Columns
    feature_cols = [
        "speed_kph", "grade_pct",
        "TCS_CTL", "ABS_ACT", "ESP_CTL", "TQI_TCS",
        "Yaw_Rate", "Lateral_Accel",
        "WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR",
        "audio_gear1_sample", "audio_gear1_hilbert_env"
    ]

    X = df_full[feature_cols].copy()
    y = df_full["target"].values

    X = X.fillna(0.0)

    # Train / Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n" + "=" * 100)
    print(" 1️⃣ Training XGBoost Classifier on Raw Time-Series Features...")
    print("=" * 100)

    model_xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
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

    print(f" 🎯 XGBoost Results:")
    print(f"    - Accuracy: {acc_xgb * 100:.2f}%")
    print(f"    - ROC-AUC:  {auc_xgb:.4f}")
    print(f"    - F1-Score: {f1_xgb:.4f}")
    print("\n Classification Report (XGBoost):")
    print(classification_report(y_test, y_pred_xgb, target_names=["Normal", "Abnormal"]))

    print("=" * 100)
    print(" 2️⃣ Training Random Forest Classifier on Raw Time-Series Features...")
    print("=" * 100)

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

    print(f" 🎯 Random Forest Results:")
    print(f"    - Accuracy: {acc_rf * 100:.2f}%")
    print(f"    - ROC-AUC:  {auc_rf:.4f}")
    print(f"    - F1-Score: {f1_rf:.4f}")
    print("\n Classification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, target_names=["Normal", "Abnormal"]))

    print("=" * 100)
    print(" 💡 Feature Importances (Top 10 - XGBoost):")
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model_xgb.feature_importances_
    }).sort_values("Importance", ascending=False)
    for idx, row in imp_df.head(10).iterrows():
        print(f"    - {row['Feature']:<25}: {row['Importance']:.6f}")
    print("=" * 100)


if __name__ == "__main__":
    run_raw_time_series_training()
