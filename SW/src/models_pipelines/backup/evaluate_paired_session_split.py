"""
evaluate_paired_session_split.py (gearbox_fault_diagnosis_fixed)

Implements Paired-Session Time-Block Split:
For every scenario (20kph, 60kph, 80kph, hills 6/12/18/30deg, steer pad),
splits 75% of Normal session time blocks + 75% of Abnormal session time blocks into Train,
and 25% of Normal + 25% of Abnormal time blocks into Test with an embargo gap.

This prevents the 47% 20kph label flip collapse and evaluates true generalizability!
"""

import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(FIXED_DIR, "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")


def run_paired_session_split_evaluation():
    print("=" * 95)
    print(" 🛠 EVALUATING PAIRED-SESSION TIME-BLOCK SPLIT (Eliminating 20kph Label Flip Collapse)")
    print("=" * 95)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] CSV not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    metadata_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X = df[feature_cols].to_numpy()
    y = df["target"].to_numpy()

    tr_indices = []
    te_indices = []

    # Perform 75:5:20 Paired Time-Block Split per scenario & state
    for scen in df["scenario"].unique():
        for st in ["normal", "abnormal"]:
            sub_idx = df[(df["scenario"] == scen) & (df["state"].str.lower() == st)].index.to_numpy()
            n = len(sub_idx)
            if n == 0:
                continue
            n_tr = int(n * 0.75)
            n_gap = max(1, int(n * 0.05)) if n >= 20 else 0
            
            tr_indices.extend(sub_idx[:n_tr])
            te_indices.extend(sub_idx[n_tr + n_gap:])

    tr_idx = np.array(tr_indices)
    te_idx = np.array(te_indices)

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    print(f"\n [PAIRED-SESSION SPLIT SUMMARY]")
    print(f" Train Samples: {len(X_tr)} (Normal 0s = {np.sum(y_tr==0)}, Abnormal 1s = {np.sum(y_tr==1)})")
    print(f" Test Samples : {len(X_te)} (Normal 0s = {np.sum(y_te==0)}, Abnormal 1s = {np.sum(y_te==1)})")

    models = {
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric="logloss"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    print("\n" + "-" * 95)
    print(f" {'Model Name':<20} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} | {'Confusion Matrix'}")
    print("-" * 95)

    for m_name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        acc = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)
        f1 = f1_score(y_te, pred, zero_division=0)
        cm = confusion_matrix(y_te, pred, labels=[0, 1])

        cm_str = f"TN:{cm[0,0]} FP:{cm[0,1]} FN:{cm[1,0]} TP:{cm[1,1]}"
        print(f" {m_name:<20} | {acc*100:8.2f}%  | {prec*100:8.2f}%  | {rec*100:8.2f}%  | {f1*100:8.2f}%  | {cm_str}")

    print("=" * 95)


if __name__ == "__main__":
    run_paired_session_split_evaluation()
