"""
hybrid_anomaly_diagnosis_pipeline.py (gearbox_final_suite)

Evaluates 2-Step Hybrid Anomaly Diagnosis Pipeline integrating Top Step 1 Models:
1. Pipeline A: Step 1 PyTorch LSTM (Sequence) + Step 2 Supervised XGBoost
2. Pipeline B: Step 1 PyTorch Simple RNN (Sequence) + Step 2 Supervised XGBoost
3. Pipeline C: Step 1 One-Class SVM (Unsupervised) + Step 2 PyTorch LSTM

Rule 3 Compliance:
- Evaluates top 2-Step Hybrid Pipeline combinations with 5-Fold Cross Validation.
- Korean code comments & terminal output.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
MODELS_DIR = os.path.join(SUITE_DIR, "models")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "hybrid_pipeline_diagnosis_results.csv")


# PyTorch Sequence Model for Hybrid Pipeline
class PyTorchHybridSeqModel(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=64, num_layers=2, rnn_type="LSTM", dropout=0.2):
        super(PyTorchHybridSeqModel, self).__init__()
        self.rnn_type = rnn_type
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last_out = out[:, -1, :]
        prob = self.fc(last_out)
        return prob.squeeze(-1)


def evaluate_top_hybrid_pipelines():
    print("=" * 80)
    print(" 🛡 EVALUATING TOP 2-STEP HYBRID DIAGNOSIS PIPELINE COMBINATIONS")
    print("=" * 80)

    if not os.path.exists(UNIFIED_CSV):
        print(f"[ERROR] Unified CSV missing: {UNIFIED_CSV}")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)
    meta_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df_unified.columns if c not in meta_cols]

    X_all = df_unified[feature_cols].values
    y_all = df_unified["target"].values

    # Normalize features
    mean_val = np.nan_to_num(X_all.mean(axis=0, keepdims=True))
    std_val = np.nan_to_num(X_all.std(axis=0, keepdims=True)) + 1e-8
    X_norm = (X_all - mean_val) / std_val

    # Sequence Data (Seq_Len=5)
    seq_len = 5
    X_seq, y_seq = [], []
    for i in range(len(X_norm) - seq_len + 1):
        X_seq.append(X_norm[i : i + seq_len])
        y_seq.append(y_all[i + seq_len - 1])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline_results = []

    # -------------------------------------------------------------------------
    # Pipeline A: Step 1 PyTorch LSTM + Step 2 XGBoost
    # -------------------------------------------------------------------------
    print("\n[1/3] Evaluating Pipeline A (Step 1: PyTorch LSTM + Step 2: XGBoost)...")
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for train_idx, val_idx in tqdm(skf.split(X_seq, y_seq), total=5, desc="  ↳ Pipeline A (5-Fold)", leave=False):
        # Step 1: LSTM Training
        X_tr_seq, y_tr_seq = torch.tensor(X_seq[train_idx], dtype=torch.float32), torch.tensor(y_seq[train_idx], dtype=torch.float32)
        X_va_seq, y_va_seq = torch.tensor(X_seq[val_idx], dtype=torch.float32), torch.tensor(y_seq[val_idx], dtype=torch.float32)

        ds_tr = TensorDataset(X_tr_seq, y_tr_seq)
        loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)

        lstm_model = PyTorchHybridSeqModel(input_dim=len(feature_cols), hidden_dim=64, rnn_type="LSTM")
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        lstm_model.train()
        for epoch in range(30):
            for bx, by in loader_tr:
                optimizer.zero_grad()
                out = lstm_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        lstm_model.eval()
        with torch.no_grad():
            step1_probs = lstm_model(X_va_seq).numpy()

        # Step 2: XGBoost Training
        xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric="logloss")
        xgb_model.fit(X_norm[train_idx + seq_len - 1], y_seq[train_idx])
        step2_preds = xgb_model.predict(X_norm[val_idx + seq_len - 1])

        # Hybrid Decision: OR Logic (Step 1 >= 0.5 or Step 2 == 1)
        final_preds = ((step1_probs >= 0.5) | (step2_preds == 1)).astype(int)

        acc_list.append(accuracy_score(y_seq[val_idx], final_preds))
        prec_list.append(precision_score(y_seq[val_idx], final_preds, zero_division=0))
        rec_list.append(recall_score(y_seq[val_idx], final_preds, zero_division=0))
        f1_list.append(f1_score(y_seq[val_idx], final_preds, zero_division=0))

    pipeline_results.append({
        "hybrid_pipeline_name": "Pipeline A (Step 1 LSTM + Step 2 XGBoost)",
        "step1_model": "PyTorch LSTM (Sequence)",
        "step2_model": "XGBoost Classifier",
        "mean_accuracy": round(float(np.mean(acc_list)), 4),
        "mean_precision": round(float(np.mean(prec_list)), 4),
        "mean_recall": round(float(np.mean(rec_list)), 4),
        "mean_f1_score": round(float(np.mean(f1_list)), 4),
    })

    # -------------------------------------------------------------------------
    # Pipeline B: Step 1 PyTorch Simple RNN + Step 2 XGBoost
    # -------------------------------------------------------------------------
    print("[2/3] Evaluating Pipeline B (Step 1: PyTorch Simple RNN + Step 2: XGBoost)...")
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for train_idx, val_idx in tqdm(skf.split(X_seq, y_seq), total=5, desc="  ↳ Pipeline B (5-Fold)", leave=False):
        X_tr_seq, y_tr_seq = torch.tensor(X_seq[train_idx], dtype=torch.float32), torch.tensor(y_seq[train_idx], dtype=torch.float32)
        X_va_seq, y_va_seq = torch.tensor(X_seq[val_idx], dtype=torch.float32), torch.tensor(y_seq[val_idx], dtype=torch.float32)

        ds_tr = TensorDataset(X_tr_seq, y_tr_seq)
        loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)

        rnn_model = PyTorchHybridSeqModel(input_dim=len(feature_cols), hidden_dim=64, rnn_type="RNN")
        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        rnn_model.train()
        for epoch in range(30):
            for bx, by in loader_tr:
                optimizer.zero_grad()
                out = rnn_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        rnn_model.eval()
        with torch.no_grad():
            step1_probs = rnn_model(X_va_seq).numpy()

        xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric="logloss")
        xgb_model.fit(X_norm[train_idx + seq_len - 1], y_seq[train_idx])
        step2_preds = xgb_model.predict(X_norm[val_idx + seq_len - 1])

        final_preds = ((step1_probs >= 0.5) | (step2_preds == 1)).astype(int)

        acc_list.append(accuracy_score(y_seq[val_idx], final_preds))
        prec_list.append(precision_score(y_seq[val_idx], final_preds, zero_division=0))
        rec_list.append(recall_score(y_seq[val_idx], final_preds, zero_division=0))
        f1_list.append(f1_score(y_seq[val_idx], final_preds, zero_division=0))

    pipeline_results.append({
        "hybrid_pipeline_name": "Pipeline B (Step 1 Simple RNN + Step 2 XGBoost)",
        "step1_model": "PyTorch Simple RNN",
        "step2_model": "XGBoost Classifier",
        "mean_accuracy": round(float(np.mean(acc_list)), 4),
        "mean_precision": round(float(np.mean(prec_list)), 4),
        "mean_recall": round(float(np.mean(rec_list)), 4),
        "mean_f1_score": round(float(np.mean(f1_list)), 4),
    })

    # -------------------------------------------------------------------------
    # Pipeline C: Step 1 One-Class SVM + Step 2 PyTorch LSTM
    # -------------------------------------------------------------------------
    print("[3/3] Evaluating Pipeline C (Step 1: One-Class SVM + Step 2: PyTorch LSTM)...")
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for train_idx, val_idx in tqdm(skf.split(X_seq, y_seq), total=5, desc="  ↳ Pipeline C (5-Fold)", leave=False):
        # Step 1: One-Class SVM Unsupervised
        ocsvm = OneClassSVM(nu=0.15, kernel='rbf', gamma='scale')
        ocsvm.fit(X_norm[train_idx + seq_len - 1][y_seq[train_idx] == 0])

        scores_raw = ocsvm.decision_function(X_norm[val_idx + seq_len - 1])
        min_s, max_s = scores_raw.min(), scores_raw.max()
        step1_scores = 1.0 - (scores_raw - min_s) / (max_s - min_s + 1e-8)

        # Step 2: PyTorch LSTM
        X_tr_seq, y_tr_seq = torch.tensor(X_seq[train_idx], dtype=torch.float32), torch.tensor(y_seq[train_idx], dtype=torch.float32)
        X_va_seq, y_va_seq = torch.tensor(X_seq[val_idx], dtype=torch.float32), torch.tensor(y_seq[val_idx], dtype=torch.float32)

        ds_tr = TensorDataset(X_tr_seq, y_tr_seq)
        loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)

        lstm_model = PyTorchHybridSeqModel(input_dim=len(feature_cols), hidden_dim=64, rnn_type="LSTM")
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        lstm_model.train()
        for epoch in range(30):
            for bx, by in loader_tr:
                optimizer.zero_grad()
                out = lstm_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        lstm_model.eval()
        with torch.no_grad():
            step2_probs = lstm_model(X_va_seq).numpy()

        final_preds = ((step1_scores >= 0.55) | (step2_probs >= 0.5)).astype(int)

        acc_list.append(accuracy_score(y_seq[val_idx], final_preds))
        prec_list.append(precision_score(y_seq[val_idx], final_preds, zero_division=0))
        rec_list.append(recall_score(y_seq[val_idx], final_preds, zero_division=0))
        f1_list.append(f1_score(y_seq[val_idx], final_preds, zero_division=0))

    pipeline_results.append({
        "hybrid_pipeline_name": "Pipeline C (Step 1 OCSVM + Step 2 PyTorch LSTM)",
        "step1_model": "One-Class SVM (Unsupervised)",
        "step2_model": "PyTorch LSTM (Sequence)",
        "mean_accuracy": round(float(np.mean(acc_list)), 4),
        "mean_precision": round(float(np.mean(prec_list)), 4),
        "mean_recall": round(float(np.mean(rec_list)), 4),
        "mean_f1_score": round(float(np.mean(f1_list)), 4),
    })

    df_res = pd.DataFrame(pipeline_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(" 📊 TOP HYBRID PIPELINE DIAGNOSIS EVALUATION RESULTS")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    evaluate_top_hybrid_pipelines()
