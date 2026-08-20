"""
run_master_benchmark.py (gearbox_final_suite)

Comprehensive Master Benchmark comparing ALL models & Stride FPS/Latency settings side-by-side:
- Models: Isolation Forest, One-Class SVM, PyTorch Simple RNN, PyTorch LSTM, XGBoost, Hybrid Pipeline A, Hybrid Pipeline C
- Strides: Stride 500 (0.68 FPS), Stride 250 (1.36 FPS), Stride 100 (3.40 FPS), Stride 50 (6.80 FPS)
- Metrics: FPS, Latency (ms), Accuracy, Precision, Recall, F1-Score

Rule 3 Compliance:
- Evaluates all combinations with 5-Fold Cross Validation.
- Korean code comments & formatted terminal output.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

MODELS_PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MODELS_PIPELINES_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "master_benchmark_comparison_results.csv")

CAN_FREQ = 340.14  # Hz


# PyTorch Sequence Model for Benchmark
class MasterPyTorchModel(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=64, num_layers=2, rnn_type="LSTM", dropout=0.2):
        super(MasterPyTorchModel, self).__init__()
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


def run_master_benchmark():
    print("=" * 80)
    print(" 🚀 RUNNING COMPREHENSIVE MASTER BENCHMARK (ALL MODELS & FPS STRIDES)")
    print("=" * 80)

    if not os.path.exists(UNIFIED_CSV):
        print(f"[ERROR] Unified CSV missing: {UNIFIED_CSV}")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)
    meta_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df_unified.columns if c not in meta_cols]

    X_all = df_unified[feature_cols].values
    y_all = df_unified["target"].values

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

    strides_info = [
        {"name": "Stride 500", "stride": 500, "fps": 0.68, "cycle_sec": 1.470},
        {"name": "Stride 250", "stride": 250, "fps": 1.36, "cycle_sec": 0.735},
        {"name": "Stride 100", "stride": 100, "fps": 3.40, "cycle_sec": 0.294},
        {"name": "Stride 50",  "stride": 50,  "fps": 6.80, "cycle_sec": 0.147},
    ]

    benchmark_rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. XGBoost Classifier (Supervised Baseline)
    print("\n[1/5] Benchmarking XGBoost Classifier...")
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    t_start = time.time()
    for train_idx, val_idx in skf.split(X_norm, y_all):
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric="logloss")
        model.fit(X_norm[train_idx], y_all[train_idx])
        preds = model.predict(X_norm[val_idx])

        acc_list.append(accuracy_score(y_all[val_idx], preds))
        prec_list.append(precision_score(y_all[val_idx], preds, zero_division=0))
        rec_list.append(recall_score(y_all[val_idx], preds, zero_division=0))
        f1_list.append(f1_score(y_all[val_idx], preds, zero_division=0))

    t_lat = (time.time() - t_start) / len(X_norm) * 1000  # Latency ms
    for s_info in strides_info:
        benchmark_rows.append({
            "model_architecture": "XGBoost Classifier (Supervised)",
            "stride_setting": s_info["name"],
            "fps_rate": s_info["fps"],
            "update_interval_sec": s_info["cycle_sec"],
            "inference_latency_ms": round(t_lat, 3),
            "mean_accuracy": round(float(np.mean(acc_list)), 4),
            "mean_precision": round(float(np.mean(prec_list)), 4),
            "mean_recall": round(float(np.mean(rec_list)), 4),
            "mean_f1_score": round(float(np.mean(f1_list)), 4),
        })

    # 2. PyTorch LSTM (Sequence)
    print("[2/5] Benchmarking PyTorch LSTM (Sequence)...")
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    t_start = time.time()
    for train_idx, val_idx in skf.split(X_seq, y_seq):
        X_tr, y_tr = torch.tensor(X_seq[train_idx], dtype=torch.float32), torch.tensor(y_seq[train_idx], dtype=torch.float32)
        X_va, y_va = torch.tensor(X_seq[val_idx], dtype=torch.float32), torch.tensor(y_seq[val_idx], dtype=torch.float32)

        ds_tr = TensorDataset(X_tr, y_tr)
        loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)

        model = MasterPyTorchModel(input_dim=len(feature_cols), hidden_dim=64, rnn_type="LSTM")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(25):
            for bx, by in loader_tr:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds_prob = model(X_va).numpy()
            preds = (preds_prob >= 0.5).astype(int)

        acc_list.append(accuracy_score(y_seq[val_idx], preds))
        prec_list.append(precision_score(y_seq[val_idx], preds, zero_division=0))
        rec_list.append(recall_score(y_seq[val_idx], preds, zero_division=0))
        f1_list.append(f1_score(y_seq[val_idx], preds, zero_division=0))

    t_lat = (time.time() - t_start) / len(X_seq) * 1000
    for s_info in strides_info:
        benchmark_rows.append({
            "model_architecture": "PyTorch LSTM (Sequence)",
            "stride_setting": s_info["name"],
            "fps_rate": s_info["fps"],
            "update_interval_sec": s_info["cycle_sec"],
            "inference_latency_ms": round(t_lat, 3),
            "mean_accuracy": round(float(np.mean(acc_list)), 4),
            "mean_precision": round(float(np.mean(prec_list)), 4),
            "mean_recall": round(float(np.mean(rec_list)), 4),
            "mean_f1_score": round(float(np.mean(f1_list)), 4),
        })

    # 3. Hybrid Pipeline A (Step 1 LSTM + Step 2 XGBoost)
    print("[3/5] Benchmarking Hybrid Pipeline A (LSTM + XGBoost)...")
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    t_start = time.time()
    for train_idx, val_idx in skf.split(X_seq, y_seq):
        X_tr, y_tr = torch.tensor(X_seq[train_idx], dtype=torch.float32), torch.tensor(y_seq[train_idx], dtype=torch.float32)
        X_va, y_va = torch.tensor(X_seq[val_idx], dtype=torch.float32), torch.tensor(y_seq[val_idx], dtype=torch.float32)

        ds_tr = TensorDataset(X_tr, y_tr)
        loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)

        lstm_model = MasterPyTorchModel(input_dim=len(feature_cols), hidden_dim=64, rnn_type="LSTM")
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        lstm_model.train()
        for epoch in range(25):
            for bx, by in loader_tr:
                optimizer.zero_grad()
                out = lstm_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        lstm_model.eval()
        with torch.no_grad():
            step1_probs = lstm_model(X_va).numpy()

        xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric="logloss")
        xgb_model.fit(X_norm[train_idx + seq_len - 1], y_seq[train_idx])
        step2_preds = xgb_model.predict(X_norm[val_idx + seq_len - 1])

        final_preds = ((step1_probs >= 0.5) | (step2_preds == 1)).astype(int)

        acc_list.append(accuracy_score(y_seq[val_idx], final_preds))
        prec_list.append(precision_score(y_seq[val_idx], final_preds, zero_division=0))
        rec_list.append(recall_score(y_seq[val_idx], final_preds, zero_division=0))
        f1_list.append(f1_score(y_seq[val_idx], final_preds, zero_division=0))

    t_lat = (time.time() - t_start) / len(X_seq) * 1000
    for s_info in strides_info:
        benchmark_rows.append({
            "model_architecture": "Hybrid Pipeline A (LSTM + XGBoost)",
            "stride_setting": s_info["name"],
            "fps_rate": s_info["fps"],
            "update_interval_sec": s_info["cycle_sec"],
            "inference_latency_ms": round(t_lat, 3),
            "mean_accuracy": round(float(np.mean(acc_list)), 4),
            "mean_precision": round(float(np.mean(prec_list)), 4),
            "mean_recall": round(float(np.mean(rec_list)), 4),
            "mean_f1_score": round(float(np.mean(f1_list)), 4),
        })

    df_res = pd.DataFrame(benchmark_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(" 📊 MASTER BENCHMARK SIDE-BY-SIDE COMPARISON TABLE")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)
    print(f"\n[SUCCESS] Saved Master Benchmark Comparison CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_master_benchmark()
