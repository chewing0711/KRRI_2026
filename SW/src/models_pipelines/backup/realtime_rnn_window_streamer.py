"""
realtime_rnn_window_streamer.py (gearbox_final_suite / models_pipelines)

Strict Verification Streamer with Random Seed Variance & 5-Fold Fold-by-Fold Breakdown:
1. Supports configurable --random_state (e.g., 42, 123, 777) for data split verification.
2. Supports 5-Fold Cross Validation Fold-by-Fold breakdown table showing individual fold scores (Fold 1~5).
3. Configurable Test Ratio (--test_size).

Rule 3 & Rule 4 Compliance:
- Strict scaling calculation on X_tab_raw[train_idx].
- Korean code comments & formatted terminal output.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

MODELS_PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MODELS_PIPELINES_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")

CAN_FREQ = 340.14  # Hz
WINDOW_SAMPLES = 500


class FullFeatureRNNAnomalyDetector(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=64, num_layers=2, rnn_type="LSTM", dropout=0.2):
        super(FullFeatureRNNAnomalyDetector, self).__init__()
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


def run_verification_streaming(stride=100, test_size=0.40, random_state=42, eval_mode="holdout", rnn_type="LSTM"):
    print("=" * 80)
    print(f" 🛡️ VERIFICATION STREAMER (RANDOM_SEED={random_state}, TEST_SIZE={int(test_size*100)}%, MODE={eval_mode.upper()})")
    print("=" * 80)

    if not os.path.exists(UNIFIED_CSV):
        print(f"[ERROR] Unified CSV missing: {UNIFIED_CSV}")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)
    meta_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df_unified.columns if c not in meta_cols]

    X_raw = df_unified[feature_cols].values
    y_raw = df_unified["target"].values

    seq_len = 5
    idx_step = max(1, int(round(stride / WINDOW_SAMPLES)))

    X_seq_raw, y_seq_raw, session_ids, scenarios, X_tab_raw = [], [], [], [], []
    for i in range(0, len(X_raw) - seq_len + 1, idx_step):
        X_seq_raw.append(X_raw[i : i + seq_len])
        y_seq_raw.append(y_raw[i + seq_len - 1])
        X_tab_raw.append(X_raw[i + seq_len - 1])
        session_ids.append(df_unified.loc[i + seq_len - 1, "session_id"])
        scenarios.append(df_unified.loc[i + seq_len - 1, "scenario"])

    X_seq_raw = np.array(X_seq_raw)
    y_seq_raw = np.array(y_seq_raw)
    X_tab_raw = np.array(X_tab_raw)

    if eval_mode == "cv":
        print("\n 🔍 EVALUATING 5-FOLD CROSS VALIDATION FOLD-BY-FOLD BREAKDOWN...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        fold_records = []

        for fold_i, (idx_tr, idx_te) in enumerate(skf.split(X_seq_raw, y_seq_raw), 1):
            mean_tr = np.nan_to_num(X_tab_raw[idx_tr].mean(axis=0, keepdims=True))
            std_tr = np.nan_to_num(X_tab_raw[idx_tr].std(axis=0, keepdims=True)) + 1e-8

            X_seq_tr = (X_seq_raw[idx_tr] - mean_tr) / std_tr
            X_seq_te = (X_seq_raw[idx_te] - mean_tr) / std_tr

            X_tab_tr = (X_tab_raw[idx_tr] - mean_tr) / std_tr
            X_tab_te = (X_tab_raw[idx_te] - mean_tr) / std_tr

            y_tr, y_te = y_seq_raw[idx_tr], y_seq_raw[idx_te]

            # Step 1 PyTorch LSTM
            model_step1 = FullFeatureRNNAnomalyDetector(input_dim=len(feature_cols), hidden_dim=64, rnn_type=rnn_type)
            optimizer = torch.optim.Adam(model_step1.parameters(), lr=1e-3)
            criterion = nn.BCELoss()

            ds_tr = TensorDataset(torch.tensor(X_seq_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
            loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)

            model_step1.train()
            for _ in range(40):
                for bx, by in loader_tr:
                    optimizer.zero_grad()
                    out = model_step1(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
            model_step1.eval()

            # Step 2 XGBoost
            model_step2 = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=random_state, eval_metric="logloss")
            model_step2.fit(X_tab_tr, y_tr)

            with torch.no_grad():
                prob_step1 = model_step1(torch.tensor(X_seq_te, dtype=torch.float32)).numpy()
            pred_step2 = model_step2.predict(X_tab_te)

            final_preds = ((prob_step1 >= 0.5) | (pred_step2 == 1)).astype(int)

            f_acc = accuracy_score(y_te, final_preds)
            f_prec = precision_score(y_te, final_preds, zero_division=0)
            f_rec = recall_score(y_te, final_preds, zero_division=0)
            f_f1 = f1_score(y_te, final_preds, zero_division=0)

            fold_records.append({
                "fold_name": f"Fold {fold_i}",
                "train_windows": len(idx_tr),
                "test_windows": len(idx_te),
                "accuracy": round(f_acc, 4),
                "precision": round(f_prec, 4),
                "recall": round(f_rec, 4),
                "f1_score": round(f_f1, 4),
            })

        df_fold = pd.DataFrame(fold_records)
        print("\n" + "=" * 80)
        print(f" 📊 5-FOLD CROSS VALIDATION BREAKDOWN (SEED={random_state})")
        print("=" * 80)
        print(df_fold.to_string(index=False))
        print("-" * 80)
        print(f" Mean Accuracy : {df_fold['accuracy'].mean():.4f}")
        print(f" Mean Precision: {df_fold['precision'].mean():.4f}")
        print(f" Mean Recall   : {df_fold['recall'].mean():.4f}")
        print(f" Mean F1-Score : {df_fold['f1_score'].mean():.4f}")
        print("=" * 80)
        return

    # Holdout Mode
    indices = np.arange(len(X_seq_raw))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=y_seq_raw)

    mean_tr = np.nan_to_num(X_tab_raw[train_idx].mean(axis=0, keepdims=True))
    std_tr = np.nan_to_num(X_tab_raw[train_idx].std(axis=0, keepdims=True)) + 1e-8

    X_seq_tr = (X_seq_raw[train_idx] - mean_tr) / std_tr
    X_seq_te = (X_seq_raw[test_idx] - mean_tr) / std_tr

    X_tab_tr = (X_tab_raw[train_idx] - mean_tr) / std_tr
    X_tab_te = (X_tab_raw[test_idx] - mean_tr) / std_tr

    y_tr, y_te = y_seq_raw[train_idx], y_seq_raw[test_idx]

    sensor_update_sec = round(stride / CAN_FREQ, 3)
    sensor_stream_fps = round(CAN_FREQ / stride, 2)
    overlap_pct = round(max(0.0, (1.0 - (stride / float(WINDOW_SAMPLES))) * 100.0), 1)

    print(f" - Train Set Size         : {len(train_idx)} Windows ({int((1-test_size)*100)}%)")
    print(f" - Unseen Test Set Size   : {len(test_idx)} Windows ({int(test_size*100)}%)")
    print(f" - Random Seed            : {random_state}")
    print(f" - Stride Setting         : Stride {stride} (Overlap {overlap_pct}%)")
    print("=" * 80)

    print("\n[1/2] Training Step 1 PyTorch LSTM (40 Epochs)...")
    model_step1 = FullFeatureRNNAnomalyDetector(input_dim=len(feature_cols), hidden_dim=64, rnn_type=rnn_type)
    optimizer = torch.optim.Adam(model_step1.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    ds_tr = TensorDataset(torch.tensor(X_seq_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)

    model_step1.train()
    for _ in range(40):
        for bx, by in loader_tr:
            optimizer.zero_grad()
            out = model_step1(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    model_step1.eval()

    print("[2/2] Training Step 2 XGBoost Classifier...")
    model_step2 = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=random_state, eval_metric="logloss")
    model_step2.fit(X_tab_tr, y_tr)

    print(f"\n 🚀 STREAMING REAL-TIME DIAGNOSIS ON UNSEEN TEST SET ({len(test_idx)} Windows)...")
    stream_records = []
    total_start_time = time.perf_counter()

    for i_idx, te_i in enumerate(test_idx):
        seq_input = torch.tensor(X_seq_te[i_idx : i_idx + 1], dtype=torch.float32)
        tab_input = X_tab_te[i_idx : i_idx + 1]

        t0 = time.perf_counter()
        with torch.no_grad():
            prob_step1 = model_step1(seq_input).item()
        pred_step2 = model_step2.predict(tab_input)[0]

        final_hybrid_pred = 1 if (prob_step1 >= 0.5 or pred_step2 == 1) else 0
        t1 = time.perf_counter()

        single_lat_ms = (t1 - t0) * 1000.0
        single_lat_us = single_lat_ms * 1000.0
        empirical_fps = round(1000.0 / single_lat_ms, 1) if single_lat_ms > 0 else 0.0

        true_label = int(y_te[i_idx])
        simulated_scenario_time_sec = round(i_idx * sensor_update_sec, 3)

        stream_records.append({
            "stream_step": i_idx,
            "raw_dataset_window_idx": te_i,
            "simulated_scenario_time_sec": simulated_scenario_time_sec,
            "stride_setting": f"Stride {stride}",
            "random_seed": random_state,
            "session_id": session_ids[te_i],
            "scenario": scenarios[te_i],
            "true_ground_truth": true_label,
            "step1_anomaly_probability": round(prob_step1, 6),
            "step2_xgboost_pred": int(pred_step2),
            "final_hybrid_pred": final_hybrid_pred,
            "prediction_match": (final_hybrid_pred == true_label),
            "single_window_lat_ms": round(single_lat_ms, 4),
            "single_window_lat_us": round(single_lat_us, 1),
            "empirical_model_fps": empirical_fps,
        })

    total_end_time = time.perf_counter()
    total_duration_sec = total_end_time - total_start_time

    df_stream = pd.DataFrame(stream_records)

    test_acc = accuracy_score(y_te, df_stream["final_hybrid_pred"])
    test_prec = precision_score(y_te, df_stream["final_hybrid_pred"], zero_division=0)
    test_rec = recall_score(y_te, df_stream["final_hybrid_pred"], zero_division=0)
    test_f1 = f1_score(y_te, df_stream["final_hybrid_pred"], zero_division=0)

    output_predictions_path = os.path.join(RESULTS_DIR, f"realtime_stream_seed_{random_state}_predictions.csv")
    df_stream.to_csv(output_predictions_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f" 📊 HOLD-OUT TEST STREAMING COMPLETED (SEED={random_state}, {len(df_stream)} Windows Evaluated)")
    print("=" * 80)
    print(f" - Total Model Inference Execution Time : {total_duration_sec:.4f} sec ({total_duration_sec*1000:.2f} ms)")
    print(f" - Average Per-Window Latency           : {df_stream['single_window_lat_ms'].mean():.4f} ms ({df_stream['single_window_lat_us'].mean():.1f} us)")
    print(f" - Average Single-Window Model FPS      : {df_stream['empirical_model_fps'].mean():.1f} FPS")
    print(f" - 🛡️ Test Accuracy   (Seed {random_state}): {test_acc*100:.2f}%")
    print(f" - 🛡️ Test Precision  (Seed {random_state}): {test_prec:.4f}")
    print(f" - 🛡️ Test Recall     (Seed {random_state}): {test_rec:.4f}")
    print(f" - 🛡️ Test F1-Score   (Seed {random_state}): {test_f1:.4f}")
    print(f" [SUCCESS] Saved Predictions CSV       : {output_predictions_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=100, choices=[500, 250, 100, 50, 10], help="Stride sample step")
    parser.add_argument("--test_size", type=float, default=0.40, help="Test set ratio (default 0.40)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for data split")
    parser.add_argument("--eval_mode", type=str, default="holdout", choices=["holdout", "cv"], help="Evaluation mode: holdout or cv")
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["LSTM", "RNN"], help="Model type")
    args = parser.parse_args()

    run_verification_streaming(
        stride=args.stride, test_size=args.test_size, random_state=args.random_state,
        eval_mode=args.eval_mode, rnn_type=args.rnn_type
    )
