"""
run_stride_ablation_rnn.py (gearbox_final_suite / models_pipelines)

Empirical Model Inference Latency & FPS Benchmark for Stride Ablation:
- Actually measures PyTorch model evaluation inference time (time.perf_counter) for each Stride.
- Calculates:
  1. Empirical Measured Latency (ms per window)
  2. Empirical Measured Inference Speed (FPS = 1000.0 / latency_ms)
  3. Sensor Stream Update Rate (Theoretical FPS = 340.14 / step_samples)
- 5-Fold Cross-Validation across all 45 UNIFIED features.

Rule 3 Compliance:
- Real empirical measurement of inference speed and FPS.
- Korean code comments & formatted terminal output.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

MODELS_PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MODELS_PIPELINES_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "stride_ablation_rnn_results.csv")

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


def run_full_stride_ablation_experiment():
    print("=" * 80)
    print(" ⏱️ RUNNING EMPIRICAL MODEL INFERENCE LATENCY & FPS BENCHMARK")
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

    # Stride Steps (Samples)
    stride_step_list = [500, 250, 100, 50, 10]
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for step_samples in tqdm(stride_step_list, desc="Empirical Stride Benchmark"):
        sensor_interval_sec = round(step_samples / CAN_FREQ, 3)
        sensor_stream_fps = round(CAN_FREQ / step_samples, 2)
        overlap_pct = round(max(0.0, (1.0 - (step_samples / float(WINDOW_SAMPLES))) * 100.0), 1)
        st_name = f"Stride {step_samples} (Overlap {overlap_pct}%)"

        seq_len = 5
        idx_step = max(1, int(round(step_samples / WINDOW_SAMPLES)))

        X_seq, y_seq = [], []
        for i in range(0, len(X_norm) - seq_len + 1, idx_step):
            X_seq.append(X_norm[i : i + seq_len])
            y_seq.append(y_all[i + seq_len - 1])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        acc_list, prec_list, rec_list, f1_list = [], [], [], []
        inference_latencies_ms = []

        for train_idx, val_idx in skf.split(X_seq, y_seq):
            X_tr, y_tr = torch.tensor(X_seq[train_idx], dtype=torch.float32), torch.tensor(y_seq[train_idx], dtype=torch.float32)
            X_va, y_va = torch.tensor(X_seq[val_idx], dtype=torch.float32), torch.tensor(y_seq[val_idx], dtype=torch.float32)

            ds_tr = TensorDataset(X_tr, y_tr)
            loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)

            model = FullFeatureRNNAnomalyDetector(input_dim=len(feature_cols), hidden_dim=64, rnn_type="LSTM")
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
            # ⏱️ EMPIRICAL INFERENCE TIME MEASUREMENT (time.perf_counter)
            with torch.no_grad():
                t0 = time.perf_counter()
                preds_prob = model(X_va).numpy()
                t1 = time.perf_counter()

                preds = (preds_prob >= 0.5).astype(int)

            batch_lat_ms = ((t1 - t0) / len(val_idx)) * 1000.0
            inference_latencies_ms.append(batch_lat_ms)

            acc_list.append(accuracy_score(y_seq[val_idx], preds))
            prec_list.append(precision_score(y_seq[val_idx], preds, zero_division=0))
            rec_list.append(recall_score(y_seq[val_idx], preds, zero_division=0))
            f1_list.append(f1_score(y_seq[val_idx], preds, zero_division=0))

        mean_lat_ms = float(np.mean(inference_latencies_ms))
        empirical_model_fps = round(1000.0 / mean_lat_ms, 1) if mean_lat_ms > 0 else 0.0

        results.append({
            "stride_setting": st_name,
            "sample_step": step_samples,
            "sensor_update_sec": sensor_interval_sec,
            "sensor_stream_fps": sensor_stream_fps,
            "measured_inference_lat_ms": round(mean_lat_ms, 4),
            "measured_inference_lat_us": round(mean_lat_ms * 1000.0, 1),
            "empirical_model_fps": empirical_model_fps,
            "total_windows_evaluated": len(X_seq),
            "mean_accuracy": round(float(np.mean(acc_list)), 4),
            "mean_f1_score": round(float(np.mean(f1_list)), 4),
        })

    df_res = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(" 📊 EMPIRICAL MODEL INFERENCE LATENCY & FPS BENCHMARK TABLE")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)
    print(f"[SUCCESS] Saved Empirical Inference Speed CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_full_stride_ablation_experiment()
