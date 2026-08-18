"""
ablation_window_stride_rnn.py (gearbox_final_suite)

Ablation Study Pipeline:
Evaluates the impact of Window Size (K = 5, 10, 20, 50) and Stride Step (S = 1, 2, 5, 10)
on 2D Feature Matrices (M_features x K_timesteps) fed into PyTorch GRU / RNN & XGBoost models.

Feature Composition (M features):
- Physical Slip Ratios (slip_ratio_fl~rr, slip_diff_front_rear)
- Speed-Normalized FFT Order Features (norm_fft_peak, log_norm_spectral_energy, entropy, centroid)
- Chassis & Motion Features (sensor_1~4, norm_yaw_rate, norm_lat_accel, time-domain stats)

Rule 3 Compliance:
- Code comments & terminal output in Korean.
- Evaluation metrics in English.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")

if not os.path.exists(INPUT_CSV):
    INPUT_CSV = os.path.join(os.path.dirname(SUITE_DIR), "gearbox_fault_diagnosis_fixed", "results", "unified_can_context_dataset.csv")


class GearboxRNNClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32, num_layers=1):
        super(GearboxRNNClassifier, self).__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)


def run_ablation_study():
    print("=" * 95)
    print(" 🔬 ABLATION STUDY: WINDOW SIZE (K) & STRIDE STEP (S) FOR (M x K) MATRIX INPUT")
    print("=" * 95)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] CSV missing: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    metadata_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X_all = df[feature_cols].to_numpy()
    y_all = df["target"].to_numpy()

    # Scale features globally for RNN matrix construction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    window_sizes = [5, 10, 20, 50]
    stride_steps = [1, 2, 5]

    results_table = []

    for K in window_sizes:
        for S in stride_steps:
            # Construct (N_samples, K, M_features) 3D tensors for RNN & flattened 2D for XGBoost
            sequences = []
            targets = []
            scenarios = []

            for scen in df["scenario"].unique():
                scen_mask = (df["scenario"] == scen).to_numpy()
                X_sub = X_scaled[scen_mask]
                y_sub = y_all[scen_mask]

                n_sub = len(X_sub)
                for i in range(0, n_sub - K + 1, S):
                    seq = X_sub[i : i + K]
                    lbl = y_sub[i + K - 1]
                    sequences.append(seq)
                    targets.append(lbl)
                    scenarios.append(scen)

            if len(sequences) == 0:
                continue

            X_seq = np.array(sequences)        # (N, K, M)
            X_flat = X_seq.reshape(len(X_seq), -1)  # (N, K * M)
            y_seq = np.array(targets)
            scen_seq = np.array(scenarios)

            # Paired 75:5:20 Split on Constructed Sequences
            tr_idx, te_idx = [], []
            for sc in np.unique(scen_seq):
                for target_val in [0, 1]:
                    sub_i = np.where((scen_seq == sc) & (y_seq == target_val))[0]
                    n_s = len(sub_i)
                    if n_s == 0:
                        continue
                    n_tr = int(n_s * 0.75)
                    n_gap = max(1, int(n_s * 0.05)) if n_s >= 20 else 0
                    tr_idx.extend(sub_i[:n_tr])
                    te_idx.extend(sub_i[n_tr + n_gap:])

            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue

            tr_idx = np.array(tr_idx)
            te_idx = np.array(te_idx)

            # 1. XGBoost on Flattened (K * M) Matrix
            model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric="logloss")
            model_xgb.fit(X_flat[tr_idx], y_seq[tr_idx])
            pred_xgb = model_xgb.predict(X_flat[te_idx])
            acc_xgb = accuracy_score(y_seq[te_idx], pred_xgb)
            f1_xgb = f1_score(y_seq[te_idx], pred_xgb, zero_division=0)

            # 2. PyTorch GRU-RNN on (K, M) Sequence Matrix
            feature_dim = X_seq.shape[2]
            train_ds = TensorDataset(torch.tensor(X_seq[tr_idx], dtype=torch.float32), torch.tensor(y_seq[tr_idx], dtype=torch.long))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

            rnn = GearboxRNNClassifier(feature_dim=feature_dim, hidden_dim=32)
            optimizer = optim.Adam(rnn.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            rnn.train()
            for _ in range(30):
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    out = rnn(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

            rnn.eval()
            with torch.no_grad():
                logits = rnn(torch.tensor(X_seq[te_idx], dtype=torch.float32))
                pred_rnn = torch.argmax(logits, dim=1).numpy()
            
            acc_rnn = accuracy_score(y_seq[te_idx], pred_rnn)
            f1_rnn = f1_score(y_seq[te_idx], pred_rnn, zero_division=0)

            results_table.append({
                "K_window": K,
                "S_stride": S,
                "matrix_shape": f"{feature_dim}x{K}",
                "num_seqs": len(X_seq),
                "xgb_acc": acc_xgb, "xgb_f1": f1_xgb,
                "rnn_acc": acc_rnn, "rnn_f1": f1_rnn
            })

    print("-" * 95)
    print(f" {'Window Size (K)':^16} | {'Stride (S)':^12} | {'Matrix Shape':^15} | {'XGBoost Acc':^14} | {'RNN-GRU Acc':^14} ")
    print("-" * 95)
    for r in results_table:
        print(f" K={r['K_window']:<13} | S={r['S_stride']:<9} | {r['matrix_shape']:^15} | {r['xgb_acc']*100:10.2f}%    | {r['rnn_acc']*100:10.2f}%    ")
    print("=" * 95)

    # Save Ablation JSON
    out_json = os.path.join(RESULTS_DIR, "ablation_window_stride_results.json")
    with open(out_json, "w") as f:
        json.dump(results_table, f, indent=4)
    print(f"\n[SUCCESS] Ablation Study Complete! Results saved to: {out_json}")


if __name__ == "__main__":
    run_ablation_study()
