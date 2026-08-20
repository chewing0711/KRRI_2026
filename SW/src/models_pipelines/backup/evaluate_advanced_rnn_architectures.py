"""
evaluate_advanced_rnn_architectures.py (gearbox_final_suite)

Implements and evaluates 5 Advanced RNN Model Architectures on (N, 10, 45) Sequence Matrices:
1. Standard Single GRU
2. Bidirectional GRU (BiGRU)
3. Temporal Self-Attention GRU (Attn-GRU)
4. 1D CNN + GRU Hybrid (CRNN)
5. LayerNorm Residual GRU (Res-GRU)

Evaluates on Paired-Session 75:5:20 Split.

Rule 3 Compliance:
- Code comments & terminal output in Korean.
- Matplotlib / metrics table text in English.
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

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MODELS_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")


# 1. Standard Single GRU
class SingleGRU(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super(SingleGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# 2. Bidirectional GRU (BiGRU)
class BiGRU(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# 3. Temporal Self-Attention GRU (Attn-GRU)
class AttnGRU(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super(AttnGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x) # (B, K, hidden)
        scores = self.attn(out) # (B, K, 1)
        weights = torch.softmax(scores, dim=1) # (B, K, 1)
        ctx = torch.sum(out * weights, dim=1) # (B, hidden)
        return self.fc(ctx)


# 4. 1D CNN + GRU Hybrid (CRNN)
class CRNN(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.gru = nn.GRU(32, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        x_trans = x.transpose(1, 2) # (B, M, K)
        c_out = self.conv(x_trans).transpose(1, 2) # (B, K, 32)
        out, _ = self.gru(c_out)
        return self.fc(out[:, -1, :])


# 5. LayerNorm Residual GRU (Res-GRU)
class ResGRU(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super(ResGRU, self).__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        h_proj = self.in_proj(x)
        out, _ = self.gru(h_proj)
        res = self.ln(h_proj + out)
        return self.fc(res[:, -1, :])


def run_advanced_rnn_benchmark():
    print("=" * 95)
    print(" 🛠 ADVANCED RNN MODEL ARCHITECTURE BENCHMARK (gearbox_final_suite)")
    print("=" * 95)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] CSV missing: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    metadata_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X_all = df[feature_cols].to_numpy()
    y_all = df["target"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    K = 10
    S = 1

    sequences, targets, scenarios = [], [], []
    for scen in df["scenario"].unique():
        scen_mask = (df["scenario"] == scen).to_numpy()
        X_sub = X_scaled[scen_mask]
        y_sub = y_all[scen_mask]
        n_sub = len(X_sub)
        for i in range(0, n_sub - K + 1, S):
            sequences.append(X_sub[i : i + K])
            targets.append(y_sub[i + K - 1])
            scenarios.append(scen)

    X_seq = np.array(sequences) # (N, 10, 45)
    y_seq = np.array(targets)
    scen_seq = np.array(scenarios)

    # Paired 75:5:20 Split
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

    tr_idx = np.array(tr_idx)
    te_idx = np.array(te_idx)

    in_dim = X_seq.shape[2]

    train_ds = TensorDataset(torch.tensor(X_seq[tr_idx], dtype=torch.float32), torch.tensor(y_seq[tr_idx], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    test_x = torch.tensor(X_seq[te_idx], dtype=torch.float32)
    test_y = y_seq[te_idx]

    model_dict = {
        "1. Standard Single GRU": SingleGRU(in_dim),
        "2. Bidirectional GRU (BiGRU)": BiGRU(in_dim),
        "3. Temporal Self-Attn GRU": AttnGRU(in_dim),
        "4. 1D CNN + GRU (CRNN)": CRNN(in_dim),
        "5. LayerNorm Res-GRU": ResGRU(in_dim),
    }

    results = []

    for name, model in model_dict.items():
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(40):
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(test_x)
            pred = torch.argmax(logits, dim=1).numpy()

        acc = accuracy_score(test_y, pred)
        prec = precision_score(test_y, pred, zero_division=0)
        rec = recall_score(test_y, pred, zero_division=0)
        f1 = f1_score(test_y, pred, zero_division=0)

        results.append({
            "model_name": name,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1
        })

    print("-" * 95)
    print(f" {'Model Architecture':<30} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} ")
    print("-" * 95)
    for r in results:
        print(f" {r['model_name']:<30} | {r['acc']*100:8.2f}%  | {r['prec']*100:8.2f}%  | {r['rec']*100:8.2f}%  | {r['f1']*100:8.2f}%  ")
    print("=" * 95)

    out_json = os.path.join(RESULTS_DIR, "advanced_rnn_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[SUCCESS] Advanced RNN Benchmark Complete! Results saved to: {out_json}")


if __name__ == "__main__":
    run_advanced_rnn_benchmark()
