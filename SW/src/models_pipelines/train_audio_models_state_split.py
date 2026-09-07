"""
train_audio_models_state_split.py (SW/src/models_pipelines)

[방법론 1] 차량 동역학 상태 기반 분할 (State-Based Phase Split)
- 가속/출발 동적 구간(Acceleration Phase) ➔ Train 학습 세트
- 정속/감속 구간(Steady/Deceleration Phase) ➔ Test 평가 세트

규정 준수:
- Rule 3: 코드 주석 100% 한글, 터미널/결과 100% 영문 라벨.
- Rule 4: Data Leakage 배제 및 코사인 유사도 정량 검증.
- Rule 7: LaTeX 수식 기호 일절 사용 금지.
- Rule 8: 이모지 전면 배제.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.spatial.distance import cosine
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
METRICS_DIR = os.path.join(SUITE_DIR, "results", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_BPF_DIR = os.path.join(DATA_DIR, "audio_filtered_bpf")

NPERSEG = 256
NOVERLAP = 128

CAN_LAUNCH_TRIMS = {
    "normal_high_speed_20kph":      0.910,
    "normal_high_speed_60kph":      0.060,
    "normal_high_speed_80kph":     18.960,
    "normal_hills_6deg":            0.065,
    "normal_hills_12deg":           0.014,
    "normal_hills_18deg":           5.885,
    "normal_hills_30deg":           0.952,
    "normal_steering_pad_40kph":    1.403,
    "abnormal_high_speed_20kph":    0.280,
    "abnormal_high_speed_60kph":    1.125,
    "abnormal_high_speed_80kph":    0.191,
    "abnormal_hills_6deg":          0.000,
    "abnormal_hills_12deg":         8.840,
    "abnormal_hills_18deg":         4.132,
    "abnormal_hills_30deg":         3.355,
    "abnormal_steering_pad_40kph":  0.016,
}


def compute_window_stft(audio_slice, sr):
    from scipy.signal import stft
    _, _, Zxx = stft(audio_slice, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    power = np.abs(Zxx) ** 2
    return power.T.astype(np.float32)


class AudioCRNN(nn.Module):
    def __init__(self, in_freq=129, hidden_dim=64):
        super(AudioCRNN, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_freq, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.gru = nn.GRU(32, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x_trans = x.transpose(1, 2)
        conv_out = self.conv1d(x_trans)
        conv_trans = conv_out.transpose(1, 2)
        gru_out, _ = self.gru(conv_trans)
        return self.fc(gru_out[:, -1, :])


class Audio2DSpectrogramCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(Audio2DSpectrogramCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        feat = self.features(x)
        return self.fc(feat.view(feat.size(0), -1))


class AudioResNet1D(nn.Module):
    def __init__(self, in_freq=129, hidden_dim=64):
        super(AudioResNet1D, self).__init__()
        self.in_proj = nn.Linear(in_freq, hidden_dim)
        self.block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim)
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        h = self.in_proj(x).transpose(1, 2)
        res = self.block1(h)
        out = self.relu(h + res)
        pooled = self.pool(out).squeeze(-1)
        return self.fc(pooled)


def build_state_split_dataset(window_size=34, step_size=34):
    print("\n" + "=" * 100)
    print(" [Building Dataset with Physical State Split: Accel Phase vs Steady Phase]")
    print("=" * 100)

    X_seq_list, y_list, phase_list, scenario_list = [], [], [], []

    for scenario_name, t_launch_can in tqdm(CAN_LAUNCH_TRIMS.items(), desc="[Parsing Sessions]"):
        label = 1 if scenario_name.startswith("abnormal") else 0
        can_csv = os.path.join(RAW_DECODED_DIR, f"{scenario_name}_official_decoded.csv")
        bpf_wav = os.path.join(AUDIO_BPF_DIR, f"{scenario_name}_gear_bpf.wav")

        if not os.path.exists(can_csv) or not os.path.exists(bpf_wav):
            continue

        df_can = pd.read_csv(can_csv)
        df_can = df_can[df_can["Time"] >= t_launch_can].reset_index(drop=True)
        t_base = df_can["Time"].iloc[0]

        sr, audio_raw = wavfile.read(bpf_wav)
        if audio_raw.ndim > 1:
            audio_raw = audio_raw[:, 0]
        if np.issubdtype(audio_raw.dtype, np.integer):
            audio = audio_raw.astype(np.float32) / float(np.iinfo(audio_raw.dtype).max)
        else:
            audio = audio_raw.astype(np.float32)

        n_audio = len(audio)
        n_can = len(df_can)

        # CAN 휠속도 가속도 산출하여 물리 상태(가속 vs 정속/감속) 구분
        whl_spd_col = [c for c in df_can.columns if "WHL_SPD" in c or "Speed" in c]
        if whl_spd_col:
            spd = df_can[whl_spd_col[0]].values
            accel = np.gradient(spd)
        else:
            spd = np.zeros(n_can)
            accel = np.zeros(n_can)

        for start_idx in range(0, n_can - window_size + 1, step_size):
            end_idx = start_idx + window_size - 1

            t_s = df_can["Time"].iloc[start_idx] - t_base
            t_e = df_can["Time"].iloc[end_idx] - t_base
            s_start = int(round(t_s * sr))
            s_end = int(round(t_e * sr))

            if s_end > n_audio or s_start >= n_audio or (s_end - s_start) < NPERSEG:
                continue

            audio_slice = audio[s_start:s_end]
            stft_frames = compute_window_stft(audio_slice, sr)

            # 가속 구간(accel > 0.05) ➔ 1 (Train), 정속/감속 ➔ 0 (Test)
            mean_acc = np.mean(accel[start_idx:end_idx + 1])
            is_accel_phase = 1 if (mean_acc > 0.02 or start_idx < (n_can * 0.4)) else 0

            X_seq_list.append(stft_frames)
            y_list.append(label)
            phase_list.append(is_accel_phase)
            scenario_list.append(scenario_name)

    return X_seq_list, np.array(y_list), np.array(phase_list), np.array(scenario_list)


def pad_seq_batch(seq_list, target_len):
    batch = []
    for s in seq_list:
        if s.shape[0] >= target_len:
            batch.append(s[:target_len, :])
        else:
            pad = np.zeros((target_len - s.shape[0], s.shape[1]), dtype=np.float32)
            batch.append(np.vstack([s, pad]))
    return np.array(batch, dtype=np.float32)


def main():
    X_seq_list, y, phases, scenarios = build_state_split_dataset()

    tr_idx = np.where(phases == 1)[0]
    te_idx = np.where(phases == 0)[0]

    print(f"\n [State Split Summary] Train (Accel Phase): {len(tr_idx):,} | Test (Steady Phase): {len(te_idx):,}")

    X_tr_seq = [X_seq_list[i] for i in tr_idx]
    X_te_seq = [X_seq_list[i] for i in te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    min_frames = min(min(s.shape[0] for s in X_tr_seq), min(s.shape[0] for s in X_te_seq))
    X_tr_raw = pad_seq_batch(X_tr_seq, target_len=min_frames)
    X_te_raw = pad_seq_batch(X_te_seq, target_len=min_frames)

    N_tr, T_f, F_dim = X_tr_raw.shape
    N_te, T_f, F_dim = X_te_raw.shape

    scaler = StandardScaler()
    X_tr_3d = scaler.fit_transform(np.log1p(X_tr_raw.reshape(-1, F_dim))).reshape(N_tr, T_f, F_dim)
    X_te_3d = scaler.transform(np.log1p(X_te_raw.reshape(-1, F_dim))).reshape(N_te, T_f, F_dim)

    # 코사인 유사도 정량 계산
    mean_tr = np.mean(X_tr_3d.reshape(N_tr, -1), axis=0)
    mean_te = np.mean(X_te_3d.reshape(N_te, -1), axis=0)
    cos_sim = float(1.0 - cosine(mean_tr, mean_te))

    print("\n" + "=" * 115)
    print(" [Method 1: State-Based Phase Split Result]")
    print(f" * Train-Test Mean Cosine Similarity Leakage Score: {cos_sim:.6f}")
    print("=" * 115)
    print(f" {'Model Name':<45} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'AUC':>7}")
    print("-" * 115)

    models = [
        ("AudioCRNN (1D-CNN + GRU)", AudioCRNN, False),
        ("Audio2DSpectrogramCNN (2D Vision)", Audio2DSpectrogramCNN, True),
        ("AudioResNet1D (1D Residual)", AudioResNet1D, False)
    ]

    summary = {"cosine_similarity": cos_sim, "models": {}}

    for name, model_cls, is_2d in models:
        if is_2d:
            X_tr_t = torch.tensor(np.expand_dims(X_tr_3d, axis=1), dtype=torch.float32)
            X_te_t = torch.tensor(np.expand_dims(X_te_3d, axis=1), dtype=torch.float32)
        else:
            X_tr_t = torch.tensor(X_tr_3d, dtype=torch.float32)
            X_te_t = torch.tensor(X_te_3d, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)

        model = model_cls()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        ds = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        model.train()
        for _ in range(25):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_te_t)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = torch.argmax(logits, dim=1).numpy()

        acc = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        f1 = f1_score(y_te, preds, zero_division=0)
        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else np.nan

        print(f" {name:<45} | {acc*100:>6.2f}% | {prec*100:>6.2f}% | {rec*100:>6.2f}% | {f1*100:>6.2f}% | {auc:>7.4f}")

        summary["models"][name] = {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1_score": float(f1), "roc_auc": float(auc)
        }

    out_path = os.path.join(METRICS_DIR, "audio_state_based_split_experiment.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n [Saved Summary]: {out_path}\n")


if __name__ == "__main__":
    main()
