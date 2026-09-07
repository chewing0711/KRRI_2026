"""
train_audio_models_stepped_noise_test.py (SW/src/models_pipelines)

[방법론 2-확장] 단계별 노이즈 침범 강도 스트레스 테스트 (Stepped SNR Noise Stress Test)
- Clean (무노이즈) ➔ SNR 20dB ➔ SNR 10dB ➔ SNR 5dB ➔ SNR 0dB 5단계 노이즈 주입 평가
- 소음 강도 증가에 따른 모델별 성능 감쇄 곡선 및 로버스트니스 한계점 정량 측정

규정 준수:
- Rule 3: 코드 주석 100% 한글, 터미널/결과 100% 영문 라벨.
- Rule 4: Data Leakage 배제 및 코사인 유사도 정량 검증.
- Rule 7: LaTeX 수식 기호 일절 사용 금지.
- Rule 8: 이모지 전면 배제.
"""

import os
import sys
import json
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


def add_road_noise(audio_slice, snr_db):
    if snr_db is None or snr_db == "clean":
        return audio_slice
    sig_power = np.mean(audio_slice ** 2) + 1e-12
    noise_power = sig_power / (10 ** (float(snr_db) / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), size=audio_slice.shape).astype(np.float32)
    return audio_slice + noise


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


def build_dataset_by_snr(snr_level):
    X_tr_seq, X_te_seq, y_tr_list, y_te_list = [], [], [], []

    for scenario_name, t_launch_can in CAN_LAUNCH_TRIMS.items():
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
        cut = int(n_can * 0.60)

        for start_idx in range(0, n_can - 34 + 1, 34):
            end_idx = start_idx + 33
            t_s = df_can["Time"].iloc[start_idx] - t_base
            t_e = df_can["Time"].iloc[end_idx] - t_base
            s_start = int(round(t_s * sr))
            s_end = int(round(t_e * sr))

            if s_end > n_audio or s_start >= n_audio or (s_end - s_start) < NPERSEG:
                continue

            audio_slice = audio[s_start:s_end]

            if start_idx < cut:
                # Train 세트 (클린 신호)
                X_tr_seq.append(compute_window_stft(audio_slice, sr))
                y_tr_list.append(label)
            elif start_idx >= (cut + 34 * 10):  # Gap 차단
                # Test 세트 (지정된 SNR 노이즈 주입)
                noisy_slice = add_road_noise(audio_slice, snr_level)
                X_te_seq.append(compute_window_stft(noisy_slice, sr))
                y_te_list.append(label)

    return X_tr_seq, X_te_seq, np.array(y_tr_list), np.array(y_te_list)


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
    print("\n" + "=" * 125)
    print(" [Stepped SNR Noise Stress Test Benchmark: Clean ➔ 20dB ➔ 10dB ➔ 5dB ➔ 0dB]")
    print("=" * 125)

    snr_steps = [
        ("Clean (No Noise)", "clean"),
        ("SNR 20 dB (Light Road Noise)", 20.0),
        ("SNR 10 dB (Moderate Wind Noise)", 10.0),
        ("SNR 5 dB (Heavy Road Noise)", 5.0),
        ("SNR 0 dB (Extreme Storm Noise)", 0.0),
    ]

    models = [
        ("AudioCRNN (1D-CNN + GRU)", AudioCRNN),
        ("AudioResNet1D (1D Residual)", AudioResNet1D)
    ]

    master_summary = {}

    for label_desc, snr_val in snr_steps:
        print(f"\n >>> Evaluating Step: {label_desc} <<<")
        X_tr_seq, X_te_seq, y_tr, y_te = build_dataset_by_snr(snr_val)

        min_frames = min(min(s.shape[0] for s in X_tr_seq), min(s.shape[0] for s in X_te_seq))
        X_tr_raw = pad_seq_batch(X_tr_seq, target_len=min_frames)
        X_te_raw = pad_seq_batch(X_te_seq, target_len=min_frames)

        N_tr, T_f, F_dim = X_tr_raw.shape
        N_te, T_f, F_dim = X_te_raw.shape

        scaler = StandardScaler()
        X_tr_3d = scaler.fit_transform(np.log1p(X_tr_raw.reshape(-1, F_dim))).reshape(N_tr, T_f, F_dim)
        X_te_3d = scaler.transform(np.log1p(X_te_raw.reshape(-1, F_dim))).reshape(N_te, T_f, F_dim)

        mean_tr = np.mean(X_tr_3d.reshape(N_tr, -1), axis=0)
        mean_te = np.mean(X_te_3d.reshape(N_te, -1), axis=0)
        cos_sim = float(1.0 - cosine(mean_tr, mean_te))

        print(f" * Train-Test Mean Cosine Similarity Leakage Score: {cos_sim:.6f}")
        print(f" {'Model Name':<38} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'AUC':>7}")
        print("-" * 105)

        step_summary = {"cosine_similarity": cos_sim, "models": {}}

        for name, model_cls in models:
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

            print(f" {name:<38} | {acc*100:>6.2f}% | {prec*100:>6.2f}% | {rec*100:>6.2f}% | {f1*100:>6.2f}% | {auc:>7.4f}")

            step_summary["models"][name] = {
                "accuracy": float(acc), "precision": float(prec),
                "recall": float(rec), "f1_score": float(f1), "roc_auc": float(auc)
            }

        master_summary[label_desc] = step_summary

    out_path = os.path.join(METRICS_DIR, "audio_stepped_noise_experiment.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False)
    print(f"\n [Saved Master Stepped Noise Summary]: {out_path}\n")


if __name__ == "__main__":
    main()
