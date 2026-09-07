"""
train_audio_deep_learning_models.py (gearbox_final_suite / src / models_pipelines)

[음향 특화 딥러닝 SOTA 아키텍처 모델 학습 및 벤치마크 평가 스크립트]

1. [AudioCRNN: 1D-CNN + GRU]
   - 1D-CNN 필터로 오디오 STFT 스펙트럼 로컬 펄스 특성을 자동 추출한 후, GRU 시계열 레이어로 기어 회전 주기성을 캡처.

2. [Audio2DSpectrogramCNN: 2D Spectrogram Vision Model]
   - 2D 시간-주파수 스펙트로그램 이미지(Time-Frequency Representation)를 공간 이미지 패치로 포착하는 2D Convolution 신경망.

3. [AudioResNet1D: 1D Residual Network]
   - Skip Connection 구조의 1D 잔차 블록을 이용하여 multi-scale 주파수 변조 패턴을 직접 학습하는 아키텍처.

[평가 프로토콜]
- 전략 1: Causal Chronological Split (75% Train / 20-window Gap / 25% Test 시간 순서 분할)
- 전략 2: Leave-One-Group-Out (5-Fold 시나리오 완전 격리 분할)

규정 준수:
- Rule 3: 코드 주석 100% 한글, 터미널/결과 100% 영문 라벨.
- Rule 4: Data Leakage 배제 및 코사인 유사도 정량 검증.
- Rule 6: 비판적 엔지니어링 자체 비교 검증.
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
from scipy.signal import stft
from scipy.spatial.distance import cosine
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupKFold
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
N_FREQ = NPERSEG // 2 + 1  # 129

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


# =========================================================================
# [음향 전용 SOTA 딥러닝 PyTorch 신경망 정의]
# =========================================================================

class AudioCRNN(nn.Module):
    """1D-CNN + GRU 아키텍처 (로컬 로컬 충격음 피처 추출 + 시계열 궤적 학습)"""
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
        # x: (B, seq_len, freq_dim) -> (B, freq_dim, seq_len)
        x_trans = x.transpose(1, 2)
        conv_out = self.conv1d(x_trans)  # (B, 32, seq_len // 2)
        conv_trans = conv_out.transpose(1, 2)  # (B, seq_len // 2, 32)
        gru_out, _ = self.gru(conv_trans)
        last_step = gru_out[:, -1, :]
        return self.fc(last_step)


class Audio2DSpectrogramCNN(nn.Module):
    """2D Spectrogram Vision Model (시간-주파수 스펙트로그램 2D Spatial Image Encoder)"""
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
        # x: (B, 1, seq_len, freq_dim)
        feat = self.features(x)
        feat_flat = feat.view(feat.size(0), -1)
        return self.fc(feat_flat)


class AudioResNet1D(nn.Module):
    """1D Residual Waveform/Spectral Block Encoder"""
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
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: (B, seq_len, freq_dim)
        h = self.in_proj(x).transpose(1, 2)  # (B, hidden_dim, seq_len)
        res = self.block1(h)
        out = self.relu(h + res)
        out_pooled = torch.mean(out, dim=2)  # Global Average Pooling
        return self.fc(out_pooled)


# =========================================================================
# [데이터 로딩: WAV -> STFT 스펙트럼 배열 구성]
# =========================================================================

def compute_window_stft(audio_slice, sr):
    if len(audio_slice) < NPERSEG:
        audio_slice = np.pad(audio_slice, (0, NPERSEG - len(audio_slice)))
    _, _, Zxx = stft(audio_slice, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    power = np.abs(Zxx) ** 2
    stft_frames = power.T.astype(np.float32)  # (n_frames, N_FREQ)
    return stft_frames


def build_raw_dataset(window_size=34, step_size=34):
    print("\n" + "=" * 100)
    print(f" [Raw Audio Spectrum Extraction] nperseg={NPERSEG}, noverlap={NOVERLAP}")
    print("=" * 100)

    X_seq_list, y_list, session_list, scenario_list = [], [], [], []

    for scenario_name, t_launch_can in tqdm(CAN_LAUNCH_TRIMS.items(), desc="[Extracting STFT Spectra]"):
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

            X_seq_list.append(stft_frames)
            y_list.append(label)
            session_list.append(scenario_name)

            scen_short = scenario_name.replace("normal_", "").replace("abnormal_", "")
            scenario_list.append(scen_short)

    return X_seq_list, np.array(y_list), np.array(session_list), np.array(scenario_list)


def pad_seq_batch(seq_list, target_len):
    batch = []
    for s in seq_list:
        if s.shape[0] >= target_len:
            batch.append(s[:target_len, :])
        else:
            pad = np.zeros((target_len - s.shape[0], s.shape[1]), dtype=np.float32)
            batch.append(np.vstack([s, pad]))
    return np.array(batch, dtype=np.float32)


def calculate_cosine_leakage(X_tr_3d, X_te_3d):
    mean_tr = np.mean(X_tr_3d.reshape(X_tr_3d.shape[0], -1), axis=0)
    mean_te = np.mean(X_te_3d.reshape(X_te_3d.shape[0], -1), axis=0)
    sim = 1.0 - cosine(mean_tr, mean_te)
    return float(sim)


def benchmark_model_latency(model, X_sample_3d, is_2d=False):
    model.eval()
    if is_2d:
        x_single = torch.tensor(np.expand_dims(X_sample_3d[:1], axis=1), dtype=torch.float32)
    else:
        x_single = torch.tensor(X_sample_3d[:1], dtype=torch.float32)

    with torch.no_grad():
        for _ in range(10):
            _ = model(x_single)

    latencies = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model(x_single)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    mean_lat = float(np.mean(latencies))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    return mean_lat, fps


# =========================================================================
# [학습 및 평가 전용 함수]
# =========================================================================

def train_eval_model(model_cls, X_tr, y_tr, X_te, y_te, model_name, epochs=25, batch_size=32, is_2d=False):
    if is_2d:
        X_tr_t = torch.tensor(np.expand_dims(X_tr, axis=1), dtype=torch.float32)
        X_te_t = torch.tensor(np.expand_dims(X_te, axis=1), dtype=torch.float32)
    else:
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)

    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    # 클래스 가중치 계산 (불균형 방지)
    n_0 = np.sum(y_tr == 0)
    n_1 = np.sum(y_tr == 1)
    weights = torch.tensor([1.0, float(n_0) / float(n_1 + 1e-5)], dtype=torch.float32)

    model = model_cls()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    ds = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    pbar_ep = tqdm(range(1, epochs + 1), desc=f"  [{model_name[:28]}]", leave=False)
    for _ in pbar_ep:
        running_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        pbar_ep.set_postfix(loss=f"{running_loss / len(loader):.4f}")

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

    return acc, prec, rec, f1, auc, model


def evaluate_deep_learning_splits(splits, X_seq_list, y, split_name):
    print("\n" + "=" * 125)
    print(f" [Audio Deep Learning Benchmark] Strategy: {split_name}")
    print("=" * 125)

    model_configs = [
        ("AudioCRNN (1D-CNN + GRU Architecture)", AudioCRNN, False),
        ("Audio2DSpectrogramCNN (2D Vision Model)", Audio2DSpectrogramCNN, True),
        ("AudioResNet1D (1D Residual Block)", AudioResNet1D, False),
    ]

    results = {name: {"acc": [], "prec": [], "rec": [], "f1": [], "auc": [], "lat": [], "fps": []} for name, _, _ in model_configs}
    leakage_sims = []

    fold_pbar = tqdm(enumerate(splits, start=1), total=len(splits), desc="[Folds Evaluation]")
    for fold_idx, (tr_idx, te_idx) in fold_pbar:
        print(f"\n --- Fold {fold_idx}/{len(splits)} (Train: {len(tr_idx):,}, Test: {len(te_idx):,}) ---")
        X_tr_seq = [X_seq_list[i] for i in tr_idx]
        X_te_seq = [X_seq_list[i] for i in te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        min_frames = min(min(s.shape[0] for s in X_tr_seq), min(s.shape[0] for s in X_te_seq))
        X_tr_raw = pad_seq_batch(X_tr_seq, target_len=min_frames)
        X_te_raw = pad_seq_batch(X_te_seq, target_len=min_frames)

        # Log 변환 및 주파수 대역별 StandardScaler 정규화 (Train 세트로만 fit)
        N_tr, T_f, F_dim = X_tr_raw.shape
        N_te, T_f, F_dim = X_te_raw.shape

        scaler = StandardScaler()
        X_tr_log = np.log1p(X_tr_raw.reshape(-1, F_dim))
        X_te_log = np.log1p(X_te_raw.reshape(-1, F_dim))

        X_tr_3d = scaler.fit_transform(X_tr_log).reshape(N_tr, T_f, F_dim)
        X_te_3d = scaler.transform(X_te_log).reshape(N_te, T_f, F_dim)

        cos_sim = calculate_cosine_leakage(X_tr_3d, X_te_3d)
        leakage_sims.append(cos_sim)

        for name, model_cls, is_2d in model_configs:
            acc, prec, rec, f1, auc, model_obj = train_eval_model(model_cls, X_tr_3d, y_tr, X_te_3d, y_te, name, epochs=25, is_2d=is_2d)
            lat_ms, fps = benchmark_model_latency(model_obj, X_te_3d, is_2d=is_2d)

            results[name]["acc"].append(acc)
            results[name]["prec"].append(prec)
            results[name]["rec"].append(rec)
            results[name]["f1"].append(f1)
            results[name]["auc"].append(auc)
            results[name]["lat"].append(lat_ms)
            results[name]["fps"].append(fps)

    mean_leakage = float(np.mean(leakage_sims))
    print("\n" + "=" * 125)
    print(f" [Results Summary] Strategy: {split_name}")
    print(f" * Train-Test Mean Cosine Similarity Leakage Score: {mean_leakage:.6f}")
    print("=" * 125)
    print(f" {'Model Name':<48} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'AUC':>7} | {'Latency':>8} | {'FPS':>8}")
    print("-" * 125)
    for name, _, _ in model_configs:
        r = results[name]
        acc_m = np.nanmean(r["acc"]) * 100
        prec_m = np.nanmean(r["prec"]) * 100
        rec_m = np.nanmean(r["rec"]) * 100
        f1_m = np.nanmean(r["f1"]) * 100
        auc_m = np.nanmean(r["auc"])
        lat_m = np.nanmean(r["lat"])
        fps_m = np.nanmean(r["fps"])
        print(f" {name:<48} | {acc_m:>6.2f}% | {prec_m:>6.2f}% | {rec_m:>6.2f}% | {f1_m:>6.2f}% | {auc_m:>7.4f} | {lat_m:>6.2f}ms | {fps_m:>7.1f}")

    return {
        "cosine_similarity_leakage": mean_leakage,
        "models": {
            name: {
                "accuracy": float(np.nanmean(v["acc"])),
                "precision": float(np.nanmean(v["prec"])),
                "recall": float(np.nanmean(v["rec"])),
                "f1_score": float(np.nanmean(v["f1"])),
                "roc_auc": float(np.nanmean(v["auc"])),
                "latency_ms": float(np.nanmean(v["lat"])),
                "throughput_fps": float(np.nanmean(v["fps"])),
            }
            for name, v in results.items()
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Audio Deep Learning SOTA Benchmark")
    parser.add_argument("--split_mode", type=str, default="all", choices=["temporal", "group_kfold", "all"],
                        help="분할 방식: temporal(Causal Temporal Split), group_kfold(Leave-One-Scenario-Out), all(둘 다)")
    args = parser.parse_args()

    X_seq_list, y, sessions, scenarios = build_raw_dataset(window_size=34, step_size=34)
    print(f"\n Extracted Windows: {len(y):,} | Normal: {(y==0).sum():,} | Abnormal: {(y==1).sum():,}")

    summary_out = {}

    # [1-A 방식] 30% Embargo Gap (시간 순서 전후 분할: 60% Train / 30% Gap 차단 / 10% Test)
    TRAIN_RATIO_30 = 0.60
    GAP_RATIO_30 = 0.30
    tr_idx, te_idx = [], []
    for sess in np.unique(sessions):
        idx = np.where(sessions == sess)[0]
        cut_train = int(len(idx) * TRAIN_RATIO_30)
        gap_len = int(len(idx) * GAP_RATIO_30)
        tr_idx.extend(idx[:cut_train])
        te_idx.extend(idx[cut_train + gap_len:])
    temporal_splits_30 = [(np.array(tr_idx), np.array(te_idx))]
    summary_out["1a_causal_temporal_30pct_gap"] = evaluate_deep_learning_splits(
        temporal_splits_30, X_seq_list, y, "1A. Causal Temporal Split (60% Train / 30% Embargo Gap Blocked / 10% Test)"
    )

    # [1-B 방식] 40% Embargo Gap (시간 순서 전후 분할: 50% Train / 40% Gap 차단 / 10% Test)
    TRAIN_RATIO_40 = 0.50
    GAP_RATIO_40 = 0.40
    tr_idx, te_idx = [], []
    for sess in np.unique(sessions):
        idx = np.where(sessions == sess)[0]
        cut_train = int(len(idx) * TRAIN_RATIO_40)
        gap_len = int(len(idx) * GAP_RATIO_40)
        tr_idx.extend(idx[:cut_train])
        te_idx.extend(idx[cut_train + gap_len:])
    temporal_splits_40 = [(np.array(tr_idx), np.array(te_idx))]
    summary_out["1b_causal_temporal_40pct_gap"] = evaluate_deep_learning_splits(
        temporal_splits_40, X_seq_list, y, "1B. Causal Temporal Split (50% Train / 40% Embargo Gap Blocked / 10% Test)"
    )

    # [2번 방식] Leave-One-Group-Out / GroupKFold (사용자 지시로 주석 처리)
    # if args.split_mode in ["group_kfold", "all"]:
    #     gkf = GroupKFold(n_splits=min(5, len(np.unique(scenarios))))
    #     group_splits = list(gkf.split(np.arange(len(y)), y, groups=scenarios))
    #     summary_out["2_leave_one_group_out"] = evaluate_deep_learning_splits(
    #         group_splits, X_seq_list, y, f"2. Leave-One-Group-Out ({len(group_splits)}-Fold Scenario Hold-Out)"
    #     )

    out_path = os.path.join(METRICS_DIR, "audio_deep_learning_experiment_w34.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    print(f"\n [Saved Summary]: {out_path}\n")


if __name__ == "__main__":
    main()
