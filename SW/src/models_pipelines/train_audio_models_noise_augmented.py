"""
train_audio_models_noise_augmented.py (SW/src/models_pipelines)

[노이즈 증강 및 기존 힐베르트 피처 융합 학습 평가 스크립트]
1. 기존 검증 스크립트(build_audio_only_dataset_w34.py)의 extract_audio_window_features 모듈을 직접 import하여 사용.
2. 학습 세트(X_tr)에 노이즈 증강(Noise Augmentation, SNR 20dB~5dB)을 적용하여, 실차 풍절음/노면 소음 하에서도 성능 붕괴 없는 강건한 모델 구현.

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

# 1. 기존 검증 코드 모듈 직접 import (새로 작성하지 않고 기존 코드 100% 재사용)
sys.path.append(SRC_DIR)
from audio_processing.build_audio_only_dataset_w34 import extract_audio_window_features, CAN_LAUNCH_TRIMS

RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_BPF_DIR = os.path.join(DATA_DIR, "audio_filtered_bpf")


def add_road_noise(audio_slice, snr_db):
    """노이즈 증강(Noise Augmentation) 전용 함수"""
    if snr_db is None or snr_db == "clean":
        return audio_slice
    sig_power = np.mean(audio_slice ** 2) + 1e-12
    noise_power = sig_power / (10 ** (float(snr_db) / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), size=audio_slice.shape).astype(np.float32)
    return audio_slice + noise


class AudioFeatureMLP(nn.Module):
    """힐베르트 + 고급 DSP 24차원 피처 입력 전용 노이즈 증강 MLP 신경망"""
    def __init__(self, in_dim=24, hidden_dim=64):
        super(AudioFeatureMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        return self.net(x)


def get_feature_vector(audio_slice, sr, use_hilbert=True):
    feat_dict = extract_audio_window_features(audio_slice, sr)
    if not use_hilbert:
        # 힐베르트 포락선 및 충격파 피처 제외 (OFF)
        filtered_dict = {
            k: v for k, v in feat_dict.items()
            if not k.startswith("audio_hilbert_") and not k.startswith("audio_shockwave_") and k != "audio_env_mod_index"
        }
        return list(filtered_dict.values())
    return list(feat_dict.values())


def build_augmented_dataset(use_hilbert=True):
    """
    extract_audio_window_features()를 호출하여 힐베르트 ON/OFF 모드로 데이터셋 구축.
    학습 세트에 노이즈 증강(Noise Augmentation) 데이터 포함 생성.
    """
    X_tr_list, y_tr_list = [], []
    X_te_clean_list, X_te_noisy_list, y_te_list = [], [], []

    desc_text = f"[Extracting Features (Hilbert={'ON' if use_hilbert else 'OFF'}) & Augmenting]"
    for scenario_name, t_launch_can in tqdm(CAN_LAUNCH_TRIMS.items(), desc=desc_text):
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

            if s_end > n_audio or s_start >= n_audio:
                continue

            audio_slice = audio[s_start:s_end]

            if start_idx < cut:
                # [Train 세트] 클린 피처 + 노이즈 증강(Noise Augmentation) 피처 생성
                feats_clean = get_feature_vector(audio_slice, sr, use_hilbert=use_hilbert)
                X_tr_list.append(feats_clean)
                y_tr_list.append(label)

                # 50% 확률로 노이즈(SNR 15dB~5dB)를 섞어 학습 세트에 노이즈 증강 추가
                if np.random.rand() > 0.5:
                    snr_aug = np.random.choice([15.0, 10.0, 5.0])
                    noisy_tr_slice = add_road_noise(audio_slice, snr_db=snr_aug)
                    feats_aug = get_feature_vector(noisy_tr_slice, sr, use_hilbert=use_hilbert)
                    X_tr_list.append(feats_aug)
                    y_tr_list.append(label)

            elif start_idx >= (cut + 34 * 10):  # 30% Embargo Gap 차단
                # [Test 세트] 클린 피처 및 노이즈(SNR 5dB) 주입 피처 저장
                feats_te_clean = get_feature_vector(audio_slice, sr, use_hilbert=use_hilbert)
                X_te_clean_list.append(feats_te_clean)

                noisy_te_slice = add_road_noise(audio_slice, snr_db=5.0)
                feats_te_noisy = get_feature_vector(noisy_te_slice, sr, use_hilbert=use_hilbert)
                X_te_noisy_list.append(feats_te_noisy)

                y_te_list.append(label)

    return (np.array(X_tr_list, dtype=np.float32), np.array(y_tr_list, dtype=int),
            np.array(X_te_clean_list, dtype=np.float32), np.array(X_te_noisy_list, dtype=np.float32),
            np.array(y_te_list, dtype=int))


def evaluate_single_mode(use_hilbert=True):
    mode_str = "ON" if use_hilbert else "OFF"
    print("\n" + "=" * 115)
    print(f" [Noise Augmented Training Evaluation - Hilbert Feature: {mode_str}]")
    print("=" * 115)

    X_tr, y_tr, X_te_clean, X_te_noisy, y_te = build_augmented_dataset(use_hilbert=use_hilbert)
    print(f" Feature Dim: {X_tr.shape[1]} | Train: {len(y_tr):,} | Test (Clean): {len(y_te):,} | Test (Noisy SNR 5dB): {len(y_te):,}")

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_clean_scaled = scaler.transform(X_te_clean)
    X_te_noisy_scaled = scaler.transform(X_te_noisy)

    # 코사인 유사도 유출 정량 계산
    mean_tr = np.mean(X_tr_scaled, axis=0)
    mean_te_c = np.mean(X_te_clean_scaled, axis=0)
    mean_te_n = np.mean(X_te_noisy_scaled, axis=0)
    sim_clean = float(1.0 - cosine(mean_tr, mean_te_c))
    sim_noisy = float(1.0 - cosine(mean_tr, mean_te_n))

    print(f"\n * Train-Test Cosine Similarity (Clean Test): {sim_clean:.6f}")
    print(f" * Train-Test Cosine Similarity (Noisy SNR 5dB Test): {sim_noisy:.6f}")

    # XGBoost 및 MLP 학습
    import xgboost as xgb
    model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    model_xgb.fit(X_tr_scaled, y_tr)

    model_mlp = AudioFeatureMLP(in_dim=X_tr.shape[1], hidden_dim=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_mlp.parameters(), lr=0.002, weight_decay=1e-4)

    ds = TensorDataset(torch.tensor(X_tr_scaled, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model_mlp.train()
    for _ in range(30):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model_mlp(bx), by)
            loss.backward()
            optimizer.step()

    model_mlp.eval()

    print("\n" + "-" * 115)
    print(f" [Evaluation: Hilbert={mode_str} | Clean Test Set]")
    print("-" * 115)
    print(f" {'Model Name':<45} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'AUC':>7}")
    print("-" * 115)

    for tag, model_obj in [("XGBoost", model_xgb), ("AudioFeatureMLP", model_mlp)]:
        name = f"[Hilbert={mode_str} + Augment] {tag}"
        if tag == "XGBoost":
            preds = model_obj.predict(X_te_clean_scaled)
            probs = model_obj.predict_proba(X_te_clean_scaled)[:, 1]
        else:
            with torch.no_grad():
                logits = model_obj(torch.tensor(X_te_clean_scaled, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1)[:, 1].numpy()
                preds = torch.argmax(logits, dim=1).numpy()

        acc = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        f1 = f1_score(y_te, preds, zero_division=0)
        auc = roc_auc_score(y_te, probs)

        print(f" {name:<45} | {acc*100:>6.2f}% | {prec*100:>6.2f}% | {rec*100:>6.2f}% | {f1*100:>6.2f}% | {auc:>7.4f}")

    print("\n" + "-" * 115)
    print(f" [Evaluation: Hilbert={mode_str} | Noisy SNR 5dB Test Set (Heavy Noise Attack)]")
    print("-" * 115)
    print(f" {'Model Name':<45} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'AUC':>7}")
    print("-" * 115)

    for tag, model_obj in [("XGBoost", model_xgb), ("AudioFeatureMLP", model_mlp)]:
        name = f"[Hilbert={mode_str} + Augment] {tag}"
        if tag == "XGBoost":
            preds = model_obj.predict(X_te_noisy_scaled)
            probs = model_obj.predict_proba(X_te_noisy_scaled)[:, 1]
        else:
            with torch.no_grad():
                logits = model_obj(torch.tensor(X_te_noisy_scaled, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1)[:, 1].numpy()
                preds = torch.argmax(logits, dim=1).numpy()

        acc = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        f1 = f1_score(y_te, preds, zero_division=0)
        auc = roc_auc_score(y_te, probs)

        print(f" {name:<45} | {acc*100:>6.2f}% | {prec*100:>6.2f}% | {rec*100:>6.2f}% | {f1*100:>6.2f}% | {auc:>7.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Noise Augmented Training with Hilbert ON/OFF Switch")
    parser.add_argument("--use_hilbert", type=str, default="both", choices=["on", "off", "both"],
                        help="힐베르트 포락선 피처 사용 여부: on(사용), off(제외), both(둘 다 비교)")
    args = parser.parse_args()

    if args.use_hilbert in ["on", "both"]:
        evaluate_single_mode(use_hilbert=True)
    if args.use_hilbert in ["off", "both"]:
        evaluate_single_mode(use_hilbert=False)


if __name__ == "__main__":
    main()
