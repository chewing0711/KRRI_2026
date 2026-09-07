"""
train_audio_raw_wav_models.py (gearbox_final_suite / src / models_pipelines)

피쳐 엔지니어링 없이 WAV 원본 스펙트럼으로 이상 탐지 실험
(No Feature Engineering - Raw STFT Spectrum Baseline)

입력: audio_filtered_bpf/ WAV (BPF 필터링 완료, 44.1 kHz)
표현 방식:
  - 전통적 ML (XGBoost, OC-SVM, iForest): STFT 파워 스펙트럼 시간 평균 (129 주파수 빈)
  - 신경망 (GRU): STFT 프레임 시퀀스 [n_frames, 129] 그대로 입력

비교 목적: train_audio_only_models.py (14개 핸드크래프트 피쳐) 대비 원본 스펙트럼 성능 비교
평가 방식: 시나리오 내 시간축 분할 (앞 75% Train / gap 20 윈도우 / 나머지 Test)

규정 준수:
- Rule 3: 코드 주석 100% 한글, 터미널 100% 영문 라벨.
- Rule 4: Data Leakage 배제 (Train fold만 fit).
- Rule 7: LaTeX 수식 기호 일절 사용 금지.
- Rule 8: 이모지 전면 배제.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR  = os.path.join(SUITE_DIR, "data")
METRICS_DIR = os.path.join(SUITE_DIR, "results", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

RAW_DECODED_DIR     = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_BPF_DIR       = os.path.join(DATA_DIR, "audio_filtered_bpf")

# STFT 파라미터: nperseg=256 → 129 주파수 빈, hop=128 → 0.1s 윈도우에서 ~32 프레임
NPERSEG   = 256
NOVERLAP  = 128
N_FREQ    = NPERSEG // 2 + 1   # 129

# 시나리오별 CAN 출발 트리밍 오프셋 (오디오 슬라이싱 시간 기준 정렬용)
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
# [데이터 로딩: WAV -> STFT 스펙트럼 배열 구성]
# =========================================================================
def compute_window_stft(audio_slice, sr):
    """
    단일 오디오 윈도우 -> STFT 파워 스펙트럼
    반환:
      mean_spec  : (N_FREQ,)         시간 평균 파워 스펙트럼 (전통 ML 입력)
      stft_frames: (n_frames, N_FREQ) 프레임별 스펙트럼  (GRU 입력)
    """
    if len(audio_slice) < NPERSEG:
        # 슬라이스가 너무 짧으면 0 패딩
        audio_slice = np.pad(audio_slice, (0, NPERSEG - len(audio_slice)))

    _, _, Zxx = stft(audio_slice, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    power = np.abs(Zxx) ** 2           # (N_FREQ, n_frames)
    mean_spec   = np.mean(power, axis=1).astype(np.float32)  # (N_FREQ,)
    stft_frames = power.T.astype(np.float32)                 # (n_frames, N_FREQ)
    return mean_spec, stft_frames


def build_raw_dataset(window_size=34, step_size=34):
    """
    CAN 타임스탬프로 윈도우 경계 계산 후 WAV에서 STFT 직접 추출.
    반환:
      X_flat     : (N, N_FREQ)          전통 ML용 평균 스펙트럼
      X_seq_list : list of (n_f, N_FREQ) GRU용 프레임 시퀀스 (가변 길이)
      y          : (N,) int             레이블
      scenarios  : (N,) str             시나리오명
    """
    print("\n" + "=" * 100)
    print(" [Raw WAV STFT Extraction]  nperseg={}, noverlap={}".format(NPERSEG, NOVERLAP))
    print("=" * 100)
    print(f" {'Scenario':<35} | {'Windows':>8} | {'Valid':>7} | {'Drop':>6}")
    print("-" * 100)

    X_flat_list, X_seq_list, y_list, scen_list = [], [], [], []
    total_can, total_valid, total_drop = 0, 0, 0

    for scenario_name, t_launch_can in CAN_LAUNCH_TRIMS.items():
        label = 1 if scenario_name.startswith("abnormal") else 0

        can_csv = os.path.join(RAW_DECODED_DIR, f"{scenario_name}_official_decoded.csv")
        bpf_wav = os.path.join(AUDIO_BPF_DIR,   f"{scenario_name}_gear_bpf.wav")

        if not os.path.exists(can_csv) or not os.path.exists(bpf_wav):
            print(f" {scenario_name:<35} | [SKIP] 파일 없음")
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
        n_can   = len(df_can)
        scen_can, scen_valid, scen_drop = 0, 0, 0

        for start_idx in range(0, n_can - window_size + 1, step_size):
            scen_can += 1
            end_idx = start_idx + window_size - 1

            t_s = df_can["Time"].iloc[start_idx] - t_base
            t_e = df_can["Time"].iloc[end_idx]   - t_base
            s_start = int(round(t_s * sr))
            s_end   = int(round(t_e * sr))

            if s_end > n_audio or s_start >= n_audio:
                scen_drop += 1
                continue

            audio_slice = audio[s_start:s_end]
            mean_spec, stft_frames = compute_window_stft(audio_slice, sr)

            X_flat_list.append(mean_spec)
            X_seq_list.append(stft_frames)
            y_list.append(label)
            scen_list.append(scenario_name)
            scen_valid += 1

        total_can   += scen_can
        total_valid += scen_valid
        total_drop  += scen_drop
        print(f" {scenario_name:<35} | {scen_can:>8} | {scen_valid:>7} | {scen_drop:>6}")

    print("=" * 100)
    print(f" Total: {total_can} windows | Valid: {total_valid} ({total_valid/total_can*100:.2f}%) | Drop: {total_drop}")

    X_flat = np.stack(X_flat_list, axis=0)      # (N, 129)
    y      = np.array(y_list, dtype=int)
    scens  = np.array(scen_list)
    return X_flat, X_seq_list, y, scens


# =========================================================================
# [모델: AutoEncoder (비지도)]
# =========================================================================
class AnomalyAutoEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, in_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_norm, epochs=25, batch_size=64, lr=0.001):
    in_dim = X_norm.shape[1]
    model  = AnomalyAutoEncoder(in_dim)
    opt    = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.tensor(X_norm, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (bx,) in loader:
            opt.zero_grad()
            loss = nn.MSELoss()(model(bx), bx)
            loss.backward()
            opt.step()
    model.eval()
    return model


def ae_recon_error(model, X):
    model.eval()
    with torch.no_grad():
        x_t  = torch.tensor(X, dtype=torch.float32)
        err  = torch.mean((model(x_t) - x_t) ** 2, dim=1).numpy()
    return err


# =========================================================================
# [모델: SingleGRU (지도학습 — STFT 프레임 시퀀스 입력)]
# =========================================================================
class SingleGRU(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def pad_seq_batch(seq_list, target_len=None):
    """
    가변 길이 STFT 프레임 시퀀스를 동일 길이로 패딩 -> (N, target_len, N_FREQ)
    target_len 미지정 시 최소 프레임 수 기준 (잘라 맞춤).
    """
    if target_len is None:
        target_len = min(s.shape[0] for s in seq_list)
    out = np.stack([s[:target_len] for s in seq_list], axis=0)  # (N, target_len, N_FREQ)
    return out.astype(np.float32)


def train_gru(X_tr_3d, y_tr, X_te_3d, epochs=25, batch_size=32, lr=0.001):
    in_dim = X_tr_3d.shape[2]
    model  = SingleGRU(in_dim)
    opt    = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_3d), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(bx), by).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te_3d))
        probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds  = torch.argmax(logits, dim=1).numpy()
    return preds, probs


# =========================================================================
# [결과 기록 및 출력]
# =========================================================================
def record(results, name, y_true, y_pred, y_prob=None):
    results[name]["acc"].append(accuracy_score(y_true, y_pred))
    results[name]["prec"].append(precision_score(y_true, y_pred, zero_division=0))
    results[name]["rec"].append(recall_score(y_true, y_pred, zero_division=0))
    results[name]["f1"].append(f1_score(y_true, y_pred, zero_division=0))
    if y_prob is not None and len(np.unique(y_true)) > 1:
        results[name]["auc"].append(roc_auc_score(y_true, y_prob))
    else:
        results[name]["auc"].append(float("nan"))


def print_results(results, model_names):
    print(f"\n {'Model':<55} | {'Acc':>8} | {'Prec':>8} | {'Rec':>8} | {'F1':>8} | {'AUC':>8}")
    print("-" * 110)
    for name in model_names:
        r = results[name]
        print(f" {name:<55} | {np.nanmean(r['acc'])*100:>7.2f}% | {np.nanmean(r['prec'])*100:>7.2f}% "
              f"| {np.nanmean(r['rec'])*100:>7.2f}% | {np.nanmean(r['f1'])*100:>7.2f}% "
              f"| {np.nanmean(r['auc']):>8.4f}")


# =========================================================================
# [메인 실험]
# =========================================================================
def evaluate_raw_wav_splits(splits, X_flat, X_seq_list, y, split_name):
    """지정된 분할 방식(Temporal 또는 GroupKFold)으로 RAW STFT 모델 평가 및 결과 기록"""
    print("\n" + "=" * 110)
    print(f" [Raw WAV STFT Spectrum Evaluation] Strategy: {split_name}")
    print("=" * 110)

    model_names = [
        "[UNSUPER]   AutoEncoder (MSE Threshold)       | STFT mean 129-dim",
        "[SEMISUPER] OC-SVM (Decision Boundary)        | STFT mean 129-dim",
        "[SEMISUPER] Isolation Forest (Anomaly Score)  | STFT mean 129-dim",
        "[SUPER]     XGBoost (Window)                  | STFT mean 129-dim",
        "[SUPER]     SingleGRU (STFT Frames Seq)       | STFT frames [n_f x 129]",
    ]
    results = {m: {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []} for m in model_names}

    for fold_idx, (tr_idx, te_idx) in enumerate(splits, start=1):
        print(f"\n --- Fold {fold_idx}/{len(splits)} (Train: {len(tr_idx):,}, Test: {len(te_idx):,}) ---")
        X_tr_flat, X_te_flat = X_flat[tr_idx], X_flat[te_idx]
        X_tr_seq  = [X_seq_list[i] for i in tr_idx]
        X_te_seq  = [X_seq_list[i] for i in te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 스케일링 (Train으로만 fit, log 스케일 후 정규화)
        X_tr_log = np.log1p(X_tr_flat)
        X_te_log = np.log1p(X_te_flat)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_log)
        X_te_s = scaler.transform(X_te_log)
        X_tr_norm = X_tr_s[y_tr == 0]

        # [1] UNSUPER: AutoEncoder
        ae = train_autoencoder(X_tr_norm, epochs=30)
        tr_err = ae_recon_error(ae, X_tr_s)
        te_err = ae_recon_error(ae, X_te_s)
        ae_thresh = np.percentile(tr_err[y_tr == 0], 95)
        record(results, model_names[0], y_te, (te_err > ae_thresh).astype(int), te_err)

        # [2] SEMISUPER: OC-SVM
        oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_tr_norm)
        tr_oc = oc_svm.decision_function(X_tr_s)
        te_oc = oc_svm.decision_function(X_te_s)
        oc_thresh = np.percentile(tr_oc[y_tr == 0], 5)
        record(results, model_names[1], y_te, (te_oc < oc_thresh).astype(int), -te_oc)

        # [3] SEMISUPER: Isolation Forest
        iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42).fit(X_tr_norm)
        tr_if = iforest.decision_function(X_tr_s)
        te_if = iforest.decision_function(X_te_s)
        if_thresh = np.percentile(tr_if[y_tr == 0], 5)
        record(results, model_names[2], y_te, (te_if < if_thresh).astype(int), -te_if)

        # [4] SUPERVISED: XGBoost
        xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    random_state=42, eval_metric="logloss", n_jobs=-1)
        xgb_m.fit(X_tr_s, y_tr)
        record(results, model_names[3], y_te, xgb_m.predict(X_te_s), xgb_m.predict_proba(X_te_s)[:, 1])

        # [5] SUPERVISED: SingleGRU
        min_frames = min(min(s.shape[0] for s in X_tr_seq), min(s.shape[0] for s in X_te_seq))
        X_tr_3d = np.log1p(pad_seq_batch(X_tr_seq, target_len=min_frames))
        X_te_3d = np.log1p(pad_seq_batch(X_te_seq, target_len=min_frames))
        pred_gru, prob_gru = train_gru(X_tr_3d, y_tr, X_te_3d, epochs=30)
        record(results, model_names[4], y_te, pred_gru, prob_gru)

    print("\n" + "=" * 110)
    print(f" [Results Summary] - Strategy: {split_name}")
    print("=" * 110)
    print_results(results, model_names)

    return {
        name: {
            "accuracy":  float(np.nanmean(v["acc"])),
            "precision": float(np.nanmean(v["prec"])),
            "recall":    float(np.nanmean(v["rec"])),
            "f1_score":  float(np.nanmean(v["f1"])),
            "roc_auc":   float(np.nanmean(v["auc"])),
        }
        for name, v in results.items()
    }


def run_raw_wav_experiment():
    import argparse
    parser = argparse.ArgumentParser(description="Raw WAV STFT Spectrum Experiment")
    parser.add_argument("--split_mode", type=str, default="all", choices=["temporal", "group_kfold", "all"],
                        help="분할 방식: temporal(1번 Causal Temporal Split + Gap), group_kfold(2번 Leave-One-Scenario-Out), all(둘 다)")
    args = parser.parse_args()

    X_flat, X_seq_list, y, scenarios = build_raw_dataset(window_size=34, step_size=34)
    print(f"\n Input shape (flat STFT): {X_flat.shape}   |  Normal: {(y==0).sum():,}  Abnormal: {(y==1).sum():,}")

    summary_out = {}

    # [1번 방식] Causal Chronological Split + Embargo Gap (시간 순서 전후 분할)
    if args.split_mode in ["temporal", "all"]:
        GAP, TRAIN_RATIO = 20, 0.75
        tr_idx, te_idx = [], []
        for scen in np.unique(scenarios):
            idx = np.where(scenarios == scen)[0]
            cut = int(len(idx) * TRAIN_RATIO)
            tr_idx.extend(idx[:cut])
            te_idx.extend(idx[cut + GAP:])
        temporal_splits = [(np.array(tr_idx), np.array(te_idx))]
        summary_out["1_causal_temporal_split"] = evaluate_raw_wav_splits(
            temporal_splits, X_flat, X_seq_list, y, "1. Causal Chronological Split (75% Train / 20-window Gap / 25% Test)"
        )

    # [2번 방식] Leave-One-Group-Out / GroupKFold Split (시나리오 단위 완전 격리)
    if args.split_mode in ["group_kfold", "all"]:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=min(5, len(np.unique(scenarios))))
        group_splits = list(gkf.split(X_flat, y, groups=scenarios))
        summary_out["2_leave_one_group_out"] = evaluate_raw_wav_splits(
            group_splits, X_flat, X_seq_list, y, f"2. Leave-One-Group-Out ({len(group_splits)}-Fold Scenario Hold-Out)"
        )

    out_path = os.path.join(METRICS_DIR, "audio_raw_wav_experiment_w34.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    print(f"\n [Saved Summary]: {out_path}\n")


if __name__ == "__main__":
    run_raw_wav_experiment()

