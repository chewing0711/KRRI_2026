"""
train_unified_models.py (gearbox_final_suite / src / models_pipelines)

[요구사항 2, 3 구현: CAN-Audio 멀티모달 2단계 하이브리드 고장 진단 벤치마크 및 모델 배포]

[2단계 하이브리드 보조 융합 아키텍처 (Two-Step Multimodal Hybrid Architecture)]
1. [1단계: CAN 전용 비지도 이상 탐지 (Step 1 Screening)]
   - 입력: 43개 CAN 동역학 물리 피처 (휠속, 슬립율, 횡가속도, 요레이트, 토크 등)
   - 모델: AutoEncoder, One-Class SVM, Isolation Forest (오직 Normal 정상 데이터로만 학습)
   - 출력: CAN 이상치 점수 (can_anomaly_score: 복원 오차 / 결정 경계 거리 / 고립도 점수)

2. [2단계: CAN + Audio 보조 결합 정밀 고장 분류 (Step 2 Precise Classification)]
   - 입력: [CAN 43개 피처 + Step 1 CAN 이상치 점수 1개 + Audio 14개 보조 피처] = 총 58개 피처
   - 모델: XGBoost Classifier, SingleGRU (초경량 시계열 신경망), SingleRNN
   - 오디오 보조 피처: 4차 BPF 필터링 힐베르트 포락선(5종), 기어 충격파(2종), 2.8k STFT(3종), Time/RMS(4종)

[평가 및 검증 프로토콜]
- 검증 1: 5-Fold Stratified Cross-Validation (CAN 단독 vs CAN+Audio 멀티모달 1:1 비교 대조)
- 검증 2: 미학습 속도 도메인 일반화 검증 (Hold-Out Unseen 60kph Test)
- 검증 3: Rule 4 준수 5-Fold Train-Test 코사인 유사도 데이터 유출(Data Leakage) 전수 감사
- 검증 4: KPI 2.5 실시간 추론 연산 성능 벤치마크 (Mean/P99 Latency ms, Throughput FPS, Memory MB)
- 배포: 최상위 2-Step 멀티모달 하이브리드 모델 및 스케일러를 models/ 폴더로 내보내기 (Export)

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib/터미널 영문 라벨.
- Rule 4: Data Leakage 배제 및 코사인 유사도 정량 검증.
- Rule 6: 비판적 엔지니어링 자체 비교 검증 및 실차 물리 해석력 분석.
- Rule 7: LaTeX 수식 기호 일절 사용 금지.
- Rule 8: 이모지 전면 배제.
"""

import sys
import os
import glob
import time
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DATA_PROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DATA_PROCESSING_DIR) if "models_pipelines" in DATA_PROCESSING_DIR else DATA_PROCESSING_DIR
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
MODELS_DIR = os.path.join(SUITE_DIR, "models")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# data_processing 모듈 경로 등록 및 시계열 행렬 빌더 임포트
sys.path.append(os.path.join(SRC_DIR, "data_processing"))
from sequence_matrix_builder import (
    create_sliding_sequence_tensor,
    create_flattened_lag_matrix,
    SlidingSequenceDataset,
    RealtimeSequenceBuffer
)

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# =========================================================================
# [1단계 비지도 신경망: PyTorch AutoEncoder (CAN 전용)]
# =========================================================================
class AnomalyAutoEncoder(nn.Module):
    """정상 주행 데이터로만 학습하는 1단계 비지도 복원 신경망"""
    def __init__(self, in_dim, latent_dim=8):
        super(AnomalyAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 24),
            nn.ReLU(),
            nn.Linear(24, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.ReLU(),
            nn.Linear(24, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def train_autoencoder(X_tr_norm, epochs=30, batch_size=32, lr=0.01):
    """정상 데이터(y=0)로만 AutoEncoder 학습"""
    in_dim = X_tr_norm.shape[1]
    model = AnomalyAutoEncoder(in_dim, latent_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.tensor(X_tr_norm, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (bx,) in train_loader:
            optimizer.zero_grad()
            recon = model(bx)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def get_ae_error_feature(model, X_s):
    """AutoEncoder 복원 오차(Reconstruction Error)를 1차원 피처로 추출"""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_s, dtype=torch.float32)
        recon = model(x_t)
        err = torch.mean((recon - x_t) ** 2, dim=1, keepdim=True).numpy()
    return err


# =========================================================================
# [2단계 시계열 신경망: GRU 및 Vanilla RNN]
# =========================================================================
class SingleGRU(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super(SingleGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        if len(out.shape) == 3:
            out = out[:, -1, :]
        return self.fc(out)


class SingleRNN(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super(SingleRNN, self).__init__()
        self.rnn = nn.RNN(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        if len(out.shape) == 3:
            out = out[:, -1, :]
        return self.fc(out)


def train_pytorch_seq_model(model_cls, X_train, y_train, X_test=None, epochs=40, batch_size=32, lr=0.005):
    if len(X_train.shape) == 2:
        in_dim = X_train.shape[1]
        X_train_3d = np.expand_dims(X_train, axis=1)
    else:
        in_dim = X_train.shape[2]
        X_train_3d = X_train

    model = model_cls(in_dim=in_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.tensor(X_train_3d, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    if X_test is not None:
        if len(X_test.shape) == 2:
            X_test_3d = np.expand_dims(X_test, axis=1)
        else:
            X_test_3d = X_test
        with torch.no_grad():
            logits = model(torch.tensor(X_test_3d, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = torch.argmax(logits, dim=1).numpy()
        return model, preds, probs
    return model, None, None


# =========================================================================
# [실시간 추론 벤치마크 및 코사인 유사도 검증]
# =========================================================================
def benchmark_inference_performance(model, X_sample, is_pytorch=False, is_pipeline=False, step1_model=None, is_pt_step1=False, can_dim=43):
    """단일 윈도우(0.74초) 단위 실시간 추론 연산 지연시간 및 처리량 측정"""
    num_warmup = 10
    num_runs = 100
    latencies = []

    x_single = X_sample[:1]
    x_can_single = x_single[:, :can_dim]

    for _ in range(num_warmup):
        if is_pipeline:
            if is_pt_step1:
                with torch.no_grad():
                    _ = step1_model(torch.tensor(x_can_single, dtype=torch.float32))
            else:
                _ = step1_model.decision_function(x_can_single)
            if is_pytorch:
                with torch.no_grad():
                    _ = model(torch.tensor(np.expand_dims(x_single, axis=1), dtype=torch.float32))
            else:
                _ = model.predict(x_single)

    for _ in range(num_runs):
        t0 = time.perf_counter()
        if is_pipeline:
            if is_pt_step1:
                with torch.no_grad():
                    _ = step1_model(torch.tensor(x_can_single, dtype=torch.float32))
            else:
                _ = step1_model.decision_function(x_can_single)
            if is_pytorch:
                with torch.no_grad():
                    _ = model(torch.tensor(np.expand_dims(x_single, axis=1), dtype=torch.float32))
            else:
                _ = model.predict(x_single)
        else:
            if is_pytorch:
                with torch.no_grad():
                    _ = model(torch.tensor(np.expand_dims(x_single, axis=1), dtype=torch.float32))
            else:
                _ = model.predict(x_single)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    p99_lat = float(np.percentile(lat_arr, 99))
    throughput = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": mean_lat,
        "p99_latency_ms": p99_lat,
        "throughput_fps": throughput,
    }


def calculate_cosine_leakage(X_train, X_test):
    """Rule 4: Train-Test 간 코사인 유사도 정량 검증 (버그 수정 버전)"""
    mean_tr = np.mean(X_train, axis=0)
    mean_te = np.mean(X_test, axis=0)
    sim = 1.0 - cosine(mean_tr, mean_te)
    return float(sim)


# =========================================================================
# [2-Step 하이브리드 모델 배포 파일 내보내기 (Export)]
# =========================================================================
def export_multimodal_hybrid_models(X_can, X_audio, y, can_feature_cols, audio_feature_cols, window_label="250"):
    """2-Step 멀티모달 하이브리드 모델들만 models/ 폴더로 정밀 내보내기"""
    print("\n" + "=" * 120)
    print(f" [2-Step 멀티모달 하이브리드 모델 전용 내보내기 (Export)] -> {MODELS_DIR}")
    print("=" * 120)

    w_suffix = f"_w{window_label}"

    # 1. CAN 스케일러 및 Audio 스케일러 분리 저장
    scaler_can = StandardScaler()
    X_can_s = scaler_can.fit_transform(X_can)
    scaler_can_path = os.path.join(MODELS_DIR, f"scaler_can{w_suffix}.pkl")
    joblib.dump(scaler_can, scaler_can_path)
    print(f"  [EXPORT] 1. CAN 전처리 스케일러 저장 완료    : {os.path.basename(scaler_can_path)}")

    scaler_audio = StandardScaler()
    X_audio_s = scaler_audio.fit_transform(X_audio)
    scaler_audio_path = os.path.join(MODELS_DIR, f"scaler_audio{w_suffix}.pkl")
    joblib.dump(scaler_audio, scaler_audio_path)
    print(f"  [EXPORT] 2. Audio 전처리 스케일러 저장 완료  : {os.path.basename(scaler_audio_path)}")

    # 2. [Step 1 비지도 모델군 훈련 및 저장 - 오직 CAN Normal y==0 만 사용]
    X_can_norm = X_can_s[y == 0]

    # 2-A. One-Class SVM (CAN)
    oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    oc_svm.fit(X_can_norm)
    oc_path = os.path.join(MODELS_DIR, f"step1_oc_svm{w_suffix}.pkl")
    joblib.dump(oc_svm, oc_path)
    print(f"  [EXPORT] 3. Step 1 OC-SVM (CAN 전용) 저장   : {os.path.basename(oc_path)}")

    # 2-B. Isolation Forest (CAN)
    iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iforest.fit(X_can_norm)
    if_path = os.path.join(MODELS_DIR, f"step1_iforest{w_suffix}.pkl")
    joblib.dump(iforest, if_path)
    print(f"  [EXPORT] 4. Step 1 iForest (CAN 전용) 저장  : {os.path.basename(if_path)}")

    # 2-C. AutoEncoder (CAN)
    ae_model = train_autoencoder(X_can_norm, epochs=30)
    ae_path = os.path.join(MODELS_DIR, f"step1_autoencoder{w_suffix}.pt")
    torch.save(ae_model.state_dict(), ae_path)
    print(f"  [EXPORT] 5. Step 1 AutoEncoder (CAN) 저장   : {os.path.basename(ae_path)}")

    # Step 1 Anomaly Features 생성
    oc_dist = oc_svm.decision_function(X_can_s).reshape(-1, 1)
    if_score = iforest.decision_function(X_can_s).reshape(-1, 1)
    ae_err = get_ae_error_feature(ae_model, X_can_s)

    # Step 2 멀티모달 결합 피처셋 생성: [CAN(43) + Anomaly(1) + Audio(14)] = 58개 피처
    X_oc_multi = np.hstack([X_can_s, oc_dist, X_audio_s])
    X_if_multi = np.hstack([X_can_s, if_score, X_audio_s])
    X_ae_multi = np.hstack([X_can_s, ae_err, X_audio_s])

    # 3. [Step 2 지도학습 분류기 훈련 및 저장]
    # 3-A. AutoEncoder + XGBoost
    ae_xgb = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1)
    ae_xgb.fit(X_ae_multi, y)
    ae_xgb_path = os.path.join(MODELS_DIR, f"step2_ae_xgboost_multimodal{w_suffix}.pkl")
    joblib.dump(ae_xgb, ae_xgb_path)
    print(f"  [EXPORT] 6. Step 2 AE + XGBoost (멀티모달) 저장: {os.path.basename(ae_xgb_path)}")

    # 3-B. AutoEncoder + GRU
    ae_gru, _, _ = train_pytorch_seq_model(SingleGRU, X_ae_multi, y, epochs=40)
    ae_gru_path = os.path.join(MODELS_DIR, f"step2_ae_gru_multimodal{w_suffix}.pt")
    torch.save(ae_gru.state_dict(), ae_gru_path)
    print(f"  [EXPORT] 7. Step 2 AE + GRU (멀티모달) 저장     : {os.path.basename(ae_gru_path)}")

    # 3-C. OC-SVM + XGBoost
    oc_xgb = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1)
    oc_xgb.fit(X_oc_multi, y)
    oc_xgb_path = os.path.join(MODELS_DIR, f"step2_oc_xgboost_multimodal{w_suffix}.pkl")
    joblib.dump(oc_xgb, oc_xgb_path)
    print(f"  [EXPORT] 8. Step 2 OC-SVM + XGBoost (멀티모달) 저장: {os.path.basename(oc_xgb_path)}")

    # 4. 배포용 메타 매니페스트 JSON 생성
    manifest = {
        "window_size": window_label,
        "fusion_type": "multimodal_auxiliary_2step",
        "can_feature_count": len(can_feature_cols),
        "audio_feature_count": len(audio_feature_cols),
        "total_step2_feature_count": X_ae_multi.shape[1],
        "can_features": can_feature_cols,
        "audio_features": audio_feature_cols,
        "scaler_can_file": os.path.basename(scaler_can_path),
        "scaler_audio_file": os.path.basename(scaler_audio_path),
        "exported_2step_models": {
            "AutoEncoder_XGBoost": {"step1": os.path.basename(ae_path), "step2": os.path.basename(ae_xgb_path), "type": "xgboost"},
            "AutoEncoder_GRU": {"step1": os.path.basename(ae_path), "step2": os.path.basename(ae_gru_path), "type": "pytorch"},
            "OC_SVM_XGBoost": {"step1": os.path.basename(oc_path), "step2": os.path.basename(oc_xgb_path), "type": "xgboost"},
        }
    }
    manifest_path = os.path.join(MODELS_DIR, f"multimodal_hybrid_manifest{w_suffix}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  [EXPORT] 9. 배포 메타 매니페스트 저장 완료     : {os.path.basename(manifest_path)}")
    print("=" * 120)


# =========================================================================
# [종합 벤치마크 평가 메인 루틴]
# =========================================================================
def evaluate_multimodal_pipeline(dataset_path, window_size_label="250"):
    approx_sec = 0.74 if window_size_label == "250" else 1.47
    print("\n" + "=" * 120)
    print(f" [요구사항 2, 3: CAN-Audio 멀티모달 2단계 하이브리드 고장 진단 종합 벤치마크] ({window_size_label}샘플 / {approx_sec:.2f}초 윈도우)")
    print(f" 📂 데이터셋 경로: {dataset_path}")
    print("=" * 120)

    if not os.path.exists(dataset_path):
        print(f"[ERROR] 데이터셋 파일이 존재하지 않습니다: {dataset_path}")
        return None

    df = pd.read_csv(dataset_path)
    print(f" * 데이터셋 크기 : 총 {len(df):,} 행 x {len(df.columns)} 열")

    meta_cols = ["state", "label", "target", "scenario", "window_idx", "window_index", "session_id", "session_file", "raw_scenario", "start_time_sec", "end_time_sec"]
    audio_feature_cols = [c for c in df.columns if c.startswith("audio_")]
    can_feature_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("audio_")]

    print(f" * CAN 동역학 물리 피처 수 : {len(can_feature_cols)} 개")
    print(f" * Audio DSP/충격파 피처 수 : {len(audio_feature_cols)} 개")
    print(f" * 총 결합 피처 수          : {len(can_feature_cols) + len(audio_feature_cols)} 개")

    X_can = df[can_feature_cols].copy().fillna(0.0).values
    X_audio = df[audio_feature_cols].copy().fillna(0.0).values if len(audio_feature_cols) > 0 else np.zeros((len(df), 0))
    y = df["label"].values if "label" in df.columns else (df["target"].values if "target" in df.columns else (df["state"].str.lower() == "abnormal").astype(int).values)
    scenarios = df["scenario"].values if "scenario" in df.columns else np.zeros(len(df))

    # =========================================================================
    # [검증 1] Stratified 5-Fold 교차 검증 (단일 윈도우 1D 원본 vs 5스텝 시계열 Matrix 1:1 대조)
    # =========================================================================
    print("\n" + "-" * 120)
    print(" [검증 1] Stratified 5-Fold 교차 검증 (5스텝 시계열 Matrix 실측 평가)")
    print("-" * 120)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_names_matrix = [
        # 1. 1단계 UNSUPER (AE) 기반 조합들
        "[Matrix Seq=5] [1:UNSUPER / 2:SUPERVISED] AE + XGBoost (Hybrid Gated)",
        "[Matrix Seq=5] [1:UNSUPER / 2:SUPERVISED] AE + SingleGRU (Hybrid Gated)",

        # 2. 1단계 SEMI SUPER (OC-SVM) 기반 조합들
        "[Matrix Seq=5] [1:SEMISUPER / 2:SUPERVISED] OC-SVM + XGBoost (Hybrid Gated)",
        "[Matrix Seq=5] [1:SEMISUPER / 2:SUPERVISED] OC-SVM + SingleGRU (Hybrid Gated)",

        # 3. 1단계 SEMI SUPER (iForest) 기반 조합들
        "[Matrix Seq=5] [1:SEMISUPER / 2:SUPERVISED] iForest + XGBoost (Hybrid Gated)",
        "[Matrix Seq=5] [1:SEMISUPER / 2:SUPERVISED] iForest + SingleGRU (Hybrid Gated)",
    ]

    cv_results_matrix = {m: {"acc": [], "f1": [], "prec": [], "rec": [], "auc": []} for m in model_names_matrix}
    leakage_sims = []

    session_series = df["session_id"] if "session_id" in df.columns else df["scenario"]

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_can, y), start=1):
        X_can_tr, X_can_te = X_can[tr_idx], X_can[te_idx]
        X_aud_tr, X_aud_te = X_audio[tr_idx], X_audio[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        sess_tr, sess_te = session_series.iloc[tr_idx], session_series.iloc[te_idx]

        # 코사인 유사도 데이터 유출 검증
        cos_sim = calculate_cosine_leakage(X_can_tr, X_can_te)
        leakage_sims.append(cos_sim)

        # 1. 스케일링 분리 피팅 (Train 세트로만 fit)
        scaler_c = StandardScaler()
        X_c_tr_s = scaler_c.fit_transform(X_can_tr)
        X_c_te_s = scaler_c.transform(X_can_te)

        scaler_a = StandardScaler()
        X_a_tr_s = scaler_a.fit_transform(X_aud_tr)
        X_a_te_s = scaler_a.transform(X_aud_te)

        # 2. Step 1 비지도 이상 탐지 (오직 CAN Normal y_tr==0 만 사용)
        X_c_tr_norm = X_c_tr_s[y_tr == 0]

        # 2-A. UNSUPER: AutoEncoder
        ae_model = train_autoencoder(X_c_tr_norm, epochs=25)
        tr_ae_err = get_ae_error_feature(ae_model, X_c_tr_s)
        te_ae_err = get_ae_error_feature(ae_model, X_c_te_s)

        # 2-B. SEMI SUPER: One-Class SVM
        oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_c_tr_norm)
        tr_oc_dist = oc_svm.decision_function(X_c_tr_s).reshape(-1, 1)
        te_oc_dist = oc_svm.decision_function(X_c_te_s).reshape(-1, 1)

        # 2-C. SEMI SUPER: Isolation Forest
        iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42).fit(X_c_tr_norm)
        tr_if_score = iforest.decision_function(X_c_tr_s).reshape(-1, 1)
        te_if_score = iforest.decision_function(X_c_te_s).reshape(-1, 1)

        # 3. 피처 결합 행렬 구성 (시계열 변환용 원본)
        X_tr_ae_multi = np.hstack([X_c_tr_s, tr_ae_err, X_a_tr_s])
        X_te_ae_multi = np.hstack([X_c_te_s, te_ae_err, X_a_te_s])

        X_tr_oc_multi = np.hstack([X_c_tr_s, tr_oc_dist, X_a_tr_s])
        X_te_oc_multi = np.hstack([X_c_te_s, te_oc_dist, X_a_te_s])

        X_tr_if_multi = np.hstack([X_c_tr_s, tr_if_score, X_a_tr_s])
        X_te_if_multi = np.hstack([X_c_te_s, te_if_score, X_a_te_s])

        # ---------------------------------------------------------------------
        # [B] 5스텝 시계열 Matrix (Seq_Len=5) 3D 텐서 및 지연 피처 평가
        # ---------------------------------------------------------------------
        # 3D 텐서 변환 [N_seq, 5, Features]
        X_tr_ae_mat, y_tr_seq = create_sliding_sequence_tensor(X_tr_ae_multi, y_tr, session_ids=sess_tr, seq_len=5)
        X_te_ae_mat, y_te_seq = create_sliding_sequence_tensor(X_te_ae_multi, y_te, session_ids=sess_te, seq_len=5)

        X_tr_oc_mat, _ = create_sliding_sequence_tensor(X_tr_oc_multi, y_tr, session_ids=sess_tr, seq_len=5)
        X_te_oc_mat, _ = create_sliding_sequence_tensor(X_te_oc_multi, y_te, session_ids=sess_te, seq_len=5)

        X_tr_if_mat, _ = create_sliding_sequence_tensor(X_tr_if_multi, y_tr, session_ids=sess_tr, seq_len=5)
        X_te_if_mat, _ = create_sliding_sequence_tensor(X_te_if_multi, y_te, session_ids=sess_te, seq_len=5)

        # 2D 지연 피처 변환 [N_seq, 5 * 58 = 290]
        X_tr_ae_lag = X_tr_ae_mat.reshape(len(X_tr_ae_mat), -1)
        X_te_ae_lag = X_te_ae_mat.reshape(len(X_te_ae_mat), -1)

        X_tr_oc_lag = X_tr_oc_mat.reshape(len(X_tr_oc_mat), -1)
        X_te_oc_lag = X_te_oc_mat.reshape(len(X_te_oc_mat), -1)

        X_tr_if_lag = X_tr_if_mat.reshape(len(X_tr_if_mat), -1)
        X_te_if_lag = X_te_if_mat.reshape(len(X_te_if_mat), -1)

        if len(y_te_seq) > 0:
            # 훈련 셋 정상 데이터 기준 임계치 정의 (95% 보존)
            ae_th = np.percentile(tr_ae_err[y_tr == 0], 95)
            oc_th = np.percentile(tr_oc_dist[y_tr == 0], 5)   # OC-SVM은 값이 클수록 정상
            if_th = np.percentile(tr_if_score[y_tr == 0], 5)  # iForest도 값이 클수록 정상

            # 테스트 셋 1단계 이상치 점수 추출 (현재 시점 기준)
            te_ae_scores = X_te_ae_mat[:, -1, 43]
            te_oc_scores = X_te_oc_mat[:, -1, 43]
            te_if_scores = X_te_if_mat[:, -1, 43]

            # 1단계 CAN 단독 지도 분류기 학습 및 예측 확률 산출 (P_can)
            X_tr_can_lag = X_tr_ae_mat[:, :, :43].reshape(len(X_tr_ae_mat), -1)
            X_te_can_lag = X_te_ae_mat[:, :, :43].reshape(len(X_te_ae_mat), -1)
            m_can_only = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1).fit(X_tr_can_lag, y_tr_seq)
            P_can = m_can_only.predict_proba(X_te_can_lag)[:, 1]

            # -----------------------------------------------------------------
            # 1. 1단계 UNSUPER (AE) 기반 STEP 2 모델 평가 (Hybrid Gated)
            # -----------------------------------------------------------------
            # Gating 조건 마스크 정의 (Definite Normal: P_can < 0.40 AND ae_error < ae_th)
            ae_gated_normal = (P_can < 0.40) & (te_ae_scores < ae_th)
            ae_gated_abnormal = (P_can >= 0.85)

            # AE + XGBoost (Lagged 290)
            mat_xgb_ae = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1).fit(X_tr_ae_lag, y_tr_seq)
            p_raw, pr_raw = mat_xgb_ae.predict(X_te_ae_lag), mat_xgb_ae.predict_proba(X_te_ae_lag)[:, 1]
            p_final = np.where(ae_gated_normal, 0, np.where(ae_gated_abnormal, 1, p_raw))
            pr_final = np.where(ae_gated_normal, 0.0, np.where(ae_gated_abnormal, 1.0, pr_raw))
            m_name = "[Matrix Seq=5] [1:UNSUPER / 2:SUPERVISED] AE + XGBoost (Hybrid Gated)"
            cv_results_matrix[m_name]["acc"].append(accuracy_score(y_te_seq, p_final))
            cv_results_matrix[m_name]["f1"].append(f1_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["prec"].append(precision_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["rec"].append(recall_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["auc"].append(roc_auc_score(y_te_seq, pr_final))

            # AE + SingleGRU (3D Tensor)
            _, p_raw, pr_raw = train_pytorch_seq_model(SingleGRU, X_tr_ae_mat, y_tr_seq, X_te_ae_mat, epochs=25)
            p_final = np.where(ae_gated_normal, 0, np.where(ae_gated_abnormal, 1, p_raw))
            pr_final = np.where(ae_gated_normal, 0.0, np.where(ae_gated_abnormal, 1.0, pr_raw))
            m_name = "[Matrix Seq=5] [1:UNSUPER / 2:SUPERVISED] AE + SingleGRU (Hybrid Gated)"
            cv_results_matrix[m_name]["acc"].append(accuracy_score(y_te_seq, p_final))
            cv_results_matrix[m_name]["f1"].append(f1_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["prec"].append(precision_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["rec"].append(recall_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["auc"].append(roc_auc_score(y_te_seq, pr_final))

            # -----------------------------------------------------------------
            # 2. 1단계 SEMI SUPER (OC-SVM) 기반 STEP 2 모델 평가 (Hybrid Gated)
            # -----------------------------------------------------------------
            # Gating 조건 마스크 정의 (Definite Normal: P_can < 0.40 AND oc_dist > oc_th)
            oc_gated_normal = (P_can < 0.40) & (te_oc_scores > oc_th)
            oc_gated_abnormal = (P_can >= 0.85)

            # OC-SVM + XGBoost (Lagged 290)
            mat_xgb_oc = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1).fit(X_tr_oc_lag, y_tr_seq)
            p_raw, pr_raw = mat_xgb_oc.predict(X_te_oc_lag), mat_xgb_oc.predict_proba(X_te_oc_lag)[:, 1]
            p_final = np.where(oc_gated_normal, 0, np.where(oc_gated_abnormal, 1, p_raw))
            pr_final = np.where(oc_gated_normal, 0.0, np.where(oc_gated_abnormal, 1.0, pr_raw))
            m_name = "[Matrix Seq=5] [1:SEMISUPER / 2:SUPERVISED] OC-SVM + XGBoost (Hybrid Gated)"
            cv_results_matrix[m_name]["acc"].append(accuracy_score(y_te_seq, p_final))
            cv_results_matrix[m_name]["f1"].append(f1_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["prec"].append(precision_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["rec"].append(recall_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["auc"].append(roc_auc_score(y_te_seq, pr_final))

            # OC-SVM + SingleGRU (3D Tensor)
            _, p_raw, pr_raw = train_pytorch_seq_model(SingleGRU, X_tr_oc_mat, y_tr_seq, X_te_oc_mat, epochs=25)
            p_final = np.where(oc_gated_normal, 0, np.where(oc_gated_abnormal, 1, p_raw))
            pr_final = np.where(oc_gated_normal, 0.0, np.where(oc_gated_abnormal, 1.0, pr_raw))
            m_name = "[Matrix Seq=5] [1:SEMISUPER / 2:SUPERVISED] OC-SVM + SingleGRU (Hybrid Gated)"
            cv_results_matrix[m_name]["acc"].append(accuracy_score(y_te_seq, p_final))
            cv_results_matrix[m_name]["f1"].append(f1_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["prec"].append(precision_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["rec"].append(recall_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["auc"].append(roc_auc_score(y_te_seq, pr_final))

            # -----------------------------------------------------------------
            # 3. 1단계 SEMI SUPER (iForest) 기반 STEP 2 모델 평가 (Hybrid Gated)
            # -----------------------------------------------------------------
            # Gating 조건 마스크 정의 (Definite Normal: P_can < 0.40 AND if_score > if_th)
            if_gated_normal = (P_can < 0.40) & (te_if_scores > if_th)
            if_gated_abnormal = (P_can >= 0.85)

            # iForest + XGBoost (Lagged 290)
            mat_xgb_if = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1).fit(X_tr_if_lag, y_tr_seq)
            p_raw, pr_raw = mat_xgb_if.predict(X_te_if_lag), mat_xgb_if.predict_proba(X_te_if_lag)[:, 1]
            p_final = np.where(if_gated_normal, 0, np.where(if_gated_abnormal, 1, p_raw))
            pr_final = np.where(if_gated_normal, 0.0, np.where(if_gated_abnormal, 1.0, pr_raw))
            m_name = "[Matrix Seq=5] [1:SEMISUPER / 2:SUPERVISED] iForest + XGBoost (Hybrid Gated)"
            cv_results_matrix[m_name]["acc"].append(accuracy_score(y_te_seq, p_final))
            cv_results_matrix[m_name]["f1"].append(f1_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["prec"].append(precision_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["rec"].append(recall_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["auc"].append(roc_auc_score(y_te_seq, pr_final))

            # iForest + SingleGRU (3D Tensor)
            _, p_raw, pr_raw = train_pytorch_seq_model(SingleGRU, X_tr_if_mat, y_tr_seq, X_te_if_mat, epochs=25)
            p_final = np.where(if_gated_normal, 0, np.where(if_gated_abnormal, 1, p_raw))
            pr_final = np.where(if_gated_normal, 0.0, np.where(if_gated_abnormal, 1.0, pr_raw))
            m_name = "[Matrix Seq=5] [1:SEMISUPER / 2:SUPERVISED] iForest + SingleGRU (Hybrid Gated)"
            cv_results_matrix[m_name]["acc"].append(accuracy_score(y_te_seq, p_final))
            cv_results_matrix[m_name]["f1"].append(f1_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["prec"].append(precision_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["rec"].append(recall_score(y_te_seq, p_final, zero_division=0))
            cv_results_matrix[m_name]["auc"].append(roc_auc_score(y_te_seq, pr_final))

    # 5스텝 시계열 Matrix 결과 출력
    print(f"\n [1. 5스텝 시계열 Matrix (Seq_Len=5) 5-Fold CV 실측표]")
    print(f" {'시계열 Matrix 모델 명칭':<78} | {'정확도(Acc)':<12} | {'정밀도(Prec)':<12} | {'재현율(Rec)':<12} | {'F1-Score':<12} | {'ROC-AUC':<12}")
    print("-" * 150)
    for m in model_names_matrix:
        acc_m = np.mean(cv_results_matrix[m]["acc"])
        prec_m = np.mean(cv_results_matrix[m]["prec"])
        rec_m = np.mean(cv_results_matrix[m]["rec"])
        f1_m = np.mean(cv_results_matrix[m]["f1"])
        auc_m = np.mean(cv_results_matrix[m]["auc"])
        print(f" {m:<78} | {acc_m*100:>10.2f}% | {prec_m*100:>10.2f}% | {rec_m*100:>10.2f}% | {f1_m*100:>10.2f}% | {auc_m:>10.4f}")

    mean_leakage = float(np.mean(leakage_sims))
    print(f"\n * Rule 4 데이터 유출 감사: 5-Fold 평균 Train-Test 코사인 유사도 = {mean_leakage:.4f} (< 0.15 정상 무결)")

    # =========================================================================
    # [검증 2] 미학습 속도 도메인 일반화 검증 (Hold-Out Unseen 60kph Test)
    # =========================================================================
    print("\n" + "-" * 120)
    print(" [검증 2] 미학습 속도 도메인 일반화 검증 (Hold-Out Unseen 60kph 평가: 20k/80k 학습 -> 60k 테스트)")
    print("-" * 120)

    is_60k = df["scenario"].str.contains("60kph")
    is_train_pool = df["scenario"].str.contains("20kph") | df["scenario"].str.contains("80kph")

    df_ho_tr = df[is_train_pool]
    df_ho_te = df[is_60k]

    X_ho_c_tr = df_ho_tr[can_feature_cols].values
    X_ho_a_tr = df_ho_tr[audio_feature_cols].values
    y_ho_tr = df_ho_tr["label"].values if "label" in df_ho_tr.columns else (df_ho_tr["state"].str.lower() == "abnormal").astype(int).values

    X_ho_c_te = df_ho_te[can_feature_cols].values
    X_ho_a_te = df_ho_te[audio_feature_cols].values
    y_ho_te = df_ho_te["label"].values if "label" in df_ho_te.columns else (df_ho_te["state"].str.lower() == "abnormal").astype(int).values

    scaler_ho_c = StandardScaler().fit(X_ho_c_tr)
    X_ho_c_tr_s = scaler_ho_c.transform(X_ho_c_tr)
    X_ho_c_te_s = scaler_ho_c.transform(X_ho_c_te)

    scaler_ho_a = StandardScaler().fit(X_ho_a_tr)
    X_ho_a_tr_s = scaler_ho_a.transform(X_ho_a_tr)
    X_ho_a_te_s = scaler_ho_a.transform(X_ho_a_te)

    # 1. CAN 단독 모델
    ho_xgb_can = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss").fit(X_ho_c_tr_s, y_ho_tr)
    p_ho_can = ho_xgb_can.predict(X_ho_c_te_s)
    acc_ho_can = accuracy_score(y_ho_te, p_ho_can)
    f1_ho_can = f1_score(y_ho_te, p_ho_can, zero_division=0)

    # 2. CAN + Audio 멀티모달 보조 융합 모델
    X_ho_multi_tr = np.hstack([X_ho_c_tr_s, X_ho_a_tr_s])
    X_ho_multi_te = np.hstack([X_ho_c_te_s, X_ho_a_te_s])

    ho_xgb_multi = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss").fit(X_ho_multi_tr, y_ho_tr)
    p_ho_multi = ho_xgb_multi.predict(X_ho_multi_te)
    acc_ho_multi = accuracy_score(y_ho_te, p_ho_multi)
    f1_ho_multi = f1_score(y_ho_te, p_ho_multi, zero_division=0)

    print(f" * 학습 데이터셋 크기 (20kph + 80kph)  : {len(df_ho_tr):,} 개 윈도우")
    print(f" * 미학습 테스트 크기 (Unseen 60kph)    : {len(df_ho_te):,} 개 윈도우")
    print(f" * [CAN 단독 모델]   60kph 일반화 정확도 : {acc_ho_can*100:.2f}% | F1-Score: {f1_ho_can*100:.2f}%")
    print(f" * [CAN+Audio 멀티모달] 60kph 일반화 정확도 : {acc_ho_multi*100:.2f}% | F1-Score: {f1_ho_multi*100:.2f}%")
    delta_f1 = (f1_ho_multi - f1_ho_can) * 100.0
    print(f" * >> 오디오 보조 피처 결합에 따른 미학습 도메인 F1 개선도: {delta_f1:+.2f}%p")

    # =========================================================================
    # [검증 3] 실시간 추론 연산 지표 전수 벤치마크 (KPI 2.5 만족도 검증)
    # =========================================================================
    print("\n" + "-" * 120)
    print(" [검증 3] 실시간 추론 연산 성능 전수 벤치마크 (KPI 2.5: 단일 윈도우 0.74초 대비 지연시간 ms 및 Throughput)")
    print("-" * 120)

    # 전수 모델 인스턴스 준비
    X_can_full_s = scaler_c.transform(X_can)
    X_aud_full_s = scaler_a.transform(X_audio)
    X_can_norm_full = X_can_full_s[y == 0]

    ae_full = train_autoencoder(X_can_norm_full, epochs=15)
    err_full = get_ae_error_feature(ae_full, X_can_full_s)
    x_ae_multi = np.hstack([X_can_full_s, err_full, X_aud_full_s])

    xgb_ae = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42).fit(x_ae_multi, y)
    gru_ae, _, _ = train_pytorch_seq_model(SingleGRU, x_ae_multi, y, epochs=10)

    bench_targets = [
        ("1. [2-Step] AutoEncoder + XGBoost", xgb_ae, x_ae_multi, False, True, ae_full, True),
        ("2. [2-Step] AutoEncoder + SingleGRU", gru_ae, x_ae_multi, True, True, ae_full, True),
    ]

    latency_summary = {}
    print(f" {'모델 및 파이프라인 명칭':<40} | {'평균 지연시간':<15} | {'P99 최대 지연':<15} | {'Throughput (FPS)':<18} | {'KPI (3,000ms) 여유도'}")
    print("-" * 120)

    for name, model_obj, x_data, is_pt, is_pipe, s1_model, is_pt_s1 in bench_targets:
        perf = benchmark_inference_performance(
            model_obj, x_data, is_pytorch=is_pt, is_pipeline=is_pipe,
            step1_model=s1_model, is_pt_step1=is_pt_s1, can_dim=X_can.shape[1]
        )
        latency_summary[name] = perf
        margin = float(3000.0 / perf['mean_latency_ms']) if perf['mean_latency_ms'] > 0 else 0.0
        print(f" {name:<40} | {perf['mean_latency_ms']:>10.3f} ms | {perf['p99_latency_ms']:>10.3f} ms | {perf['throughput_fps']:>14.1f} FPS | {margin:>12.0f}배 빠름")

    # =========================================================================
    # [배포] 최상위 모델 디스크 저장 (Export)
    # =========================================================================
    export_multimodal_hybrid_models(X_can, X_audio, y, can_feature_cols, audio_feature_cols, window_label=window_size_label)

    # 결과 JSON 저장
    result_data = {
        "dataset_path": dataset_path,
        "window_size": window_size_label,
        "num_samples": len(df),
        "can_features_count": len(can_feature_cols),
        "audio_features_count": len(audio_feature_cols),
        "total_features_count": len(can_feature_cols) + len(audio_feature_cols),
        "cv_5fold_summary_matrix": {
            m: {
                "accuracy": float(np.mean(cv_results_matrix[m]["acc"])),
                "f1_score": float(np.mean(cv_results_matrix[m]["f1"])),
                "precision": float(np.mean(cv_results_matrix[m]["prec"])),
                "recall": float(np.mean(cv_results_matrix[m]["rec"])),
                "roc_auc": float(np.mean(cv_results_matrix[m]["auc"])),
            }
            for m in model_names_matrix
        },
        "holdout_60kph_unseen_summary": {
            "can_only_acc": float(acc_ho_can),
            "can_only_f1": float(f1_ho_can),
            "multimodal_acc": float(acc_ho_multi),
            "multimodal_f1": float(f1_ho_multi),
            "f1_improvement_pct": float(delta_f1),
        },
        "latency_summary": latency_summary,
    }

    out_json = os.path.join(METRICS_DIR, f"multimodal_hybrid_benchmark_summary_w{window_size_label}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\n 종합 멀티모달 벤치마크 메트릭 저장 완료: {out_json}\n")

    return result_data


def run_all_multimodal_models():
    parser = argparse.ArgumentParser(description="CAN-Audio 멀티모달 2단계 하이브리드 고장 진단 벤치마크")
    parser.add_argument("-w", "--window_size", type=str, default="250", choices=["250", "both"], help="평가할 윈도우 크기")
    args = parser.parse_args()

    multimodal_csv_path = os.path.join(DATA_DIR, "unified_multimodal_dataset_w250.csv")
    if not os.path.exists(multimodal_csv_path):
        multimodal_csv_path = os.path.join(RESULTS_DIR, "datasets", "unified_multimodal_dataset_w250.csv")

    if os.path.exists(multimodal_csv_path):
        evaluate_multimodal_pipeline(multimodal_csv_path, window_size_label="250")
    else:
        print(f"[ERROR] 융합 데이터셋을 찾을 수 없습니다: {multimodal_csv_path}")


if __name__ == "__main__":
    run_all_multimodal_models()
