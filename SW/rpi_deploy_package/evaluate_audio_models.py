"""
evaluate_audio_models.py (오디오 전용 배포 모델 검증)

[배포 모델 전용 검증 스크립트]
1. models/ 폴더에 저장된 오디오 전용 배포 모델 파일(총 9종) 로드
2. 벤치마크:
   - [검증 1] 정확도 평가: 
       * MLP/XGB 계열 6종: unified_audio_only_dataset_w34.csv 전체 데이터(11,939개) 대상
       * CRNN 계열 3종: 원본 .wav 파일에서 추출한 전체 STFT 데이터(11,939개) 대상
   - [검증 2] 실시간 연산 성능(Latency/FPS): 전체 9종 모델 전수 벤치마크
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
DATA_DIR = os.path.join(CURRENT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "unified_audio_only_dataset_w34.csv")

# src 폴더 임포트 설정
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "models_pipelines"))

# =========================================================
# 모델 뼈대(클래스) 정의
# =========================================================
class AudioFeatureMLP(nn.Module):
    def __init__(self, in_dim=24, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 2)
        )
    def forward(self, x):
        return self.net(x)

class AudioCRNN(nn.Module):
    def __init__(self, in_freq=129, hidden_dim=64):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_freq, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.gru = nn.GRU(32, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        conv_out = self.conv1d(x.transpose(1, 2))
        gru_out, _ = self.gru(conv_out.transpose(1, 2))
        return self.fc(gru_out[:, -1, :])

class AudioHybridFeatureMLP(nn.Module):
    def __init__(self, in_dim=25, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 2)
        )
    def forward(self, x):
        return self.net(x)

class AudioHybridCRNN(nn.Module):
    def __init__(self, in_freq=129, hidden_dim=64):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_freq, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.gru = nn.GRU(32, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim + 1, 2)
    def forward(self, x_spec, x_score):
        conv_out = self.conv1d(x_spec.transpose(1, 2))
        gru_out, _ = self.gru(conv_out.transpose(1, 2))
        return self.fc(torch.cat([gru_out[:, -1, :], x_score], dim=1))

def benchmark_single_model_latency(predict_fn, sample_input, sample_input2=None, num_runs=100):
    for _ in range(10):
        if sample_input2 is not None: _ = predict_fn(sample_input, sample_input2)
        else: _ = predict_fn(sample_input)
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        if sample_input2 is not None: _ = predict_fn(sample_input, sample_input2)
        else: _ = predict_fn(sample_input)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.array(times)
    return float(np.mean(arr)), float(np.percentile(arr, 99)), float(1000.0 / np.mean(arr))


def evaluate_audio_models():
    print("=" * 115)
    print(" [오디오 배포 모델 종합 벤치마크] (Zero-Training / Pure Deployed Models)")
    print("=" * 115)

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] 오디오 평가 데이터셋이 존재하지 않습니다: {CSV_PATH}")
        return

    # ---------------------------------------------------------
    # 1. 1D Feature 데이터 로드 (CSV 기반 - 6종 모델 평가용)
    # ---------------------------------------------------------
    print(f"[1] 1D Feature CSV Data Loading...")
    df = pd.read_csv(CSV_PATH)
    meta_cols = ["state", "label", "target", "scenario", "window_idx", "window_index", "session_id", "session_file", "raw_scenario", "start_time_sec", "end_time_sec"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    y_true_csv = df["label"].values
    X_raw_csv = df[feature_cols].values
    
    # ---------------------------------------------------------
    # 2. 2D STFT 데이터 로드 (WAV 원본 기반 - CRNN 3종 평가용)
    # ---------------------------------------------------------
    try:
        from train_rpi_models_no_gap import build_rpi_no_gap_hybrid_dataset
        _, _, X_s_tr, X_s_te, y_tr, y_te = build_rpi_no_gap_hybrid_dataset()
        # CSV 전체(Train+Test)를 1D로 평가하듯, STFT도 전체를 병합하여 통일
        X_s_all = np.concatenate([X_s_tr, X_s_te], axis=0)
        y_true_wav = np.concatenate([y_tr, y_te], axis=0)
        wav_loaded = True
    except Exception as e:
        print(f"[ERROR] WAV 데이터 추출 실패: {e}")
        print(" -> CRNN 정확도 측정은 생략되며, 속도 테스트만 진행됩니다.")
        wav_loaded = False

    print(f" * CSV 데이터셋 크기 : {len(df):,} 개 윈도우")
    if wav_loaded:
        print(f" * WAV 데이터셋 크기 : {len(X_s_all):,} 개 윈도우")
    
    # =========================================================
    # 스케일러 및 모델 로드
    # =========================================================
    scaler_feat = joblib.load(os.path.join(MODELS_DIR, "rpi_scaler_feat.joblib"))
    scaler_stft = joblib.load(os.path.join(MODELS_DIR, "rpi_scaler_stft.joblib"))
    
    # 1D Feature 스케일링
    X_feat_s = scaler_feat.transform(X_raw_csv)
    X_feat_t = torch.tensor(X_feat_s, dtype=torch.float32)

    # STFT 스케일링
    if wav_loaded:
        orig_shape = X_s_all.shape
        X_s_s = scaler_stft.transform(np.log1p(X_s_all).reshape(orig_shape[0], -1))
        X_s_t = torch.tensor(X_s_s, dtype=torch.float32).view(orig_shape[0], orig_shape[1], orig_shape[2])

    # 1단계 Anomaly 모델 로드
    try:
        class AnomalyAutoEncoder(nn.Module):
            def __init__(self, input_dim=24, hidden_dim=32, bottleneck_dim=8):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, bottleneck_dim), nn.ReLU())
                self.decoder = nn.Sequential(nn.Linear(bottleneck_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
            def forward(self, x): return self.decoder(self.encoder(x))

        ae = AnomalyAutoEncoder(input_dim=24, hidden_dim=32, bottleneck_dim=8)
        ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, "rpi_step1_ae.pt"), map_location="cpu", weights_only=True))
        ae.eval()
        with torch.no_grad():
            ae_err_tr = torch.mean((ae(X_feat_t) - X_feat_t)**2, dim=1, keepdim=True).numpy()
    except Exception as e:
        print(f"[Warning] AE Load Failed: {e}")
        ae_err_tr = np.zeros((len(X_raw_csv), 1), dtype=np.float32)

    ocsvm = joblib.load(os.path.join(MODELS_DIR, "rpi_step1_ocsvm.joblib"))
    oc_err_tr = (-ocsvm.decision_function(X_feat_s)).reshape(-1, 1).astype(np.float32)

    # Hybrid 입력 생성 (1D)
    h_ae_s = np.hstack([X_feat_s, ae_err_tr])
    h_oc_s = np.hstack([X_feat_s, oc_err_tr])
    h_ae_t = torch.tensor(h_ae_s, dtype=torch.float32)
    h_oc_t = torch.tensor(h_oc_s, dtype=torch.float32)

    # =========================================================================
    # [검증 1] 전체 9개 배포 모델 정확도 벤치마크
    # =========================================================================
    print("\n" + "=" * 115)
    print(" [검증 1] 오디오 배포 모델 정확도(Accuracy) 및 F1-Score 평가")
    print(" * MLP/XGBoost 계열 6종: CSV 파일 기반 평가")
    print(" * CRNN 계열 3종: 원본 WAV 파일 실시간 추출 기반 평가")
    print("-" * 115)
    print(f" {'배포 모델 명칭':<35} | {'Acc':<10} | {'F1':<10} | {'Prec':<10} | {'Rec':<10} | {'AUC':<10}")
    print("-" * 115)

    def print_acc(name, y_true, probs, preds):
        acc = accuracy_score(y_true, preds) * 100
        f1 = f1_score(y_true, preds, zero_division=0) * 100
        prec = precision_score(y_true, preds, zero_division=0) * 100
        rec = recall_score(y_true, preds, zero_division=0) * 100
        auc = roc_auc_score(y_true, probs)
        print(f" {name:<35} | {acc:>9.2f}% | {f1:>9.2f}% | {prec:>9.2f}% | {rec:>9.2f}% | {auc:>10.4f}")

    with torch.no_grad():
        # [단독 3종]
        mlp = AudioFeatureMLP(in_dim=24)
        mlp.load_state_dict(torch.load(os.path.join(MODELS_DIR, "rpi_audio_feature_mlp.pt"), map_location="cpu", weights_only=True))
        mlp.eval()
        mlp_logits = mlp(X_feat_t)
        print_acc("[단독] AudioFeatureMLP", y_true_csv, torch.softmax(mlp_logits, 1)[:, 1].numpy(), torch.argmax(mlp_logits, 1).numpy())

        xgb_model = joblib.load(os.path.join(MODELS_DIR, "rpi_xgboost_no_gap.joblib"))
        print_acc("[단독] XGBoost", y_true_csv, xgb_model.predict_proba(X_feat_s)[:, 1], xgb_model.predict(X_feat_s))

        crnn = AudioCRNN(in_freq=129)
        crnn.load_state_dict(torch.load(os.path.join(MODELS_DIR, "rpi_supervised_crnn.pt"), map_location="cpu", weights_only=True))
        crnn.eval()
        if wav_loaded:
            crnn_logits = crnn(X_s_t)
            print_acc("[단독] AudioCRNN", y_true_wav, torch.softmax(crnn_logits, 1)[:, 1].numpy(), torch.argmax(crnn_logits, 1).numpy())

        print("-" * 115)
        # [하이브리드 AE 3종]
        hmlp_ae = AudioHybridFeatureMLP(in_dim=25)
        hmlp_ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, "rpi_step2_audio_hybrid_mlp_ae.pt"), map_location="cpu", weights_only=True))
        hmlp_ae.eval()
        hmlp_ae_logits = hmlp_ae(h_ae_t)
        print_acc("[하이브리드 AE] AudioHybridMLP", y_true_csv, torch.softmax(hmlp_ae_logits, 1)[:, 1].numpy(), torch.argmax(hmlp_ae_logits, 1).numpy())

        hx_ae = joblib.load(os.path.join(MODELS_DIR, "rpi_step2_xgboost_ae.joblib"))
        print_acc("[하이브리드 AE] XGBoost", y_true_csv, hx_ae.predict_proba(h_ae_s)[:, 1], hx_ae.predict(h_ae_s))

        hcrnn_ae = AudioHybridCRNN(in_freq=129)
        hcrnn_ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, "rpi_hybrid_crnn_ae.pt"), map_location="cpu", weights_only=True))
        hcrnn_ae.eval()
        if wav_loaded:
            ae_err_wav = torch.mean((ae(torch.tensor(scaler_feat.transform(X_raw_csv), dtype=torch.float32)) - torch.tensor(scaler_feat.transform(X_raw_csv), dtype=torch.float32))**2, dim=1, keepdim=True).numpy()
            hcrnn_ae_logits = hcrnn_ae(X_s_t, torch.tensor(ae_err_wav, dtype=torch.float32))
            print_acc("[하이브리드 AE] AudioHybridCRNN", y_true_wav, torch.softmax(hcrnn_ae_logits, 1)[:, 1].numpy(), torch.argmax(hcrnn_ae_logits, 1).numpy())

        print("-" * 115)
        # [하이브리드 OCSVM 3종]
        hmlp_oc = AudioHybridFeatureMLP(in_dim=25)
        hmlp_oc.load_state_dict(torch.load(os.path.join(MODELS_DIR, "rpi_step2_audio_hybrid_mlp_oc.pt"), map_location="cpu", weights_only=True))
        hmlp_oc.eval()
        hmlp_oc_logits = hmlp_oc(h_oc_t)
        print_acc("[하이브리드 OCSVM] AudioHybridMLP", y_true_csv, torch.softmax(hmlp_oc_logits, 1)[:, 1].numpy(), torch.argmax(hmlp_oc_logits, 1).numpy())

        hx_oc = joblib.load(os.path.join(MODELS_DIR, "rpi_step2_xgboost_ocsvm.joblib"))
        print_acc("[하이브리드 OCSVM] XGBoost", y_true_csv, hx_oc.predict_proba(h_oc_s)[:, 1], hx_oc.predict(h_oc_s))

        hcrnn_oc = AudioHybridCRNN(in_freq=129)
        hcrnn_oc.load_state_dict(torch.load(os.path.join(MODELS_DIR, "rpi_hybrid_crnn_ocsvm.pt"), map_location="cpu", weights_only=True))
        hcrnn_oc.eval()
        if wav_loaded:
            oc_err_wav = (-ocsvm.decision_function(scaler_feat.transform(X_raw_csv))).reshape(-1, 1).astype(np.float32)
            hcrnn_oc_logits = hcrnn_oc(X_s_t, torch.tensor(oc_err_wav, dtype=torch.float32))
            print_acc("[하이브리드 OCSVM] AudioHybridCRNN", y_true_wav, torch.softmax(hcrnn_oc_logits, 1)[:, 1].numpy(), torch.argmax(hcrnn_oc_logits, 1).numpy())


    # =========================================================================
    # [검증 2] 9개 모델 전수 Latency & FPS 벤치마크
    # =========================================================================
    print("\n" + "=" * 115)
    print(" [검증 2] 9개 모델 전수 실시간 연산 성능 벤치마크 (1 윈도우 단일 추론)")
    print("-" * 115)
    print(f" {'배포 모델 명칭':<35} | {'Mean Latency':<15} | {'P99 Latency':<15} | {'Throughput':<18} | {'KPI (3000ms)'}")
    print("-" * 115)

    dummy_feat_np = X_feat_s[:1]
    dummy_feat_th = X_feat_t[:1]
    dummy_ae_np = h_ae_s[:1]
    dummy_ae_th = h_ae_t[:1]
    dummy_oc_np = h_oc_s[:1]
    dummy_oc_th = h_oc_t[:1]

    dummy_stft = torch.randn(1, 36, 129, dtype=torch.float32)
    dummy_stft_s = scaler_stft.transform(torch.log1p(dummy_stft).view(1, -1).numpy())
    dummy_stft_t = torch.tensor(dummy_stft_s, dtype=torch.float32).view(1, 36, 129)

    with torch.no_grad():
        def print_lat(name, fn, in1, in2=None):
            lat, p99, fps = benchmark_single_model_latency(fn, in1, in2)
            print(f" {name:<35} | {lat:>10.3f} ms | {p99:>10.3f} ms | {fps:>14.1f} FPS | {3000/lat:>12.0f}x faster")

        print_lat("[단독] AudioFeatureMLP", lambda x: mlp(x), dummy_feat_th)
        print_lat("[단독] XGBoost", lambda x: xgb_model.predict(x), dummy_feat_np)
        print_lat("[단독] AudioCRNN", lambda x: crnn(x), dummy_stft_t)
        
        print("-" * 115)
        print_lat("[하이브리드 AE] AudioHybridMLP", lambda x: hmlp_ae(x), dummy_ae_th)
        print_lat("[하이브리드 AE] XGBoost", lambda x: hx_ae.predict(x), dummy_ae_np)
        def inf_ae_crnn(stft, feat):
            err = torch.mean((ae(feat) - feat)**2, dim=1, keepdim=True)
            return hcrnn_ae(stft, err)
        print_lat("[하이브리드 AE] AudioHybridCRNN", inf_ae_crnn, dummy_stft_t, dummy_feat_th)
        
        print("-" * 115)
        print_lat("[하이브리드 OCSVM] AudioHybridMLP", lambda x: hmlp_oc(x), dummy_oc_th)
        print_lat("[하이브리드 OCSVM] XGBoost", lambda x: hx_oc.predict(x), dummy_oc_np)
        def inf_oc_crnn(stft, feat_np):
            err = torch.tensor((-ocsvm.decision_function(feat_np)).reshape(-1, 1), dtype=torch.float32)
            return hcrnn_oc(stft, err)
        print_lat("[하이브리드 OCSVM] AudioHybridCRNN", inf_oc_crnn, dummy_stft_t, dummy_feat_np)

    print("=" * 115)
    print(" [완료] 종합 벤치마크 종료.\n")

if __name__ == "__main__":
    evaluate_audio_models()
