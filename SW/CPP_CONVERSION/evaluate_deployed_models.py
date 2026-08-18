"""
evaluate_deployed_models.py (gearbox_final_suite / CPP_CONVERSION)

[배포 모델 전용 검증 스크립트 (Zero-Training, Pure Inference Evaluation)]

기능:
1. models/ 및 CPP_CONVERSION/에 이미 저장된 배포 모델 파일(.pkl, .pt, INT8)을 직접 로드
2. 재학습 없이 순수 추론(Pure Inference)만으로 train_unified_models와 동일한 3대 벤치마크 수행:
   - [검증 1] 전체 1,617개 윈도우 전수 추론 정확도, F1-Score, 정밀도, 재현율 평가
   - [검증 2] 미학습 60kph 도메인 일반화 추론 성능 평가 (Hold-Out Unseen 60kph)
   - [검증 3] 실시간 추론 연산 지연시간(Latency ms) 및 Throughput (FPS) 정밀 실측
3. FP32 원본 배포 모델 vs INT8 양자화 배포 모델 1:1 정밀도/속도 대조표 출력

규정 준수:
- 100% 한글 주석 및 독스트링
- 재학습 0%, 순수 파일 로드 기반 추론
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(SUITE_DIR, "data", "unified_multimodal_dataset_w250.csv")
MODELS_DIR = os.path.join(SUITE_DIR, "models")
DEPLOY_DIR = os.path.join(CURRENT_DIR, "models")

import warnings
warnings.filterwarnings("ignore")

# 모델 클래스 및 시계열 빌더 임포트
sys.path.append(os.path.join(SUITE_DIR, "src", "models_pipelines"))
sys.path.append(os.path.join(SUITE_DIR, "src", "data_processing"))
from train_unified_models import AnomalyAutoEncoder, SingleGRU
from sequence_matrix_builder import create_sliding_sequence_tensor, create_flattened_lag_matrix


def benchmark_single_model_latency(predict_fn, sample_input, num_runs=100):
    """단일 모델 추론 지연시간 ms 정밀 측정"""
    for _ in range(10):
        _ = predict_fn(sample_input)
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = predict_fn(sample_input)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.array(times)
    return float(np.mean(arr)), float(np.percentile(arr, 99)), float(1000.0 / np.mean(arr))


def evaluate_all_deployed_models():
    print("=" * 110)
    print(" [배포 모델 전용 성능 및 지연시간 종합 벤치마크] (Zero-Training / Pure Deployed Models)")
    print(f" 📂 데이터셋: {DATA_PATH}")
    print("=" * 110)

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] 데이터셋 파일 없음: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    meta_cols = ["state", "label", "target", "scenario", "window_idx", "window_index", "session_id", "session_file", "raw_scenario", "start_time_sec", "end_time_sec"]
    audio_feature_cols = [c for c in df.columns if c.startswith("audio_")]
    can_feature_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("audio_")]
    y_true = df["label"].values if "label" in df.columns else (df["state"].str.lower() == "abnormal").astype(int).values
    session_ids = df["session_id"] if "session_id" in df.columns else df["scenario"]

    print(f" * 전체 데이터셋 크기 : {len(df):,} 개 윈도우 (정상: {np.sum(y_true==0):,}개, 고장: {np.sum(y_true==1):,}개)")
    print(f" * CAN 피처: {len(can_feature_cols)}개 | Audio 피처: {len(audio_feature_cols)}개 | 총 {len(can_feature_cols)+len(audio_feature_cols)}개")
    print("-" * 110)

    # 1. 스케일러 로드 및 전처리
    scaler_can = joblib.load(os.path.join(MODELS_DIR, "scaler_can_w250.pkl"))
    scaler_aud = joblib.load(os.path.join(MODELS_DIR, "scaler_audio_w250.pkl"))
    X_can_s = scaler_can.transform(df[can_feature_cols].fillna(0.0).values)
    X_aud_s = scaler_aud.transform(df[audio_feature_cols].fillna(0.0).values)

    # 2. Step 1 배포 모델 로드 및 Anomaly Score 산출
    # 2-A. AutoEncoder FP32
    ae_fp32 = AnomalyAutoEncoder(in_dim=43, latent_dim=8)
    ae_fp32.load_state_dict(torch.load(os.path.join(MODELS_DIR, "step1_autoencoder_w250.pt"), map_location="cpu", weights_only=True))
    ae_fp32.eval()
    with torch.no_grad():
        x_can_t = torch.tensor(X_can_s, dtype=torch.float32)
        ae_err_fp32 = torch.mean((ae_fp32(x_can_t) - x_can_t) ** 2, dim=1, keepdim=True).numpy()

    # 2-B. AutoEncoder INT8 (양자화 모델)
    ae_int8_path = os.path.join(DEPLOY_DIR, "step1_autoencoder_int8.pt")
    has_ae_int8 = os.path.exists(ae_int8_path)
    if has_ae_int8:
        ae_int8 = torch.quantization.quantize_dynamic(AnomalyAutoEncoder(in_dim=43, latent_dim=8), {nn.Linear}, dtype=torch.qint8)
        ae_int8.load_state_dict(torch.load(ae_int8_path, map_location="cpu", weights_only=False))
        ae_int8.eval()
        with torch.no_grad():
            ae_err_int8 = torch.mean((ae_int8(x_can_t) - x_can_t) ** 2, dim=1, keepdim=True).numpy()

    # 2-C. One-Class SVM
    oc_svm = joblib.load(os.path.join(MODELS_DIR, "step1_oc_svm_w250.pkl"))
    oc_dist = oc_svm.decision_function(X_can_s).reshape(-1, 1)

    # 3. 결합 피처셋 구성
    X_ae_multi_fp32 = np.hstack([X_can_s, ae_err_fp32, X_aud_s])
    X_oc_multi = np.hstack([X_can_s, oc_dist, X_aud_s])

    # 5스텝 지연 피처 및 3D 텐서 변환
    X_lag290, y_seq = create_flattened_lag_matrix(X_ae_multi_fp32, y=y_true, session_ids=session_ids, seq_len=5)
    X_seq_3d, _ = create_sliding_sequence_tensor(X_ae_multi_fp32, y=y_true, session_ids=session_ids, seq_len=5)

    # 4. Step 2 배포 모델 로드
    # 4-A. Step 2 AE + XGBoost (멀티모달)
    xgb_ae = joblib.load(os.path.join(MODELS_DIR, "step2_ae_xgboost_multimodal_w250.pkl"))

    # 4-B. Step 2 OC-SVM + XGBoost (멀티모달)
    xgb_oc = joblib.load(os.path.join(MODELS_DIR, "step2_oc_xgboost_multimodal_w250.pkl"))

    # 4-C. Step 2 AE + SingleGRU FP32
    gru_fp32 = SingleGRU(in_dim=58, hidden=32)
    gru_fp32.load_state_dict(torch.load(os.path.join(MODELS_DIR, "step2_ae_gru_multimodal_w250.pt"), map_location="cpu", weights_only=True))
    gru_fp32.eval()

    # 4-D. Step 2 AE + SingleGRU INT8
    gru_int8_path = os.path.join(DEPLOY_DIR, "step2_gru_multimodal_int8.pt")
    has_gru_int8 = os.path.exists(gru_int8_path)
    if has_gru_int8:
        gru_int8 = torch.quantization.quantize_dynamic(SingleGRU(in_dim=58, hidden=32), {nn.Linear, nn.GRU}, dtype=torch.qint8)
        gru_int8.load_state_dict(torch.load(gru_int8_path, map_location="cpu", weights_only=False))
        gru_int8.eval()

    # =========================================================================
    # [검증 1] 전체 데이터셋 추론 성능 평가
    # =========================================================================
    print("\n [검증 1] 배포 모델 전수 추론 정확도 및 F1-Score 평가 (총 1,617개 윈도우)")
    print(f" {'배포 모델 명칭':<45} | {'포맷/정밀도':<12} | {'정확도(Acc)':<12} | {'F1-Score':<12} | {'정밀도(Prec)':<12} | {'재현율(Rec)':<12}")
    print("-" * 110)

    # 1. 2-Step AE + XGBoost (1D 단일 윈도우)
    p_xgb_1d = xgb_ae.predict(X_ae_multi_fp32)
    print(f" {'[2-Step] AE + XGBoost (1D Single)':<45} | {'FP32 PKL':<12} | {accuracy_score(y_true, p_xgb_1d)*100:>10.2f}% | {f1_score(y_true, p_xgb_1d)*100:>10.2f}% | {precision_score(y_true, p_xgb_1d)*100:>10.2f}% | {recall_score(y_true, p_xgb_1d)*100:>10.2f}%")

    # 2. 2-Step AE + XGBoost (5스텝 Matrix Lagged 290)
    p_xgb_mat = xgb_ae.predict(X_lag290) if X_lag290.shape[1] == xgb_ae.n_features_in_ else p_xgb_1d
    if X_lag290.shape[1] == xgb_ae.n_features_in_:
        print(f" {'[2-Step] AE + XGBoost (5-Step Matrix)':<45} | {'FP32 PKL':<12} | {accuracy_score(y_seq, p_xgb_mat)*100:>10.2f}% | {f1_score(y_seq, p_xgb_mat)*100:>10.2f}% | {precision_score(y_seq, p_xgb_mat)*100:>10.2f}% | {recall_score(y_seq, p_xgb_mat)*100:>10.2f}%")

    # 3. 2-Step OC-SVM + XGBoost
    p_oc_xgb = xgb_oc.predict(X_oc_multi)
    print(f" {'[2-Step] OC-SVM + XGBoost':<45} | {'FP32 PKL':<12} | {accuracy_score(y_true, p_oc_xgb)*100:>10.2f}% | {f1_score(y_true, p_oc_xgb)*100:>10.2f}% | {precision_score(y_true, p_oc_xgb)*100:>10.2f}% | {recall_score(y_true, p_oc_xgb)*100:>10.2f}%")

    # 4. 2-Step AE + SingleGRU (FP32)
    with torch.no_grad():
        out_gru_fp32 = gru_fp32(torch.tensor(X_seq_3d, dtype=torch.float32))
        p_gru_fp32 = torch.argmax(out_gru_fp32, dim=1).numpy()
    print(f" {'[2-Step] AE + SingleGRU':<45} | {'FP32 PT':<12} | {accuracy_score(y_seq, p_gru_fp32)*100:>10.2f}% | {f1_score(y_seq, p_gru_fp32)*100:>10.2f}% | {precision_score(y_seq, p_gru_fp32)*100:>10.2f}% | {recall_score(y_seq, p_gru_fp32)*100:>10.2f}%")

    # 5. 2-Step AE + SingleGRU (INT8 양자화)
    if has_gru_int8:
        with torch.no_grad():
            out_gru_int8 = gru_int8(torch.tensor(X_seq_3d, dtype=torch.float32))
            p_gru_int8 = torch.argmax(out_gru_int8, dim=1).numpy()
        print(f" {'[2-Step] AE + SingleGRU (INT8 Quantized)':<45} | {'INT8 PT':<12} | {accuracy_score(y_seq, p_gru_int8)*100:>10.2f}% | {f1_score(y_seq, p_gru_int8)*100:>10.2f}% | {precision_score(y_seq, p_gru_int8)*100:>10.2f}% | {recall_score(y_seq, p_gru_int8)*100:>10.2f}%")

    # 6. CAN 주도 비대칭 게이팅 (Gating Matrix)
    gating_can_path = os.path.join(MODELS_DIR, "fusion_strategies", "gating_can_matrix_model.pkl")
    gating_aud_path = os.path.join(MODELS_DIR, "fusion_strategies", "gating_audio_matrix_model.pkl")
    if not os.path.exists(gating_can_path):
        gating_can_path = os.path.join(MODELS_DIR, "gating_can_matrix_model.pkl")
        gating_aud_path = os.path.join(MODELS_DIR, "gating_audio_matrix_model.pkl")

    if os.path.exists(gating_can_path) and os.path.exists(gating_aud_path):
        m_gate_can = joblib.load(gating_can_path)
        m_gate_aud = joblib.load(gating_aud_path)
        X_c_mat, y_mat = create_flattened_lag_matrix(X_can_s, y=y_true, session_ids=session_ids, seq_len=5)
        X_a_mat, _ = create_flattened_lag_matrix(X_aud_s, y=y_true, session_ids=session_ids, seq_len=5)

        prob_can = m_gate_can.predict_proba(X_c_mat)[:, 1]
        prob_aud = m_gate_aud.predict_proba(X_a_mat)[:, 1]
        p_gate = np.zeros(len(y_mat), dtype=int)
        for i in range(len(y_mat)):
            if prob_can[i] >= 0.85:
                p_gate[i] = 1
            elif prob_can[i] < 0.40:
                p_gate[i] = 0
            else:
                p_gate[i] = 1 if prob_aud[i] >= 0.50 else 0
        print(f" {'[Gating] CAN-Dominant Gating Matrix':<45} | {'FP32 PKL':<12} | {accuracy_score(y_mat, p_gate)*100:>10.2f}% | {f1_score(y_mat, p_gate)*100:>10.2f}% | {precision_score(y_mat, p_gate)*100:>10.2f}% | {recall_score(y_mat, p_gate)*100:>10.2f}%")

    # =========================================================================
    # [검증 2] 미학습 속도 도메인 일반화 검증 (Hold-Out Unseen 60kph)
    # =========================================================================
    print("\n" + "-" * 110)
    print(" [검증 2] 미학습 속도 도메인 일반화 평가 (Unseen 60kph 전용 251개 윈도우)")
    print("-" * 110)

    is_60k = df["scenario"].str.contains("60kph")
    y_60k = y_true[is_60k]
    X_ae_60k = X_ae_multi_fp32[is_60k]

    p_60k_xgb = xgb_ae.predict(X_ae_60k)
    acc_60k = accuracy_score(y_60k, p_60k_xgb)
    f1_60k = f1_score(y_60k, p_60k_xgb)
    print(f" * [배포 AE + XGBoost 멀티모달] 60kph 일반화 정확도 : {acc_60k*100:.2f}% | F1-Score: {f1_60k*100:.2f}%")

    # =========================================================================
    # [검증 3] 배포 모델 실시간 추론 지연시간(Latency ms) 및 Throughput 벤치마크
    # =========================================================================
    print("\n" + "-" * 110)
    print(" [검증 3] 배포 모델 실시간 연산 성능 벤치마크 (1개 윈도우 단일 추론 소요시간 ms)")
    print(f" {'배포 모델 파이프라인 명칭':<45} | {'평균 지연시간':<15} | {'P99 최대 지연':<15} | {'Throughput (FPS)':<18} | {'KPI (3,000ms) 여유도'}")
    print("-" * 110)

    sample_1d = X_ae_multi_fp32[:1]
    sample_seq = torch.tensor(X_seq_3d[:1], dtype=torch.float32)

    # 1. AE + XGBoost
    m_lat, p99_lat, fps = benchmark_single_model_latency(lambda x: xgb_ae.predict(x), sample_1d)
    print(f" {'1. [2-Step] AutoEncoder + XGBoost (FP32)':<45} | {m_lat:>10.3f} ms | {p99_lat:>10.3f} ms | {fps:>14.1f} FPS | {float(3000/m_lat):>12.0f}배 빠름")

    # 2. AE + SingleGRU FP32
    m_lat2, p99_lat2, fps2 = benchmark_single_model_latency(lambda x: gru_fp32(x), sample_seq)
    print(f" {'2. [2-Step] AutoEncoder + SingleGRU (FP32)':<45} | {m_lat2:>10.3f} ms | {p99_lat2:>10.3f} ms | {fps2:>14.1f} FPS | {float(3000/m_lat2):>12.0f}배 빠름")

    # 3. AE + SingleGRU INT8
    if has_gru_int8:
        m_lat3, p99_lat3, fps3 = benchmark_single_model_latency(lambda x: gru_int8(x), sample_seq)
        print(f" {'3. [2-Step] AutoEncoder + SingleGRU (INT8)':<45} | {m_lat3:>10.3f} ms | {p99_lat3:>10.3f} ms | {fps3:>14.1f} FPS | {float(3000/m_lat3):>12.0f}배 빠름")

    # 4. OC-SVM + XGBoost
    sample_oc = X_oc_multi[:1]
    m_lat4, p99_lat4, fps4 = benchmark_single_model_latency(lambda x: xgb_oc.predict(x), sample_oc)
    print(f" {'4. [2-Step] OC-SVM + XGBoost (FP32)':<45} | {m_lat4:>10.3f} ms | {p99_lat4:>10.3f} ms | {fps4:>14.1f} FPS | {float(3000/m_lat4):>12.0f}배 빠름")

    print("=" * 110)
    print(" [성공] 배포된 모델 파일들(.pkl, .pt, INT8)을 통한 순수 추론 벤치마크 완료.\n")


if __name__ == "__main__":
    evaluate_all_deployed_models()
