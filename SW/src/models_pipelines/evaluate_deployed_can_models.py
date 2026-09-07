"""
evaluate_deployed_can_models.py (w34 CAN 전용 배포 모델 검증)

[배포 모델 전용 검증 스크립트 (Zero-Training, Pure Inference Evaluation)]

기능:
1. models/ 폴더에 저장된 CAN 전용 배포 모델 파일(.pkl, .pt)을 직접 로드
2. 재학습 없이 순수 추론(Pure Inference)만으로 벤치마크 수행:
   - [검증 1] 전체 윈도우 전수 추론 정확도, F1-Score, 정밀도, 재현율 평가
   - [검증 2] 미학습 60kph 도메인 일반화 추론 성능 평가
   - [검증 3] 실시간 추론 연산 지연시간(Latency ms) 및 Throughput (FPS) 정밀 실측
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
MODELS_DIR = os.path.join(SUITE_DIR, "models")

# CAN 전용 데이터셋 경로 (w34)
DATA_PATH = os.path.join(DATA_DIR, "unified_can_context_dataset_w34.csv")

# 모델 클래스 임포트
sys.path.insert(0, CURRENT_DIR)
from train_unified_models import AnomalyAutoEncoder, SingleGRU, get_ae_error_feature, create_sliding_sequence_tensor


def benchmark_single_model_latency(predict_fn, sample_input, num_runs=100):
    """단일 모델 추론 지연시간 ms 정밀 측정"""
    # Warm-up
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


def evaluate_can_only_deployed_models():
    W_SUFFIX = "_w34"

    print("=" * 110)
    print(" [CAN 전용 배포 모델 성능 및 지연시간 종합 벤치마크] (Zero-Training / Pure Deployed Models)")
    print(f" Dataset: {DATA_PATH}")
    print("=" * 110)

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] 데이터셋 파일 없음: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    meta_cols = ["state", "label", "target", "scenario", "window_idx", "window_index",
                 "session_id", "session_file", "raw_scenario", "start_time_sec", "end_time_sec"]
    can_feature_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("audio_")]
    y_true = df["label"].values if "label" in df.columns else \
             (df["target"].values if "target" in df.columns else
              (df["state"].str.lower() == "abnormal").astype(int).values)
    session_ids = df["session_id"] if "session_id" in df.columns else df["scenario"]

    print(f" * 전체 데이터셋 크기 : {len(df):,} 개 윈도우 (정상: {np.sum(y_true==0):,}개, 고장: {np.sum(y_true==1):,}개)")
    print(f" * CAN 피처: {len(can_feature_cols)}개")
    print("-" * 110)

    # =========================================================================
    # 1. 스케일러 로드 및 전처리
    # =========================================================================
    scaler_can_path = os.path.join(MODELS_DIR, f"scaler_can{W_SUFFIX}.pkl")
    if not os.path.exists(scaler_can_path):
        print(f"[ERROR] CAN 스케일러 없음: {scaler_can_path}")
        return
    scaler_can = joblib.load(scaler_can_path)
    X_can_s = scaler_can.transform(df[can_feature_cols].fillna(0.0).values)

    # =========================================================================
    # 2. Step 1 배포 모델 로드 및 Anomaly Score 산출
    # =========================================================================
    # 2-A. AutoEncoder
    ae_path = os.path.join(MODELS_DIR, f"step1_autoencoder{W_SUFFIX}.pt")
    ae_model = AnomalyAutoEncoder(in_dim=X_can_s.shape[1], latent_dim=8)
    ae_model.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
    ae_model.eval()
    with torch.no_grad():
        x_can_t = torch.tensor(X_can_s, dtype=torch.float32)
        ae_err = torch.mean((ae_model(x_can_t) - x_can_t) ** 2, dim=1, keepdim=True).numpy()

    # 2-B. One-Class SVM
    oc_svm = joblib.load(os.path.join(MODELS_DIR, f"step1_oc_svm{W_SUFFIX}.pkl"))
    oc_dist = oc_svm.decision_function(X_can_s).reshape(-1, 1)

    # =========================================================================
    # 3. CAN 전용 결합 피처셋 구성: [CAN + Anomaly(1)] = 44개
    # =========================================================================
    X_ae_can = np.hstack([X_can_s, ae_err])
    X_oc_can = np.hstack([X_can_s, oc_dist])

    # 시계열 Matrix 변환 (10스텝)
    X_seq_3d_ae, y_seq = create_sliding_sequence_tensor(X_ae_can, y_true, session_ids=session_ids, seq_len=10)

    # =========================================================================
    # 4. Step 2 배포 모델 로드
    # =========================================================================
    # 4-A. AE + XGBoost (CAN)
    xgb_ae = joblib.load(os.path.join(MODELS_DIR, f"step2_ae_xgboost_can{W_SUFFIX}.pkl"))

    # 4-B. OC-SVM + XGBoost (CAN)
    xgb_oc = joblib.load(os.path.join(MODELS_DIR, f"step2_oc_xgboost_can{W_SUFFIX}.pkl"))

    # 4-C. AE + SingleGRU (CAN)
    gru_in_dim = X_ae_can.shape[1]  # 44
    gru_model = SingleGRU(in_dim=gru_in_dim, hidden=32)
    gru_path = os.path.join(MODELS_DIR, f"step2_ae_gru_can{W_SUFFIX}.pt")
    gru_model.load_state_dict(torch.load(gru_path, map_location="cpu", weights_only=True))
    gru_model.eval()

    # =========================================================================
    # [검증 1] 전체 데이터셋 추론 성능 평가
    # =========================================================================
    print("\n [검증 1] 배포 모델 전수 추론 정확도 및 F1-Score 평가")
    print(f" {'배포 모델 명칭':<45} | {'Acc':<12} | {'F1':<12} | {'Prec':<12} | {'Rec':<12}")
    print("-" * 110)

    # 1. AE + XGBoost (1D)
    p_xgb_ae = xgb_ae.predict(X_ae_can)
    print(f" {'[2-Step] AE + XGBoost (CAN 1D)':<45} | {accuracy_score(y_true, p_xgb_ae)*100:>10.2f}% | {f1_score(y_true, p_xgb_ae)*100:>10.2f}% | {precision_score(y_true, p_xgb_ae)*100:>10.2f}% | {recall_score(y_true, p_xgb_ae)*100:>10.2f}%")

    # 2. OC-SVM + XGBoost (1D)
    p_xgb_oc = xgb_oc.predict(X_oc_can)
    print(f" {'[2-Step] OC-SVM + XGBoost (CAN 1D)':<45} | {accuracy_score(y_true, p_xgb_oc)*100:>10.2f}% | {f1_score(y_true, p_xgb_oc)*100:>10.2f}% | {precision_score(y_true, p_xgb_oc)*100:>10.2f}% | {recall_score(y_true, p_xgb_oc)*100:>10.2f}%")

    # 3. AE + SingleGRU (10-Step Matrix)
    if len(y_seq) > 0:
        with torch.no_grad():
            out_gru = gru_model(torch.tensor(X_seq_3d_ae, dtype=torch.float32))
            p_gru = torch.argmax(out_gru, dim=1).numpy()
        print(f" {'[2-Step] AE + SingleGRU (10-Step Matrix)':<45} | {accuracy_score(y_seq, p_gru)*100:>10.2f}% | {f1_score(y_seq, p_gru)*100:>10.2f}% | {precision_score(y_seq, p_gru)*100:>10.2f}% | {recall_score(y_seq, p_gru)*100:>10.2f}%")

    # =========================================================================
    # [검증 2] 미학습 속도 도메인 일반화 검증 (Hold-Out Unseen 60kph)
    # =========================================================================
    print("\n" + "-" * 110)
    print(" [검증 2] 미학습 속도 도메인 일반화 평가 (Unseen 60kph)")
    print("-" * 110)

    is_60k = df["scenario"].str.contains("60kph")
    if is_60k.sum() > 0:
        y_60k = y_true[is_60k]
        X_ae_60k = X_ae_can[is_60k]
        p_60k = xgb_ae.predict(X_ae_60k)
        print(f" * [AE + XGBoost CAN] 60kph Acc : {accuracy_score(y_60k, p_60k)*100:.2f}% | F1: {f1_score(y_60k, p_60k, zero_division=0)*100:.2f}%")
    else:
        print(" * 60kph 시나리오 데이터가 없습니다.")

    # =========================================================================
    # [검증 3] 배포 모델 실시간 추론 지연시간(Latency ms) 벤치마크
    # =========================================================================
    print("\n" + "-" * 110)
    print(" [검증 3] 배포 모델 실시간 연산 성능 벤치마크 (1 윈도우 단일 추론)")
    print(f" {'배포 모델 명칭':<45} | {'Mean Latency':<15} | {'P99 Latency':<15} | {'Throughput':<18} | {'KPI (3,000ms)'}")
    print("-" * 110)

    sample_1d_ae = X_ae_can[:1]
    sample_1d_oc = X_oc_can[:1]

    # 1. AE + XGBoost
    m_lat, p99, fps = benchmark_single_model_latency(lambda x: xgb_ae.predict(x), sample_1d_ae)
    print(f" {'[2-Step] AE + XGBoost (CAN)':<45} | {m_lat:>10.3f} ms | {p99:>10.3f} ms | {fps:>14.1f} FPS | {3000/m_lat:>12.0f}x faster")

    # 2. OC-SVM + XGBoost
    m_lat2, p992, fps2 = benchmark_single_model_latency(lambda x: xgb_oc.predict(x), sample_1d_oc)
    print(f" {'[2-Step] OC-SVM + XGBoost (CAN)':<45} | {m_lat2:>10.3f} ms | {p992:>10.3f} ms | {fps2:>14.1f} FPS | {3000/m_lat2:>12.0f}x faster")

    # 3. AE + SingleGRU
    if len(y_seq) > 0:
        sample_seq = torch.tensor(X_seq_3d_ae[:1], dtype=torch.float32)
        m_lat3, p993, fps3 = benchmark_single_model_latency(lambda x: gru_model(x), sample_seq)
        print(f" {'[2-Step] AE + SingleGRU (10-Step)':<45} | {m_lat3:>10.3f} ms | {p993:>10.3f} ms | {fps3:>14.1f} FPS | {3000/m_lat3:>12.0f}x faster")

    print("=" * 110)
    print(" [완료] CAN 전용 배포 모델 순수 추론 벤치마크 종료.\n")


if __name__ == "__main__":
    evaluate_can_only_deployed_models()
