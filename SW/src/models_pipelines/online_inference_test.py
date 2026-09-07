"""
online_inference_test.py (git_uploads/HW)

실시간 socketcan 인터페이스로부터 수신되는 CAN 메시지 스트림에 대해
1초 주기로 디코딩, ZOH 정합, feature extraction 및 2-Step 모델 추론을 수행하는 실시간 검증 스크립트.

사용 방법:
1. CAN 인터페이스 활성화 (가상 CAN 활성화 또는 실차 연결):
   sudo ip link add dev can0 type vcan
   sudo ip link set up can0
2. 스크립트 가동:
   python online_inference_test.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import can

# 모듈 로드 및 디렉토리 경로 구성
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR) if "models_pipelines" in CURRENT_DIR else CURRENT_DIR
SUITE_DIR = os.path.dirname(SRC_DIR)
HW_DIR = os.path.abspath(os.path.join(SUITE_DIR, "..", "HW"))
DATA_PROCESSING_DIR = os.path.abspath(os.path.join(SRC_DIR, "data_processing"))

sys.path.append(HW_DIR)
sys.path.append(DATA_PROCESSING_DIR)

import can_parser
import decoding_functions
from realtime_inference_wrapper import process_decoded_1s

# 모델 저장 폴더
MODELS_DIR = os.path.abspath(os.path.join(SUITE_DIR, "models"))

# -------------------------------------------------------------------------
# [네트워크 아키텍처 정의]
# -------------------------------------------------------------------------
class AnomalyAutoEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=8):
        super(AnomalyAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 24), nn.ReLU(),
            nn.Linear(24, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.ReLU(),
            nn.Linear(24, in_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class SingleGRU(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super(SingleGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# -------------------------------------------------------------------------
# [모델 로드 헬퍼]
# -------------------------------------------------------------------------
def load_models():
    scaler_path = os.path.join(MODELS_DIR, "scaler_w34.pkl")
    ae_path = os.path.join(MODELS_DIR, "step1_autoencoder_w34.pt")
    gru_path = os.path.join(MODELS_DIR, "step2_ae_gru_w34.pt")

    for path in [scaler_path, ae_path, gru_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] 모델 파일을 찾을 수 없습니다 (먼저 train 스크립트를 가동하여 모델을 생성하세요): {path}")

    scaler = joblib.load(scaler_path)
    in_dim = len(scaler.mean_)  # 41차원 예상

    # Step 1 AutoEncoder 로드
    ae = AnomalyAutoEncoder(in_dim=in_dim, latent_dim=8)
    ae.load_state_dict(torch.load(ae_path, map_location=torch.device('cpu')))
    ae.eval()

    # Step 2 GRU 로드 (입력 차원은 CAN 피처 차원 + AE Anomaly Score 1차원 = in_dim + 1)
    gru = SingleGRU(in_dim=in_dim + 1, hidden=32)
    gru.load_state_dict(torch.load(gru_path, map_location=torch.device('cpu')))
    gru.eval()

    print(f" [INFO] Real-time inference 모델 로드 성공 (Feature Dim: {in_dim})")
    return scaler, ae, gru

# -------------------------------------------------------------------------
# [메인 실행 루프]
# -------------------------------------------------------------------------
def main():
    print("=" * 100)
    print(" [온라인 실시간 고장 진단 인퍼런스 테스트 루프 시작]")
    print("=" * 100)

    try:
        scaler, ae, gru = load_models()
    except Exception as e:
        print(e)
        sys.exit(1)

    # socketcan 버스 초기화 (채널명: can0)
    try:
        bus = can.Bus(interface="socketcan", channel="can0")
        print(" [INFO] socketcan 'can0' 버스 연결 완료. 데이터 대기 중...")
    except Exception as e:
        print(f"[ERROR] CAN 버스 연결 실패 (can_parser 가상 소켓 또는 인터페이스 구동 상태를 확인하세요): {e}")
        sys.exit(1)

    start_time = None
    window_idx = 0

    try:
        while True:
            msg = bus.recv(timeout=1.0)
            if msg is None:
                continue

            received_time, can_id, data = can_parser.parser(msg)

            if start_time is None:
                start_time = received_time

            if can_id is not None:
                relative_time = round(received_time - start_time, 5)
                results_1s = can_parser.collect_data(relative_time, can_id, data)

                # 1.0초 단위 버퍼 획득 시점
                if results_1s is not None:
                    # 1) 디코딩 수행
                    dataset = decoding_functions.decoding(results_1s)
                    
                    # 2) 실시간 ZOH 및 sub-window 피처 추출 -> [1, 10, 41] sequence matrix 획득
                    sequence_matrix = process_decoded_1s(dataset, spd_target=20.0, grd_target=0.0)
                    
                    if sequence_matrix is not None:
                        t0 = time.perf_counter()
                        
                        # 3) Scaler 및 AutoEncoder Anomaly Score 연산
                        # sequence_matrix 형상 [1, 10, 41]에서 각 step(0.1초)별 정규화 수행
                        seq_len = sequence_matrix.shape[1]
                        feats_dim = sequence_matrix.shape[2]
                        
                        seq_flat = sequence_matrix.reshape(-1, feats_dim)
                        seq_scaled = scaler.transform(seq_flat)
                        
                        # PyTorch tensor 변환
                        tensor_scaled = torch.tensor(seq_scaled, dtype=torch.float32)
                        with torch.no_grad():
                            # Step 1: AE 재구성 오차 계산
                            reconstructed = ae(tensor_scaled)
                            reconstruction_errors = torch.mean((reconstructed - tensor_scaled) ** 2, dim=1, keepdim=True)
                            
                            # CAN 피처에 Anomaly Score 병합 -> [10, 42]
                            combined_features = torch.cat([tensor_scaled, reconstruction_errors], dim=1)
                            
                            # GRU 입력용 형상 변환 -> [1, 10, 42]
                            gru_input = combined_features.unsqueeze(0)
                            
                            # Step 2: GRU 고장 감지 추론
                            logits = gru(gru_input)
                            fault_prob = float(torch.softmax(logits, dim=1)[:, 1].item())
                            pred_label = 1 if fault_prob >= 0.50 else 0
                        
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                        diag_str = "⚠️ [ABNORMAL FAULT]" if pred_label == 1 else "✅ [NORMAL]"
                        
                        print(f" [Win {window_idx:03d}] 진단 결과: {diag_str} | 고장 확률: {fault_prob * 100:>6.2f}% | 연산 지연: {latency_ms:>5.3f} ms")
                        window_idx += 1

    except KeyboardInterrupt:
        print("\n [INFO] 사용자에 의해 강제 종료되었습니다.")
    finally:
        bus.shutdown()
        print(" [INFO] CAN 버스 리소스가 성공적으로 해제되었습니다.")

if __name__ == "__main__":
    main()
