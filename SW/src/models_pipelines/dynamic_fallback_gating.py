"""
dynamic_fallback_gating.py (gearbox_final_suite / src / models_pipelines)

[STEP 1 - STEP 2 하이브리드 고장 진단 및 오디오 폴백 제어 시스템]

설명:
- 1단계(STEP 1): CAN 전용 비지도 이상 탐지(AutoEncoder)를 수행하여 이상치 점수(복원 오차) 산출.
  - 이상치 점수가 임계치 미만이면 최종 정상(Normal)으로 조기 판정하여 불필요한 연산 차단 (Filtering).
  - 이상치 점수가 임계치 이상이면 2단계(STEP 2) 정밀 분류 세션으로 진입.
- 2단계(STEP 2): 수집 윈도우 범위 내 오디오 수집 유효성(audio_is_valid)을 검사.
  - 오디오 유효 시: 2단계 멀티모달 모델(CAN + AE_Score + Audio 피처, 58차원)로 최종 고장 여부 판정.
  - 오디오 누락 시: 2단계 CAN 전용 모델(CAN + AE_Score 피처, 44차원)로 즉시 폴백하여 진단 누락 방지.

규정 준수:
- Rule 3: 코드 주석 및 독스트링 100% 한글.
- Rule 7: LaTeX 수식 기호 일절 사용 금지.
- Rule 8: 이모지 전면 배제.
"""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn


class AnomalyAutoEncoder(nn.Module):
    """1단계 CAN 전용 비지도 복원 신경망 클래스 정의"""
    def __init__(self, in_dim=43, latent_dim=8):
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
        return self.decoder(z)


class TwoStepFallbackSystem:
    """
    STEP 1 스크리닝 및 STEP 2 오디오 동적 폴백 제어 통합 시스템
    """
    def __init__(self, models_dir):
        """
        인자:
        - models_dir: 스케일러 및 pkl/pt 모델들이 저장된 폴더 경로
        """
        # 스케일러 로드
        self.scaler_can = joblib.load(os.path.join(models_dir, "scaler_can_w250.pkl"))
        self.scaler_aud = joblib.load(os.path.join(models_dir, "scaler_audio_w250.pkl"))

        # STEP 1: AutoEncoder 로드 (FP32)
        self.ae_model = AnomalyAutoEncoder(in_dim=43, latent_dim=8)
        self.ae_model.load_state_dict(
            torch.load(os.path.join(models_dir, "step1_autoencoder_w250.pt"), map_location="cpu", weights_only=True)
        )
        self.ae_model.eval()

        # STEP 2: XGBoost 모델 로드
        # - 멀티모달 (58개 피처: CAN 43 + AE_Score 1 + Audio 14)
        self.xgb_multimodal = joblib.load(os.path.join(models_dir, "step2_ae_xgboost_multimodal_w250.pkl"))
        # - CAN 전용 (44개 피처: CAN 43 + AE_Score 1)
        self.xgb_can_only = joblib.load(os.path.join(models_dir, "step2_ae_xgboost_w250.pkl"))

    def diagnose_window(self, can_raw_vector, audio_raw_vector=None, audio_is_valid=True, step1_threshold=0.02):
        """
        단일 데이터 윈도우에 대해 STEP 1 스크리닝 및 STEP 2 동적 폴백 고장 판정을 실행합니다.

        인자:
        - can_raw_vector: 정규화 전 CAN 입력 피처 벡터 (1D numpy array, 43차원)
        - audio_raw_vector: 정규화 전 오디오 입력 피처 벡터 (1D numpy array, 14차원 또는 None)
        - audio_is_valid: 오디오 수집 유효성 플래그 (True / False)
        - step1_threshold: STEP 1 조기 필터링을 위한 복원 오차(MSE) 임계치 (기본값: 0.02)

        반환값:
        - prediction: 최종 진단 결과 (0: 정상, 1: 고장)
        - ae_error: STEP 1 AutoEncoder 복원 오차 값
        - step_active: 실행된 단계 제어 문자열 ('STEP1_FILTERED_NORMAL', 'STEP2_MULTIMODAL', 'STEP2_CAN_ONLY_FALLBACK')
        """
        # =====================================================================
        # [STEP 1] CAN 전용 비지도 이상 탐지 및 필터링
        # =====================================================================
        # 1. CAN 피처 정규화
        can_reshaped = can_raw_vector.reshape(1, -1)
        can_scaled = self.scaler_can.transform(can_reshaped)

        # 2. AutoEncoder 복원 오차(Anomaly Score) 산출
        with torch.no_grad():
            can_t = torch.tensor(can_scaled, dtype=torch.float32)
            recon_t = self.ae_model(can_t)
            ae_error = float(torch.mean((recon_t - can_t) ** 2).item())

        # 3. 조기 필터링 판단
        if ae_error < step1_threshold:
            # 정상 기준선을 통과했으므로 STEP 2 구동 없이 즉시 정상(0) 판정
            return 0, ae_error, "STEP1_FILTERED_NORMAL"

        # =====================================================================
        # [STEP 2] 정밀 고장 분류 및 오디오 동적 폴백 제어
        # =====================================================================
        # 1단계에서 걸러지지 않은 경우 2단계 정밀 분류 수행
        if audio_is_valid and (audio_raw_vector is not None):
            # 2-A. 오디오가 유효함: 58차원 멀티모달 모델 구동
            audio_reshaped = audio_raw_vector.reshape(1, -1)
            audio_scaled = self.scaler_aud.transform(audio_reshaped)

            # 피처 결합 (CAN 43 + AE_Score 1 + Audio 14 = 58)
            X_step2_multi = np.hstack([can_scaled, [[ae_error]], audio_scaled])
            
            prediction = int(self.xgb_multimodal.predict(X_step2_multi)[0])
            return prediction, ae_error, "STEP2_MULTIMODAL"

        else:
            # 2-B. 오디오 단선/누락 발생: 44차원 CAN 전용 모델로 즉시 폴백 (윈도우 드롭 방지)
            # 피처 결합 (CAN 43 + AE_Score 1 = 44)
            X_step2_can = np.hstack([can_scaled, [[ae_error]]])
            
            prediction = int(self.xgb_can_only.predict(X_step2_can)[0])
            return prediction, ae_error, "STEP2_CAN_ONLY_FALLBACK"


if __name__ == "__main__":
    print(" [dynamic_fallback_gating] STEP 1/2 고장 진단 및 폴백 시스템 기동 테스트")
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SUITE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
    MODELS_DIR = os.path.join(SUITE_DIR, "models")

    try:
        sys_test = TwoStepFallbackSystem(models_dir=MODELS_DIR)
        print(" * 모델 및 스케일러 로드 완료.")

        mock_can = np.random.normal(size=43)
        mock_aud = np.random.normal(size=14)

        # 1. 스크리닝 필터 통과 (임계치 크게 설정)
        pred, err, step = sys_test.diagnose_window(mock_can, mock_aud, audio_is_valid=True, step1_threshold=9.9)
        print(f" [테스트 1 - 조기필터링] 예측: {pred} | 에러: {err:.4f} | 동작단계: {step}")

        # 2. 2단계 멀티모달 진단 (임계치 작게 설정, 오디오 유효)
        pred, err, step = sys_test.diagnose_window(mock_can, mock_aud, audio_is_valid=True, step1_threshold=0.0)
        print(f" [테스트 2 - 멀티모달]   예측: {pred} | 에러: {err:.4f} | 동작단계: {step}")

        # 3. 2단계 폴백 진단 (임계치 작게 설정, 오디오 누락)
        pred, err, step = sys_test.diagnose_window(mock_can, None, audio_is_valid=False, step1_threshold=0.0)
        print(f" [테스트 3 - CAN폴백]    예측: {pred} | 에러: {err:.4f} | 동작단계: {step}")

    except Exception as e:
        print(f" [테스트 공지] 로컬 모델 파일 결손으로 동작 모의 테스트는 생략합니다. (에러: {e})")
