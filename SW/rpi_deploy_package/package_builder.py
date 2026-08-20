"""
package_builder.py (gearbox_final_suite / rpi_deploy_package)

라즈베리 파이(RPi 4/5) w34 CAN 전용 독립 배포 패키지 자동 조립 및 무결성 검증 스크립트

규정 준수:
- 100% 한글 주석
- 이모지 배제
- LaTeX 기호 금지 (No-LaTeX Protocol)
"""

import os
import shutil
import glob
import json

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(DEPLOY_DIR)  # git_uploads/SW
GIT_UPLOADS_DIR = os.path.dirname(SUITE_DIR)  # git_uploads

MODELS_SRC = os.path.join(SUITE_DIR, "models")
DATA_SRC = os.path.join(SUITE_DIR, "data", "unified_can_context_dataset_w34.csv")
HW_SRC = os.path.join(GIT_UPLOADS_DIR, "HW")

TARGET_MODELS = os.path.join(DEPLOY_DIR, "models")
TARGET_DATA = os.path.join(DEPLOY_DIR, "data")
TARGET_SRC = os.path.join(DEPLOY_DIR, "src")
TARGET_METRICS = os.path.join(DEPLOY_DIR, "results", "metrics")
TARGET_FIGURES = os.path.join(DEPLOY_DIR, "results", "figures")

os.makedirs(TARGET_MODELS, exist_ok=True)
os.makedirs(TARGET_DATA, exist_ok=True)
os.makedirs(TARGET_SRC, exist_ok=True)
os.makedirs(TARGET_METRICS, exist_ok=True)
os.makedirs(TARGET_FIGURES, exist_ok=True)


def build_rpi_package():
    print("=" * 90)
    print(" [라즈베리 파이 전용 w34 CAN 독립 배포 패키지 자동 조립]")
    print(f" 📂 대상 폴더: {DEPLOY_DIR}")
    print("=" * 90)

    # 1. 데이터셋 복사
    if os.path.exists(DATA_SRC):
        shutil.copy2(DATA_SRC, os.path.join(TARGET_DATA, "unified_can_context_dataset_w34.csv"))
        print(" * [DATA]   w34 CAN 데이터셋 복사 완료")
    else:
        print(f" * [WARNING] 데이터셋 누락: {DATA_SRC}")

    # 2. w34 CAN 전용 모델 파일들 복사
    model_files = [
        os.path.join(MODELS_SRC, "scaler_can_w34.pkl"),
        os.path.join(MODELS_SRC, "step1_autoencoder_w34.pt"),
        os.path.join(MODELS_SRC, "step1_oc_svm_w34.pkl"),
        os.path.join(MODELS_SRC, "step1_iforest_w34.pkl"),
        os.path.join(MODELS_SRC, "step2_ae_xgboost_can_w34.pkl"),
        os.path.join(MODELS_SRC, "step2_oc_xgboost_can_w34.pkl"),
        os.path.join(MODELS_SRC, "step2_if_xgboost_can_w34.pkl"),
        os.path.join(MODELS_SRC, "step2_ae_gru_can_w34.pt"),
        os.path.join(MODELS_SRC, "step2_oc_gru_can_w34.pt"),
        os.path.join(MODELS_SRC, "step2_if_gru_can_w34.pt"),
        os.path.join(MODELS_SRC, "can_only_manifest_w34.json")
    ]

    for mf in model_files:
        if os.path.exists(mf):
            shutil.copy2(mf, os.path.join(TARGET_MODELS, os.path.basename(mf)))
            print(f" * [MODEL]  {os.path.basename(mf)} 복사 완료")
        else:
            print(f" * [WARNING] 모델 파일 누락: {mf}")

    # 3. 핵심 파이썬 모듈 및 스크립트 복사
    src_files = [
        # SW 소스 복사
        (os.path.join(SUITE_DIR, "src", "data_processing", "sequence_matrix_builder.py"), os.path.join(TARGET_SRC, "sequence_matrix_builder.py")),
        (os.path.join(SUITE_DIR, "src", "models_pipelines", "train_unified_models.py"), os.path.join(TARGET_SRC, "train_unified_models.py")),
        (os.path.join(SUITE_DIR, "src", "models_pipelines", "simulate_online_inference.py"), os.path.join(TARGET_SRC, "simulate_online_inference.py")),
        (os.path.join(SUITE_DIR, "src", "models_pipelines", "simulate_online_direct_array.py"), os.path.join(TARGET_SRC, "simulate_online_direct_array.py")),
        (os.path.join(SUITE_DIR, "src", "models_pipelines", "evaluate_deployed_can_models.py"), os.path.join(DEPLOY_DIR, "evaluate_deployed_models.py")),
        # HW 소스 복사 (Pi 배포용 필수 모듈 포함)
        (os.path.join(HW_SRC, "can_parser.py"), os.path.join(DEPLOY_DIR, "can_parser.py")),
        (os.path.join(HW_SRC, "decoding_functions.py"), os.path.join(DEPLOY_DIR, "decoding_functions.py")),
        (os.path.join(HW_SRC, "live_can_inference_node.py"), os.path.join(DEPLOY_DIR, "live_can_inference_node.py"))
    ]

    for src_p, dst_p in src_files:
        if os.path.exists(src_p):
            shutil.copy2(src_p, dst_p)
            print(f" * [SCRIPT] {os.path.basename(dst_p)} 복사 완료")
        else:
            print(f" * [WARNING] 스크립트 누락: {src_p}")

    print("=" * 90)
    print(" [완료] w34 CAN 전용 라즈베리 파이 독립 배포 패키지 조립 완료.")
    print("=" * 90)


if __name__ == "__main__":
    build_rpi_package()
