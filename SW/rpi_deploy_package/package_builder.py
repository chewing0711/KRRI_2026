"""
package_builder.py (gearbox_final_suite / rpi_deploy_package)

라즈베리 파이(RPi 4/5) 독립 배포 패키지 자동 조립 및 무결성 검증 스크립트
"""

import os
import shutil
import glob
import json

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(DEPLOY_DIR)

MODELS_SRC = os.path.join(SUITE_DIR, "models")
CPP_SRC = os.path.join(SUITE_DIR, "CPP_CONVERSION")
DATA_SRC = os.path.join(SUITE_DIR, "data", "unified_multimodal_dataset_w250.csv")

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
    print(" [라즈베리 파이 전용 독립 배포 패키지 자동 조립]")
    print(f" 📂 대상 폴더: {DEPLOY_DIR}")
    print("=" * 90)

    # 1. 데이터셋 복사
    if os.path.exists(DATA_SRC):
        shutil.copy2(DATA_SRC, os.path.join(TARGET_DATA, "unified_multimodal_dataset_w250.csv"))
        print(" * [DATA]   통합 멀티모달 데이터셋 복사 완료 (1,617 윈도우)")

    # 2. 모델 및 스케일러 복사 (FP32 원본 + INT8 양자화 + Gating Matrix 모델)
    FUSION_MODELS_SRC = os.path.join(MODELS_SRC, "fusion_strategies")
    model_files = [
        os.path.join(MODELS_SRC, "scaler_can_w250.pkl"),
        os.path.join(MODELS_SRC, "scaler_audio_w250.pkl"),
        os.path.join(MODELS_SRC, "step1_autoencoder_w250.pt"),
        os.path.join(MODELS_SRC, "step1_oc_svm_w250.pkl"),
        os.path.join(MODELS_SRC, "step2_ae_xgboost_multimodal_w250.pkl"),
        os.path.join(MODELS_SRC, "step2_ae_gru_multimodal_w250.pt"),
        os.path.join(MODELS_SRC, "step2_oc_xgboost_multimodal_w250.pkl"),
        os.path.join(CPP_SRC, "models", "step1_autoencoder_int8.pt"),
        os.path.join(CPP_SRC, "models", "step2_gru_multimodal_int8.pt"),
        os.path.join(CPP_SRC, "models", "step2_ae_xgboost_multimodal.json"),
        os.path.join(CPP_SRC, "models", "rpi_deployment_manifest.json"),
        os.path.join(FUSION_MODELS_SRC, "gating_can_matrix_model.pkl"),
        os.path.join(FUSION_MODELS_SRC, "gating_audio_matrix_model.pkl"),
        os.path.join(FUSION_MODELS_SRC, "feature_attenuation_matrix_model.pkl"),
        os.path.join(FUSION_MODELS_SRC, "gating_fusion_manifest.json")
    ]

    for mf in model_files:
        if os.path.exists(mf):
            shutil.copy2(mf, os.path.join(TARGET_MODELS, os.path.basename(mf)))
            print(f" * [MODEL]  {os.path.basename(mf)} 복사 완료")

    # 3. 핵심 파이썬 모듈 복사
    src_files = [
        (os.path.join(SUITE_DIR, "src", "data_processing", "sequence_matrix_builder.py"), os.path.join(TARGET_SRC, "sequence_matrix_builder.py")),
        (os.path.join(SUITE_DIR, "src", "models_pipelines", "train_unified_models.py"), os.path.join(TARGET_SRC, "train_unified_models.py")),
        (os.path.join(SUITE_DIR, "src", "models_pipelines", "experiment_multimodal_fusion_strategies.py"), os.path.join(TARGET_SRC, "experiment_multimodal_fusion_strategies.py")),
        (os.path.join(CPP_SRC, "run_2hours_thermal_stress_test.py"), os.path.join(DEPLOY_DIR, "run_2hours_thermal_stress_test.py")),
        (os.path.join(CPP_SRC, "thermal_live_dashboard.py"), os.path.join(DEPLOY_DIR, "thermal_live_dashboard.py")),
        (os.path.join(CPP_SRC, "evaluate_deployed_models.py"), os.path.join(DEPLOY_DIR, "evaluate_deployed_models.py"))
    ]

    for src_p, dst_p in src_files:
        if os.path.exists(src_p):
            shutil.copy2(src_p, dst_p)
            print(f" * [SCRIPT] {os.path.basename(dst_p)} 복사 완료")

    print("=" * 90)
    print(" [완료] 라즈베리 파이 독립 배포 패키지 조립 완료.")
    print("=" * 90)


if __name__ == "__main__":
    build_rpi_package()
