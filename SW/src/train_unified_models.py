"""
train_unified_models.py (gearbox_final_suite / src)

신규 CAN 통합 데이터셋(500샘플 및 250샘플) 기반 모델 벤치마크 학습 및 엄격한 검증 실행 진입점.
"""

from models_pipelines.train_unified_models import run_all_models

if __name__ == "__main__":
    run_all_models()
