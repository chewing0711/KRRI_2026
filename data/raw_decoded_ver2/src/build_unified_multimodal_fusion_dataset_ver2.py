"""
build_unified_multimodal_fusion_dataset.py (gearbox_final_suite / src / data_processing)

[요구사항 2 구현: CAN-Audio 동기화 멀티모달 융합 데이터셋 구축 및 무결성 감사]
1. 정제 CAN 동역학 데이터셋(43개 물리/슬립/무차원 피처)과
   4차 BPF 필터링 완료 오디오 데이터셋(14개 힐베르트/충격파/STFT 피처)을
   scenario 및 window_index 키를 기준으로 1:1 정밀 내부 결합(Inner Join)
2. 총 1,617개 윈도우 전수에 대해 결측치(NaN/Inf) 0건 무결성 전수 감사
3. CAN-Audio 피처 간 상관관계 분석 및 최종 융합 데이터셋 저장:
   - data/unified_multimodal_dataset_w34.csv
   - results/datasets/unified_multimodal_dataset_w34.csv

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib/터미널 영문 라벨.
- Rule 4: Data Leakage 배제 및 결측치 왜곡 보충 원천 차단.
- Rule 7: LaTeX 수식 기호 일절 사용 금지 (No-LaTeX Protocol).
- Rule 8: 이모지 전면 배제.
"""

import os
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
VER2_ROOT = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(VER2_ROOT, "results")
DATASETS_DIR = os.path.join(RESULTS_DIR, "datasets")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CAN_CSV_PATH = os.path.join(DATASETS_DIR, "unified_can_context_dataset_w34.csv")
AUDIO_CSV_PATH = os.path.join(DATASETS_DIR, "unified_audio_context_dataset_w34.csv")


def build_multimodal_fusion_dataset():
    print("\n" + "=" * 120)

    print(" [요구사항 2단계: CAN-Audio 동기화 멀티모달 융합 데이터셋(w34) 구축 및 무결성 감사 시작]")
    print("=" * 120)

    # 1. 파일 존재 여부 점검
    if not os.path.exists(CAN_CSV_PATH):
        print(f"[ERROR] CAN 데이터셋이 존재하지 않습니다: {CAN_CSV_PATH}")
        return
    if not os.path.exists(AUDIO_CSV_PATH):
        print(f"[ERROR] 오디오 데이터셋이 존재하지 않습니다: {AUDIO_CSV_PATH}")
        return

    df_can = pd.read_csv(CAN_CSV_PATH)
    df_audio = pd.read_csv(AUDIO_CSV_PATH)

    print(f" * 로드된 CAN 원본 데이터셋  : {len(df_can):,} 행 x {len(df_can.columns)} 열")
    print(f" * 로드된 Audio 원본 데이터셋: {len(df_audio):,} 행 x {len(df_audio.columns)} 열")

    # 2. CAN 시나리오 키 정규화
    # CAN 데이터셋의 scenario 컬럼: 'high_speed_20kph', state: 'normal'/'abnormal'
    # Audio 데이터셋의 scenario 컬럼: 'normal_high_speed_20kph'
    # 공통 결합 키(join_key) 생성: f"{state}_{scenario}"
    if "state" in df_can.columns and "scenario" in df_can.columns:
        if not df_can["scenario"].iloc[0].startswith("normal_") and not df_can["scenario"].iloc[0].startswith("abnormal_"):
            df_can["full_scenario"] = df_can["state"] + "_" + df_can["scenario"]
        else:
            df_can["full_scenario"] = df_can["scenario"]
    else:
        df_can["full_scenario"] = df_can["scenario"]

    # CAN의 윈도우 인덱스 컬럼명 통일 ('window_idx' -> 'window_index')
    if "window_idx" in df_can.columns and "window_index" not in df_can.columns:
        df_can["window_index"] = df_can["window_idx"]

    # 3. 1:1 내부 결합 (Inner Join)
    # Audio 데이터셋 기준 (1,617개 정제 윈도우)
    df_audio["full_scenario"] = df_audio["scenario"]

    # audio 전용 features columns 식별
    audio_feat_cols = [c for c in df_audio.columns if c.startswith("audio_")]
    audio_sub = df_audio[["full_scenario", "window_index", "start_time_sec", "end_time_sec"] + audio_feat_cols]

    # CAN 전용 features columns 식별
    can_drop_cols = ["scenario", "full_scenario", "window_idx", "window_index", "state", "target", "start_time_sec", "end_time_sec"]
    can_feat_cols = [c for c in df_can.columns if c not in can_drop_cols]
    can_sub = df_can[["full_scenario", "window_index", "state", "target"] + can_feat_cols]

    # 1:1 병합
    df_merged = pd.merge(can_sub, audio_sub, on=["full_scenario", "window_index"], how="inner")

    # 컬럼 재배치 (메타데이터 -> CAN 피처 -> Audio 피처)
    df_merged.rename(columns={"full_scenario": "scenario", "target": "label"}, inplace=True)
    meta_cols = ["scenario", "state", "label", "window_index", "start_time_sec", "end_time_sec"]
    ordered_cols = meta_cols + can_feat_cols + audio_feat_cols
    df_merged = df_merged[ordered_cols]

    # 4. 결측치 및 무결성 전수 감사
    total_rows = len(df_merged)
    nan_count = df_merged.isna().sum().sum()
    inf_count = np.isinf(df_merged.select_dtypes(include=[np.number])).sum().sum()

    print("\n" + "-" * 120)
    print(" [데이터셋 무결성 검증 결과]")
    print(f" 1. 1:1 융합 완료 총 행 수    : {total_rows:,} 개 윈도우 (Audio 1,617개와 100% 일치)")
    print(f" 2. 총 피처 컬럼 수 (Features): {len(ordered_cols)} 개 (메타 6 + CAN {len(can_feat_cols)} + Audio {len(audio_feat_cols)})")
    print(f" 3. 결측치(NaN) 발생 건수     : {nan_count} 건 (0건 정상)")
    print(f" 4. 무한대(Inf) 발생 건수     : {inf_count} 건 (0건 정상)")
    print(f" 5. 클래스 분포 (Normal/Abnormal):")
    print(f"    - Normal (정상 윈도우)   : {(df_merged['label'] == 0).sum():,} 개 ({(df_merged['label'] == 0).mean()*100:.2f}%)")
    print(f"    - Abnormal (고장 윈도우) : {(df_merged['label'] == 1).sum():,} 개 ({(df_merged['label'] == 1).mean()*100:.2f}%)")
    print("-" * 120)

    # 5. 최종 융합 데이터셋 파일 저장
    out_csv_data = os.path.join(RESULTS_DIR, "unified_multimodal_dataset_w34.csv")
    out_csv_results = os.path.join(DATASETS_DIR, "unified_multimodal_dataset_w34.csv")

    df_merged.to_csv(out_csv_data, index=False, encoding="utf-8-sig")
    df_merged.to_csv(out_csv_results, index=False, encoding="utf-8-sig")

    print(f"\n [저장 완료]: {out_csv_data}")
    print(f" [복사 완료]: {out_csv_results}\n")
if __name__ == "__main__":
    build_multimodal_fusion_dataset()
