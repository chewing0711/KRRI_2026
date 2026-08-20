"""
extract_raw_can_extra_features.py (gearbox_final_suite / src)

2단계: 디코딩된 16개 CAN 시계열 데이터에서 사용자가 지정한 슬라이딩 윈도우 단위 30차원 CAN 특성 추출 스크립트.
"""

import argparse
from data_processing.extract_raw_can_extra_features import run_extra_feature_extraction, OUTPUT_CSV

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2단계: 슬라이딩 윈도우 기반 30차원 CAN 특성 추출기")
    parser.add_argument("-w", "--window_size", type=int, default=500, help="슬라이딩 윈도우 크기 (샘플 수, 기본값: 500, 예: 250, 500, 1000)")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_CSV, help="저장할 출력 CSV 파일 경로")
    args = parser.parse_args()

    run_extra_feature_extraction(window_size=args.window_size, output_csv=args.output)
