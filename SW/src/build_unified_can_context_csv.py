"""
build_unified_can_context_csv.py (gearbox_final_suite / src)

3단계: 500샘플 및 250샘플 슬라이딩 윈도우 기반 최종 Unified CAN 데이터셋 생성 스크립트.
"""

import argparse
from data_processing.build_unified_can_context_csv import (
    build_unified_dataset_for_window,
    run_all_requested_window_sizes
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3단계: 500샘플 및 250샘플 최종 Unified CAN 데이터셋 생성기")
    parser.add_argument("-w", "--window_size", type=int, default=None, help="특정 슬라이딩 윈도우 크기만 생성 (예: 500 또는 250). 생략 시 500과 250 둘 다 일괄 생성.")
    parser.add_argument("--all", action="store_true", help="500과 250 둘 다 일괄 생성")
    args = parser.parse_args()

    if args.window_size is not None and not args.all:
        build_unified_dataset_for_window(window_size=args.window_size)
    else:
        run_all_requested_window_sizes()
