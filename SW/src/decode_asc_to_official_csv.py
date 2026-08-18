"""
decode_asc_to_official_csv.py (gearbox_final_suite / src)

1단계: 16개 전체 세션 폴더의 4대 원본 CAN CSV 파일 및 ASC 로그를 통합하여
누락 없는 공식 16개 디코딩 시계열 CSV (data/*_official_decoded.csv)를 복원하는 스크립트.
"""

from data_processing.decode_asc_to_official_csv import run_all_sessions_decoding

if __name__ == "__main__":
    run_all_sessions_decoding()
