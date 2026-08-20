"""
trim_decoded_kona_csv.py

디코딩 완료된 코나 EV 주행 데이터셋(kona_ev_decoded_221217-2-2021-pcan-bcan-drive-modes.csv)에서
초기 정지 대기 구간(TS 362s 이전) 및 패킷 유실 구간(TS 557s 이후)을 전면 제거하고,
실제 100% 동적 주행 구간(TS 362.0s ~ 557.0s)만 정밀 크롭(Crop)하여 저장하는 스크립트.

입력: /mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_decoded_221217-2-2021-pcan-bcan-drive-modes.csv
출력: /mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_decoded_221217-2_trimmed_valid.csv

규정 준수:
- 주석 및 터미널 출력 100% 한글 작성.
"""

import os
import pandas as pd

INPUT_DECODED_CSV = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_decoded_221217-2-2021-pcan-bcan-drive-modes.csv"
OUTPUT_TRIMMED_CSV = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_decoded_221217-2_trimmed_valid.csv"

# README 검증 공식 타임스탬프 (Microseconds)
TS_MIN = 362000000  # TS 362.0초 (주행 시작)
TS_MAX = 557000000  # TS 557.0초 (패킷 유실 이전 끝)


def trim_decoded_file():
    print("=" * 100)
    print(" ✂️ KONA EV 디코딩 데이터셋 정밀 크롭(Crop) 작업")
    print(f" 📄 대상 파일: {INPUT_DECODED_CSV}")
    print(f" 💾 저장 경로: {OUTPUT_TRIMMED_CSV}")
    print("=" * 100)

    if not os.path.exists(INPUT_DECODED_CSV):
        print(f"[ERROR] 디코딩 파일이 존재하지 않습니다: {INPUT_DECODED_CSV}")
        return

    print(" ⏳ 디코딩된 CSV 로딩 중...")
    df = pd.read_csv(INPUT_DECODED_CSV, low_memory=False)
    raw_len = len(df)
    print(f"   - 로드된 총 행 수: {raw_len:,} 행")

    if "Time Stamp" not in df.columns:
        print("[ERROR] Time Stamp 컬럼이 존재하지 않습니다.")
        return

    # Time Stamp 숫자 변환
    df["Time Stamp Num"] = pd.to_numeric(df["Time Stamp"], errors="coerce")

    # 유효 구간 슬라이싱 (TS 362s ~ 557s)
    df_valid = df[(df["Time Stamp Num"] >= TS_MIN) & (df["Time Stamp Num"] <= TS_MAX)].copy()
    df_valid = df_valid.drop(columns=["Time Stamp Num"])

    trimmed_len = len(df_valid)

    # 저장
    df_valid.to_csv(OUTPUT_TRIMMED_CSV, index=False)

    print("\n" + "=" * 100)
    print(f" 🎉 디코딩 주행 데이터셋 크롭(Crop) 완료!")
    print(f" 💾 저장된 파일 경로: {OUTPUT_TRIMMED_CSV}")
    print(f" 📊 원본 총 행 수 : {raw_len:,} 행")
    print(f" 📊 크롭 후 행 수 : {trimmed_len:,} 행 (초기 정지 및 유실 구간 100% 제거)")
    print(f" ⏱️ 최종 시계열 범위: {TS_MIN / 1e6:.1f}초 ~ {TS_MAX / 1e6:.1f}초 (총 { (TS_MAX - TS_MIN) / 1e6:.1f}초 유효 주행)")
    print("=" * 100)


if __name__ == "__main__":
    trim_decoded_file()
