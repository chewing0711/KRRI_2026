"""
trim_and_copy_kona_logs.py

코나 EV 유효 주행 6개 CSV 로그를 복사본 폴더(data/kona_ev_trimmed_raw_logs/)에 복사하고,
README 검증 기준에 맞추어 유실(Intermittent) 이전 유효 구간만 엄격하게 정밀 슬라이싱(Crop)하는 스크립트.

출력 디렉토리: /mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_trimmed_raw_logs/

규정 준수:
- 원본 로그 절대 변경하지 않음 (복사본 디렉토리에 잘라낸 정제본 보관).
- 주석 및 터미널 출력 100% 한글 작성.
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm

KONA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/hyundai-kona-ev-can-logs"
TRIMMED_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_trimmed_raw_logs"

# 의미 있는 6개 주행 로그 및 유효 타임스탬프 범위 정의 (Microseconds)
# README 공식 메타데이터 검증 기준
LOG_CROP_RULES = {
    "221217-2-2021-pcan-bcan-drive-modes.csv": {
        "ts_min": 362000000,   # TS 362초 주행 시작
        "ts_max": 557000000,   # TS 557초 이전 유효 (이후 패킷 유실 구간 자름)
        "desc": "드라이브 모드, 급가속, 급제동 (TS 362s~557s 정밀 자르기)"
    },
    "221217-3-2021-pcan-ccan-drive-modes-abs-traction.csv": {
        "ts_min": 0,
        "ts_max": 695000000,   # TS 695초 이전 유효
        "desc": "ABS & 트랙션 제어 주행 (TS 695s 이하 자르기)"
    },
    "230112-4-2019-pcan-ccan-driving.csv": {
        "ts_min": 0,
        "ts_max": float("inf"), # 전체 100% 유효
        "desc": "실도로 일반 주행 (전체 100% 보존)"
    },
    "240114-08-2022-rolling.csv": {
        "ts_min": 0,
        "ts_max": float("inf"), # 전체 100% 유효
        "desc": "관성 주행 Rolling (전체 100% 보존)"
    },
    "240114-12-2022-cruise.csv": {
        "ts_min": 0,
        "ts_max": float("inf"), # 전체 100% 유효
        "desc": "크루즈 주행 Cruise (전체 100% 보존)"
    },
    "240512-2-pcan-regen-brake.csv": {
        "ts_min": 0,
        "ts_max": float("inf"), # 전체 100% 유효
        "desc": "회생 제동 주행 Regen Brake (전체 100% 보존)"
    }
}


def prepare_and_trim_logs():
    print("=" * 100)
    print(" 📂 KONA EV 6개 주행 로그 복사본 폴더 생성 및 유효 구간 슬라이싱(Crop)")
    print(f" 📁 복사 및 정제 대상 경로: {TRIMMED_DIR}")
    print("=" * 100)

    os.makedirs(TRIMMED_DIR, exist_ok=True)

    for fname, rule in LOG_CROP_RULES.items():
        src_path = os.path.join(KONA_DIR, fname)
        dst_path = os.path.join(TRIMMED_DIR, fname)

        print("\n" + "-" * 100)
        print(f" 📄 파일 처리 중: {fname}")
        print(f" ℹ️ 설명: {rule['desc']}")
        print(f" 🎯 슬라이싱 조건: {rule['ts_min'] / 1e6:.1f}초 ~ {rule['ts_max'] / 1e6 if rule['ts_max'] != float('inf') else 'END'}초")

        if not os.path.exists(src_path):
            print(f" [ERROR] 원본 파일 없음: {src_path}")
            continue

        # 원본 읽기
        print(" ⏳ Raw 패킷 읽는 중...")
        df_raw = pd.read_csv(src_path, low_memory=False)
        raw_rows = len(df_raw)

        # Time Stamp 슬라이싱
        if "Time Stamp" in df_raw.columns:
            ts_series = pd.to_numeric(df_raw["Time Stamp"], errors="coerce")
            mask = (ts_series >= rule["ts_min"]) & (ts_series <= rule["ts_max"])
            df_trimmed = df_raw[mask].copy()
        else:
            df_trimmed = df_raw.copy()

        trimmed_rows = len(df_trimmed)

        # 슬라이싱 결과 복사본 폴더에 CSV 저장
        df_trimmed.to_csv(dst_path, index=False)

        drop_rate = (1.0 - (trimmed_rows / raw_rows)) * 100.0 if raw_rows > 0 else 0.0
        print(f" 💾 [저장 완료] -> {dst_path}")
        print(f"    - 원본 패킷 행 수 : {raw_rows:,} 개")
        print(f"    - 정제 후 행 수   : {trimmed_rows:,} 개 (유실 무효 패킷 {drop_rate:.1f}% 제거됨)")

    print("\n" + "=" * 100)
    print(f" 🎉 KONA EV 유효 6개 주행 로그 복사 및 슬라이싱 완전 완수!")
    print(f" 📁 저장된 폴더: {TRIMMED_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    prepare_and_trim_logs()
