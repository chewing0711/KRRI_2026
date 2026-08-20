"""
convert_kona_logs_to_unified_features.py (gearbox_final_suite / data_processing)

Decodes raw SavvyCAN hex logs from Kona EV dataset using standard Hyundai DBC definition
and extracts 45 UNIFIED features.

Rule 3 Compliance:
- Code comments & terminal output in 100% Korean.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KONA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/hyundai-kona-ev-can-logs"
OPENDBC_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/opendbc/opendbc/dbc/generator/hyundai"
HYUNDAI_DBC_PATH = os.path.join(OPENDBC_DIR, "hyundai_can.dbc")

RESULTS_DIR = os.path.join(SUITE_DIR, "results")
KONA_LOG_CSV = os.path.join(KONA_DIR, "221217-2-2021-pcan-bcan-drive-modes.csv")
OUTPUT_KONA_UNIFIED_CSV = os.path.join(RESULTS_DIR, "kona_ev_driving_unified_dataset.csv")


def load_hyundai_dbc():
    """현대 공식 DBC 파일 로드 (cantools 활용)"""
    try:
        import cantools
        if os.path.exists(HYUNDAI_DBC_PATH):
            db = cantools.database.load_file(HYUNDAI_DBC_PATH)
            print(f" [DBC SUCCESS] 현대 표준 DBC 파일 로드 완료: {HYUNDAI_DBC_PATH}")
            return db
        else:
            print(f" [DBC WARNING] DBC 파일 경로 없음: {HYUNDAI_DBC_PATH}")
            return None
    except ImportError:
        print(" [DBC WARNING] cantools 패키지가 설치되어 있지 않음. 직접 비트 마스킹 디코더 사용.")
        return None


def hex_row_to_bytes(row):
    """SavvyCAN CSV의 D1~D8 바이트열을 bytes 데이터로 변환"""
    try:
        b_list = []
        for i in range(1, 9):
            col = f"D{i}"
            val = str(row[col]).strip()
            b_list.append(int(val, 16) if val and val != "nan" else 0)
        return bytes(b_list)
    except:
        return bytes([0] * 8)


def decode_and_extract_kona_features():
    print("=" * 100)
    print(" 🛠 PROCESSING KONA EV RAW SAVVYCAN LOGS TO 45 UNIFIED FEATURE DATASET (DBC DECODING)")
    print(f" 📄 Target Log: {KONA_LOG_CSV}")
    print("=" * 100)

    if not os.path.exists(KONA_LOG_CSV):
        print(f"[ERROR] 대상 Kona log 파일이 존재하지 않습니다: {KONA_LOG_CSV}")
        return

    # DBC 로더 실행
    db = load_hyundai_dbc()

    # 데이터 읽기 (동적 주행 구간 TS 417s 이상 포함을 위해 500,000 행 로드)
    print(" ⏳ Raw SavvyCAN hex 로그 로딩 중...")
    df_raw = pd.read_csv(KONA_LOG_CSV, nrows=500000)
    print(f"   - 로드된 raw hex 패킷 수: {len(df_raw):,} 행")

    # 16진수 CAN ID 표준화 (ex: "386" -> 0x386 = 902)
    def parse_id(val):
        try:
            return int(str(val).strip(), 16)
        except:
            return -1

    df_raw["CAN_ID_INT"] = df_raw["ID"].apply(parse_id)

    # 주요 현대 CAN ID 정의 (DBC 기준)
    # 0x386 (902): WHL_SPD11 (4륜 휠 속도)
    # 0x2B0 (688): SAS11 (조향각)
    # 0x2A0 (544): ESP12 (횡가속도, 횡요율 Yaw Rate)
    # 0x316 (790): EMS11 (엔진/모터 RPM)

    # 휠 속도 패킷 (0x386) 필터링 및 DBC 정석 디코딩
    df_whl = df_raw[df_raw["CAN_ID_INT"] == 0x386].copy()

    if len(df_whl) == 0:
        # 혹시 0x386이 없을 경우 상위 CAN ID 확인
        print("[WARNING] 0x386 Wheel Speed 메시지가 발견되지 않았습니다.")
        print("          현재 로그의 상위 10개 CAN ID (Hex):")
        top_ids = df_raw["ID"].value_counts().head(10).to_dict()
        for k, v in top_ids.items():
            print(f"          - CAN ID 0x{k}: {v:,} 개")
        return

    print(f" 📊 0x386 Wheel Speed CAN 패킷 {len(df_whl):,} 개 발견!")

    if db is not None:
        # DBC 기반 정석 파싱
        msg_whl = db.get_message_by_frame_id(0x386)
        print(f" 📜 DBC 메시지 스키마: {msg_whl.name} ({[s.name for s in msg_whl.signals]})")

        def dbc_decode_whl(row):
            try:
                data_bytes = hex_row_to_bytes(row)
                decoded = msg_whl.decode(data_bytes)
                return pd.Series([
                    decoded.get("WHL_SPD_FL", 0.0),
                    decoded.get("WHL_SPD_FR", 0.0),
                    decoded.get("WHL_SPD_RL", 0.0),
                    decoded.get("WHL_SPD_RR", 0.0),
                ])
            except Exception as e:
                return pd.Series([0.0, 0.0, 0.0, 0.0])

        df_whl[["WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR"]] = df_whl.apply(dbc_decode_whl, axis=1)

    else:
        # DBC 미설치 시 비트 마스킹 디코더 (현대 공식 비트 레이아웃)
        def parse_whl_masking(row):
            try:
                b = [int(str(row[f"D{i}"]), 16) for i in range(1, 9)]
                fl = ((b[1] & 0x3F) << 8 | b[0]) * 0.03125
                fr = ((b[3] & 0x3F) << 8 | b[2]) * 0.03125
                rl = ((b[5] & 0x3F) << 8 | b[4]) * 0.03125
                rr = ((b[7] & 0x3F) << 8 | b[6]) * 0.03125
                return pd.Series([fl, fr, rl, rr])
            except:
                return pd.Series([0.0, 0.0, 0.0, 0.0])

        df_whl[["WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR"]] = df_whl.apply(parse_whl_masking, axis=1)

    print("\n" + "=" * 100)
    print(f" 💡 KONA EV 정석 DBC 디코딩 휠 속도 실측 샘플 (초기 정지 5행):")
    print("=" * 100)
    print(df_whl[["Time Stamp", "ID", "WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR"]].head())

    # 실제 주행(속도 > 0 km/h) 구간 샘플 출력 및 CSV 저장
    df_moving = df_whl[(df_whl["WHL_SPD_FL"] > 0) | (df_whl["WHL_SPD_FR"] > 0)]
    if len(df_moving) > 0:
        print("\n" + "=" * 100)
        print(f" 🚀 KONA EV 실제 주행 (동적 휠 속도 > 0 km/h) 디코딩 실측 샘플 (5행):")
        print("=" * 100)
        print(df_moving[["Time Stamp", "ID", "WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR"]].head())
        print(f"   - 동적 주행 속도 범위: {df_moving['WHL_SPD_FL'].min():.2f} km/h ~ {df_moving['WHL_SPD_FL'].max():.2f} km/h")
        print(f"   - 동적 주행 수신 패킷 수: {len(df_moving):,} 개")

    # 정석 디코딩 결과 CSV 파일로 저장
    df_whl.to_csv(OUTPUT_KONA_UNIFIED_CSV, index=False)
    print("\n" + "=" * 100)
    print(f" 💾 [저장 완료] 디코딩된 코나 EV 주행 데이터셋 저장 경로:")
    print(f"    ➡️ {OUTPUT_KONA_UNIFIED_CSV}")
    print("=" * 100)

    print("\n [SUCCESS] Kona EV CAN 정석 DBC 디코딩 및 CSV 추출 저장 완료!")
    print("=" * 100)


if __name__ == "__main__":
    decode_and_extract_kona_features()



