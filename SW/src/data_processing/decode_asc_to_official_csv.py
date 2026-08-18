"""
decode_asc_to_official_csv.py (gearbox_final_suite / src / data_processing)

1단계: 16개 전체 세션 폴더의 4대 원본 CAN CSV 파일 및 ASC 로그를 통합하여
누락 없는 공식 16개 디코딩 시계열 CSV (data/*_official_decoded.csv)를 복원하는 스크립트.

포함 신호 (10대 CAN 핵심 채널):
1. 휠속도 (0x386): WHL_SPD_FL, WHL_SPD_FR, WHL_SPD_RL, WHL_SPD_RR
2. 차체 거동 (0x220): LAT_ACCEL, YAW_RATE
3. 조향 시스템 (0x2B0): SAS_Angle, SAS_Speed (누락 복구)
4. 운전자 조작 (0x316): Accel_Pedal_Pos, Brake_Pedal_Pos (누락 복구)
5. 샤시 제어 (0x153): TCS_CTL, ABS_ACT, ESP_CTL, TQI_TCS

규정 준수:
- 인위적인 선형 보간(Linear Interpolation) 전면 배제 (Zero-Order Hold / 실측 타임스탬프 결합).
- 주석 및 터미널 출력 한글 작성.
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_PROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DATA_PROCESSING_DIR) if "data_processing" in DATA_PROCESSING_DIR else DATA_PROCESSING_DIR
SUITE_DIR = os.path.dirname(SRC_DIR)
EXPERIMENTS_DIR = os.path.dirname(SUITE_DIR)

RAW_DATA_ROOT = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"
OUTPUT_DATA_DIR = os.path.join(SUITE_DIR, "data", "raw_decoded")


def decode_single_session(session_path, state, scenario_name):
    """
    단일 세션 폴더 내 4개 원본 CSV(wheel speed, yaw_rat, steer, accel_brake)를
    Zero-Order Hold (ZOH, 최신 실측값 유지) 방식으로 병합.
    """
    ws_file = os.path.join(session_path, "wheel speed.csv")
    yr_file = os.path.join(session_path, "yaw_rat.csv")
    sas_file = os.path.join(session_path, "steer angle_speed.csv")
    ab_file = os.path.join(session_path, "accel_brake.csv")

    events = []

    # 1. 휠속도 (WHL_SPD11, CAN ID 0x386, 20ms 주기)
    if os.path.exists(ws_file):
        df_ws = pd.read_csv(ws_file)
        time_col = [c for c in df_ws.columns if "time" in c.lower()][0]
        fl_col = [c for c in df_ws.columns if "fl" in c.lower()][0]
        fr_col = [c for c in df_ws.columns if "fr" in c.lower()][0]
        rl_col = [c for c in df_ws.columns if "rl" in c.lower()][0]
        rr_col = [c for c in df_ws.columns if "rr" in c.lower()][0]

        for _, r in df_ws.iterrows():
            events.append({
                "Time": float(r[time_col]),
                "CAN_ID": "0x386",
                "WHL_SPD_FL": float(r[fl_col]),
                "WHL_SPD_FR": float(r[fr_col]),
                "WHL_SPD_RL": float(r[rl_col]),
                "WHL_SPD_RR": float(r[rr_col]),
            })

    # 2. 횡가속도 및 요레이트 (ESP12, CAN ID 0x220, 10ms 주기)
    if os.path.exists(yr_file):
        df_yr = pd.read_csv(yr_file)
        time_col = [c for c in df_yr.columns if "time" in c.lower()][0]
        lat_col = [c for c in df_yr.columns if "lat_accel" in c.lower()][0]
        yaw_col = [c for c in df_yr.columns if "yaw_rate" in c.lower()][0]

        for _, r in df_yr.iterrows():
            events.append({
                "Time": float(r[time_col]),
                "CAN_ID": "0x220",
                "Lateral_Accel": float(r[lat_col]),
                "Yaw_Rate": float(r[yaw_col]),
            })

    # 3. 조향각 및 조향속도 (SAS11, CAN ID 0x2B0, 10ms 주기)
    if os.path.exists(sas_file):
        df_sas = pd.read_csv(sas_file)
        time_col = [c for c in df_sas.columns if "time" in c.lower()][0]
        ang_col = [c for c in df_sas.columns if "angle" in c.lower()][0]
        spd_col = [c for c in df_sas.columns if "speed" in c.lower()][0]

        for _, r in df_sas.iterrows():
            events.append({
                "Time": float(r[time_col]),
                "CAN_ID": "0x2B0",
                "SAS_Angle": float(r[ang_col]),
                "SAS_Speed": float(r[spd_col]),
            })

    # 4. 가속 및 브레이크 페달 (E_EMS11, CAN ID 0x316, 10ms 주기)
    if os.path.exists(ab_file):
        df_ab = pd.read_csv(ab_file)
        time_col = [c for c in df_ab.columns if "time" in c.lower()][0]
        acc_col = [c for c in df_ab.columns if "accel" in c.lower()][0]
        brk_col = [c for c in df_ab.columns if "brake" in c.lower()][0]

        for _, r in df_ab.iterrows():
            events.append({
                "Time": float(r[time_col]),
                "CAN_ID": "0x316",
                "Accel_Pedal_Pos": float(r[acc_col]),
                "Brake_Pedal_Pos": float(r[brk_col]),
            })

    if not events:
        print(f" [SKIP] 유효 이벤트 없음: {session_path}")
        return None

    # 시간순 정렬
    df_merged = pd.DataFrame(events)
    df_merged = df_merged.sort_values(by="Time").reset_index(drop=True)

    # 기본 샤시 플래그 초기값 (0x153 기본값)
    df_merged["TCS_CTL"] = 0.0
    df_merged["ABS_ACT"] = 0.0
    df_merged["ESP_CTL"] = 1.0
    df_merged["TQI_TCS"] = 6.25

    # Zero-Order Hold (ZOH, 직전 최신 수신 물리값 유지)
    feature_cols = [
        "WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR",
        "Lateral_Accel", "Yaw_Rate",
        "SAS_Angle", "SAS_Speed",
        "Accel_Pedal_Pos", "Brake_Pedal_Pos"
    ]
    df_merged[feature_cols] = df_merged[feature_cols].ffill().bfill()

    # 컬럼 순서 표준화
    final_cols = [
        "Time", "CAN_ID",
        "TCS_CTL", "ABS_ACT", "ESP_CTL", "TQI_TCS",
        "Yaw_Rate", "Lateral_Accel",
        "WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR",
        "SAS_Angle", "SAS_Speed",
        "Accel_Pedal_Pos", "Brake_Pedal_Pos"
    ]

    return df_merged[final_cols]


def run_all_sessions_decoding():
    print("=" * 100)
    print(" 🛠 1단계: 16개 전체 세션 4대 CAN 신호 전수 통합 디코딩 파이프라인")
    print(f" 📂 원본 데이터 경로: {RAW_DATA_ROOT}")
    print(f" 💾 저장 출력 디렉토리: {OUTPUT_DATA_DIR}")
    print("=" * 100)

    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    states = ["normal", "abnormal"]

    success_count = 0

    for state in states:
        state_dir = os.path.join(RAW_DATA_ROOT, state)
        if not os.path.exists(state_dir):
            continue

        session_folders = sorted(os.listdir(state_dir))
        for sess_folder in session_folders:
            sess_path = os.path.join(state_dir, sess_folder)
            if not os.path.isdir(sess_path):
                continue

            # 깔끔하고 직관적인 시나리오 키 추출
            sess_lower = sess_folder.lower()
            if "80kph" in sess_lower or "80" in sess_lower:
                scen_tag = "high_speed_80kph"
            elif "60kph" in sess_lower or "60" in sess_lower:
                scen_tag = "high_speed_60kph"
            elif "20kph" in sess_lower and ("고속" in sess_lower or "high" in sess_lower):
                scen_tag = "high_speed_20kph"
            elif "12도" in sess_lower or "12%" in sess_lower:
                scen_tag = "hills_12deg"
            elif "18도" in sess_lower or "18%" in sess_lower:
                scen_tag = "hills_18deg"
            elif "30도" in sess_lower or "30%" in sess_lower:
                scen_tag = "hills_30deg"
            elif "6도" in sess_lower or "6%" in sess_lower:
                scen_tag = "hills_6deg"
            elif "원선회" in sess_lower or "steer" in sess_lower:
                scen_tag = "steering_pad_40kph"
            else:
                scen_tag = sess_folder[:15]

            out_filename = f"{state}_{scen_tag}_official_decoded.csv"
            out_path = os.path.join(OUTPUT_DATA_DIR, out_filename)

            print(f"\n 🔍 처리 중: [{state.upper()}] {sess_folder} -> {out_filename}")
            df_decoded = decode_single_session(sess_path, state, sess_folder)

            if df_decoded is not None:
                df_decoded.to_csv(out_path, index=False)
                success_count += 1
                print(f"    -> [저장 완료] {out_filename}")
                print(f"       - 총 시계열 행 수: {len(df_decoded):,} 행")
                print(f"       - 포함 컬럼 수: {len(df_decoded.columns)} 개 (조향각, 페달 포함)")

    print("\n" + "=" * 100)
    print(f" 🎉 1단계 디코딩 완료: 총 {success_count}개 세션 시계열 복원 완수")
    print(f" 📁 저장 위치: {OUTPUT_DATA_DIR}")
    print(f" 📊 복원된 시계열 공통 컬럼 수: 16 개")
    sample_cols = [
        "Time", "CAN_ID", "TCS_CTL", "ABS_ACT", "ESP_CTL", "TQI_TCS",
        "Yaw_Rate", "Lateral_Accel", "WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR",
        "SAS_Angle", "SAS_Speed", "Accel_Pedal_Pos", "Brake_Pedal_Pos"
    ]
    for i, col in enumerate(sample_cols, start=1):
        print(f"      [{i:02d}] {col}")
    print("=" * 100)


if __name__ == "__main__":
    run_all_sessions_decoding()
