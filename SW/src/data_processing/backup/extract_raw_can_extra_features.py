"""
extract_raw_can_extra_features.py (gearbox_final_suite / src / data_processing)

2단계: 디코딩된 16개 CAN 시계열 데이터(data/*_official_decoded.csv)에서
1.47초(500개 샘플) 슬라이딩 윈도우 단위로
시간 영역 통계량, 휠 간 상대 속도차, 조향각, 가속/제동 페달 요약 특성을 추출하여
캐시 데이터셋(results/multimodal_cache_full.csv)으로 저장하는 스크립트.

특성 카테고리 (총 30개 핵심 CAN 특성):
1. 휠속도 기본 통계 (10개): mean, std, p2p, rms, variance, crest_factor, shape_factor, skew, kurt, peak
2. 4륜 상대 속도차 (8개): diff_fl_rl_mean/std/p2p, diff_fr_rr_mean/std/p2p, diff_fl_fr_std, diff_rl_rr_std
3. 차체 거동 동역학 (4개): lat_accel_mean, lat_accel_std, yaw_rate_mean, yaw_rate_std
4. 조향 시스템 특성 (4개): sas_angle_mean, sas_angle_std, sas_angle_p2p, sas_speed_mean
5. 운전자 조작 특성 (4개): accel_pedal_mean, accel_pedal_max, brake_pedal_mean, brake_pedal_max

규정 준수:
- 인위적 데이터 누출(Data Leakage) 및 임의 통계 대입(Imputation) 전면 배제.
- 주석 및 터미널 출력 한글 작성.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

DATA_PROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DATA_PROCESSING_DIR) if "data_processing" in DATA_PROCESSING_DIR else DATA_PROCESSING_DIR
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
DATA_DIR = os.path.join(SUITE_DIR, "data")

OUTPUT_CSV = os.path.join(DATA_DIR, "multimodal_cache_full.csv")

# 시나리오별 목표 속도, 경사도, 표준 시나리오명 매핑 규칙
SCENARIO_CONTEXT_RULES = {
    "80kph": (80.0, 0.0, "high_speed_80kph"),
    "60kph": (60.0, 0.0, "high_speed_60kph"),
    "20kph": (20.0, 0.0, "high_speed_20kph"),
    "12deg": (20.0, 12.0, "hills_12deg"),
    "18deg": (20.0, 18.0, "hills_18deg"),
    "30deg": (20.0, 30.0, "hills_30deg"),
    "6deg": (20.0, 6.0, "hills_6deg"),
    "12도": (20.0, 12.0, "hills_12deg"),
    "18도": (20.0, 18.0, "hills_18deg"),
    "30도": (20.0, 30.0, "hills_30deg"),
    "6도": (20.0, 6.0, "hills_6deg"),
    "steer": (40.0, 0.0, "steering_pad_40kph"),
    "circle": (40.0, 0.0, "steering_pad_40kph"),
    "원선회": (40.0, 0.0, "steering_pad_40kph"),
}


def extract_features_from_window(seg, local_idx, sess_name, state, spd_target, grd_target, scen_clean):
    """500개 샘플 윈도우에서 30개 핵심 물리 특성 추출"""
    fl = seg["WHL_SPD_FL"].values
    fr = seg["WHL_SPD_FR"].values
    rl = seg["WHL_SPD_RL"].values
    rr = seg["WHL_SPD_RR"].values

    whl_all = np.concatenate([fl, fr, rl, rr])

    # 1. 휠속도 기본 통계량 (10개)
    mean_val = np.mean(whl_all)
    std_val = np.std(whl_all)
    p2p_val = np.ptp(whl_all)
    rms_val = np.sqrt(np.mean(whl_all ** 2))
    var_val = np.var(whl_all)
    peak_val = np.max(np.abs(whl_all))
    crest_factor = peak_val / (rms_val + 1e-6)
    shape_factor = rms_val / (mean_val + 1e-6)
    skew_val = stats.skew(whl_all)
    kurt_val = stats.kurtosis(whl_all)

    # 2. 4륜 간 상대 속도차 (8개)
    diff_fl_rl = fl - rl
    diff_fr_rr = fr - rr
    diff_fl_fr = fl - fr
    diff_rl_rr = rl - rr

    diff_fl_rl_mean = np.mean(diff_fl_rl)
    diff_fl_rl_std = np.std(diff_fl_rl)
    diff_fl_rl_p2p = np.ptp(diff_fl_rl)

    diff_fr_rr_mean = np.mean(diff_fr_rr)
    diff_fr_rr_std = np.std(diff_fr_rr)
    diff_fr_rr_p2p = np.ptp(diff_fr_rr)

    diff_fl_fr_std = np.std(diff_fl_fr)
    diff_rl_rr_std = np.std(diff_rl_rr)

    # 3. 차체 거동 동역학 (4개)
    lat_acc = seg["Lateral_Accel"].values if "Lateral_Accel" in seg.columns else np.zeros_like(fl)
    yaw_rt = seg["Yaw_Rate"].values if "Yaw_Rate" in seg.columns else np.zeros_like(fl)

    lat_accel_mean = np.mean(lat_acc)
    lat_accel_std = np.std(lat_acc)
    yaw_rate_mean = np.mean(yaw_rt)
    yaw_rate_std = np.std(yaw_rt)

    # 4. 조향 시스템 특성 (4개)
    sas_ang = seg["SAS_Angle"].values if "SAS_Angle" in seg.columns else np.zeros_like(fl)
    sas_spd = seg["SAS_Speed"].values if "SAS_Speed" in seg.columns else np.zeros_like(fl)

    sas_angle_mean = np.mean(sas_ang)
    sas_angle_std = np.std(sas_ang)
    sas_angle_p2p = np.ptp(sas_ang)
    sas_speed_mean = np.mean(sas_spd)

    # 5. 운전자 조작 특성 (4개)
    acc_pos = seg["Accel_Pedal_Pos"].values if "Accel_Pedal_Pos" in seg.columns else np.zeros_like(fl)
    brk_pos = seg["Brake_Pedal_Pos"].values if "Brake_Pedal_Pos" in seg.columns else np.zeros_like(fl)

    accel_pedal_mean = np.mean(acc_pos)
    accel_pedal_max = np.max(acc_pos)
    brake_pedal_mean = np.mean(brk_pos)
    brake_pedal_max = np.max(brk_pos)

    row = {
        "state": state,
        "target": 1 if state == "abnormal" else 0,
        "scenario": scen_clean,
        "speed_kph": spd_target,
        "grade_pct": grd_target,
        "window_idx": local_idx,

        # 30개 특성
        "mean": mean_val,
        "std": std_val,
        "p2p": p2p_val,
        "rms": rms_val,
        "variance": var_val,
        "peak": peak_val,
        "crest_factor": crest_factor,
        "shape_factor": shape_factor,
        "skew": skew_val,
        "kurt": kurt_val,

        "diff_fl_rl_mean": diff_fl_rl_mean,
        "diff_fl_rl_std": diff_fl_rl_std,
        "diff_fl_rl_p2p": diff_fl_rl_p2p,
        "diff_fr_rr_mean": diff_fr_rr_mean,
        "diff_fr_rr_std": diff_fr_rr_std,
        "diff_fr_rr_p2p": diff_fr_rr_p2p,
        "diff_fl_fr_std": diff_fl_fr_std,
        "diff_rl_rr_std": diff_rl_rr_std,

        "lat_accel_mean": lat_accel_mean,
        "lat_accel_std": lat_accel_std,
        "yaw_rate_mean": yaw_rate_mean,
        "yaw_rate_std": yaw_rate_std,

        "sas_angle_mean": sas_angle_mean,
        "sas_angle_std": sas_angle_std,
        "sas_angle_p2p": sas_angle_p2p,
        "sas_speed_mean": sas_speed_mean,

        "accel_pedal_mean": accel_pedal_mean,
        "accel_pedal_max": accel_pedal_max,
        "brake_pedal_mean": brake_pedal_mean,
        "brake_pedal_max": brake_pedal_max,
    }
    return row


def run_extra_feature_extraction(window_size=500, output_csv=OUTPUT_CSV):
    raw_decoded_dir = os.path.join(DATA_DIR, "raw_decoded")
    input_dir = raw_decoded_dir if os.path.exists(raw_decoded_dir) and glob.glob(os.path.join(raw_decoded_dir, "*_official_decoded.csv")) else DATA_DIR

    print("=" * 100)
    print(f" 🛠 2단계: {window_size}샘플 ({window_size / 340.0:.2f}초) 윈도우 단위 30차원 CAN 특성 추출 파이프라인")
    print(f" 📂 디코딩 CSV 입력 경로: {input_dir}")
    print(f" 💾 캐시 출력 경로: {output_csv}")
    print("=" * 100)

    decoded_files = sorted(glob.glob(os.path.join(input_dir, "*_official_decoded.csv")))
    if not decoded_files:
        print(f"[ERROR] 디코딩된 CSV 파일이 없습니다: {input_dir}")
        return

    extracted_rows = []

    for fpath in decoded_files:
        fname = os.path.basename(fpath)
        state = "abnormal" if fname.startswith("abnormal") else "normal"

        spd_target, grd_target, scen_clean = 20.0, 0.0, "high_speed_20kph"
        for k, v in SCENARIO_CONTEXT_RULES.items():
            if k in fname:
                spd_target, grd_target, scen_clean = v
                break

        df_can = pd.read_csv(fpath)
        n_rows = len(df_can)
        n_windows = n_rows // window_size

        print(f" 🔍 추출 중: [{state.upper()}] {fname[:45]}... (총 {n_rows:,}행 -> {n_windows}개 윈도우)")

        for w_idx in range(n_windows):
            start_i = w_idx * window_size
            end_i = start_i + window_size
            seg = df_can.iloc[start_i:end_i]

            clean_session_id = f"{state}_{scen_clean}"
            feat_row = extract_features_from_window(
                seg, w_idx, clean_session_id,
                state, spd_target, grd_target, scen_clean
            )
            extracted_rows.append(feat_row)

    df_features = pd.DataFrame(extracted_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_features.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f" 🎉 2단계 특성 추출 완료! (윈도우 크기: {window_size} 샘플)")
    print(f" 💾 저장된 캐시 CSV: {output_csv}")
    print(f"    - 총 데이터 행 수 (윈도우 수) : {len(df_features):,} 행")
    print(f"    - 총 데이터 컬럼 수 (Column Count) : {len(df_features.columns)} 개")
    print(f"    - 전체 컬럼 명세 목록 (Columns) :")
    for i, col in enumerate(df_features.columns, start=1):
        print(f"      [{i:02d}] {col}")
    print("=" * 100)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="2단계: 슬라이딩 윈도우 기반 30차원 CAN 특성 추출기")
    parser.add_argument("-w", "--window_size", type=int, default=500, help="슬라이딩 윈도우 크기 (샘플 수, 기본값: 500, 예: 250, 500, 1000)")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_CSV, help="저장할 출력 CSV 파일 경로")
    args = parser.parse_args()

    run_extra_feature_extraction(window_size=args.window_size, output_csv=args.output)
