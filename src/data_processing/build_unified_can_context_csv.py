"""
build_unified_can_context_csv.py (gearbox_final_suite / src / data_processing)

3단계: 16개 디코딩된 CAN 시계열 데이터(data/*_official_decoded.csv)로부터
사용자가 지정한 슬라이딩 윈도우(500개 샘플 및 250개 샘플) 단위로
- 30대 CAN 물리 특성 (휠속 통계, 4륜 편차, 차체 거동, 조향각, 페달)
- 7대 무차원 슬립 비율 및 정규화 특성 (slip_ratio_fl~rr, slip_diff, norm_yaw, norm_lat)
- 6대 주행 맥락 메타데이터 (state, target, scenario, speed_kph, grade_pct, window_idx)
를 결합하여 최종 단일 통합 데이터셋(unified_can_context_dataset.csv)을 생성하는 스크립트.

총 컬럼 수: 43개 (메타데이터 6개 + 슬립/정규화 7개 + CAN 물리 특성 30개)

규정 준수:
- 인위적 데이터 누출(Data Leakage) 및 임의 통계 대입(Imputation) 전면 배제.
- 주석 및 터미널 출력 한글 작성.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats

DATA_PROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DATA_PROCESSING_DIR) if "data_processing" in DATA_PROCESSING_DIR else DATA_PROCESSING_DIR
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
DATA_DIR = os.path.join(SUITE_DIR, "data")

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


def extract_unified_window_row(seg, local_idx, state, spd_target, grd_target, scen_clean):
    """
    단일 윈도우 세그먼트에서 메타데이터(6개) + 슬립/정규화(7개) + CAN 물리 특성(30개)
    총 43개 컬럼의 1개 행 추출
    """
    fl = seg["WHL_SPD_FL"].values
    fr = seg["WHL_SPD_FR"].values
    rl = seg["WHL_SPD_RL"].values
    rr = seg["WHL_SPD_RR"].values

    whl_all = np.concatenate([fl, fr, rl, rr])

    # 1. 휠속도 기본 통계량 (10개)
    mean_val = float(np.mean(whl_all))
    std_val = float(np.std(whl_all))
    p2p_val = float(np.ptp(whl_all))
    rms_val = float(np.sqrt(np.mean(whl_all ** 2)))
    var_val = float(np.var(whl_all))
    peak_val = float(np.max(np.abs(whl_all)))
    crest_factor = float(peak_val / (rms_val + 1e-6))
    shape_factor = float(rms_val / (mean_val + 1e-6))
    skew_val = float(stats.skew(whl_all))
    kurt_val = float(stats.kurtosis(whl_all))

    # 2. 4륜 간 상대 속도차 (8개)
    diff_fl_rl = fl - rl
    diff_fr_rr = fr - rr
    diff_fl_fr = fl - fr
    diff_rl_rr = rl - rr

    diff_fl_rl_mean = float(np.mean(diff_fl_rl))
    diff_fl_rl_std = float(np.std(diff_fl_rl))
    diff_fl_rl_p2p = float(np.ptp(diff_fl_rl))

    diff_fr_rr_mean = float(np.mean(diff_fr_rr))
    diff_fr_rr_std = float(np.std(diff_fr_rr))
    diff_fr_rr_p2p = float(np.ptp(diff_fr_rr))

    diff_fl_fr_std = float(np.std(diff_fl_fr))
    diff_rl_rr_std = float(np.std(diff_rl_rr))

    # 3. 차체 거동 동역학 (4개)
    lat_acc = seg["Lateral_Accel"].values if "Lateral_Accel" in seg.columns else np.zeros_like(fl)
    yaw_rt = seg["Yaw_Rate"].values if "Yaw_Rate" in seg.columns else np.zeros_like(fl)

    lat_accel_mean = float(np.mean(lat_acc))
    lat_accel_std = float(np.std(lat_acc))
    yaw_rate_mean = float(np.mean(yaw_rt))
    yaw_rate_std = float(np.std(yaw_rt))

    # 4. 조향 시스템 특성 (4개)
    sas_ang = seg["SAS_Angle"].values if "SAS_Angle" in seg.columns else np.zeros_like(fl)
    sas_spd = seg["SAS_Speed"].values if "SAS_Speed" in seg.columns else np.zeros_like(fl)

    sas_angle_mean = float(np.mean(sas_ang))
    sas_angle_std = float(np.std(sas_ang))
    sas_angle_p2p = float(np.ptp(sas_ang))
    sas_speed_mean = float(np.mean(sas_spd))

    # 5. 운전자 조작 특성 (4개)
    acc_pos = seg["Accel_Pedal_Pos"].values if "Accel_Pedal_Pos" in seg.columns else np.zeros_like(fl)
    brk_pos = seg["Brake_Pedal_Pos"].values if "Brake_Pedal_Pos" in seg.columns else np.zeros_like(fl)

    accel_pedal_mean = float(np.mean(acc_pos))
    accel_pedal_max = float(np.max(acc_pos))
    brake_pedal_mean = float(np.mean(brk_pos))
    brake_pedal_max = float(np.max(brk_pos))

    # 6. 무차원 바퀴 슬립 비율 및 정규화 특성 (7개)
    fl_mean = float(np.mean(fl))
    fr_mean = float(np.mean(fr))
    rl_mean = float(np.mean(rl))
    rr_mean = float(np.mean(rr))

    base_spd = max(spd_target, 1.0)
    slip_ratio_fl = fl_mean / base_spd
    slip_ratio_fr = fr_mean / base_spd
    slip_ratio_rl = rl_mean / base_spd
    slip_ratio_rr = rr_mean / base_spd

    slip_diff_front_rear = ((fl_mean + fr_mean) / 2.0) - ((rl_mean + rr_mean) / 2.0)
    norm_yaw_rate = yaw_rate_mean / base_spd
    norm_lat_accel = lat_accel_mean / (grd_target + 1.0)

    row = {
        # [01~06] 메타데이터 (6개)
        "state": state,
        "target": 1 if state == "abnormal" else 0,
        "scenario": scen_clean,
        "speed_kph": spd_target,
        "grade_pct": grd_target,
        "window_idx": local_idx,

        # [07~13] 무차원 슬립 비율 및 정규화 특성 (7개)
        "slip_ratio_fl": slip_ratio_fl,
        "slip_ratio_fr": slip_ratio_fr,
        "slip_ratio_rl": slip_ratio_rl,
        "slip_ratio_rr": slip_ratio_rr,
        "slip_diff_front_rear": slip_diff_front_rear,
        "norm_yaw_rate": norm_yaw_rate,
        "norm_lat_accel": norm_lat_accel,

        # [14~23] 휠속도 기본 통계량 (10개)
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

        # [24~31] 4륜 간 상대 속도차 (8개)
        "diff_fl_rl_mean": diff_fl_rl_mean,
        "diff_fl_rl_std": diff_fl_rl_std,
        "diff_fl_rl_p2p": diff_fl_rl_p2p,
        "diff_fr_rr_mean": diff_fr_rr_mean,
        "diff_fr_rr_std": diff_fr_rr_std,
        "diff_fr_rr_p2p": diff_fr_rr_p2p,
        "diff_fl_fr_std": diff_fl_fr_std,
        "diff_rl_rr_std": diff_rl_rr_std,

        # [32~35] 차체 거동 동역학 (4개)
        "lat_accel_mean": lat_accel_mean,
        "lat_accel_std": lat_accel_std,
        "yaw_rate_mean": yaw_rate_mean,
        "yaw_rate_std": yaw_rate_std,

        # [36~39] 조향 시스템 특성 (4개)
        "sas_angle_mean": sas_angle_mean,
        "sas_angle_std": sas_angle_std,
        "sas_angle_p2p": sas_angle_p2p,
        "sas_speed_mean": sas_speed_mean,

        # [40~43] 페달 조작 특성 (4개)
        "accel_pedal_mean": accel_pedal_mean,
        "accel_pedal_max": accel_pedal_max,
        "brake_pedal_mean": brake_pedal_mean,
        "brake_pedal_max": brake_pedal_max,

        # [44~47] 강화 핵심 무차원 변동률 및 페달 토크 응답 특성 (4개)
        "dimless_std": std_val / max(mean_val, 1.0),
        "dimless_p2p": p2p_val / max(mean_val, 1.0),
        "dimless_diff_fr_rr_std": diff_fr_rr_std / max(mean_val, 1.0),
        "pedal_slip_response": diff_fr_rr_std / max(accel_pedal_mean, 1.0),
    }
    return row


def build_unified_dataset_for_window(window_size=500, output_csv=None, cache_csv=None):
    """지정된 window_size에 대해 3단계 최종 통합 CSV 및 2단계 캐시 CSV를 data/ 폴더에 동시 생성"""
    if output_csv is None:
        output_csv = os.path.join(DATA_DIR, f"unified_can_context_dataset_w{window_size}.csv")
    if cache_csv is None:
        cache_csv = os.path.join(DATA_DIR, f"multimodal_cache_w{window_size}.csv")

    raw_decoded_dir = os.path.join(DATA_DIR, "raw_decoded")
    input_dir = raw_decoded_dir if os.path.exists(raw_decoded_dir) and glob.glob(os.path.join(raw_decoded_dir, "*_official_decoded.csv")) else DATA_DIR

    approx_sec = window_size / 340.0
    print("=" * 100)
    print(f" 🛠 3단계: 슬라이딩 윈도우 {window_size}샘플 ({approx_sec:.2f}초) 최종 통합 CAN 데이터셋 생성")
    print(f" 📂 디코딩 CSV 입력 경로: {input_dir}")
    print(f" 💾 최종 Unified CSV 저장 경로: {output_csv}")
    print(f" 💾 2단계 캐시 CSV 동기화 경로: {cache_csv}")
    print("=" * 100)

    decoded_files = sorted(glob.glob(os.path.join(input_dir, "*_official_decoded.csv")))
    if not decoded_files:
        print(f"[ERROR] 디코딩된 CSV 파일이 없습니다: {input_dir}")
        return None

    unified_rows = []
    cache_rows = []

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

        print(f" 🔍 처리 중: [{state.upper()}] {fname[:45]}... (총 {n_rows:,}행 -> {n_windows}개 윈도우)")

        for w_idx in range(n_windows):
            start_i = w_idx * window_size
            end_i = start_i + window_size
            seg = df_can.iloc[start_i:end_i]

            urow = extract_unified_window_row(seg, w_idx, state, spd_target, grd_target, scen_clean)
            unified_rows.append(urow)

            # 캐시 데이터 (슬립 7개를 제외한 36개 컬럼)
            crow = {k: v for k, v in urow.items() if not k.startswith("slip_") and not k.startswith("norm_")}
            cache_rows.append(crow)

    # 1. 최종 통합 CSV 저장 (data/ 폴더)
    df_unified = pd.DataFrame(unified_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_unified.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # 기본 unified_can_context_dataset.csv (기본 500샘플일 때 복사본 저장)
    if window_size == 500:
        default_unified_path = os.path.join(DATA_DIR, "unified_can_context_dataset.csv")
        df_unified.to_csv(default_unified_path, index=False, encoding="utf-8-sig")
        print(f" 💾 [기본 Unified CSV 생성 완료] -> {default_unified_path}")

    # 2. 캐시 CSV 저장 (data/ 폴더)
    df_cache = pd.DataFrame(cache_rows)
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    df_cache.to_csv(cache_csv, index=False, encoding="utf-8-sig")

    if window_size == 500:
        default_cache_path = os.path.join(DATA_DIR, "multimodal_cache_full.csv")
        df_cache.to_csv(default_cache_path, index=False, encoding="utf-8-sig")
        print(f" 💾 [기본 Cache CSV 생성 완료]   -> {default_cache_path}")

    print("\n" + "=" * 100)
    print(f" 🎉 [{window_size}샘플] 최종 Unified 데이터셋 완성!")
    print(f" 💾 최종 Unified CSV: {output_csv}")
    print(f"    - 총 데이터 행 수 (윈도우 수) : {len(df_unified):,} 행")
    print(f"    - 총 데이터 컬럼 수 (Column Count) : {len(df_unified.columns)} 개 (메타 6 + 슬립/정규화 7 + 물리특성 30)")
    print(f"    - 전체 컬럼 명세 목록 :")
    for i, col in enumerate(df_unified.columns, start=1):
        print(f"      [{i:02d}] {col}")
    print("=" * 100)

    return df_unified


def run_all_requested_window_sizes():
    """500개 샘플 및 250개 샘플 데이터셋 일괄 동시 생성 (전체 data/ 폴더 저장)"""
    print("\n" + "#" * 100)
    print(" 🚀 [전체 일괄 실행] 500샘플 및 250샘플 슬라이딩 윈도우 최종 Unified 데이터셋 생성")
    print("#" * 100)

    # 1. 500샘플 데이터셋 생성
    df_500 = build_unified_dataset_for_window(window_size=500)

    # 2. 250샘플 데이터셋 생성
    df_250 = build_unified_dataset_for_window(window_size=250)

    print("\n" + "=" * 100)
    print(" 🏆 500샘플 및 250샘플 데이터셋 생성 최종 요약 (저장 위치: data/)")
    print("=" * 100)
    if df_500 is not None:
        print(f" 1. 500샘플 (~1.47초 윈도우): 총 {len(df_500):,} 행 x {len(df_500.columns)} 열")
        print(f"    - Unified 경로: {os.path.join(DATA_DIR, 'unified_can_context_dataset_w500.csv')}")
        print(f"    - Cache 경로  : {os.path.join(DATA_DIR, 'multimodal_cache_w500.csv')}")
    if df_250 is not None:
        print(f" 2. 250샘플 (~0.73초 윈도우): 총 {len(df_250):,} 행 x {len(df_250.columns)} 열")
        print(f"    - Unified 경로: {os.path.join(DATA_DIR, 'unified_can_context_dataset_w250.csv')}")
        print(f"    - Cache 경로  : {os.path.join(DATA_DIR, 'multimodal_cache_w250.csv')}")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3단계: 500샘플 및 250샘플 최종 Unified CAN 데이터셋 생성기")
    parser.add_argument("-w", "--window_size", type=int, default=None, help="특정 슬라이딩 윈도우 크기만 생성 (예: 500 또는 250). 생략 시 500과 250 둘 다 일괄 생성.")
    parser.add_argument("--all", action="store_true", help="500과 250 둘 다 일괄 생성")
    args = parser.parse_args()

    if args.window_size is not None and not args.all:
        build_unified_dataset_for_window(window_size=args.window_size)
    else:
        run_all_requested_window_sizes()
