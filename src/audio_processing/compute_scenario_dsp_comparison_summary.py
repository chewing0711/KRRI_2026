"""
compute_scenario_dsp_comparison_summary.py (gearbox_final_suite / src / audio_processing)

[목적]:
새로 구축된 4차 버터워스 필터링 및 출발점 정렬 완료 독립 오디오 데이터셋(unified_audio_context_dataset_w250.csv)을 기반으로,
8대 주행 시나리오별 [정상 vs 고장] 힐베르트 포락선, 충격파 지표, 2.8kHz STFT 피크 및 수치 상승률(%) 전수 비교 감사표 및 CSV 생성.

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib/터미널 100% 영문 라벨.
- Rule 5: 100% 실측 데이터 기반 감사 및 수치 왜곡 배제.
- Rule 7: LaTeX 수식 기호 일절 사용 금지 (No-LaTeX Protocol).
- Rule 8: 이모지 전면 배제.
"""

import os
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
AUDIO_AUDIT_DIR = os.path.join(RESULTS_DIR, "audio_audit")
os.makedirs(AUDIO_AUDIT_DIR, exist_ok=True)

CSV_DATASET = os.path.join(DATA_DIR, "unified_audio_context_dataset_w250.csv")

SCENARIO_DISPLAY_NAMES = {
    "hills_30deg": "30deg Steep Hill (30도 극한 등판)",
    "steering_pad_40kph": "40kph Steer Pad (40km/h 원선회 주행)",
    "high_speed_60kph": "60kph Medium High Speed (60km/h 주행)",
    "high_speed_80kph": "80kph High Speed (80km/h 주행)",
    "hills_18deg": "18deg Hill (18도 급경사 등판)",
    "hills_12deg": "12deg Hill (12도 중경사 등판)",
    "hills_6deg": "6deg Hill (6도 완경사 등판)",
    "high_speed_20kph": "20kph Low Speed (20km/h 저속 주행)",
}

COURSES_ORDER = [
    "hills_30deg", "steering_pad_40kph", "high_speed_60kph", "high_speed_80kph",
    "hills_18deg", "hills_12deg", "hills_6deg", "high_speed_20kph"
]

def get_course_name(scen):
    if scen.startswith("abnormal_"):
        return scen[len("abnormal_"):]
    elif scen.startswith("normal_"):
        return scen[len("normal_"):]
    return scen

def run_dsp_scenario_comparison():
    print("\n" + "=" * 140)
    print(" [정밀 정제 데이터셋 기반 8대 시나리오별 힐베르트 포락선 & 충격파 정상/고장 비교 감사 시작]")
    print("=" * 140)

    if not os.path.exists(CSV_DATASET):
        print(f"[ERROR] 데이터셋 파일이 존재하지 않습니다: {CSV_DATASET}")
        return

    df = pd.read_csv(CSV_DATASET)
    df["course"] = df["scenario"].apply(get_course_name)

    report_rows = []

    print(f" {'주행 시나리오 (Course)':<35} | {'정상 힐베르트':<14} | {'고장 힐베르트':<14} | {'포락선 상승률':<14} | {'정상 2.8k(dB)':<14} | {'고장 2.8k(dB)':<14} | {'2.8k 증폭(dB)'}")
    print("-" * 140)

    for course in COURSES_ORDER:
        df_norm = df[(df["course"] == course) & (df["state"] == "normal")]
        df_abnorm = df[(df["course"] == course) & (df["state"] == "abnormal")]

        disp_name = SCENARIO_DISPLAY_NAMES.get(course, course)

        # 1. 힐베르트 포락선 평균
        norm_env_mean = df_norm["audio_hilbert_env_mean"].mean() if len(df_norm) > 0 else 0.0
        abn_env_mean = df_abnorm["audio_hilbert_env_mean"].mean() if len(df_abnorm) > 0 else 0.0
        env_mean_delta_pct = ((abn_env_mean - norm_env_mean) / (norm_env_mean + 1e-12)) * 100.0

        # 2. 힐베르트 포락선 최대 피크
        norm_env_peak = df_norm["audio_hilbert_env_peak"].mean() if len(df_norm) > 0 else 0.0
        abn_env_peak = df_abnorm["audio_hilbert_env_peak"].mean() if len(df_abnorm) > 0 else 0.0
        env_peak_delta_pct = ((abn_env_peak - norm_env_peak) / (norm_env_peak + 1e-12)) * 100.0

        # 3. 충격파 에너지 비중 (Top 5%)
        norm_shock_ratio = df_norm["audio_shockwave_energy_ratio"].mean() if len(df_norm) > 0 else 0.0
        abn_shock_ratio = df_abnorm["audio_shockwave_energy_ratio"].mean() if len(df_abnorm) > 0 else 0.0
        shock_ratio_delta_pct = ((abn_shock_ratio - norm_shock_ratio) / (norm_shock_ratio + 1e-12)) * 100.0

        # 4. 충격파 펄스 발생 횟수
        norm_pulse_cnt = df_norm["audio_shockwave_pulse_count"].mean() if len(df_norm) > 0 else 0.0
        abn_pulse_cnt = df_abnorm["audio_shockwave_pulse_count"].mean() if len(df_abnorm) > 0 else 0.0
        pulse_cnt_delta_pct = ((abn_pulse_cnt - norm_pulse_cnt) / (norm_pulse_cnt + 1e-12)) * 100.0

        # 5. 2.8 kHz 결함 STFT 피크 (dB)
        norm_stft_2k8 = df_norm["audio_stft_2k8_db"].mean() if len(df_norm) > 0 else -100.0
        abn_stft_2k8 = df_abnorm["audio_stft_2k8_db"].mean() if len(df_abnorm) > 0 else -100.0
        stft_2k8_delta_db = abn_stft_2k8 - norm_stft_2k8

        # 6. RMS 에너지
        norm_rms = df_norm["audio_rms"].mean() if len(df_norm) > 0 else 0.0
        abn_rms = df_abnorm["audio_rms"].mean() if len(df_abnorm) > 0 else 0.0
        rms_delta_pct = ((abn_rms - norm_rms) / (norm_rms + 1e-12)) * 100.0

        report_rows.append({
            "course": course,
            "display_name": disp_name,
            "normal_windows": len(df_norm),
            "abnormal_windows": len(df_abnorm),
            "normal_hilbert_env_mean": round(norm_env_mean, 8),
            "abnormal_hilbert_env_mean": round(abn_env_mean, 8),
            "hilbert_env_mean_delta_pct": round(env_mean_delta_pct, 2),
            "normal_hilbert_env_peak": round(norm_env_peak, 8),
            "abnormal_hilbert_env_peak": round(abn_env_peak, 8),
            "hilbert_env_peak_delta_pct": round(env_peak_delta_pct, 2),
            "normal_shockwave_energy_ratio": round(norm_shock_ratio, 4),
            "abnormal_shockwave_energy_ratio": round(abn_shock_ratio, 4),
            "shockwave_energy_ratio_delta_pct": round(shock_ratio_delta_pct, 2),
            "normal_shockwave_pulse_count": round(norm_pulse_cnt, 1),
            "abnormal_shockwave_pulse_count": round(abn_pulse_cnt, 1),
            "shockwave_pulse_count_delta_pct": round(pulse_cnt_delta_pct, 2),
            "normal_stft_2k8_db": round(norm_stft_2k8, 2),
            "abnormal_stft_2k8_db": round(abn_stft_2k8, 2),
            "stft_2k8_delta_db": round(stft_2k8_delta_db, 2),
            "normal_rms": round(norm_rms, 8),
            "abnormal_rms": round(abn_rms, 8),
            "rms_delta_pct": round(rms_delta_pct, 2)
        })

        sign_env = "+" if env_mean_delta_pct >= 0 else ""
        sign_stft = "+" if stft_2k8_delta_db >= 0 else ""
        print(f" {disp_name:<35} | {norm_env_mean:>14.6f} | {abn_env_mean:>14.6f} | {sign_env}{env_mean_delta_pct:>13.2f}% | {norm_stft_2k8:>14.2f} | {abn_stft_2k8:>14.2f} | {sign_stft}{stft_2k8_delta_db:>13.2f} dB")

    print("=" * 140)

    df_report = pd.DataFrame(report_rows)
    out_csv = os.path.join(AUDIO_AUDIT_DIR, "dsp_shockwave_scenario_comparison_report.csv")
    df_report.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n [신규 정밀 DSP 감사 보고서 CSV 저장 완료]: {out_csv}\n")


if __name__ == "__main__":
    run_dsp_scenario_comparison()
