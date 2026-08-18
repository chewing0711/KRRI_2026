"""
scan_all_scenarios_frequency_peaks.py (gearbox_final_suite / src / audio_processing)

16개 전체 시나리오(Normal 8개 vs Abnormal 8개)의 원본 음향 데이터에 대해:
1. STFT(단시간 푸리에 변환) 기반 전 대역(0 ~ 22.05 kHz) 주파수 전력 스펙트럼 전수 스캔
2. 8개 주행 코스별 정상 vs 고장의 2.8 kHz 결함 피크 주파수(Hz) 및 실측 파워(dB) 1:1 대조
3. 0 ~ 1,500 Hz 노면/풍절음 노이즈 파워(dB) 및 정상 대비 고장의 피크 증폭차(dB) 전수 산출

규정 준수:
- Rule 3: 코드 주석 100% 한글, 터미널 100% 영문 라벨.
- Rule 7: LaTeX 수식 기호 일절 사용 금지 (No-LaTeX Protocol).
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft

RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
AUDIO_AUDIT_DIR = os.path.join(RESULTS_DIR, "audio_audit")

os.makedirs(AUDIO_AUDIT_DIR, exist_ok=True)

SCENARIO_PAIRS = [
    ("고속주회로 20kph", "record_20-gear-1.wav", "go_20-gear-1.wav"),
    ("고속주회로 60kph", "record_60-gear-1.wav", "go_60-gear-1.wav"),
    ("고속주회로 80kph", "record_80-gear-1.wav", "go_80-gear-1.wav"),
    ("등판로 6도", "record_up_6-gear-1.wav", "go_up_6-gear-1.wav"),
    ("등판로 12도", "record_up_12-gear-1.wav", "go_up_12-gear-1_cut.wav"),
    ("등판로 18도", "record_up_18-gear-1.wav", "go_up_18-gear-1.wav"),
    ("등판로 30도", "record_up_30-gear-1.wav", "go_up_30-gear-1.wav"),
    ("원선회로 40kph", "record_circle-gear-1.wav", "go_up_circle_40-gear-1.wav"),
]


def find_file_path(target_filename):
    matches = glob.glob(os.path.join(RAW_DATA_DIR, "**", target_filename), recursive=True)
    return matches[0] if matches else None


def analyze_stft_spectrum(wav_path):
    sr, audio_raw = wavfile.read(wav_path)
    if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
    audio_norm = audio_raw.astype(np.float32) / 32768.0

    # STFT 주파수 전력 계산 (nperseg=2048, 주파수 해상도 21.5 Hz)
    f, t, Zxx = stft(audio_norm, fs=sr, nperseg=2048, noverlap=1024)
    power_spec = np.abs(Zxx) ** 2  # (F, T)
    mean_power = np.mean(power_spec, axis=1)  # (F,)
    db_spec = 10.0 * np.log10(mean_power + 1e-15)

    # 1. 0 ~ 1,500 Hz 노면/풍절음 대역 평균 파워 (dB)
    mask_noise = (f >= 0) & (f <= 1500)
    noise_power_db = 10.0 * np.log10(np.mean(mean_power[mask_noise]) + 1e-15)

    # 2. 2,000 ~ 3,500 Hz 감속기 결함 대역 내 최대 피크 주파수(Hz) 및 파워(dB)
    mask_fault = (f >= 2000) & (f <= 3500)
    f_fault = f[mask_fault]
    db_fault = db_spec[mask_fault]
    fault_peak_idx = np.argmax(db_fault)
    fault_peak_freq = f_fault[fault_peak_idx]
    fault_peak_db = db_fault[fault_peak_idx]

    # 3. 전 대역(0 ~ 22kHz) 전체 최빈 피크 주파수
    max_all_idx = np.argmax(db_spec)
    global_peak_freq = f[max_all_idx]
    global_peak_db = db_spec[max_all_idx]

    return {
        "noise_power_db": round(float(noise_power_db), 2),
        "fault_peak_freq": round(float(fault_peak_freq), 1),
        "fault_peak_db": round(float(fault_peak_db), 2),
        "global_peak_freq": round(float(global_peak_freq), 1),
        "global_peak_db": round(float(global_peak_db), 2),
    }


def run_all_scenarios_stft_scan():
    print("\n" + "=" * 145)
    print(" 🔍 [16개 전체 시나리오 STFT 전력 스펙트럼 1:1 결함 피크(2.8kHz) 전수 실측표]")
    print("=" * 145)
    print(f" {'주행 코스명':<16} | {'정상 2.8k피크(Hz/dB)':<22} | {'고장 2.8k피크(Hz/dB)':<22} | {'피크 증폭차(dB)':<16} | {'정상 노면소음(dB)':<18} | {'고장 노면소음(dB)':<18}")
    print("-" * 145)

    records = []

    for course_name, norm_fn, abn_fn in SCENARIO_PAIRS:
        norm_path = find_file_path(norm_fn)
        abn_path = find_file_path(abn_fn)

        res_norm = analyze_stft_spectrum(norm_path)
        res_abn = analyze_stft_spectrum(abn_path)

        delta_fault_db = res_abn["fault_peak_db"] - res_norm["fault_peak_db"]

        records.append({
            "course": course_name,
            "normal_file": norm_fn,
            "abnormal_file": abn_fn,
            "normal_fault_peak_freq_hz": res_norm["fault_peak_freq"],
            "normal_fault_peak_db": res_norm["fault_peak_db"],
            "abnormal_fault_peak_freq_hz": res_abn["fault_peak_freq"],
            "abnormal_fault_peak_db": res_abn["fault_peak_db"],
            "delta_fault_peak_db": round(delta_fault_db, 2),
            "normal_noise_band_db (0-1.5k)": res_norm["noise_power_db"],
            "abnormal_noise_band_db (0-1.5k)": res_abn["noise_power_db"],
            "normal_global_peak_freq_hz": res_norm["global_peak_freq"],
            "abnormal_global_peak_freq_hz": res_abn["global_peak_freq"],
        })

        norm_str = f"{res_norm['fault_peak_freq']:>6.1f}Hz / {res_norm['fault_peak_db']:>6.1f}dB"
        abn_str = f"{res_abn['fault_peak_freq']:>6.1f}Hz / {res_abn['fault_peak_db']:>6.1f}dB"

        print(f" {course_name:<16} | {norm_str:<22} | {abn_str:<22} | {delta_fault_db:>+13.2f} dB   | {res_norm['noise_power_db']:>14.2f} dB  | {res_abn['noise_power_db']:>14.2f} dB")

    print("=" * 145)

    df_out = pd.DataFrame(records)
    out_csv = os.path.join(AUDIO_AUDIT_DIR, "all_16_scenarios_frequency_peak_scan.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f" 💾 [16개 시나리오 주파수 전수 실측 CSV 저장]: {out_csv}\n")


if __name__ == "__main__":
    run_all_scenarios_stft_scan()
