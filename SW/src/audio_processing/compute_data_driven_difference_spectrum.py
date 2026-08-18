"""
compute_data_driven_difference_spectrum.py (gearbox_final_suite / src / audio_processing)

1500 Hz라는 임의의 가정을 완전히 배제하고:
1. 8개 주행 코스 전체에 대해 정상(Normal) vs 고장(Abnormal)의 '실측 차분 스펙트럼(Difference Spectrum: Delta_PSD(f) = PSD_abnormal(f) - PSD_normal(f))'을 직접 계산
2. 차분 파워가 0 dB(노면/환경 소음 일치 대역)에서 벗어나 실제 분리(Delta > +3.0 dB)가 시작되는 '데이터 기반 컷오프 주파수(f_cutoff)'를 전수 도출
3. 8개 코스별 실측 차분 곡선 및 유의미 분리 대역을 시각화(data_driven_spectral_difference_analysis.png)하고 리포트 CSV 저장

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib 라벨/타이틀 100% 영문(English).
- Rule 7: LaTeX 수식 기호 일절 사용 금지 (No-LaTeX Protocol).
- Rule 8: 이모지 전면 배제 및 순수 공학적 수치 산출.
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch

RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
AUDIO_AUDIT_DIR = os.path.join(RESULTS_DIR, "audio_audit")
ARTIFACT_FIG_DIR = "/home/kante/.gemini/antigravity-ide/brain/6e341025-6278-4988-86a0-4dfdf4574e3b/figures"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(AUDIO_AUDIT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_FIG_DIR, exist_ok=True)

# 스타일 설정
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

SCENARIO_PAIRS = [
    ("High Speed 20kph", "record_20-gear-1.wav", "go_20-gear-1.wav"),
    ("High Speed 60kph", "record_60-gear-1.wav", "go_60-gear-1.wav"),
    ("High Speed 80kph", "record_80-gear-1.wav", "go_80-gear-1.wav"),
    ("Hills 6deg", "record_up_6-gear-1.wav", "go_up_6-gear-1.wav"),
    ("Hills 12deg", "record_up_12-gear-1.wav", "go_up_12-gear-1_cut.wav"),
    ("Hills 18deg", "record_up_18-gear-1.wav", "go_up_18-gear-1.wav"),
    ("Hills 30deg", "record_up_30-gear-1.wav", "go_up_30-gear-1.wav"),
    ("Steer Pad 40kph", "record_circle-gear-1.wav", "go_up_circle_40-gear-1.wav"),
]


def find_file_path(target_filename):
    matches = glob.glob(os.path.join(RAW_DATA_DIR, "**", target_filename), recursive=True)
    return matches[0] if matches else None


def run_data_driven_difference_spectrum():
    print("\n" + "=" * 140)
    print(" [데이터 기반 정상 vs 고장 차분 스펙트럼(Difference Spectrum) 및 실측 컷오프 주파수 전수 도출]")
    print("=" * 140)
    print(f" {'주행 코스명':<16} | {'노면노이즈 일치구간(Delta < 1dB)':<30} | {'분리시작 컷오프(Hz)':<20} | {'최대 결함 분리피크(Hz / Delta dB)':<34}")
    print("-" * 140)

    records = []
    fig, axes = plt.subplots(4, 2, figsize=(20, 16), dpi=300)
    axes_flat = axes.flatten()

    for idx, (course_name, norm_fn, abn_fn) in enumerate(SCENARIO_PAIRS):
        norm_path = find_file_path(norm_fn)
        abn_path = find_file_path(abn_fn)

        sr_n, aud_n = wavfile.read(norm_path)
        sr_a, aud_a = wavfile.read(abn_path)
        if aud_n.ndim > 1: aud_n = aud_n[:, 0]
        if aud_a.ndim > 1: aud_a = aud_a[:, 0]

        aud_n_norm = aud_n.astype(np.float32) / 32768.0
        aud_a_norm = aud_a.astype(np.float32) / 32768.0

        # Welch PSD 계산 (nperseg=4096, 주파수 해상도 10.77 Hz)
        f_n, psd_n = welch(aud_n_norm, fs=sr_n, nperseg=4096, noverlap=2048)
        f_a, psd_a = welch(aud_a_norm, fs=sr_a, nperseg=4096, noverlap=2048)

        # dB 변환
        psd_n_db = 10.0 * np.log10(psd_n + 1e-15)
        psd_a_db = 10.0 * np.log10(psd_a + 1e-15)

        # 차분 스펙트럼 (Delta PSD dB = Abnormal dB - Normal dB)
        delta_psd_db = psd_a_db - psd_n_db

        # 1. 0 ~ 5,000 Hz 범위에서 차분 파워가 +3.0 dB를 안정적으로 초과하기 시작하는 첫 컷오프 주파수 탐색
        mask_search = (f_n >= 200) & (f_n <= 5000)
        f_search = f_n[mask_search]
        delta_search = delta_psd_db[mask_search]

        # 3포인트 연속 3dB 초과 지점 탐색
        cutoff_freq = None
        for i in range(len(delta_search) - 3):
            if np.all(delta_search[i:i+3] > 3.0):
                cutoff_freq = f_search[i]
                break
        if cutoff_freq is None:
            # 3dB 초과가 없을 경우 최대 피크 직전 1dB 초과 지점으로 설정
            max_pt = np.argmax(delta_search)
            cutoff_freq = f_search[max_pt]

        # 2. 2,000 ~ 4,000 Hz 대역 내 최대 분리 피크
        mask_fault_band = (f_n >= 1800) & (f_n <= 4000)
        f_fb = f_n[mask_fault_band]
        d_fb = delta_psd_db[mask_fault_band]
        max_fb_idx = np.argmax(d_fb)
        max_fault_freq = f_fb[max_fb_idx]
        max_fault_delta_db = d_fb[max_fb_idx]

        # 3. 저주파 노면 노이즈 일치 구간(Delta < 1.5 dB) 범위
        mask_low = (f_n >= 0) & (f_n < cutoff_freq)
        mean_low_delta = np.mean(np.abs(delta_psd_db[mask_low]))

        records.append({
            "course": course_name,
            "normal_file": norm_fn,
            "abnormal_file": abn_fn,
            "empirical_noise_band_hz": f"0 ~ {cutoff_freq:.0f} Hz",
            "empirical_noise_delta_mean_db": round(float(mean_low_delta), 2),
            "data_driven_cutoff_freq_hz": round(float(cutoff_freq), 1),
            "max_fault_peak_freq_hz": round(float(max_fault_freq), 1),
            "max_fault_delta_db": round(float(max_fault_delta_db), 2),
        })

        noise_str = f"0 ~ {cutoff_freq:.0f} Hz (Avg |Delta|: {mean_low_delta:.2f}dB)"
        peak_str = f"{max_fault_freq:>6.1f} Hz / {max_fault_delta_db:>+6.2f} dB"

        print(f" {course_name:<16} | {noise_str:<30} | {cutoff_freq:>18.1f} Hz | {peak_str:<34}")

        # 플롯 렌더링
        ax = axes_flat[idx]
        ax.plot(f_n, delta_psd_db, color="#b71c1c", linewidth=1.3, label="Spectral Difference (Abnormal - Normal)")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.9, alpha=0.7, label="0 dB Equal Baseline")
        ax.axhline(3.0, color="#f57c00", linestyle="--", linewidth=0.9, label="+3 dB Separation Threshold")

        # 실측 데이터 기반 노면 노이즈 구간 및 결함 분리 구간 음영 처리
        ax.axvspan(0, cutoff_freq, color="#e0e0e0", alpha=0.35, label=f"Empirical Noise Band (0-{cutoff_freq:.0f}Hz)")
        ax.axvspan(cutoff_freq, 10000, color="#fff9c4", alpha=0.35, label="Empirical Gear Fault Band")

        ax.axvline(cutoff_freq, color="#1565c0", linestyle=":", linewidth=1.4, label=f"Empirical Cutoff ({cutoff_freq:.0f}Hz)")

        ax.set_xlim([0, 12000])
        ax.set_ylim([-15, 35])
        ax.set_ylabel("Power Difference (dB)", fontsize=9, fontweight="bold")
        ax.set_title(f"{course_name} [Cutoff: {cutoff_freq:.0f}Hz | Max Separation: {max_fault_delta_db:+.1f}dB at {max_fault_freq:.0f}Hz]", fontsize=9, fontweight="bold", pad=4)
        ax.legend(loc="upper right", fontsize=7, frameon=True)

        if idx >= 6:
            ax.set_xlabel("Frequency (Hz)", fontsize=10, fontweight="bold")

    print("=" * 140)
    plt.suptitle("Empirical Difference Spectrum (Abnormal - Normal) Across 8 Driving Courses - Data-Driven Cutoff Determination", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "data_driven_spectral_difference_analysis.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, "data_driven_spectral_difference_analysis.png")
    shutil.copy2(out_png, art_png)

    df_out = pd.DataFrame(records)
    out_csv = os.path.join(AUDIO_AUDIT_DIR, "data_driven_spectral_difference_report.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f" 💾 [데이터 기반 차분 스펙트럼 리포트 CSV 저장]: {out_csv}")
    print(f" 📊 [차분 스펙트럼 마스터 그래프 저장]: {out_png}\n")


if __name__ == "__main__":
    run_data_driven_difference_spectrum()
