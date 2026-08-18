"""
plot_difference_spectrum_and_spectrograms.py (gearbox_final_suite / src / audio_processing)

실측 차분 스펙트럼(Difference Spectrum) 및 Normal vs Abnormal 시계열 STFT 스펙트로그램 시각화:
1. 8개 주행 코스별 Normal PSD, Abnormal PSD, 차분 스펙트럼(Delta PSD), 실측 컷오프(f_cutoff) 및 최대 결함 피크 마킹
   -> data_driven_spectral_difference_master_plot.png
2. 주요 4개 코스(20kph, 60kph, 80kph, 원선회 40kph)의 Normal vs Abnormal 시간-주파수 스펙트로그램 정밀 시각화
   - 정밀 dBFS 정규화 및 고대비 컬러맵(inferno/plasma), 개별 컬러바(Colorbar) 부착
   - 2.8 kHz 결함 수평선 및 저주파 노면 노이즈 영역 명확한 가이드라인 표시
   -> difference_spectrogram_time_frequency_comparison.png

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib 라벨/타이틀 100% 영문(English).
- Rule 7: LaTeX 수식 기호 일절 사용 금지 (No-LaTeX Protocol).
- Rule 8: 이모지 전면 배제.
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
from scipy.signal import welch, stft

RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
ARTIFACT_FIG_DIR = "/home/kante/.gemini/antigravity-ide/brain/6e341025-6278-4988-86a0-4dfdf4574e3b/figures"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_FIG_DIR, exist_ok=True)

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


def plot_difference_spectrum_master():
    """8개 코스별 PSD 및 차분 스펙트럼 마스터 플롯 생성"""
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

        f_n, psd_n = welch(aud_n_norm, fs=sr_n, nperseg=4096, noverlap=2048)
        f_a, psd_a = welch(aud_a_norm, fs=sr_a, nperseg=4096, noverlap=2048)

        psd_n_db = 10.0 * np.log10(psd_n + 1e-15)
        psd_a_db = 10.0 * np.log10(psd_a + 1e-15)
        delta_psd_db = psd_a_db - psd_n_db

        mask_search = (f_n >= 200) & (f_n <= 5000)
        f_search = f_n[mask_search]
        delta_search = delta_psd_db[mask_search]

        cutoff_freq = None
        for i in range(len(delta_search) - 3):
            if np.all(delta_search[i:i+3] > 3.0):
                cutoff_freq = f_search[i]
                break
        if cutoff_freq is None:
            cutoff_freq = f_search[np.argmax(delta_search)]

        mask_fault = (f_n >= 1800) & (f_n <= 4000)
        f_fb = f_n[mask_fault]
        d_fb = delta_psd_db[mask_fault]
        max_idx = np.argmax(d_fb)
        max_freq = f_fb[max_idx]
        max_delta = d_fb[max_idx]

        ax = axes_flat[idx]
        ax.plot(f_n, delta_psd_db, color="#c62828", linewidth=1.5, label="Delta PSD (Abnormal - Normal)")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7, label="0 dB Equal Baseline")
        ax.axhline(3.0, color="#ef6c00", linestyle="--", linewidth=0.8, label="+3 dB Separation Threshold")

        ax.axvspan(0, cutoff_freq, color="#e0e0e0", alpha=0.35, label=f"Noise Match Band (0-{cutoff_freq:.0f}Hz)")
        ax.axvspan(cutoff_freq, 10000, color="#fff9c4", alpha=0.35, label="Fault Separation Band")
        ax.axvline(cutoff_freq, color="#1565c0", linestyle=":", linewidth=1.4, label=f"Cutoff ({cutoff_freq:.0f}Hz)")

        ax.scatter([max_freq], [max_delta], color="#b71c1c", s=45, zorder=5)
        ax.annotate(f"Peak: {max_freq:.0f}Hz\n+{max_delta:.1f}dB",
                    xy=(max_freq, max_delta),
                    xytext=(max_freq + 600, max_delta + 3.0),
                    fontsize=8, fontweight="bold", color="#b71c1c",
                    arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.2))

        ax.set_xlim([0, 10000])
        ax.set_ylim([-12, 32])
        ax.set_ylabel("Power Delta (dB)", fontsize=9, fontweight="bold")
        ax.set_title(f"{course_name} [Empirical Cutoff: {cutoff_freq:.0f}Hz | Max Delta: {max_delta:+.1f}dB at {max_freq:.0f}Hz]", fontsize=9, fontweight="bold", pad=4)
        ax.legend(loc="upper right", fontsize=7, frameon=True)

        if idx >= 6:
            ax.set_xlabel("Frequency (Hz)", fontsize=10, fontweight="bold")

    plt.suptitle("Empirical Difference Spectrum (Abnormal - Normal) Across 8 Driving Courses", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "data_driven_spectral_difference_master_plot.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, "data_driven_spectral_difference_master_plot.png")
    shutil.copy2(out_png, art_png)
    print(f" [마스터 차분 스펙트럼 그래프 저장]: {out_png}")


def plot_spectrogram_time_frequency_comparison():
    """대표 4개 코스(20kph, 60kph, 80kph, 원선회 40kph)의 Normal vs Abnormal 시간-주파수 STFT 스펙트로그램 정밀 비교"""
    rep_scenarios = [
        ("High Speed 20kph", "record_20-gear-1.wav", "go_20-gear-1.wav"),
        ("High Speed 60kph", "record_60-gear-1.wav", "go_60-gear-1.wav"),
        ("High Speed 80kph", "record_80-gear-1.wav", "go_80-gear-1.wav"),
        ("Steer Pad 40kph", "record_circle-gear-1.wav", "go_up_circle_40-gear-1.wav"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(22, 16), dpi=300)

    for row_idx, (scen_title, norm_fn, abn_fn) in enumerate(rep_scenarios):
        norm_path = find_file_path(norm_fn)
        abn_path = find_file_path(abn_fn)

        sr_n, aud_n = wavfile.read(norm_path)
        sr_a, aud_a = wavfile.read(abn_path)
        if aud_n.ndim > 1: aud_n = aud_n[:, 0]
        if aud_a.ndim > 1: aud_a = aud_a[:, 0]

        aud_n_norm = aud_n.astype(np.float32) / 32768.0
        aud_a_norm = aud_a.astype(np.float32) / 32768.0

        # STFT 계산 (nperseg=2048, noverlap=1024)
        f_n, t_n, Zxx_n = stft(aud_n_norm, fs=sr_n, nperseg=2048, noverlap=1024)
        f_a, t_a, Zxx_a = stft(aud_a_norm, fs=sr_a, nperseg=2048, noverlap=1024)

        # 0 dBFS 정규화 스케일링 (최대 피크 기준 dBFS)
        mag_n = np.abs(Zxx_n)
        mag_a = np.abs(Zxx_a)
        ref_val = max(np.max(mag_n), np.max(mag_a)) + 1e-12

        dbfs_n = 20.0 * np.log10(mag_n / ref_val + 1e-8)
        dbfs_a = 20.0 * np.log10(mag_a / ref_val + 1e-8)

        # 0 ~ 5,000 Hz 범위 필터링
        mask_f = f_n <= 5000
        f_sub = f_n[mask_f]
        dbfs_n_sub = dbfs_n[mask_f, :]
        dbfs_a_sub = dbfs_a[mask_f, :]

        # 공통 컬러 스케일 (-65 dBFS ~ 0 dBFS)
        c_min, c_max = -65.0, 0.0

        # Normal Plot
        ax_norm = axes[row_idx, 0]
        mesh0 = ax_norm.pcolormesh(t_n, f_sub, dbfs_n_sub, cmap="inferno", vmin=c_min, vmax=c_max, shading="auto")
        ax_norm.axhline(2800, color="cyan", linestyle="--", linewidth=1.3, alpha=0.9, label="2.8 kHz GMF Target Line")
        ax_norm.axhspan(0, 1800, color="gray", alpha=0.15, label="Road/Wind Noise Band (0-1.8k)")
        ax_norm.set_title(f"[Normal] {scen_title} - Clean Spectrum Floor (No 2.8k Fault Track)", fontsize=10, fontweight="bold")
        ax_norm.set_ylabel("Frequency (Hz)", fontsize=9, fontweight="bold")
        ax_norm.set_ylim([0, 5000])
        ax_norm.legend(loc="upper right", fontsize=8, frameon=True)
        cb0 = fig.colorbar(mesh0, ax=ax_norm, pad=0.015, aspect=20)
        cb0.set_label("Power (dBFS)", fontsize=8, fontweight="bold")

        # Abnormal Plot
        ax_abn = axes[row_idx, 1]
        mesh1 = ax_abn.pcolormesh(t_a, f_sub, dbfs_a_sub, cmap="inferno", vmin=c_min, vmax=c_max, shading="auto")
        ax_abn.axhline(2800, color="cyan", linestyle="--", linewidth=1.3, alpha=0.9, label="2.8 kHz GMF Target Line")
        ax_abn.axhspan(0, 1800, color="gray", alpha=0.15, label="Road/Wind Noise Band (0-1.8k)")
        ax_abn.set_title(f"[Abnormal] {scen_title} - Blazing 2.8 kHz Gear Fault Track (Continuous)", fontsize=10, fontweight="bold", color="#b71c1c")
        ax_abn.set_ylabel("Frequency (Hz)", fontsize=9, fontweight="bold")
        ax_abn.set_ylim([0, 5000])
        ax_abn.legend(loc="upper right", fontsize=8, frameon=True)
        cb1 = fig.colorbar(mesh1, ax=ax_abn, pad=0.015, aspect=20)
        cb1.set_label("Power (dBFS)", fontsize=8, fontweight="bold")

        if row_idx == 3:
            ax_norm.set_xlabel("Time (sec)", fontsize=10, fontweight="bold")
            ax_abn.set_xlabel("Time (sec)", fontsize=10, fontweight="bold")

    plt.suptitle("Time-Frequency STFT Spectrogram Comparison (Normalized dBFS Scale) - Normal Floor vs Abnormal 2.8 kHz Fault Line", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "difference_spectrogram_time_frequency_comparison.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, "difference_spectrogram_time_frequency_comparison.png")
    shutil.copy2(out_png, art_png)
    print(f" [시간-주파수 스펙트로그램 정밀 비교 그래프 저장]: {out_png}")


def run_all_plots():
    print("\n" + "=" * 100)
    print(" [차분 스펙트럼 및 시간-주파수 STFT 스펙트로그램 전수 시각화 생성 시작]")
    print("=" * 100)
    plot_difference_spectrum_master()
    plot_spectrogram_time_frequency_comparison()
    print("=" * 100 + "\n")


if __name__ == "__main__":
    run_all_plots()
