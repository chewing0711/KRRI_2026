"""
profile_and_filter_audio_noise.py (gearbox_final_suite / src / audio_processing)

트림(Trim)하기 전의 '원본 32개 오디오 파일(Gear-1, OBD-2)' 전체에 대해:
1. 저상형 개조 구동부 소음 데이터의 주파수 대역별 정밀 프로파일링(Profiling)
   - 노면 소음 및 섀시 럼블 대역 (0 ~ 500 Hz)
   - 풍절음 및 실내 공진 대역 (500 ~ 1,500 Hz)
   - 감속기 기어 맞물림 결함 고유 대역 (2,000 ~ 3,500 Hz, 2.8 kHz 중심)
   - 기어 마모 고차 조화파 대역 (4,000 ~ 10,000 Hz)
2. 복합 외부 노이즈(노면/풍절음)를 제거하고 감속기 결함 대역을 분리하는 4차 버터워스 디지털 대역통과 필터(BPF, 1.8k~10.5kHz) 적용
3. 필터링 전/후 스펙트럼의 Y축 범위를 데이터 실제 동적 범위(dBFS / Auto-scaling)에 완벽 정렬하여 시각화

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib 라벨/타이틀 100% 영문(English).
- Rule 7: LaTeX 수식 기호 일절 사용 금지 (No-LaTeX Protocol).
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
from scipy.signal import butter, filtfilt, welch

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


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(data, lowcut=1800.0, highcut=10500.0, fs=44100, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def compute_band_powers(freqs, psd):
    """주파수 대역별 에너지 파워(%) 계산"""
    total_power = np.sum(psd) + 1e-12

    mask_road = (freqs >= 0) & (freqs < 500)
    mask_wind = (freqs >= 500) & (freqs < 1500)
    mask_gear_2k8 = (freqs >= 2000) & (freqs < 3500)
    mask_harmonics = (freqs >= 4000) & (freqs < 10000)

    p_road = np.sum(psd[mask_road])
    p_wind = np.sum(psd[mask_wind])
    p_gear = np.sum(psd[mask_gear_2k8])
    p_harm = np.sum(psd[mask_harmonics])

    pct_road = (p_road / total_power) * 100.0
    pct_wind = (p_wind / total_power) * 100.0
    pct_gear = (p_gear / total_power) * 100.0
    pct_harm = (p_harm / total_power) * 100.0

    return {
        "p_road": p_road, "p_wind": p_wind, "p_gear": p_gear, "p_harm": p_harm,
        "pct_road": pct_road, "pct_wind": pct_wind, "pct_gear": pct_gear, "pct_harm": pct_harm,
    }


def run_profiling_and_filtering_analysis():
    print("\n" + "=" * 135)
    print(" 🔊 [원본 32개 오디오 파일 소음 프로파일링 및 디지털 대역통과 필터링(BPF) 전수 실측 검증]")
    print("=" * 135)

    wav_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.wav"), recursive=True))
    if not wav_files:
        print("[ERROR] 원본 WAV 파일을 찾을 수 없습니다:", RAW_DATA_DIR)
        return

    records = []

    print(f" {'파일명 (Filename)':<32} | {'채널':<5} | {'상태':<8} | {'노면/풍절음 비중':<16} | {'2.8kHz 결함 비중':<16} | {'노이즈 제거율':<14} | {'SNR 개선도':<12}")
    print("-" * 135)

    plot_data = []

    for wav_path in wav_files:
        scen_folder = os.path.dirname(wav_path)
        parts = scen_folder.split(os.sep)
        state_str = parts[-2].strip().lower()
        scen_str = parts[-1].strip().lower()
        filename = os.path.basename(wav_path)

        state = "abnormal" if "abnormal" in state_str else "normal"
        channel = "GEAR" if "gear" in filename.lower() else "OBD"

        sr, audio_raw = wavfile.read(wav_path)
        if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
        audio_norm = audio_raw.astype(np.float32) / 32768.0

        # 1. 필터링 전 원본 PSD 계산 (Welch's method)
        f_raw, psd_raw = welch(audio_norm, fs=sr, nperseg=2048, noverlap=1024)
        raw_bands = compute_band_powers(f_raw, psd_raw)

        # 2. 4차 버터워스 디지털 BPF (1800 Hz ~ 10500 Hz) 적용
        audio_filtered = apply_bandpass_filter(audio_norm, lowcut=1800.0, highcut=10500.0, fs=sr, order=4)

        # 3. 필터링 후 PSD 계산
        f_flt, psd_flt = welch(audio_filtered, fs=sr, nperseg=2048, noverlap=1024)
        flt_bands = compute_band_powers(f_flt, psd_flt)

        # 4. 정량 지표 산출
        noise_before = raw_bands["p_road"] + raw_bands["p_wind"]
        noise_after = flt_bands["p_road"] + flt_bands["p_wind"]
        noise_red_pct = ((noise_before - noise_after) / (noise_before + 1e-12)) * 100.0

        gear_retention_pct = (flt_bands["p_gear"] / (raw_bands["p_gear"] + 1e-12)) * 100.0

        snr_raw = 10.0 * np.log10((raw_bands["p_gear"] + raw_bands["p_harm"]) / (noise_before + 1e-12) + 1e-10)
        snr_flt = 10.0 * np.log10((flt_bands["p_gear"] + flt_bands["p_harm"]) / (noise_after + 1e-12) + 1e-10)
        snr_gain_db = snr_flt - snr_raw

        raw_noise_pct = raw_bands["pct_road"] + raw_bands["pct_wind"]
        raw_gear_pct = raw_bands["pct_gear"]

        records.append({
            "state": state,
            "channel": channel,
            "filename": filename,
            "raw_duration_sec": round(len(audio_norm) / float(sr), 2),
            "raw_noise_band_pct (0-1500Hz)": round(raw_noise_pct, 2),
            "raw_gear_fault_pct (2-3.5kHz)": round(raw_gear_pct, 2),
            "raw_harmonics_pct (4-10kHz)": round(raw_bands["pct_harm"], 2),
            "filtered_noise_reduction_pct": round(noise_red_pct, 2),
            "gear_peak_retention_pct": round(gear_retention_pct, 2),
            "snr_improvement_gain_db": round(snr_gain_db, 2),
        })

        print(f" {filename:<32} | {channel:<5} | {state:<8} | {raw_noise_pct:>13.2f} %  | {raw_gear_pct:>13.2f} %  | {noise_red_pct:>11.2f} % | {snr_gain_db:>+9.2f} dB")

        if channel == "GEAR":
            plot_data.append((filename, state, f_raw, psd_raw, psd_flt))

    print("=" * 135)

    df_out = pd.DataFrame(records)
    out_csv = os.path.join(AUDIO_AUDIT_DIR, "acoustic_noise_profiling_and_filtering_report.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f" 💾 [프로파일링 및 필터링 리포트 저장]: {out_csv}")

    # =========================================================================
    # [16개 시나리오 필터링 전/후 스펙트럼 정밀 시각화]
    # =========================================================================
    fig, axes = plt.subplots(4, 4, figsize=(24, 18), dpi=300)
    axes_flat = axes.flatten()

    for idx, (fn, st, freqs, p_raw, p_flt) in enumerate(plot_data[:16]):
        ax = axes_flat[idx]

        # 원본 피크 기준 정규화 dBFS (최대값 0 dB 기준)
        max_p = np.max(p_raw) + 1e-12
        p_raw_norm_db = 10.0 * np.log10(p_raw / max_p + 1e-12)
        p_flt_norm_db = 10.0 * np.log10(p_flt / max_p + 1e-12)

        # 2.8 kHz 및 노이즈 밴드 음영 표시
        ax.axvspan(2000, 3500, color="#ffeb3b", alpha=0.3, label="Gear Fault Band (2.8 kHz)")
        ax.axvspan(0, 1500, color="#e0e0e0", alpha=0.35, label="Filtered Road/Wind Noise (0-1.5 kHz)")

        # 스펙트럼 곡선 그리기
        ax.plot(freqs, p_raw_norm_db, color="#757575", linewidth=1.1, alpha=0.7, label="Raw Acoustic (dBFS)")
        ax.plot(freqs, p_flt_norm_db, color="#c62828" if st == "abnormal" else "#1565c0", linewidth=1.5, label="Filtered BPF (1.8k-10.5kHz)")

        ax.set_xlim([0, 12000])
        ax.set_ylim([-70, 5])
        ax.set_ylabel("Normalized Power (dBFS)", fontsize=9, fontweight="bold")
        ax.set_title(f"{fn}\n[{st.upper()} | Noise Cut: {records[idx]['filtered_noise_reduction_pct']}% | SNR Gain: {records[idx]['snr_improvement_gain_db']:+.1f}dB]", fontsize=9, fontweight="bold", pad=4)
        ax.legend(loc="upper right", fontsize=7, frameon=True)

        if idx >= 12:
            ax.set_xlabel("Frequency (Hz)", fontsize=10, fontweight="bold")

    plt.suptitle("Raw Acoustic Spectrum vs Digital Bandpass Filtered Spectrum (16 Scenarios - Road/Wind Noise Removal)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "digital_filtering_before_after_spectral_profile.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, "digital_filtering_before_after_spectral_profile.png")
    shutil.copy2(out_png, art_png)

    print(f" 📊 [필터링 전/후 스펙트럼 비교 그래프 저장]: {out_png}")
    print(f" 📊 [Artifact 동기화 완료]: {art_png}\n")


if __name__ == "__main__":
    run_profiling_and_filtering_analysis()
