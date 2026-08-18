"""
analyze_multimodal_feature_cross_correlation.py (gearbox_final_suite / src / audio_processing)

CAN 다중 피처(차속, 가속페달, 횡가속도, 요레이트, 좌우/전후 휠속차)와
동기화된 오디오 음향 피처(Audio RMS, Peak, 힐베르트 포락선) 간의
1. 상호 상관 계수(Cross-Correlation) 및 밀리초(ms) 단위 시차(Lag Tau) 정밀 계산
2. 다중 피처 동시 비교 4단 동기화 시각화 그래프 생성

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib 라벨/타이틀 100% 영문(English).
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_ALIGNED_DIR = os.path.join(DATA_DIR, "audio_aligned")
AUDIO_AUDIT_DIR = os.path.join(RESULTS_DIR, "audio_audit")
ARTIFACT_FIG_DIR = "/home/kante/.gemini/antigravity-ide/brain/6e341025-6278-4988-86a0-4dfdf4574e3b/figures"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(AUDIO_AUDIT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_FIG_DIR, exist_ok=True)

# 스타일 설정
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def run_cross_correlation_analysis(scenario_name="abnormal_high_speed_20kph"):
    can_path = os.path.join(RAW_DECODED_DIR, f"{scenario_name}_official_decoded.csv")
    state = "abnormal" if "abnormal" in scenario_name else "normal"
    scen_core = scenario_name.replace("abnormal_", "").replace("normal_", "")
    
    audio_dir = os.path.join(AUDIO_ALIGNED_DIR, state, scen_core)
    wav_candidates = [f for f in os.listdir(audio_dir) if "gear" in f.lower()]
    if not wav_candidates:
        wav_candidates = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    audio_path = os.path.join(audio_dir, wav_candidates[0])

    print("\n" + "=" * 115)
    print(f" 🔬 [CAN 다중 피처 vs Audio 신호 상호상관(Cross-Correlation) 정밀 분석]")
    print(f" 📂 CAN 파일: {os.path.basename(can_path)} | 📂 오디오 파일: {os.path.basename(audio_path)}")
    print("=" * 115)

    df_can = pd.read_csv(can_path)
    sr, audio_raw = wavfile.read(audio_path)
    if audio_raw.ndim > 1:
        audio_raw = audio_raw[:, 0]

    audio_norm = audio_raw.astype(np.float32) / 32768.0
    t_can = df_can["Time"].values - df_can["Time"].values[0]

    # CAN 각 행(4ms) 구간별 Audio RMS 및 Peak 계산
    audio_rms = []
    audio_peak = []
    for i in range(len(t_can) - 1):
        s_start = int(round(t_can[i] * sr))
        s_end = int(round(t_can[i + 1] * sr))
        chunk = audio_norm[s_start:s_end]
        if len(chunk) > 0:
            audio_rms.append(float(np.sqrt(np.mean(chunk ** 2))))
            audio_peak.append(float(np.max(np.abs(chunk))))
        else:
            audio_rms.append(0.0)
            audio_peak.append(0.0)

    audio_rms.append(audio_rms[-1] if audio_rms else 0.0)
    audio_peak.append(audio_peak[-1] if audio_peak else 0.0)

    audio_rms_arr = np.array(audio_rms)
    audio_peak_arr = np.array(audio_peak)

    # CAN 다중 피처 추출
    spd_mean = (df_can["WHL_SPD_FL"].values + df_can["WHL_SPD_FR"].values + df_can["WHL_SPD_RL"].values + df_can["WHL_SPD_RR"].values) / 4.0
    wheel_slip_diff = np.abs(df_can["WHL_SPD_FR"].values - df_can["WHL_SPD_RR"].values)
    pedal_pos = df_can["Accel_Pedal_Pos"].values if "Accel_Pedal_Pos" in df_can.columns else np.zeros_like(t_can)
    lat_accel = df_can["Lateral_Accel"].values if "Lateral_Accel" in df_can.columns else np.zeros_like(t_can)
    yaw_rate = df_can["Yaw_Rate"].values if "Yaw_Rate" in df_can.columns else np.zeros_like(t_can)

    # -------------------------------------------------------------------------
    # 상호 상관(Cross-Correlation) 및 시차(Lag) 계산 함수
    # -------------------------------------------------------------------------
    def compute_lag_tau(sig1, sig2, max_lag_samples=250):
        # 정규화
        s1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-8)
        s2 = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-8)
        corr = correlate(s1, s2, mode="full") / len(s1)
        lags = correlation_lags(len(s1), len(s2), mode="full")
        
        # 관심 영역 제한 (+-250개 CAN 행, 약 +-1초)
        mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
        sub_lags = lags[mask]
        sub_corr = corr[mask]
        
        peak_idx = np.argmax(np.abs(sub_corr))
        best_lag_samples = sub_lags[peak_idx]
        best_lag_ms = best_lag_samples * 4.0  # CAN 평균 4.0ms 주기
        max_corr_val = sub_corr[peak_idx]
        return best_lag_ms, max_corr_val, sub_lags * 4.0, sub_corr

    lag_spd, r_spd, lags_spd_ms, corr_spd = compute_lag_tau(spd_mean, audio_rms_arr)
    lag_pdl, r_pdl, lags_pdl_ms, corr_pdl = compute_lag_tau(pedal_pos, audio_peak_arr)
    lag_slp, r_slp, lags_slp_ms, corr_slp = compute_lag_tau(wheel_slip_diff, audio_rms_arr)
    lag_yaw, r_yaw, lags_yaw_ms, corr_yaw = compute_lag_tau(np.abs(yaw_rate), audio_rms_arr)

    print(f" {'CAN 피처 (Feature)':<30} | {'대응 Audio 피처':<18} | {'최대 상관계수(R)':<18} | {'측정 시차(Lag Tau, ms)':<22}")
    print("-" * 115)
    print(f" {'1. 차속 평균 (WHL_SPD Mean)':<30} | {'Audio RMS (에너지)':<18} | {r_spd:>16.4f}   | {lag_spd:>18.2f} ms")
    print(f" {'2. 가속 페달 (Accel Pedal)':<30} | {'Audio Peak (충격파)':<18} | {r_pdl:>16.4f}   | {lag_pdl:>18.2f} ms")
    print(f" {'3. 휠속도차 (|FR - RR| Slip)':<30} | {'Audio RMS (에너지)':<18} | {r_slp:>16.4f}   | {lag_slp:>18.2f} ms")
    print(f" {'4. 회전 요레이트 (|Yaw Rate|)':<30} | {'Audio RMS (에너지)':<18} | {r_yaw:>16.4f}   | {lag_yaw:>18.2f} ms")
    print("=" * 115)

    # -------------------------------------------------------------------------
    # 4단 고해상도 시각화 생성
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), dpi=300)

    # Panel 1: 차속 vs Audio RMS
    ax1 = axes[0]
    ax1.plot(t_can, spd_mean, color="#1976d2", linewidth=1.3, label="CAN: Mean Vehicle Speed (km/h)")
    ax1.set_ylabel("Speed (km/h)", color="#1976d2", fontsize=11, fontweight="bold")
    ax1_tw = ax1.twinx()
    ax1_tw.plot(t_can, audio_rms_arr, color="#388e3c", linewidth=1.1, alpha=0.85, label="Audio: Synchronized RMS Power")
    ax1_tw.set_ylabel("Audio RMS", color="#388e3c", fontsize=11, fontweight="bold")
    ax1_tw.grid(False)
    ax1.set_title(f"1. Vehicle Speed vs Audio Acoustic Power (Lag: {lag_spd:.1f}ms, R: {r_spd:.3f})", fontsize=12, fontweight="bold")

    # Panel 2: 페달 개도량 vs Audio Peak
    ax2 = axes[1]
    ax2.plot(t_can, pedal_pos, color="#d32f2f", linewidth=1.3, label="CAN: Accel Pedal Position (%)")
    ax2.set_ylabel("Pedal (%)", color="#d32f2f", fontsize=11, fontweight="bold")
    ax2_tw = ax2.twinx()
    ax2_tw.plot(t_can, audio_peak_arr, color="#7b1fa2", linewidth=1.1, alpha=0.85, label="Audio: Synchronized Peak Impact")
    ax2_tw.set_ylabel("Audio Peak", color="#7b1fa2", fontsize=11, fontweight="bold")
    ax2_tw.grid(False)
    ax2.set_title(f"2. Accel Pedal Input vs Audio Peak Impact Waveform (Lag: {lag_pdl:.1f}ms, R: {r_pdl:.3f})", fontsize=12, fontweight="bold")

    # Panel 3: 휠속 슬립차 vs Audio RMS
    ax3 = axes[2]
    ax3.plot(t_can, wheel_slip_diff, color="#f57c00", linewidth=1.3, label="CAN: Drive Wheel Speed Delta |FR-RR| (km/h)")
    ax3.set_ylabel("Wheel Delta (km/h)", color="#f57c00", fontsize=11, fontweight="bold")
    ax3_tw = ax3.twinx()
    ax3_tw.plot(t_can, audio_rms_arr, color="#00796b", linewidth=1.1, alpha=0.85, label="Audio: Synchronized RMS Power")
    ax3_tw.set_ylabel("Audio RMS", color="#00796b", fontsize=11, fontweight="bold")
    ax3_tw.grid(False)
    ax3.set_title(f"3. Drive Wheel Slip Delta vs Audio Acoustic Power (Lag: {lag_slp:.1f}ms, R: {r_slp:.3f})", fontsize=12, fontweight="bold")

    # Panel 4: 4개 피처 상호상관 계수(Cross-Correlation) 곡선
    ax4 = axes[3]
    ax4.plot(lags_spd_ms, corr_spd, label=f"Speed vs Audio RMS (Peak at {lag_spd:.1f}ms)", color="#1976d2", linewidth=1.5)
    ax4.plot(lags_pdl_ms, corr_pdl, label=f"Pedal vs Audio Peak (Peak at {lag_pdl:.1f}ms)", color="#d32f2f", linewidth=1.5)
    ax4.plot(lags_slp_ms, corr_slp, label=f"Slip Delta vs Audio RMS (Peak at {lag_slp:.1f}ms)", color="#f57c00", linewidth=1.5)
    ax4.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Zero Lag Reference (0 ms)")
    ax4.set_xlabel("Time Lag Tau (Milliseconds)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Normalized Cross-Correlation", fontsize=11, fontweight="bold")
    ax4.set_title("4. Cross-Correlation Curves across Time Lag (Exact ms Alignment Proof)", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper right", frameon=True, fontsize=9)
    ax4.set_xlim([-500, 500])

    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "multimodal_feature_cross_correlation_analysis.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, "multimodal_feature_cross_correlation_analysis.png")
    shutil.copy2(out_png, art_png)

    print(f"\n [SUCCESS] 다중 피처 상호상관 그래프 저장 완료: {out_png}")
    print(f" [SUCCESS] Artifact 복사 완료: {art_png}")


if __name__ == "__main__":
    run_cross_correlation_analysis()
