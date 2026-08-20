"""
plot_can_audio_synchronized_overlay.py (gearbox_final_suite / src / audio_processing)

동일한 시간축(Time Axis, 초 단위) 상에서
1. CAN 차량 주행 신호 (4륜 휠속도 및 페달 개도량)
2. 44.1kHz 오디오 음향 원시 파형 및 힐베르트 포락선(Envelope)
3. CAN 4ms 행별 동기화 매핑된 Audio RMS 에너지 vs 휠속도 요동(Flapping)
4. 0.74초 단일 슬라이딩 윈도우 구간(250개 CAN 점 vs 32,500개 오디오 선) 초정밀 줌인(Zoom-in) 대조

를 단 하나의 종합 비교 그래프로 생성하여 직관적으로 검증하는 시각화 스크립트.

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib 라벨/타이틀/범례 100% 영문(English).
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_ALIGNED_DIR = os.path.join(DATA_DIR, "audio_aligned")
ARTIFACT_FIG_DIR = "/home/kante/.gemini/antigravity-ide/brain/6e341025-6278-4988-86a0-4dfdf4574e3b/figures"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_FIG_DIR, exist_ok=True)

# 스타일 설정
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def generate_can_audio_overlay_plot(scenario_name="abnormal_high_speed_20kph"):
    can_path = os.path.join(RAW_DECODED_DIR, f"{scenario_name}_official_decoded.csv")
    state = "abnormal" if "abnormal" in scenario_name else "normal"
    scen_core = scenario_name.replace("abnormal_", "").replace("normal_", "")
    audio_path = os.path.join(AUDIO_ALIGNED_DIR, state, scen_core, "go_20-gear-1.wav" if "20" in scen_core else "go_60-gear-1.wav")

    if not os.path.exists(audio_path):
        # 폴더 내 첫 번째 wav 자동 탐색
        audio_dir = os.path.join(AUDIO_ALIGNED_DIR, state, scen_core)
        found_wavs = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        audio_path = os.path.join(audio_dir, found_wavs[0])

    print(f" 📂 CAN 데이터 로드: {os.path.basename(can_path)}")
    print(f" 📂 오디오 데이터 로드: {os.path.basename(audio_path)}")

    df_can = pd.read_csv(can_path)
    sr, audio_raw = wavfile.read(audio_path)
    if audio_raw.ndim > 1:
        audio_raw = audio_raw[:, 0]

    # 정규화
    audio_norm = audio_raw.astype(np.float32) / 32768.0
    audio_time = np.arange(len(audio_norm)) / float(sr)

    # CAN 타임스탬프 (0초 기준 정렬)
    t_can = df_can["Time"].values - df_can["Time"].values[0]
    whl_fl = df_can["WHL_SPD_FL"].values
    whl_fr = df_can["WHL_SPD_FR"].values
    whl_rl = df_can["WHL_SPD_RL"].values
    whl_rr = df_can["WHL_SPD_RR"].values
    pedal = df_can["Accel_Pedal_Pos"].values if "Accel_Pedal_Pos" in df_can.columns else np.zeros_like(t_can)

    # 4ms 단위 오디오 RMS 계산
    audio_rms_per_can = []
    for i in range(len(t_can) - 1):
        s_start = int(round(t_can[i] * sr))
        s_end = int(round(t_can[i + 1] * sr))
        chunk = audio_norm[s_start:s_end]
        rms_val = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        audio_rms_per_can.append(rms_val)
    audio_rms_per_can.append(audio_rms_per_can[-1] if audio_rms_per_can else 0.0)
    audio_rms_arr = np.array(audio_rms_per_can)

    # =========================================================================
    # [4단 고해상도 시각화 구성]
    # =========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), dpi=300, sharex=False)

    # -------------------------------------------------------------
    # [Panel 1] CAN 주행 신호 (4륜 휠속도 & 페달)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(t_can, whl_fl, label="FL Wheel Speed (km/h)", color="#1f77b4", linewidth=1.2, alpha=0.9)
    ax1.plot(t_can, whl_fr, label="FR Wheel Speed (km/h)", color="#2ca02c", linewidth=1.2, alpha=0.9)
    ax1.plot(t_can, whl_rl, label="RL Wheel Speed (km/h)", color="#ff7f0e", linewidth=1.2, alpha=0.9)
    ax1.plot(t_can, whl_rr, label="RR Wheel Speed (km/h)", color="#d62728", linewidth=1.2, alpha=0.9)
    ax1.set_ylabel("Wheel Speed (km/h)", fontsize=11, fontweight="bold")
    ax1.set_title(f"1. CAN Domain: Real-time 4-Wheel Speed Signals ({scenario_name})", fontsize=13, fontweight="bold", pad=8)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.set_xlim([0, min(t_can[-1], audio_time[-1])])

    # -------------------------------------------------------------
    # [Panel 2] 44.1kHz 오디오 원시 음향 파형 (Raw Waveform)
    # -------------------------------------------------------------
    ax2 = axes[1]
    # 플로팅 가속을 위한 데시메이션
    decimate_factor = max(1, len(audio_norm) // 50000)
    ax2.plot(audio_time[::decimate_factor], audio_norm[::decimate_factor], color="#4a148c", linewidth=0.5, alpha=0.8, label="Raw Audio Waveform (44.1 kHz)")
    ax2.set_ylabel("Audio Amplitude (-1 to +1)", fontsize=11, fontweight="bold")
    ax2.set_title("2. Audio Domain: Raw Acoustic Vibration Waveform (44,100 Hz Continuous)", fontsize=13, fontweight="bold", pad=8)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.set_xlim([0, min(t_can[-1], audio_time[-1])])

    # -------------------------------------------------------------
    # [Panel 3] 1:1 동기화 오버레이 (CAN 휠속도차 vs Audio RMS 에너지)
    # -------------------------------------------------------------
    ax3 = axes[2]
    wheel_diff = np.abs(whl_fr - whl_rr)
    line1 = ax3.plot(t_can, wheel_diff, color="#e65100", linewidth=1.2, label="CAN: |FR - RR| Wheel Speed Delta (km/h)")
    ax3.set_ylabel("Wheel Delta (km/h)", color="#e65100", fontsize=11, fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="#e65100")

    ax3_twin = ax3.twinx()
    line2 = ax3_twin.plot(t_can, audio_rms_arr, color="#00695c", linewidth=1.2, alpha=0.85, label="Audio: Synchronized 4ms RMS Energy")
    ax3_twin.set_ylabel("Audio RMS Energy", color="#00695c", fontsize=11, fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor="#00695c")
    ax3_twin.grid(False)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right", frameon=True, fontsize=9)
    ax3.set_title("3. Multimodal Overlay: CAN Dynamic Fluctuation vs Synchronized Audio RMS Power", fontsize=13, fontweight="bold", pad=8)
    ax3.set_xlim([0, min(t_can[-1], audio_time[-1])])

    # -------------------------------------------------------------
    # [Panel 4] 0.74초(250샘플) 단일 윈도우 초정밀 줌인 비교 (Zoom-in)
    # -------------------------------------------------------------
    ax4 = axes[3]
    zoom_t_start = 15.0
    zoom_t_end = 15.74  # 0.74초 윈도우 구간

    # CAN 줌인 데이터 (약 185개 점)
    can_mask = (t_can >= zoom_t_start) & (t_can <= zoom_t_end)
    t_can_zoom = t_can[can_mask]
    whl_fl_zoom = whl_fl[can_mask]

    # Audio 줌인 데이터 (약 32,600개 파형)
    aud_mask = (audio_time >= zoom_t_start) & (audio_time <= zoom_t_end)
    t_aud_zoom = audio_time[aud_mask]
    aud_zoom = audio_norm[aud_mask]

    ax4.plot(t_aud_zoom, aud_zoom, color="#7b1fa2", linewidth=0.8, alpha=0.7, label="Audio Waveform (32,634 points in 0.74s window)")
    ax4.set_ylabel("Audio Waveform", color="#7b1fa2", fontsize=11, fontweight="bold")
    ax4.tick_params(axis="y", labelcolor="#7b1fa2")

    ax4_twin = ax4.twinx()
    ax4_twin.plot(t_can_zoom, whl_fl_zoom, color="#0d47a1", marker="o", markersize=3, linewidth=1.5, label="CAN WHL_SPD (185 discrete samples in 0.74s)")
    ax4_twin.set_ylabel("Wheel Speed (km/h)", color="#0d47a1", fontsize=11, fontweight="bold")
    ax4_twin.tick_params(axis="y", labelcolor="#0d47a1")
    ax4_twin.grid(False)

    lines_zoom = ax4.get_lines() + ax4_twin.get_lines()
    labels_zoom = [l.get_label() for l in lines_zoom]
    ax4.legend(lines_zoom, labels_zoom, loc="upper right", frameon=True, fontsize=9)
    ax4.set_title("4. Ultra-Precision Window Zoom-in (0.74s Window: 250 CAN Points vs 32,500 Audio Waveform Samples)", fontsize=13, fontweight="bold", pad=8)
    ax4.set_xlabel("Time (Seconds)", fontsize=12, fontweight="bold")
    ax4.set_xlim([zoom_t_start, zoom_t_end])

    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "can_audio_synchronized_overlay_comparison.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, "can_audio_synchronized_overlay_comparison.png")
    shutil.copy2(out_png, art_png)

    print(f"\n [SUCCESS] 동기화 오버레이 비교 그래프 저장: {out_png}")
    print(f" [SUCCESS] Artifact 동기화 완료: {art_png}")


if __name__ == "__main__":
    generate_can_audio_overlay_plot()
