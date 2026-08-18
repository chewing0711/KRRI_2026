"""
plot_before_after_departure_alignment.py (gearbox_final_suite / src / audio_processing)

출발 시작점(Launch Point)을 기준으로 CAN과 오디오 신호가 어떻게 보정(Trim)되었는지
Before (보정 전) vs After (보정 후) 비교 그래프를 생성하여 저장하는 스크립트.

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib 라벨/타이틀 100% 영문(English).
- Rule 7: LaTeX 수식 기호 일절 사용 금지 (No-LaTeX Protocol).
- Rule 8: 이모지 전면 배제.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_LAUNCH_ALIGNED_DIR = os.path.join(DATA_DIR, "audio_launch_aligned")
ARTIFACT_FIG_DIR = "/home/kante/.gemini/antigravity-ide/brain/6e341025-6278-4988-86a0-4dfdf4574e3b/figures"
RAW_DATA_ROOT = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_FIG_DIR, exist_ok=True)

# 스타일 설정
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 시각화할 대표 시나리오 2개 설정
# 1. normal_high_speed_80kph: 보정 전 시차가 -18.8초로 가장 큼
# 2. abnormal_hills_12deg: 보정 전 시차가 -8.7초로 큼
SCENARIOS = [
    {
        "scen_key": "normal_high_speed_80kph",
        "state": "normal",
        "scen_core": "high_speed_80kph",
        "raw_subpath": "normal/고속주회로 한바퀴(High Speed circut, one round) 80kph",
        "wav_filename": "record_80-gear-1.wav",
        "label": "Scenario 1: Normal 80kph (Offset: -18.87s)",
        "t_can_launch": 18.960,
        "t_aud_launch": 0.095
    },
    {
        "scen_key": "abnormal_hills_12deg",
        "state": "abnormal",
        "scen_core": "hills_12deg",
        "raw_subpath": "abnormal/등판로 12도(Test Hills 12% grade) 20kph",
        "wav_filename": "go_up_12-gear-1_cut.wav",
        "label": "Scenario 2: Abnormal 12deg Hill (Offset: -8.73s)",
        "t_can_launch": 8.840,
        "t_aud_launch": 0.115
    }
]


def generate_before_after_plot():
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
    
    for col_idx, s_info in enumerate(SCENARIOS):
        scen_key = s_info["scen_key"]
        state = s_info["state"]
        scen_core = s_info["scen_core"]
        
        # 1. CAN 데이터 로드
        can_path = os.path.join(RAW_DECODED_DIR, f"{scen_key}_official_decoded.csv")
        df_can = pd.read_csv(can_path)
        t_can_raw = df_can["Time"].values - df_can["Time"].values[0]
        can_spd_raw = (df_can["WHL_SPD_FL"].values + df_can["WHL_SPD_FR"].values + df_can["WHL_SPD_RL"].values + df_can["WHL_SPD_RR"].values) / 4.0
        
        # 2. 오디오 데이터 로드 (보정 전 - raw_data)
        wav_path_before = os.path.join(RAW_DATA_ROOT, s_info["raw_subpath"], s_info["wav_filename"])
        sr, aud_raw_before = wavfile.read(wav_path_before)
        if aud_raw_before.ndim > 1: aud_raw_before = aud_raw_before[:, 0]
        # 32768.0 나누어 [-1, 1] 범위로 변환 (master sync plot과 기준 일치)
        aud_norm_before = aud_raw_before.astype(np.float32) / 32768.0
        
        # 오디오 RMS 에너지 계산 (보정 전)
        audio_rms_before = []
        for i in range(len(t_can_raw) - 1):
            s_start = int(round(t_can_raw[i] * sr))
            s_end = int(round(t_can_raw[i + 1] * sr))
            if s_end <= len(aud_norm_before) and s_start >= 0:
                chunk = aud_norm_before[s_start:s_end]
                audio_rms_before.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0)
            else:
                audio_rms_before.append(0.0)
        audio_rms_before.append(audio_rms_before[-1] if audio_rms_before else 0.0)
        audio_rms_before = np.array(audio_rms_before)

        # 3. 오디오 데이터 로드 (보정 후 - audio_launch_aligned)
        aud_dir_after = os.path.join(AUDIO_LAUNCH_ALIGNED_DIR, state, scen_core)
        wav_path_after = os.path.join(aud_dir_after, s_info["wav_filename"])
        _, aud_raw_after = wavfile.read(wav_path_after)
        if aud_raw_after.ndim > 1: aud_raw_after = aud_raw_after[:, 0]
        # 32768.0 나누어 [-1, 1] 범위로 변환 (master sync plot과 기준 일치)
        aud_norm_after = aud_raw_after.astype(np.float32) / 32768.0
        
        # CAN 데이터 트리밍 (보정 후)
        df_can_after = df_can[df_can["Time"] >= s_info["t_can_launch"]].reset_index(drop=True)
        t_can_after = df_can_after["Time"].values - df_can_after["Time"].values[0]
        can_spd_after = (df_can_after["WHL_SPD_FL"].values + df_can_after["WHL_SPD_FR"].values + df_can_after["WHL_SPD_RL"].values + df_can_after["WHL_SPD_RR"].values) / 4.0
        
        # 오디오 RMS 에너지 계산 (보정 후)
        audio_rms_after = []
        for i in range(len(t_can_after) - 1):
            s_start = int(round(t_can_after[i] * sr))
            s_end = int(round(t_can_after[i + 1] * sr))
            if s_end <= len(aud_norm_after) and s_start >= 0:
                chunk = aud_norm_after[s_start:s_end]
                audio_rms_after.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0)
            else:
                audio_rms_after.append(0.0)
        audio_rms_after.append(audio_rms_after[-1] if audio_rms_after else 0.0)
        audio_rms_after = np.array(audio_rms_after)

        # ---------------------------------------------------------------------
        # [PLOT 1] BEFORE ALIGNMENT (보정 전)
        # ---------------------------------------------------------------------
        ax_bef = axes[0, col_idx]
        color_can = "#1565c0"
        color_aud = "#e65100"
        
        line1 = ax_bef.plot(t_can_raw, can_spd_raw, color=color_can, linewidth=1.5, label="CAN: Wheel Speed (km/h)")
        ax_bef.axvline(x=s_info["t_can_launch"], color="#2e7d32", linestyle="--", linewidth=1.5, label=f"CAN Departure ({s_info['t_can_launch']}s)")
        ax_bef.set_ylabel("Speed (km/h)", color=color_can, fontsize=11, fontweight="bold")
        ax_bef.tick_params(axis="y", labelcolor=color_can)
        ax_bef.set_title(f"{s_info['label']}\n[BEFORE Alignment] Departure points are displaced", fontsize=12, fontweight="bold", pad=5)
        
        ax_bef_tw = ax_bef.twinx()
        line2 = ax_bef_tw.plot(t_can_raw, audio_rms_before, color=color_aud, linewidth=1.2, alpha=0.75, label="Audio: RMS Energy")
        ax_bef_tw.axvline(x=s_info["t_aud_launch"], color="#c62828", linestyle="--", linewidth=1.5, label=f"Audio Departure ({s_info['t_aud_launch']}s)")
        ax_bef_tw.set_ylabel("Audio RMS", color=color_aud, fontsize=11, fontweight="bold")
        ax_bef_tw.tick_params(axis="y", labelcolor=color_aud)
        ax_bef_tw.grid(False)
        
        # 범례 결합
        lines = line1 + [ax_bef.axvline(x=s_info["t_can_launch"], color="#2e7d32", linestyle="--")] + line2 + [ax_bef_tw.axvline(x=s_info["t_aud_launch"], color="#c62828", linestyle="--")]
        labels = ["CAN Speed", "CAN Launch", "Audio RMS", "Audio Launch"]
        ax_bef.legend(lines, labels, loc="upper right", fontsize=8, frameon=True)
        ax_bef.set_xlabel("Recording Raw Time (Seconds)", fontsize=10)

        # ---------------------------------------------------------------------
        # [PLOT 2] AFTER ALIGNMENT (보정 후 - 0초 정합 완료)
        # ---------------------------------------------------------------------
        ax_aft = axes[1, col_idx]
        
        line3 = ax_aft.plot(t_can_after, can_spd_after, color=color_can, linewidth=1.5, label="CAN: Aligned Speed")
        ax_aft.set_ylabel("Speed (km/h)", color=color_can, fontsize=11, fontweight="bold")
        ax_aft.tick_params(axis="y", labelcolor=color_can)
        ax_aft.set_title(f"[AFTER Aligned at 0.0s]\nBoth departure events mapped to 0s", fontsize=12, fontweight="bold", pad=5)
        
        ax_aft_tw = ax_aft.twinx()
        line4 = ax_aft_tw.plot(t_can_after, audio_rms_after, color=color_aud, linewidth=1.2, alpha=0.8, label="Audio: Aligned RMS")
        ax_aft_tw.set_ylabel("Audio RMS", color=color_aud, fontsize=11, fontweight="bold")
        ax_aft_tw.tick_params(axis="y", labelcolor=color_aud)
        ax_aft_tw.grid(False)
        
        lines_aft = line3 + line4
        labels_aft = ["Aligned CAN Speed", "Aligned Audio RMS"]
        ax_aft.legend(lines_aft, labels_aft, loc="upper right", fontsize=9, frameon=True)
        ax_aft.set_xlabel("Aligned Process Time (Seconds, 0s = Launch)", fontsize=10)

    plt.suptitle("Comparative Analysis of CAN-Audio Synchronization: BEFORE vs AFTER Departure Event Alignment", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    out_png = os.path.join(FIG_DIR, "departure_alignment_before_after_comparison.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    art_png = os.path.join(ARTIFACT_FIG_DIR, "departure_alignment_before_after_comparison.png")
    import shutil
    shutil.copy2(out_png, art_png)
    print(f" * [성공] Before-After 비교 그래프 저장 완료: {out_png}")
    print(f" * [성공] Artifact 복사 완료: {art_png}")


if __name__ == "__main__":
    generate_before_after_plot()
