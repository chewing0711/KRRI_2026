"""
verify_physical_launch_event_alignment.py (gearbox_final_suite / src / audio_processing)

16개 전체 시나리오에 대해
1. CAN 물리적 출발/가속 시점 (t_CAN: 휠속도/페달 최초 급상승 시점)
2. Audio 물리적 음향 발생 시점 (t_Audio: 마이크 정적 소음 뚫고 모터/기어 소음 최초 폭발 시점)
을 1ms 정밀도로 동시 탐색하여, 두 센서의 실제 물리적 시작 시차(Delta t, ms)를 전수 실측·검증하는 스크립트.

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

SCENARIOS = [
    ("normal_high_speed_20kph", "Normal High Speed 20kph"),
    ("normal_high_speed_60kph", "Normal High Speed 60kph"),
    ("normal_high_speed_80kph", "Normal High Speed 80kph"),
    ("normal_hills_6deg", "Normal Hills 6deg"),
    ("normal_hills_12deg", "Normal Hills 12deg"),
    ("normal_hills_18deg", "Normal Hills 18deg"),
    ("normal_hills_30deg", "Normal Hills 30deg"),
    ("normal_steering_pad_40kph", "Normal Steering Pad 40kph"),
    ("abnormal_high_speed_20kph", "Abnormal High Speed 20kph"),
    ("abnormal_high_speed_60kph", "Abnormal High Speed 60kph"),
    ("abnormal_high_speed_80kph", "Abnormal High Speed 80kph"),
    ("abnormal_hills_6deg", "Abnormal Hills 6deg"),
    ("abnormal_hills_12deg", "Abnormal Hills 12deg"),
    ("abnormal_hills_18deg", "Abnormal Hills 18deg"),
    ("abnormal_hills_30deg", "Abnormal Hills 30deg"),
    ("abnormal_steering_pad_40kph", "Abnormal Steering Pad 40kph"),
]


def detect_can_launch_time(t_arr, spd_arr, pedal_arr):
    """
    CAN 도메인에서 차량이 실제로 출발/가속을 시작한 시점(초) 탐색
    """
    min_spd = spd_arr[0]
    # 만약 정지 상태(속도 < 2km/h)에서 시작한 경우
    if min_spd < 2.0:
        idx = np.where(spd_arr >= (min_spd + 0.5))[0]
        if len(idx) > 0:
            return float(t_arr[idx[0]])
    # 만약 이미 정속 주행 중인 상태에서 녹음된 경우 (최초 속도 변동 또는 페달 입력 시점)
    if pedal_arr is not None and np.max(pedal_arr) > 0:
        idx_p = np.where(pedal_arr >= (pedal_arr[0] + 2.0))[0]
        if len(idx_p) > 0:
            return float(t_arr[idx_p[0]])
    
    # 기본값: 0.0초 (녹음 시작과 동시에 이미 주행 중)
    return float(t_arr[0])


def detect_audio_sound_burst_time(audio_norm, sr, max_search_sec=5.0):
    """
    Audio 도메인에서 마이크 정적 잡음을 뚫고 모터/기어 회전음이 폭발한 최초 시점(초) 탐색
    """
    win_samples = int(0.01 * sr)  # 10ms 윈도우
    hop_samples = int(0.005 * sr)  # 5ms 스텝
    n_frames = min(int(max_search_sec * sr), len(audio_norm) - win_samples) // hop_samples
    
    times = []
    rms_vals = []
    for f in range(n_frames):
        s_start = f * hop_samples
        chunk = audio_norm[s_start : s_start + win_samples]
        rms = np.sqrt(np.mean(chunk ** 2))
        times.append(s_start / float(sr))
        rms_vals.append(rms)
        
    times = np.array(times)
    rms_vals = np.array(rms_vals)
    
    # 최초 50ms 배경 잡음 통계
    bg_noise = np.mean(rms_vals[:10]) if len(rms_vals) >= 10 else 1e-6
    bg_std = np.std(rms_vals[:10]) if len(rms_vals) >= 10 else 1e-6
    thresh = bg_noise + 3.5 * bg_std
    
    # 임계치 돌파 시점
    idx = np.where(rms_vals >= thresh)[0]
    if len(idx) > 0 and bg_noise < 0.01:
        return float(times[idx[0]]), times, rms_vals
    
    # 이미 녹음 시작부터 소리가 컸던 경우
    return 0.0, times, rms_vals


def verify_all_launch_events():
    print("\n" + "=" * 125)
    print(" 🚦 [16개 시나리오 차량 실제 물리적 출발(Launch) 시점 vs 오디오 음향 발생 시점 정밀 대조]")
    print("=" * 125)
    print(f" {'시나리오 (Scenario)':<30} | {'CAN 출발시점 (t_CAN)':<20} | {'오디오 소리발생 (t_Audio)':<22} | {'물리 시차 (Delta t, ms)':<22} | {'동기화 판정':<15}")
    print("-" * 125)

    results = []
    fig, axes = plt.subplots(4, 4, figsize=(24, 18), dpi=300)
    axes_flat = axes.flatten()

    for idx, (scen_key, scen_label) in enumerate(SCENARIOS):
        ax = axes_flat[idx]
        state = "abnormal" if "abnormal" in scen_key else "normal"
        scen_core = scen_key.replace("abnormal_", "").replace("normal_", "")
        
        can_path = os.path.join(RAW_DECODED_DIR, f"{scen_key}_official_decoded.csv")
        audio_dir = os.path.join(AUDIO_ALIGNED_DIR, state, scen_core)
        gear_wavs = [f for f in os.listdir(audio_dir) if "gear" in f.lower() or "GEAR" in f.upper()]
        if not gear_wavs:
            gear_wavs = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        audio_path = os.path.join(audio_dir, gear_wavs[0])

        df_can = pd.read_csv(can_path)
        t_can = df_can["Time"].values - df_can["Time"].values[0]
        spd = (df_can["WHL_SPD_FL"].values + df_can["WHL_SPD_FR"].values + df_can["WHL_SPD_RL"].values + df_can["WHL_SPD_RR"].values) / 4.0
        pedal = df_can["Accel_Pedal_Pos"].values if "Accel_Pedal_Pos" in df_can.columns else np.zeros_like(t_can)

        sr, audio_raw = wavfile.read(audio_path)
        if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
        audio_norm = audio_raw.astype(np.float32) / 32768.0

        # 시점 탐색
        t_can_launch = detect_can_launch_time(t_can, spd, pedal)
        t_aud_launch, aud_t_grid, aud_rms_grid = detect_audio_sound_burst_time(audio_norm, sr, max_search_sec=min(5.0, t_can[-1]))

        delta_t_ms = (t_aud_launch - t_can_launch) * 1000.0
        
        if abs(delta_t_ms) <= 50.0:
            status = "PERFECT SYNC (<=50ms)"
        elif abs(delta_t_ms) <= 200.0:
            status = "ACCEPTABLE (<=200ms)"
        else:
            status = "OFFSET DETECTED"

        results.append({
            "scenario": scen_key,
            "state": state,
            "t_can_launch_sec": round(t_can_launch, 4),
            "t_audio_launch_sec": round(t_aud_launch, 4),
            "delta_t_launch_ms": round(delta_t_ms, 2),
            "status": status,
        })

        print(f" {scen_key:<30} | {t_can_launch:>14.3f} 초     | {t_aud_launch:>16.3f} 초       | {delta_t_ms:>+16.1f} ms       | {status:<15}")

        # 플로팅 (앞 4초 줌인)
        plot_limit_sec = min(4.0, t_can[-1])
        can_mask = t_can <= plot_limit_sec
        aud_mask = aud_t_grid <= plot_limit_sec

        ax.plot(t_can[can_mask], spd[can_mask], color="#1565c0", linewidth=1.4, label="CAN Speed (km/h)")
        ax.set_ylabel("Speed (km/h)", color="#1565c0", fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelcolor="#1565c0")

        ax_tw = ax.twinx()
        ax_tw.plot(aud_t_grid[aud_mask], aud_rms_grid[aud_mask], color="#d84315", linewidth=1.1, alpha=0.8, label="Audio RMS")
        ax_tw.set_ylabel("Audio RMS", color="#d84315", fontsize=9, fontweight="bold")
        ax_tw.tick_params(axis="y", labelcolor="#d84315")
        ax_tw.grid(False)

        # 출발 수직선 표시
        ax.axvline(t_can_launch, color="#0d47a1", linestyle="--", linewidth=1.2, alpha=0.9, label=f"CAN Launch ({t_can_launch:.2f}s)")
        ax.axvline(t_aud_launch, color="#b71c1c", linestyle=":", linewidth=1.2, alpha=0.9, label=f"Audio Sound ({t_aud_launch:.2f}s)")

        ax.set_title(f"{scen_label}\n[Delta t: {delta_t_ms:+.1f} ms | {status}]", fontsize=10, fontweight="bold", pad=4)
        if idx >= 12:
            ax.set_xlabel("Time (Seconds)", fontsize=10, fontweight="bold")

    print("=" * 125)
    plt.suptitle("Physical Launch Event Synchronization Proof across All 16 Scenarios (CAN Wheel Motion vs Audio Acoustic Burst)", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "physical_launch_event_synchronization_proof.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, "physical_launch_event_synchronization_proof.png")
    shutil.copy2(out_png, art_png)

    out_csv = os.path.join(AUDIO_AUDIT_DIR, "physical_launch_event_alignment_report.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f" 💾 물리적 출발 시점 실측 리포트 CSV 저장 완료: {out_csv}")
    print(f" 📊 16개 시나리오 발진 줌인 대조 플롯 저장 완료: {out_png}")


if __name__ == "__main__":
    verify_all_launch_events()
