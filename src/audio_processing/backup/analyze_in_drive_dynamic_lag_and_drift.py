"""
analyze_in_drive_dynamic_lag_and_drift.py (gearbox_final_suite / src / audio_processing)

출발 시점(Launch Point) 정렬 이후의 순수 주행 전체 구간에 대해:
1. 주행 중 시간 경과에 따른 실시간 동적 위상 지연 (Time-Varying Dynamic Phase Lag, ms)
2. 주행 시작부터 종료까지의 하드웨어 시계 누적 드리프트율 (Clock Drift Rate, ms/sec)
3. 주행 전체 구간의 평균 위상차(Mean Lag, ms) 및 최대 지터 편차(Max Jitter, ms)
를 16개 시나리오 전수에 대해 단구간 상호상관(STCC) 알고리즘으로 정밀 계산하고 리포트와 시각화 그래프를 생성하는 스크립트.

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib 라벨/타이틀 100% 영문(English).
- Rule 7: LaTeX 수식 기호 일절 사용 금지 (No-LaTeX Protocol).
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
AUDIO_LAUNCH_ALIGNED_DIR = os.path.join(DATA_DIR, "audio_launch_aligned")
AUDIO_AUDIT_DIR = os.path.join(RESULTS_DIR, "audio_audit")
ARTIFACT_FIG_DIR = "/home/kante/.gemini/antigravity-ide/brain/6e341025-6278-4988-86a0-4dfdf4574e3b/figures"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(AUDIO_AUDIT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_FIG_DIR, exist_ok=True)

# 스타일 설정
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 시나리오별 CAN 출발 트리밍 오프셋
CAN_LAUNCH_TRIMS = {
    "normal_high_speed_20kph": 0.910,
    "normal_high_speed_60kph": 0.060,
    "normal_high_speed_80kph": 18.960,
    "normal_hills_6deg": 0.065,
    "normal_hills_12deg": 0.014,
    "normal_hills_18deg": 5.885,
    "normal_hills_30deg": 0.952,
    "normal_steering_pad_40kph": 1.403,
    "abnormal_high_speed_20kph": 0.280,
    "abnormal_high_speed_60kph": 1.125,
    "abnormal_high_speed_80kph": 0.191,
    "abnormal_hills_6deg": 0.000,
    "abnormal_hills_12deg": 8.840,
    "abnormal_hills_18deg": 4.132,
    "abnormal_hills_30deg": 3.355,
    "abnormal_steering_pad_40kph": 0.016,
}


def analyze_in_drive_lag_and_drift():
    print("\n" + "=" * 125)
    print(" [주행 중 동적 위상 지연(Dynamic Lag) 및 클럭 드리프트(Clock Drift) 전수 정밀 계산]")
    print("=" * 125)
    print(f" {'시나리오 (Scenario)':<28} | {'순수주행시간':<10} | {'평균위상차(Mean)':<15} | {'최대지터(Max)':<14} | {'클럭드리프트율':<15} | {'종료시점누적오차':<15}")
    print("-" * 125)

    summary_records = []
    fig, axes = plt.subplots(4, 4, figsize=(24, 18), dpi=300)
    axes_flat = axes.flatten()

    scen_idx = 0
    for scen_key, can_trim in CAN_LAUNCH_TRIMS.items():
        state = "abnormal" if "abnormal" in scen_key else "normal"
        scen_core = scen_key.replace("abnormal_", "").replace("normal_", "")

        can_path = os.path.join(RAW_DECODED_DIR, f"{scen_key}_official_decoded.csv")
        audio_dir = os.path.join(AUDIO_LAUNCH_ALIGNED_DIR, state, scen_core)
        gear_wavs = [f for f in os.listdir(audio_dir) if "gear" in f.lower() or "GEAR" in f.upper()]
        if not gear_wavs:
            gear_wavs = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        audio_path = os.path.join(audio_dir, gear_wavs[0])

        df_can = pd.read_csv(can_path)
        t_can_raw = df_can["Time"].values - df_can["Time"].values[0]

        # CAN 출발점 이후 주행 구간만 슬라이싱
        can_mask = t_can_raw >= can_trim
        t_can = t_can_raw[can_mask] - can_trim
        spd = ((df_can["WHL_SPD_FL"].values + df_can["WHL_SPD_FR"].values + df_can["WHL_SPD_RL"].values + df_can["WHL_SPD_RR"].values) / 4.0)[can_mask]

        sr, audio_raw = wavfile.read(audio_path)
        if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
        audio_norm = audio_raw.astype(np.float32) / 32768.0

        # 공통 유효 주행 시간
        drive_dur = min(t_can[-1], len(audio_norm) / float(sr))

        # 4ms 단위 오디오 RMS 시계열 생성
        audio_rms = []
        for i in range(len(t_can) - 1):
            if t_can[i] > drive_dur: break
            s_start = int(round(t_can[i] * sr))
            s_end = int(round(t_can[i + 1] * sr))
            if s_end <= len(audio_norm) and s_start >= 0:
                chunk = audio_norm[s_start:s_end]
                audio_rms.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0)
            else:
                audio_rms.append(0.0)
        
        audio_rms_arr = np.array(audio_rms)
        spd_sub = spd[:len(audio_rms_arr)]
        t_sub = t_can[:len(audio_rms_arr)]

        # 단구간(5초 슬라이딩 윈도우) 상호상관으로 실시간 위상 지연 추적
        seg_len_sec = 5.0
        seg_step_sec = 1.0
        seg_len_samples = int(seg_len_sec / 0.004)
        seg_step_samples = int(seg_step_sec / 0.004)

        seg_times = []
        seg_lags_ms = []

        for start_idx in range(0, len(spd_sub) - seg_len_samples, seg_step_samples):
            end_idx = start_idx + seg_len_samples
            s_can = spd_sub[start_idx:end_idx]
            s_aud = audio_rms_arr[start_idx:end_idx]

            std_can = np.std(s_can)
            std_aud = np.std(s_aud)
            if std_can > 1e-4 and std_aud > 1e-4:
                c1 = (s_can - np.mean(s_can)) / std_can
                c2 = (s_aud - np.mean(s_aud)) / std_aud
                corr = correlate(c1, c2, mode="full") / len(c1)
                lags = correlation_lags(len(c1), len(c2), mode="full")
                # +-1초 탐색 범위
                m = (lags >= -250) & (lags <= 250)
                best_lag_samples = lags[m][np.argmax(np.abs(corr[m]))]
                lag_ms = best_lag_samples * 4.0
            else:
                lag_ms = 0.0

            cur_t = t_sub[start_idx + seg_len_samples // 2]
            seg_times.append(cur_t)
            seg_lags_ms.append(lag_ms)

        seg_times = np.array(seg_times)
        seg_lags_ms = np.array(seg_lags_ms)

        if len(seg_lags_ms) > 0:
            mean_lag = float(np.mean(seg_lags_ms))
            max_jitter = float(np.max(np.abs(seg_lags_ms - mean_lag)))
            # 선형 회귀로 클럭 드리프트 속도(ms/sec) 추정
            if len(seg_times) > 2:
                poly = np.polyfit(seg_times, seg_lags_ms, 1)
                drift_rate_ms_per_sec = float(poly[0])
                end_drift_ms = float(poly[0] * drive_dur + poly[1])
            else:
                drift_rate_ms_per_sec = 0.0
                end_drift_ms = mean_lag
        else:
            mean_lag = 0.0
            max_jitter = 0.0
            drift_rate_ms_per_sec = 0.0
            end_drift_ms = 0.0

        summary_records.append({
            "scenario": scen_key,
            "state": state,
            "drive_duration_sec": round(drive_dur, 2),
            "mean_dynamic_lag_ms": round(mean_lag, 2),
            "max_jitter_ms": round(max_jitter, 2),
            "clock_drift_rate_ms_per_sec": round(drift_rate_ms_per_sec, 4),
            "end_drive_drift_ms": round(end_drift_ms, 2),
        })

        print(f" {scen_key:<28} | {drive_dur:>7.2f} sec | {mean_lag:>+10.1f} ms    | {max_jitter:>9.1f} ms   | {drift_rate_ms_per_sec:>+10.3f} ms/s   | {end_drift_ms:>+10.1f} ms")

        # 플롯
        ax = axes_flat[scen_idx]
        if len(seg_times) > 0:
            ax.plot(seg_times, seg_lags_ms, color="#1565c0", linewidth=1.2, marker="o", markersize=2, label="Dynamic Lag (ms)")
            # 드리프트 추세선
            fit_line = drift_rate_ms_per_sec * seg_times + (mean_lag - drift_rate_ms_per_sec * np.mean(seg_times))
            ax.plot(seg_times, fit_line, color="#d32f2f", linestyle="--", linewidth=1.1, label=f"Drift Trend ({drift_rate_ms_per_sec:+.2f} ms/s)")
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_ylabel("Lag (ms)", fontsize=9, fontweight="bold")
        ax.set_title(f"{scen_key}\n[Mean Lag: {mean_lag:+.1f}ms | Jitter: {max_jitter:.1f}ms]", fontsize=9, fontweight="bold", pad=3)
        ax.legend(loc="upper right", fontsize=7, frameon=True)
        if scen_idx >= 12:
            ax.set_xlabel("Drive Time (sec)", fontsize=9, fontweight="bold")

        scen_idx += 1

    print("=" * 125)
    plt.suptitle("In-Drive Continuous Dynamic Phase Lag and Hardware Clock Drift Analysis (16 Scenarios)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "in_drive_dynamic_lag_and_drift_curves.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, "in_drive_dynamic_lag_and_drift_curves.png")
    shutil.copy2(out_png, art_png)

    out_csv = os.path.join(AUDIO_AUDIT_DIR, "in_drive_dynamic_lag_and_clock_drift_report.csv")
    pd.DataFrame(summary_records).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f" 💾 주행 중 동적 지연 및 클럭 드리프트 리포트 CSV 저장 완료: {out_csv}")
    print(f" 📊 16개 시나리오 실시간 지연 추세선 그래프 저장 완료: {out_png}\n")


if __name__ == "__main__":
    analyze_in_drive_lag_and_drift()
