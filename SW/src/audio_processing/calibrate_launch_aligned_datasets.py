"""
calibrate_launch_aligned_datasets.py (gearbox_final_suite / src / audio_processing)

16개 전체 시나리오에 대해:
1. '차량 실제 물리적 출발 시점(Launch Point)'을 기준으로 앞부분 정차 대기 구간을 정밀 절단(Trim)
2. 출발 이후 순수 주행 전체 구간에 대해:
   - CAN 도메인: 구동축 미세 슬립 변동(|WHL_SPD_FR - WHL_SPD_RR|의 미분/변동)
   - Audio 도메인: 힐베르트 변환 포락선(Hilbert Envelope) 충격파
   간의 단구간 상호상관(STCC) 알고리즘으로:
   - 주행 중 실시간 평균 동적 위상 지연 (Mean Dynamic Lag, ms)
   - 주행 중 최대 지터 편차 (Max Jitter, ms)
   - 하드웨어 시계 누적 드리프트 속도 (Clock Drift Rate, ms/sec)
   - 주행 종료 시점 최종 누적 오차 (End Drive Drift, ms)
를 정밀 계산하여 리포트와 마스터 그래프를 생성하는 통합 스크립트.

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib 라벨/타이틀 100% 영문(English).
- Rule 7: LaTeX 수식 문법 전면 금지 (No-LaTeX Protocol).
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags, hilbert

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_ALIGNED_DIR = os.path.join(DATA_DIR, "audio_aligned")
AUDIO_LAUNCH_ALIGNED_DIR = os.path.join(DATA_DIR, "audio_launch_aligned")
AUDIO_AUDIT_DIR = os.path.join(RESULTS_DIR, "audio_audit")
ARTIFACT_FIG_DIR = "/home/kante/.gemini/antigravity-ide/brain/6e341025-6278-4988-86a0-4dfdf4574e3b/figures"

os.makedirs(AUDIO_LAUNCH_ALIGNED_DIR, exist_ok=True)
os.makedirs(AUDIO_AUDIT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_FIG_DIR, exist_ok=True)

# 스타일 설정
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 실측된 16개 시나리오 물리적 출발 시점 (초 단위)
LAUNCH_OFFSETS = {
    "normal_high_speed_20kph": (0.910, 0.080),
    "normal_high_speed_60kph": (0.060, 0.065),
    "normal_high_speed_80kph": (18.960, 0.095),
    "normal_hills_6deg": (0.065, 0.130),
    "normal_hills_12deg": (0.014, 0.055),
    "normal_hills_18deg": (5.885, 0.070),
    "normal_hills_30deg": (0.952, 0.065),
    "normal_steering_pad_40kph": (1.403, 0.075),
    "abnormal_high_speed_20kph": (0.280, 0.180),
    "abnormal_high_speed_60kph": (1.125, 0.145),
    "abnormal_high_speed_80kph": (0.191, 0.120),
    "abnormal_hills_6deg": (0.000, 1.177),
    "abnormal_hills_12deg": (8.840, 0.115),
    "abnormal_hills_18deg": (4.132, 0.070),
    "abnormal_hills_30deg": (3.355, 0.100),
    "abnormal_steering_pad_40kph": (0.016, 0.175),
}


def run_launch_calibration_and_in_drive_analysis():
    print("\n" + "=" * 135)
    print(" [휠 가속도/슬립 변동 vs 힐베르트 포락선 기반 주행 중 동적 위상 지연 및 클럭 드리프트 정밀 계산]")
    print("=" * 135)
    print(f" {'시나리오 (Scenario)':<28} | {'순수주행':<8} | {'보정전출발시차':<14} | {'주행중평균위상차':<16} | {'최대지터':<10} | {'클럭드리프트속도':<16} | {'종료누적오차':<12}")
    print("-" * 135)

    records = []
    fig, axes = plt.subplots(4, 4, figsize=(24, 18), dpi=300)
    axes_flat = axes.flatten()

    scen_idx = 0
    for scen_key, (t_can_launch, t_aud_launch) in LAUNCH_OFFSETS.items():
        state = "abnormal" if "abnormal" in scen_key else "normal"
        scen_core = scen_key.replace("abnormal_", "").replace("normal_", "")

        raw_launch_delta_ms = (t_aud_launch - t_can_launch) * 1000.0

        # 타겟 폴더 생성 및 정제 오디오 저장
        target_dir = os.path.join(AUDIO_LAUNCH_ALIGNED_DIR, state, scen_core)
        os.makedirs(target_dir, exist_ok=True)

        src_audio_dir = os.path.join(AUDIO_ALIGNED_DIR, state, scen_core)
        wav_files = [f for f in os.listdir(src_audio_dir) if f.endswith(".wav")]

        for wav_name in wav_files:
            in_wav_path = os.path.join(src_audio_dir, wav_name)
            out_wav_path = os.path.join(target_dir, wav_name)
            sr, audio_raw = wavfile.read(in_wav_path)
            start_sample = int(round(t_aud_launch * sr))
            aligned_audio = audio_raw[start_sample:]
            wavfile.write(out_wav_path, sr, aligned_audio)

        # Gear-1 오디오 파일 로드
        gear_wavs = [f for f in os.listdir(target_dir) if "gear" in f.lower() or "GEAR" in f.upper()]
        if not gear_wavs: gear_wavs = wav_files
        sr_g, aud_g_raw = wavfile.read(os.path.join(target_dir, gear_wavs[0]))
        if aud_g_raw.ndim > 1: aud_g_raw = aud_g_raw[:, 0]
        audio_norm = aud_g_raw.astype(np.float32) / 32768.0

        # CAN 로드 및 출발점 이후 슬라이싱
        can_path = os.path.join(RAW_DECODED_DIR, f"{scen_key}_official_decoded.csv")
        df_can = pd.read_csv(can_path)
        t_can_raw = df_can["Time"].values - df_can["Time"].values[0]

        can_mask = t_can_raw >= t_can_launch
        t_can = t_can_raw[can_mask] - t_can_launch

        # 1. CAN 동적 변동 신호 (구동축 휠속도차 슬립 변동량)
        whl_fr = df_can["WHL_SPD_FR"].values[can_mask]
        whl_rr = df_can["WHL_SPD_RR"].values[can_mask]
        can_slip_delta = np.abs(whl_fr - whl_rr)
        # 고주파 변동 추출 (미분)
        can_dynamic_sig = np.gradient(can_slip_delta)

        drive_dur = min(t_can[-1], len(audio_norm) / float(sr_g))

        # 2. Audio 힐베르트 포락선 계산
        # 연산 효율을 위해 4ms 그리드로 다운샘플링된 힐베르트 포락선 에너지 생성
        audio_env = []
        for i in range(len(t_can) - 1):
            if t_can[i] > drive_dur: break
            s_start = int(round(t_can[i] * sr_g))
            s_end = int(round(t_can[i + 1] * sr_g))
            if s_end <= len(audio_norm) and s_start >= 0:
                chunk = audio_norm[s_start:s_end]
                if len(chunk) > 4:
                    env_chunk = np.abs(hilbert(chunk))
                    audio_env.append(float(np.mean(env_chunk)))
                else:
                    audio_env.append(float(np.mean(np.abs(chunk))) if len(chunk) > 0 else 0.0)
            else:
                audio_env.append(0.0)

        audio_env_arr = np.array(audio_env)
        # 고주파 충격 변동 추출 (미분)
        audio_dynamic_sig = np.gradient(audio_env_arr)

        n_pts = min(len(can_dynamic_sig), len(audio_dynamic_sig))
        sig_can_use = can_dynamic_sig[:n_pts]
        sig_aud_use = audio_dynamic_sig[:n_pts]
        t_use = t_can[:n_pts]

        # 3. 단구간(4초 슬라이딩 윈도우) 상호상관으로 실시간 위상 지연 추적
        seg_len_sec = 4.0
        seg_step_sec = 1.0
        seg_len_samples = int(seg_len_sec / 0.004)
        seg_step_samples = int(seg_step_sec / 0.004)

        seg_times = []
        seg_lags_ms = []

        for start_idx in range(0, len(sig_can_use) - seg_len_samples, seg_step_samples):
            end_idx = start_idx + seg_len_samples
            s_c = sig_can_use[start_idx:end_idx]
            s_a = sig_aud_use[start_idx:end_idx]

            std_c = np.std(s_c)
            std_a = np.std(s_a)
            if std_c > 1e-6 and std_a > 1e-6:
                c1 = (s_c - np.mean(s_c)) / std_c
                c2 = (s_a - np.mean(s_a)) / std_a
                corr = correlate(c1, c2, mode="full") / len(c1)
                lags = correlation_lags(len(c1), len(c2), mode="full")
                # +-250ms 탐색 범위
                m = (lags >= -62) & (lags <= 62)
                best_lag_samples = lags[m][np.argmax(corr[m])]
                lag_ms = best_lag_samples * 4.0
            else:
                lag_ms = 0.0

            cur_t = t_use[start_idx + seg_len_samples // 2]
            seg_times.append(cur_t)
            seg_lags_ms.append(lag_ms)

        seg_times = np.array(seg_times)
        seg_lags_ms = np.array(seg_lags_ms)

        if len(seg_lags_ms) > 0:
            mean_lag = float(np.mean(seg_lags_ms))
            max_jitter = float(np.max(np.abs(seg_lags_ms - mean_lag)))
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

        records.append({
            "scenario": scen_key,
            "state": state,
            "can_launch_trim_sec": round(t_can_launch, 3),
            "audio_launch_trim_sec": round(t_aud_launch, 3),
            "raw_launch_delta_ms": round(raw_launch_delta_ms, 1),
            "drive_duration_sec": round(drive_dur, 2),
            "mean_dynamic_lag_ms": round(mean_lag, 2),
            "max_jitter_ms": round(max_jitter, 2),
            "clock_drift_rate_ms_per_sec": round(drift_rate_ms_per_sec, 4),
            "end_drive_drift_ms": round(end_drift_ms, 2),
        })

        print(f" {scen_key:<28} | {drive_dur:>5.1f}s   | {raw_launch_delta_ms:>+10.1f} ms    | {mean_lag:>+12.1f} ms    | {max_jitter:>8.1f} ms | {drift_rate_ms_per_sec:>+12.3f} ms/s  | {end_drift_ms:>+10.1f} ms")

        # 플롯 생성
        ax = axes_flat[scen_idx]
        if len(seg_times) > 0:
            ax.plot(seg_times, seg_lags_ms, color="#1565c0", linewidth=1.2, marker="o", markersize=2, label="Dynamic Lag (ms)")
            fit_line = drift_rate_ms_per_sec * seg_times + (mean_lag - drift_rate_ms_per_sec * np.mean(seg_times))
            ax.plot(seg_times, fit_line, color="#d32f2f", linestyle="--", linewidth=1.1, label=f"Drift Trend ({drift_rate_ms_per_sec:+.2f} ms/s)")
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_ylabel("Lag (ms)", fontsize=9, fontweight="bold")
        ax.set_title(f"{scen_key}\n[Mean Lag: {mean_lag:+.1f}ms | Drift: {drift_rate_ms_per_sec:+.2f}ms/s]", fontsize=9, fontweight="bold", pad=3)
        ax.legend(loc="upper right", fontsize=7, frameon=True)
        if scen_idx >= 12:
            ax.set_xlabel("Drive Time (sec)", fontsize=9, fontweight="bold")

        scen_idx += 1

    print("=" * 135)
    plt.suptitle("In-Drive Continuous Dynamic Phase Lag and Hardware Clock Drift Analysis (16 Scenarios)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "launch_aligned_all_scenarios_master_plot.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, "launch_aligned_all_scenarios_master_plot.png")
    shutil.copy2(out_png, art_png)

    df_out = pd.DataFrame(records)
    out_csv = os.path.join(AUDIO_AUDIT_DIR, "launch_aligned_calibration_summary.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f" 💾 [교정 및 주행 중 오차 계산 완료] 리포트 CSV 저장: {out_csv}")
    print(f" 📊 [16개 시나리오 통합 그래프 저장]: {out_png}\n")


if __name__ == "__main__":
    run_launch_calibration_and_in_drive_analysis()
