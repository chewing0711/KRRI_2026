"""
plot_all_scenarios_multimodal_sync.py (gearbox_final_suite / src / audio_processing)

16개 전체 시나리오(Normal 8개 + Abnormal 8개)에 대해
1. 총 시간 길이 오차 (|CAN 시간 - Audio 시간|, 초/ms)
2. 상호상관(Cross-Correlation) 신호 시차 오차 (Lag Tau, ms)
3. 250샘플 슬라이딩 윈도우 유효 매칭률 및 드롭(Drop) 개수
를 정밀 계산하여 터미널 표로 상세 출력하고, 서브플롯 타이틀에 오차 지표를 명기하는 마스터 시각화 스크립트.

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

SCENARIOS_ORDER = [
    ("high_speed_20kph", "1. High Speed 20kph"),
    ("high_speed_60kph", "2. High Speed 60kph"),
    ("high_speed_80kph", "3. High Speed 80kph"),
    ("hills_6deg", "4. Hills 6deg Grade"),
    ("hills_12deg", "5. Hills 12deg Grade"),
    ("hills_18deg", "6. Hills 18deg Grade"),
    ("hills_30deg", "7. Hills 30deg Grade"),
    ("steering_pad_40kph", "8. Steering Pad 40kph (Circle)"),
]

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


def compute_cross_correlation_lag_ms(sig1, sig2, dt_ms=4.0, max_lag_samples=250):
    s1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-8)
    s2 = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-8)
    corr = correlate(s1, s2, mode="full") / len(s1)
    lags = correlation_lags(len(s1), len(s2), mode="full")
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    sub_lags = lags[mask]
    sub_corr = corr[mask]
    peak_idx = np.argmax(np.abs(sub_corr))
    best_lag_ms = sub_lags[peak_idx] * dt_ms
    max_r = sub_corr[peak_idx]
    return best_lag_ms, max_r


def plot_master_grid_with_error_analysis(target_state="normal"):
    fig, axes = plt.subplots(4, 2, figsize=(22, 19), dpi=300)
    axes_flat = axes.flatten()

    state_title = "NORMAL (Baseline Drive)" if target_state == "normal" else "ABNORMAL (Gearbox Fault Drive)"
    print("\n" + "=" * 125)
    print(f" 📊 [{state_title}] 8개 시나리오 CAN vs Audio 시간축 동기화 정밀 오차 분석 (Launch Trim 적용)")
    print("=" * 125)
    print(f" {'시나리오 (Scenario)':<28} | {'CAN 시간':<9} | {'Audio 시간':<10} | {'길이 오차(초)':<13} | {'상호상관 시차(ms)':<16} | {'윈도우 매칭률':<12}")
    print("-" * 125)

    error_summary = []

    for idx, (scen_key, scen_label) in enumerate(SCENARIOS_ORDER):
        ax = axes_flat[idx]
        can_filename = f"{target_state}_{scen_key}_official_decoded.csv"
        can_path = os.path.join(RAW_DECODED_DIR, can_filename)

        if not os.path.exists(can_path):
            continue

        # CAN 로드 및 출발점 트리밍 적용
        df_can_raw = pd.read_csv(can_path)
        full_scen_key = f"{target_state}_{scen_key}"
        t_trim = CAN_LAUNCH_TRIMS.get(full_scen_key, 0.0)
        df_can = df_can_raw[df_can_raw["Time"] >= t_trim].reset_index(drop=True)
        
        t_can = df_can["Time"].values - df_can["Time"].values[0]
        can_total_sec = t_can[-1]
        spd = (df_can["WHL_SPD_FL"].values + df_can["WHL_SPD_FR"].values + df_can["WHL_SPD_RL"].values + df_can["WHL_SPD_RR"].values) / 4.0

        audio_dir = os.path.join(AUDIO_LAUNCH_ALIGNED_DIR, target_state, scen_key)
        gear_wavs = [f for f in os.listdir(audio_dir) if "gear" in f.lower() or "GEAR" in f.upper()]
        if not gear_wavs:
            gear_wavs = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        audio_path = os.path.join(audio_dir, gear_wavs[0])

        sr, audio_raw = wavfile.read(audio_path)
        if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
        audio_norm = audio_raw.astype(np.float32) / 32768.0
        audio_total_sec = len(audio_norm) / float(sr)

        # 4ms 오디오 RMS 계산
        audio_rms = []
        for i in range(len(t_can) - 1):
            s_start = int(round(t_can[i] * sr))
            s_end = int(round(t_can[i + 1] * sr))
            if s_end <= len(audio_norm) and s_start >= 0:
                chunk = audio_norm[s_start:s_end]
                audio_rms.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0)
            else:
                audio_rms.append(0.0)
        audio_rms.append(audio_rms[-1] if audio_rms else 0.0)
        audio_rms_arr = np.array(audio_rms)

        # 오차 분석 1: 전체 길이 오차 (출발점 보정 후)
        dur_err_sec = audio_total_sec - can_total_sec

        # 오차 분석 2: 상호상관 최적 시차 (Lag Tau, ms)
        lag_ms, max_r = compute_cross_correlation_lag_ms(spd, audio_rms_arr, dt_ms=4.0)

        # 오차 분석 3: 250샘플 윈도우 매칭률
        window_size, step_size = 250, 125
        n_windows = (len(df_can) - window_size) // step_size + 1
        matched_w = sum(1 for w in range(n_windows) if int(round((t_can[min(w * step_size + window_size - 1, len(t_can)-1)]) * sr)) <= len(audio_norm))
        match_rate = (matched_w / n_windows) * 100.0 if n_windows > 0 else 0.0
        drop_w = n_windows - matched_w

        error_summary.append({
            "state": target_state,
            "scenario": scen_key,
            "can_sec": round(can_total_sec, 2),
            "audio_sec": round(audio_total_sec, 2),
            "dur_error_sec": round(dur_err_sec, 3),
            "xcorr_lag_ms": round(lag_ms, 1),
            "xcorr_r": round(max_r, 4),
            "can_windows": n_windows,
            "matched_windows": matched_w,
            "dropped_windows": drop_w,
            "match_rate_pct": round(match_rate, 2),
        })

        print(f" {target_state + '_' + scen_key:<28} | {can_total_sec:>7.2f}s | {audio_total_sec:>8.2f}s  | {dur_err_sec:>+10.3f}s    | {lag_ms:>10.1f} ms (R={max_r:.2f}) | {match_rate:>7.2f}% ({matched_w}/{n_windows})")

        # 플로팅
        can_color = "#1565c0" if target_state == "normal" else "#c62828"
        aud_color = "#2e7d32" if target_state == "normal" else "#e65100"

        line1 = ax.plot(t_can, spd, color=can_color, linewidth=1.3, label="CAN: Speed (km/h)")
        ax.set_ylabel("Speed (km/h)", color=can_color, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelcolor=can_color)
        ax.set_xlim([0, can_total_sec])

        ax_tw = ax.twinx()
        line2 = ax_tw.plot(t_can, audio_rms_arr, color=aud_color, linewidth=1.1, alpha=0.8, label="Audio: RMS Energy")
        ax_tw.set_ylabel("Audio RMS", color=aud_color, fontsize=10, fontweight="bold")
        ax_tw.tick_params(axis="y", labelcolor=aud_color)
        ax_tw.grid(False)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", frameon=True, fontsize=8)

        # 타이틀에 오차 지표 명시
        title_text = f"{scen_label}\n[Len Err: {dur_err_sec:+.2f}s | Lag: {lag_ms:+.0f}ms | Match: {match_rate:.1f}%]"
        ax.set_title(title_text, fontsize=10, fontweight="bold", pad=5)

        if idx in [6, 7]:
            ax.set_xlabel("Time (Seconds)", fontsize=11, fontweight="bold")

    print("=" * 125)
    plt.suptitle(f"Multimodal CAN vs Audio Time-Synchronized Master Layout - {state_title} (Launch Trimmed)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, f"multimodal_sync_all_{target_state}_scenarios.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, f"multimodal_sync_all_{target_state}_scenarios.png")
    shutil.copy2(out_png, art_png)

    return error_summary


def main():
    e_norm = plot_master_grid_with_error_analysis("normal")
    e_abn = plot_master_grid_with_error_analysis("abnormal")
    df_all = pd.DataFrame(e_norm + e_abn)
    out_csv = os.path.join(AUDIO_AUDIT_DIR, "all_scenarios_multimodal_synchronization_error_report.csv")
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n 💾 전체 16개 시나리오 동기화 오차 종합 리포트 CSV 저장: {out_csv}")


if __name__ == "__main__":
    main()
