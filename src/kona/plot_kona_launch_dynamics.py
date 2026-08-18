"""
plot_kona_launch_dynamics.py (gearbox_final_suite / data_processing)

KONA EV 221217-2 주행 로그의 170초~178초 가속 구간에 대한
4륜 개별 휠속도, 좌우/전후 속도 편차, 종가속도, Yaw Rate, 슬립률 다중 서브플롯 정밀 시각화 스크립트.

저장 경로:
- /mnt/c/Users/kante/Documents/KWU/etrl_task/experiments/gearbox_final_suite/results/kona_launch_dynamics_analysis.png

규정 준수:
- Rule 3 준수: 코드 주석 100% 한글, Matplotlib 라벨/타이틀 영문(English) 작성.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SUITE_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/experiments/gearbox_final_suite"
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
CSV_PATH = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_decoded_unified_20ms/kona_ev_common_73features_unified_20ms.csv"
OUTPUT_PNG = os.path.join(RESULTS_DIR, "kona_launch_dynamics_analysis.png")


def plot_dynamics():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV 파일이 없습니다: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 221217-2 파일 중 170초~178초 구간 필터링
    sub = df[(df["Source_File"].str.contains("221217-2")) & (df["Time_Rel_Sec"] >= 170.0) & (df["Time_Rel_Sec"] <= 178.0)].copy()

    if len(sub) == 0:
        print("[ERROR] 해당 시간 범위 데이터를 찾을 수 없습니다.")
        return

    time_sec = sub["Time_Rel_Sec"].values
    fl = sub["WHL_SPD_FL"].values
    fr = sub["WHL_SPD_FR"].values
    rl = sub["WHL_SPD_RL"].values
    rr = sub["WHL_SPD_RR"].values
    
    long_acc = sub["LONG_ACCEL"].values if "LONG_ACCEL" in sub.columns else np.zeros_like(time_sec)
    yaw_rate = sub["YAW_RATE"].values if "YAW_RATE" in sub.columns else np.zeros_like(time_sec)

    mean_front = (fl + fr) / 2.0
    mean_rear = (rl + rr) / 2.0
    mean_all = (fl + fr + rl + rr) / 4.0
    
    diff_lr_front = fr - fl
    diff_front_rear = mean_front - mean_rear
    slip_ratio = (mean_front - mean_rear) / (mean_all + 1e-5)

    # 4단 정밀 진단 플롯 생성
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, dpi=150)
    plt.subplots_adjust(hspace=0.25)

    # 1. 4륜 개별 휠속도 비교
    ax1 = axes[0]
    ax1.plot(time_sec, fl, label="Front Left (FL, Drive)", color="#1f77b4", linewidth=2.0)
    ax1.plot(time_sec, fr, label="Front Right (FR, Drive)", color="#d62728", linewidth=2.0, linestyle="--")
    ax1.plot(time_sec, rl, label="Rear Left (RL, Non-Drive)", color="#2ca02c", linewidth=1.5, alpha=0.8)
    ax1.plot(time_sec, rr, label="Rear Right (RR, Non-Drive)", color="#ff7f0e", linewidth=1.5, alpha=0.8)
    ax1.set_ylabel("Wheel Speed (km/h)", fontsize=11, fontweight="bold")
    ax1.set_title("Kona EV Transient Acceleration Dynamics (Time: 170s ~ 178s)", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # 2. 좌우 구동륜 편차 및 전후륜 차이
    ax2 = axes[1]
    ax2.plot(time_sec, diff_lr_front, label="Front L-R Speed Diff (FR - FL)", color="#9467bd", linewidth=2.0)
    ax2.plot(time_sec, diff_front_rear, label="Front-Rear Speed Diff (Drive - NonDrive)", color="#8c564b", linewidth=2.0, linestyle=":")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("Speed Diff (km/h)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)

    # 3. 종가속도 및 Yaw Rate 거동
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    l1 = ax3.plot(time_sec, long_acc, label="Longitudinal Accel (m/s^2)", color="#008080", linewidth=2.0)
    l2 = ax3_twin.plot(time_sec, yaw_rate, label="Yaw Rate (deg/s)", color="#e377c2", linewidth=1.8, linestyle="--")
    ax3.set_ylabel("Long Accel (m/s^2)", color="#008080", fontsize=11, fontweight="bold")
    ax3_twin.set_ylabel("Yaw Rate (deg/s)", color="#e377c2", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper left", frameon=True, fontsize=10)

    # 4. 슬립률 (Slip Ratio) 및 고장 임계치
    ax4 = axes[3]
    ax4.plot(time_sec, slip_ratio, label="Front-Rear Slip Ratio", color="#ff1493", linewidth=2.0)
    ax4.axhline(0.25, color="red", linestyle="--", linewidth=1.5, label="Fixed Anomaly Threshold (0.25)")
    ax4.axvspan(174.1, 175.2, color="yellow", alpha=0.25, label="Transient Slip Shock Window")
    ax4.set_xlabel("Relative Time (Seconds)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Slip Ratio (Fraction)", fontsize=11, fontweight="bold")
    ax4.set_ylim(-0.2, 0.8)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left", frameon=True, fontsize=10)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 100)
    print(f" 🎉 [동역학 정밀 진단 그래프 생성 완료] -> {OUTPUT_PNG}")
    print("=" * 100)


if __name__ == "__main__":
    plot_dynamics()
