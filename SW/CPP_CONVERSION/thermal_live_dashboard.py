"""
thermal_live_dashboard.py (gearbox_final_suite / CPP_CONVERSION)

라즈베리 파이 2시간 내구 시험 실시간 Thermal Heatmap 및 모니터링 UI 대시보드

기능:
1. 2x2 CPU 코어 로드/써멀 실시간 히트맵 (Core 0 ~ Core 3)
2. 2시간 시계열 롤링 그래프: CPU 온도 (°C), 동작 주파수 (GHz), 총 CPU 점유율 (%)
3. 실시간 고장 진단 상태 및 연산 지연시간(Latency ms) 게이지
4. 서멀 쓰로틀링(80°C) 임계선 및 페일세이프 경고 시각화

규정 준수:
- Rule 3: 그래프 라벨, 타이틀, 범례 영문(English) 100% 표기
- Matplotlib Live Animation 기반
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(CURRENT_DIR)
METRICS_DIR = os.path.join(SUITE_DIR, "results", "metrics")
CSV_PATH = os.path.join(METRICS_DIR, "thermal_profile_2hours.csv")
FIGURES_DIR = os.path.join(SUITE_DIR, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def update_dashboard(frame, axes, csv_path):
    if not os.path.exists(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return

    if len(df) == 0:
        return

    # 최근 120개 샘플 (최근 2분 롤링)
    df_recent = df.tail(120)

    # 4개 서브플롯 초기화
    for ax in axes.flat:
        ax.clear()

    # -------------------------------------------------------------------------
    # Panel 1: 4-Core CPU Load & Thermal Heatmap (2x2 Matrix)
    # -------------------------------------------------------------------------
    last_row = df.iloc[-1]
    core_loads = np.array([
        [last_row.get("core0_pct", 0.0), last_row.get("core1_pct", 0.0)],
        [last_row.get("core2_pct", 0.0), last_row.get("core3_pct", 0.0)]
    ])

    im = axes[0, 0].imshow(core_loads, cmap="YlOrRd", vmin=0, vmax=100)
    axes[0, 0].set_title(f"4-Core CPU Usage Heatmap (Avg Temp: {last_row.get('cpu_temp_degc', 45):.1f}C)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_xticklabels(["Core 0 / 2", "Core 1 / 3"])
    axes[0, 0].set_yticklabels(["Row 0", "Row 1"])

    for i in range(2):
        for j in range(2):
            core_id = i * 2 + j
            val = core_loads[i, j]
            axes[0, 0].text(j, i, f"Core {core_id}\n{val:.1f}%", ha="center", va="center", color="black" if val < 50 else "white", fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 2: CPU Temperature Time-Series (°C)
    # -------------------------------------------------------------------------
    times = df_recent["elapsed_sec"]
    temps = df_recent["cpu_temp_degc"]

    axes[0, 1].plot(times, temps, color="#D9534F", linewidth=2, label="CPU Core Temp (C)")
    axes[0, 1].axhline(y=80.0, color="red", linestyle="--", alpha=0.7, label="Throttling Limit (80C)")
    axes[0, 1].axhline(y=50.0, color="green", linestyle=":", alpha=0.7, label="Target Safe (50C)")
    axes[0, 1].set_title("Real-Time CPU Temperature Profile", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Elapsed Time (sec)")
    axes[0, 1].set_ylabel("Temperature (C)")
    axes[0, 1].set_ylim(30, 90)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="upper left", fontsize=8)

    # -------------------------------------------------------------------------
    # Panel 3: CPU Frequency (GHz) & Total CPU Usage (%)
    # -------------------------------------------------------------------------
    freqs = df_recent["cpu_freq_ghz"]
    usage = df_recent["total_cpu_pct"]

    axes[1, 0].plot(times, freqs, color="#0275D8", linewidth=2, label="CPU Clock (GHz)")
    axes[1, 0].set_title("CPU Clock Frequency & Dynamic DVFS State", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Elapsed Time (sec)")
    axes[1, 0].set_ylabel("Clock Frequency (GHz)", color="#0275D8")
    axes[1, 0].set_ylim(0.4, 2.0)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    ax3_twin = axes[1, 0].twinx()
    ax3_twin.plot(times, usage, color="#5CB85C", linewidth=1.5, linestyle="--", label="Total CPU Usage (%)")
    ax3_twin.set_ylabel("CPU Usage (%)", color="#5CB85C")
    ax3_twin.set_ylim(0, 100)

    # -------------------------------------------------------------------------
    # Panel 4: Real-Time Inference Latency & Diagnosis Status Badge
    # -------------------------------------------------------------------------
    latencies = df_recent["latency_ms"]
    axes[1, 1].plot(times, latencies, color="#6F42C1", linewidth=1.8, label="Latency (ms)")
    axes[1, 1].axhline(y=740.0, color="red", linestyle="--", alpha=0.5, label="Window Deadline (740ms)")
    axes[1, 1].set_title("Inference Latency & Health Status", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Elapsed Time (sec)")
    axes[1, 1].set_ylabel("Latency (ms)")
    axes[1, 1].set_ylim(0, max(5.0, np.max(latencies) * 1.5))
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper left", fontsize=8)

    # Status Banner
    cur_state = str(last_row.get("pred_state", "NORMAL"))
    cur_lat = float(last_row.get("latency_ms", 0.8))
    status_color = "#5CB85C" if cur_state == "NORMAL" else "#D9534F"

    axes[1, 1].text(0.95, 0.20, f"Diagnosis: {cur_state}\nLatency: {cur_lat:.3f} ms",
                    transform=axes[1, 1].transAxes, ha="right", va="center",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.8),
                    color="white", fontweight="bold", fontsize=10)


def start_live_dashboard(refresh_ms=1000):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Raspberry Pi 2-Hour Thermal Profiling & Real-Time Diagnosis Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    ani = animation.FuncAnimation(fig, update_dashboard, fargs=(axes, CSV_PATH), interval=refresh_ms)

    out_snapshot = os.path.join(FIGURES_DIR, "thermal_live_dashboard_snapshot.png")
    plt.savefig(out_snapshot, dpi=150)
    print(f" [대시보드 시작] 실시간 모니터링 활성화 (스냅샷 저장: {out_snapshot})")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실시간 Thermal Heatmap 및 모니터링 UI 대시보드")
    parser.add_argument("-r", "--refresh", type=int, default=1000, help="UI 갱신 주기 (ms, 기본값: 1000)")
    args = parser.parse_args()

    start_live_dashboard(refresh_ms=args.refresh)
