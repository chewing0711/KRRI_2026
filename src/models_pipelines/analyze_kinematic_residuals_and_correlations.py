"""
analyze_kinematic_residuals_and_correlations.py (gearbox_final_suite / src / models_pipelines)

차량 2-트랙 아커만 조향 기하학(Ackermann Kinematic Bicycle Model)을 기반으로
1. 이론적 좌우 휠속차(d * omega_z)와 실측 휠속차 간의 동역학 잔차(Kinematic Residual) 계산
2. 원선회로(Steering pad) 및 고속 주행 시 정상 vs 고장 분리 변별력 정량 분석
3. 종방향 가속/페달 대비 휠속 변동률 상관관계 분석 및 시각화

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib 라벨 100% 영문(English).
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DATA_PROCESSING_DIR) if "models_pipelines" in DATA_PROCESSING_DIR else DATA_PROCESSING_DIR
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
ARTIFACT_FIG_DIR = "/home/kante/.gemini/antigravity-ide/brain/6e341025-6278-4988-86a0-4dfdf4574e3b/figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_FIG_DIR, exist_ok=True)

# 코나 EV 제원: 윤거 (Track Width) = 1.565 m
TRACK_WIDTH_M = 1.565

# 스타일 설정
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def run_kinematic_residual_analysis():
    csv_path = os.path.join(RESULTS_DIR, "unified_can_context_dataset_w250.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(DATA_DIR, "unified_can_context_dataset_w250.csv")

    df = pd.read_csv(csv_path)

    # 1. 아커만 조향 기하학 이론 휠속도차 계산: Delta_v_kinematic = d * omega_z (km/h 단위 변환)
    # yaw_rate (deg/s) -> rad/s 변환 -> * TRACK_WIDTH_M (m/s) -> * 3.6 (km/h)
    yaw_rad_s = df["yaw_rate_mean"].values * (np.pi / 180.0)
    theo_diff_lr_kph = TRACK_WIDTH_M * yaw_rad_s * 3.6

    # 2. 실측 휠속차 (전륜 좌우 및 후륜 좌우)
    meas_diff_fl_fr = df["diff_fl_fr_std"].values  # 좌우 변동
    meas_diff_rl_rr = df["diff_rl_rr_std"].values

    # 3. 조향 기하학 잔차(Kinematic Residual) 계산
    # 잔차 = |실측 좌우 분산 - 기하학적 선회 회전차|
    df["theo_diff_lr_kph"] = np.abs(theo_diff_lr_kph)
    df["kinematic_residual_fl_fr"] = np.abs(df["diff_fl_fr_std"] - (df["theo_diff_lr_kph"] * 0.1))
    df["kinematic_residual_rl_rr"] = np.abs(df["diff_rl_rr_std"] - (df["theo_diff_lr_kph"] * 0.1))

    # 4. 페달 개도량 대비 휠속도 분산 비율 (Drive Torque Response Ratio)
    pedal_val = np.maximum(df["accel_pedal_mean"].values, 1.0)
    df["pedal_slip_response_ratio"] = df["diff_fr_rr_std"] / pedal_val

    # =========================================================================
    # [정량적 통계 분석 출력]
    # =========================================================================
    print("\n" + "=" * 105)
    print(" 🚗 [동역학 아커만 잔차 및 페달 응답비 정량 분석 결과]")
    print("=" * 105)

    scenarios = df["scenario"].unique()
    print(f" {'시나리오 (Scenario)':<28} | {'정상 잔차(Normal)':<20} | {'고장 잔차(Abnormal)':<20} | {'고장/정상 잔차 배율':<16}")
    print("-" * 105)

    for scen in scenarios:
        sub_n = df[(df["scenario"] == scen) & (df["state"] == "normal")]
        sub_a = df[(df["scenario"] == scen) & (df["state"] == "abnormal")]

        res_n = np.mean(sub_n["kinematic_residual_rl_rr"]) if len(sub_n) > 0 else 0.0
        res_a = np.mean(sub_a["kinematic_residual_rl_rr"]) if len(sub_a) > 0 else 0.0
        ratio = (res_a / res_n) if res_n > 0 else 0.0

        print(f" {scen:<28} | {res_n:>14.4f}        | {res_a:>14.4f}        | {ratio:>10.2f} 배")

    # =========================================================================
    # [4분할 고해상도 시각화 생성]
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # Subplot 1: 원선회로(Steering Pad)에서의 이론 휠속차 vs 실측 좌우 분산
    pad_df = df[df["scenario"].str.contains("steering_pad", na=False)]
    if len(pad_df) > 0:
        sns.boxplot(
            data=pad_df, x="state", y="kinematic_residual_rl_rr",
            palette={"normal": "#2b5c8f", "abnormal": "#d95f02"},
            ax=axes[0, 0]
        )
        axes[0, 0].set_title("1. Steering Pad (40kph): Kinematic Residual (RL-RR)", fontsize=13, fontweight="bold", pad=10)
        axes[0, 0].set_xlabel("State", fontsize=11)
        axes[0, 0].set_ylabel("Kinematic Residual Value", fontsize=11)
    else:
        sns.boxplot(
            data=df, x="speed_kph", y="kinematic_residual_rl_rr", hue="state",
            palette={"normal": "#2b5c8f", "abnormal": "#d95f02"},
            ax=axes[0, 0]
        )
        axes[0, 0].set_title("1. Kinematic Residual (RL-RR) across Speed", fontsize=13, fontweight="bold", pad=10)

    # Subplot 2: 조향각 대비 이론적 선회 속도차 상관 곡선
    sns.scatterplot(
        data=df, x="sas_angle_mean", y="theo_diff_lr_kph", hue="state",
        palette={"normal": "#2b5c8f", "abnormal": "#d95f02"},
        alpha=0.7, s=40, ax=axes[0, 1]
    )
    axes[0, 1].set_title("2. Steering Angle vs Theoretical Kinematic Delta (d * omega)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 1].set_xlabel("Mean Steering Angle (deg)", fontsize=11)
    axes[0, 1].set_ylabel("Theoretical Kinematic Diff (km/h)", fontsize=11)

    # Subplot 3: 전 차속 대역에서의 아커만 잔차 분포
    sns.boxplot(
        data=df, x="speed_kph", y="kinematic_residual_rl_rr", hue="state",
        palette={"normal": "#2b5c8f", "abnormal": "#d95f02"},
        ax=axes[1, 0], showfliers=False
    )
    axes[1, 0].set_title("3. Kinematic Residual Distribution across All Speeds", fontsize=13, fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("Vehicle Speed (km/h)", fontsize=11)
    axes[1, 0].set_ylabel("Kinematic Residual (RL-RR)", fontsize=11)
    axes[1, 0].legend(title="State", loc="upper left")

    # Subplot 4: 가속 페달 개도량 대비 구동륜 슬립 응답비
    sns.scatterplot(
        data=df, x="accel_pedal_mean", y="pedal_slip_response_ratio", hue="state",
        palette={"normal": "#2b5c8f", "abnormal": "#d95f02"},
        alpha=0.7, s=40, ax=axes[1, 1]
    )
    axes[1, 1].set_title("4. Pedal Input vs Drive Wheel Slip Response Ratio", fontsize=13, fontweight="bold", pad=10)
    axes[1, 1].set_xlabel("Mean Accel Pedal Position (%)", fontsize=11)
    axes[1, 1].set_ylabel("Slip Variance per Pedal Ratio (diff_std / Pedal)", fontsize=11)

    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "kinematic_residual_and_correlation_analysis.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    art_png = os.path.join(ARTIFACT_FIG_DIR, "kinematic_residual_and_correlation_analysis.png")
    shutil.copy2(out_png, art_png)

    print(f"\n [SUCCESS] 동역학 잔차 시각화 플롯 저장: {out_png}")
    print(f" [SUCCESS] Artifact 동기화 완료: {art_png}")
    print("=" * 105)


if __name__ == "__main__":
    run_kinematic_residual_analysis()
