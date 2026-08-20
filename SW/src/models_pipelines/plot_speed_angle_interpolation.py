"""
plot_speed_angle_interpolation.py (gearbox_final_suite / src / models_pipelines)

CAN 신호의 차속(20, 60, 80 kph) 및 조향각/경사도 간의 상관관계와
절대 물리량 vs 무차원 보간 피처(Dimensionless Normalized Features)의 분포를 비교 시각화하는 스크립트.

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib 그래프 텍스트 100% 영문(English).
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

# 스타일 설정
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def generate_speed_angle_interpolation_plots():
    csv_path = os.path.join(RESULTS_DIR, "unified_can_context_dataset_w250.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(DATA_DIR, "unified_can_context_dataset_w250.csv")

    df = pd.read_csv(csv_path)

    # 1. 무차원 상대 변동률(Dimensionless Fluctuations) 계산
    base_spd = np.maximum(df["mean"].values, 1.0)
    df["dimless_std"] = df["std"] / base_spd
    df["dimless_diff_fr_rr_std"] = df["diff_fr_rr_std"] / base_spd
    df["dimless_diff_fl_rl_mean"] = df["diff_fl_rl_mean"] / base_spd

    # 4분할 종합 시각화 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # -------------------------------------------------------------
    # [Subplot 1] 절대 휠속도 분산(Raw std, km/h) vs 차속
    # -------------------------------------------------------------
    sns.boxplot(
        data=df, x="speed_kph", y="std", hue="state",
        palette={"normal": "#2b5c8f", "abnormal": "#d95f02"},
        ax=axes[0, 0], showfliers=False
    )
    axes[0, 0].set_title("1. Raw Wheel Speed Fluctuation (Absolute std in km/h)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 0].set_xlabel("Vehicle Speed (km/h)", fontsize=11)
    axes[0, 0].set_ylabel("Wheel Speed Standard Deviation (km/h)", fontsize=11)
    axes[0, 0].legend(title="State", loc="upper left")

    # -------------------------------------------------------------
    # [Subplot 2] 무차원화 휠속도 변동률 (Dimensionless std / Speed)
    # -------------------------------------------------------------
    sns.boxplot(
        data=df, x="speed_kph", y="dimless_std", hue="state",
        palette={"normal": "#2b5c8f", "abnormal": "#d95f02"},
        ax=axes[0, 1], showfliers=False
    )
    axes[0, 1].set_title("2. Dimensionless Speed Fluctuation Ratio (std / Speed)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 1].set_xlabel("Vehicle Speed (km/h)", fontsize=11)
    axes[0, 1].set_ylabel("Dimensionless Fluctuation Ratio", fontsize=11)
    axes[0, 1].legend(title="State", loc="upper left")

    # -------------------------------------------------------------
    # [Subplot 3] 조향각(SAS Angle) vs 요레이트(Yaw Rate) 동역학 상관 곡선
    # -------------------------------------------------------------
    sns.scatterplot(
        data=df, x="sas_angle_mean", y="yaw_rate_mean", hue="state", style="speed_kph",
        palette={"normal": "#2b5c8f", "abnormal": "#d95f02"},
        alpha=0.7, s=45, ax=axes[1, 0]
    )
    axes[1, 0].set_title("3. Kinematic Correlation: Steering Angle vs Yaw Rate", fontsize=13, fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("Mean Steering Angle (deg)", fontsize=11)
    axes[1, 0].set_ylabel("Mean Yaw Rate (deg/s)", fontsize=11)

    # -------------------------------------------------------------
    # [Subplot 4] 4륜 구동 속도차 무차원 변동률 분포 (Front vs Rear Diff)
    # -------------------------------------------------------------
    sns.scatterplot(
        data=df, x="mean", y="dimless_diff_fr_rr_std", hue="state",
        palette={"normal": "#2b5c8f", "abnormal": "#d95f02"},
        alpha=0.6, s=40, ax=axes[1, 1]
    )
    axes[1, 1].set_title("4. Dimensionless Front-Rear Wheel Speed Delta vs Speed", fontsize=13, fontweight="bold", pad=10)
    axes[1, 1].set_xlabel("Mean Vehicle Speed (km/h)", fontsize=11)
    axes[1, 1].set_ylabel("Dimensionless FR-RR Diff Std", fontsize=11)

    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "speed_angle_dimensionless_interpolation.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Artifact 경로로 복사
    art_png = os.path.join(ARTIFACT_FIG_DIR, "speed_angle_dimensionless_interpolation.png")
    shutil.copy2(out_png, art_png)

    print(f"[SUCCESS] 시각화 플롯 저장 완료: {out_png}")
    print(f"[SUCCESS] Artifact 복사 완료: {art_png}")


if __name__ == "__main__":
    generate_speed_angle_interpolation_plots()
