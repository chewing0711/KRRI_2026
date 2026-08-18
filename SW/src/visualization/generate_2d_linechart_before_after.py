"""
generate_2d_linechart_before_after.py (gearbox_final_suite)

Full-Dataset Sweep 2D Line Chart Comparison:
1. BEFORE: Sweeps the dataset to locate scenario/speed boundary shifts.
2. AFTER: Sweeps the dataset to locate a single-session segment where
   nunique(speed_kph) == 1, nunique(scenario) == 1, and nunique(state) == 1 across 10 steps.

Rule 3 Compliance:
- Code comments & terminal output in Korean.
- Matplotlib labels, titles, legends, axes in English.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
INPUT_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")

if not os.path.exists(INPUT_CSV):
    INPUT_CSV = os.path.join(os.path.dirname(SUITE_DIR), "gearbox_fault_diagnosis_fixed", "results", "unified_can_context_dataset.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_2d_linecharts_full_sweep():
    print("=" * 80)
    print(" 🎨 FULL-DATASET SWEEP 2D TIME-SERIES LINE CHARTS (BEFORE vs AFTER)")
    print("=" * 80)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] CSV missing: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    # 1. Full Dataset Sweep for BEFORE (Boundary Shift)
    dirty_indices = []
    for i in range(len(df) - 10):
        sub_scen = df.iloc[i : i + 10]["scenario"].nunique()
        sub_spd = df.iloc[i : i + 10]["speed_kph"].nunique()
        if sub_scen > 1 or sub_spd > 1:
            dirty_indices.append(i)

    print(f" [FULL DATASET SWEEP AUDIT]")
    print(f"  - Total Dataset Rows: {len(df)}")
    print(f"  - Total Boundary-Crossing Windows Found: {len(dirty_indices)}")

    if len(dirty_indices) > 0:
        dirty_idx = dirty_indices[0]
    else:
        dirty_idx = 0

    df_dirty = df.iloc[dirty_idx : dirty_idx + 10].reset_index(drop=True)

    # 2. Full Dataset Sweep for AFTER (Single-Session Segment)
    clean_idx = None
    for i in range(len(df) - 10):
        sub = df.iloc[i : i + 10]
        if sub["speed_kph"].nunique() == 1 and sub["scenario"].nunique() == 1 and sub["state"].nunique() == 1:
            clean_idx = i
            break

    if clean_idx is None:
        clean_idx = 0

    df_clean = df.iloc[clean_idx : clean_idx + 10].reset_index(drop=True)
    clean_spd_val = df_clean["speed_kph"].iloc[0]

    print(f"  - Dirty BEFORE Speed Values (10 steps) : {df_dirty['speed_kph'].tolist()}")
    print(f"  - Clean AFTER Speed Values (10 steps)  : {df_clean['speed_kph'].tolist()}")
    print(f"  - Clean AFTER Scenario                 : {df_clean['scenario'].iloc[0]}")
    print(f"  - Clean AFTER State                    : {df_clean['state'].iloc[0]}")

    t_axis = np.arange(10)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=160)

    # --- Subplot 1: BEFORE Vehicle Speed ---
    axes[0, 0].plot(t_axis, df_dirty["speed_kph"], "r-o", linewidth=3.0, markersize=8, label="speed_kph (Boundary Drop)")
    change_steps = np.where(df_dirty["speed_kph"].values != df_dirty["speed_kph"].iloc[0])[0]
    if len(change_steps) > 0:
        c_step = change_steps[0] - 0.5
        axes[0, 0].axvline(x=c_step, color="black", linestyle="--", linewidth=2.0, alpha=0.8, label="Boundary Shift")

    axes[0, 0].set_ylabel("Speed (km/h)", fontsize=10, fontweight="bold")
    axes[0, 0].set_title(f"BEFORE: Vehicle Speed Shift ({df_dirty['speed_kph'].iloc[0]:.0f} -> {df_dirty['speed_kph'].iloc[-1]:.0f} km/h)", fontsize=11, fontweight="bold", color="red")
    axes[0, 0].set_ylim(-5, max(df_dirty["speed_kph"]) + 15)
    axes[0, 0].legend(fontsize=9, loc="upper right")
    axes[0, 0].grid(True, linestyle="--", alpha=0.3)

    # --- Subplot 2: AFTER Vehicle Speed ---
    axes[0, 1].plot(t_axis, df_clean["speed_kph"], "g-s", linewidth=3.5, markersize=10, label=f"speed_kph = {clean_spd_val:.1f} km/h")
    axes[0, 1].set_ylabel("Speed (km/h)", fontsize=10, fontweight="bold")
    axes[0, 1].set_title(f"AFTER: Vehicle Speed ({clean_spd_val:.1f} km/h Segment)", fontsize=11, fontweight="bold", color="green")
    axes[0, 1].set_ylim(clean_spd_val - 10, clean_spd_val + 10)
    axes[0, 1].legend(fontsize=10, loc="lower right")
    axes[0, 1].grid(True, linestyle="--", alpha=0.3)

    # --- Subplot 3: BEFORE Physical Slip Ratios ---
    axes[1, 0].plot(t_axis, df_dirty["slip_ratio_fl"], "r-o", linewidth=2.0, label="slip_ratio_fl")
    axes[1, 0].plot(t_axis, df_dirty["slip_ratio_fr"], "m-^", linewidth=2.0, label="slip_ratio_fr")
    axes[1, 0].plot(t_axis, df_dirty["slip_diff_front_rear"], "b-v", linewidth=2.0, label="slip_diff_front_rear")
    if len(change_steps) > 0:
        axes[1, 0].axvline(x=change_steps[0] - 0.5, color="black", linestyle="--", linewidth=2.0, alpha=0.8, label="Boundary Shift")
    axes[1, 0].set_xlabel("Time Steps (t=0 to t=9)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Slip Ratio Magnitude", fontsize=10, fontweight="bold")
    axes[1, 0].set_title("BEFORE: Slip Signals Shift at Boundary", fontsize=11, fontweight="bold", color="red")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    # --- Subplot 4: AFTER Physical Slip Ratios ---
    axes[1, 1].plot(t_axis, df_clean["slip_ratio_fl"], "g-s", linewidth=2.0, label="slip_ratio_fl")
    axes[1, 1].plot(t_axis, df_clean["slip_ratio_fr"], "c-d", linewidth=2.0, label="slip_ratio_fr")
    axes[1, 1].plot(t_axis, df_clean["slip_diff_front_rear"], "b-x", linewidth=2.0, label="slip_diff_front_rear")
    axes[1, 1].set_xlabel("Time Steps (t=0 to t=9)", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Slip Ratio Magnitude", fontsize=10, fontweight="bold")
    axes[1, 1].set_title("AFTER: Single-Session Physical Slip Waveforms", fontsize=11, fontweight="bold", color="green")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("2D Time-Series Line Chart Comparison: BEFORE (Boundary Shift) vs AFTER (Single Session)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "matrix_2d_linechart_before_after.png")
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[SAVED] Line Chart Comparison PNG saved: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_2d_linecharts_full_sweep()
