"""
generate_before_after_3d_comparison.py (gearbox_final_suite)

Generates a 2-subplot side-by-side 3D Surface comparison PNG:
- Subplot A (Left): BEFORE - Boundary-crossing window (80kph -> 60kph cliff at t=6)
- Subplot B (Right): AFTER - Clean single-session window (Continuous 80kph flow)

Rule 3 Compliance:
- Code comments & terminal output in Korean.
- Matplotlib labels, titles, legends, axes in English.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
INPUT_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")

if not os.path.exists(INPUT_CSV):
    INPUT_CSV = os.path.join(os.path.dirname(SUITE_DIR), "gearbox_fault_diagnosis_fixed", "results", "unified_can_context_dataset.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_before_after_comparison():
    print("=" * 80)
    print(" 🎨 GENERATING BEFORE vs AFTER 3D SURFACE COMPARISON PNG")
    print("=" * 80)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] CSV missing: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    metadata_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X = df[feature_cols].to_numpy()

    # 1. Extract Dirty BEFORE Window (crossing 80kph to 60kph boundary)
    dirty_idx = None
    for i in range(len(df) - 10):
        if df.iloc[i]["speed_kph"] == 80.0 and df.iloc[i + 9]["speed_kph"] == 60.0:
            dirty_idx = i
            break

    if dirty_idx is None:
        dirty_idx = 100

    mat_dirty = X[dirty_idx : dirty_idx + 10].T # (45, 10)

    # 2. Extract Clean AFTER Window (strictly 80kph continuous single session)
    clean_idx = None
    for i in range(len(df) - 10):
        if (df.iloc[i:i+10]["speed_kph"] == 80.0).all() and (df.iloc[i:i+10]["state"] == df.iloc[i]["state"]).all():
            clean_idx = i
            break

    if clean_idx is None:
        clean_idx = 0

    mat_clean = X[clean_idx : clean_idx + 10].T # (45, 10)

    scaler_d = StandardScaler()
    Z_dirty = scaler_d.fit_transform(mat_dirty.T).T

    scaler_c = StandardScaler()
    Z_clean = scaler_c.fit_transform(mat_clean.T).T

    x_steps = np.arange(10)
    y_feats = np.arange(45)
    X_mesh, Y_mesh = np.meshgrid(x_steps, y_feats)

    # Plot Side-by-Side 3D Comparison
    fig = plt.figure(figsize=(16, 7), dpi=160)

    # Subplot 1: BEFORE (Dirty Boundary Crossing)
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(X_mesh, Y_mesh, Z_dirty, cmap="Reds", edgecolor="none", alpha=0.85)
    ax1.set_xlabel("Time Steps (t=0 to t=9)", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Feature Channels (0 to 44)", fontsize=9, fontweight="bold")
    ax1.set_zlabel("Normalized Value", fontsize=9, fontweight="bold")
    ax1.set_title("BEFORE: Boundary-Crossing Cliff (80kph -> 60kph Step)", fontsize=11, fontweight="bold", color="red")
    ax1.view_init(elev=30, azim=-55)

    # Subplot 2: AFTER (Clean Single Session)
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(X_mesh, Y_mesh, Z_clean, cmap="Greens", edgecolor="none", alpha=0.85)
    ax2.set_xlabel("Time Steps (t=0 to t=9)", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Feature Channels (0 to 44)", fontsize=9, fontweight="bold")
    ax2.set_zlabel("Normalized Value", fontsize=9, fontweight="bold")
    ax2.set_title("AFTER: Clean Single-Session (Continuous 80kph Flow)", fontsize=11, fontweight="bold", color="green")
    ax2.view_init(elev=30, azim=-55)

    plt.suptitle("3D Surface Matrix Input Comparison: BEFORE (Boundary Cliff) vs AFTER (Clean Flow)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "matrix_3d_before_after_comparison.png")
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[SAVED] Before vs After Comparison PNG saved: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_before_after_comparison()
