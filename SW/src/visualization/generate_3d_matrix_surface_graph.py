"""
generate_3d_matrix_surface_graph.py (gearbox_final_suite)

Generates a publication-quality 3D Surface Mesh plot of the (45 x 10) Feature Matrix
from clean single-session sample_matrix_45x10_t10.csv:
- X-axis: Time Steps (t=0 to t=9)
- Y-axis: Feature Channels (0 to 44)
- Z-axis: Normalized Signal Value

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
MATRIX_DIR = os.path.join(RESULTS_DIR, "extracted_matrices")
SAMPLE_CSV = os.path.join(MATRIX_DIR, "sample_matrix_45x10_t10.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_3d_surface_graph_clean():
    print("=" * 80)
    print(" 🎨 GENERATING CLEAN 3D SURFACE MESH PLOT OF (45 x 10) FEATURE MATRIX")
    print("=" * 80)

    if not os.path.exists(SAMPLE_CSV):
        print(f"[ERROR] Sample matrix CSV missing: {SAMPLE_CSV}")
        return

    df_mat = pd.read_csv(SAMPLE_CSV, index_col=0)
    
    # Values matrix: (45 features, 10 timesteps)
    Z_raw = df_mat.to_numpy() # (45, 10)
    
    # Scale per feature for 3D visualization clarity
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z_raw.T).T # Normalize across timesteps

    num_features, num_timesteps = Z_scaled.shape

    # Construct Meshgrid
    x_steps = np.arange(num_timesteps) # 0 to 9
    y_feats = np.arange(num_features)  # 0 to 44
    X_mesh, Y_mesh = np.meshgrid(x_steps, y_feats)

    # 3D Surface Plot
    fig = plt.figure(figsize=(14, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X_mesh, Y_mesh, Z_scaled, cmap="viridis", edgecolor="none", alpha=0.85)

    ax.set_xlabel("Time Steps (t=0 to t=9)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("Feature Channels (0 to 44)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_zlabel("Normalized Value", fontsize=10, fontweight="bold", labelpad=8)

    ax.set_title("3D Surface Visualization of Clean (45 x 10) Time-Series Matrix Input", fontsize=13, fontweight="bold", pad=15)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.08)
    cbar.set_label("Normalized Amplitude", fontsize=9)

    ax.view_init(elev=30, azim=-55)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "matrix_3d_surface_visualization.png")
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[SAVED] Clean 3D Surface Plot exported successfully: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_3d_surface_graph_clean()
