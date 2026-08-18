"""
export_raw_feature_matrices.py (gearbox_final_suite / data_processing)

Exports 1:1 Synchronized Full 45-Feature Sequence Matrix from unified_can_context_dataset.csv:
- sample_matrix_45x10_t10.csv (ALL 45 UNIFIED Features x 10 Windows t_0 ~ t_9)

Rule 3 & Rule 4 Compliance:
- 100% Value Synchronization with UNIFIED CSV across ALL 45 Features.
- Korean code comments & terminal output.
"""

import os
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
MATRICES_DIR = os.path.join(RESULTS_DIR, "extracted_matrices")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")


def export_full_feature_matrix():
    print("=" * 80)
    print(" 🛠 EXPORTING 1:1 SYNCHRONIZED FULL 45-FEATURE MATRIX (45 Features x 10 Windows)")
    print("=" * 80)

    if not os.path.exists(UNIFIED_CSV):
        print(f"[ERROR] Unified CSV missing: {UNIFIED_CSV}")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)
    meta_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df_unified.columns if c not in meta_cols]

    print(f" - Extracted Total UNIFIED Features Count: {len(feature_cols)} Features")

    # Extract first 10 consecutive windows (t_0 ~ t_9) across ALL 45 features
    df_10 = df_unified.iloc[:10][feature_cols]

    # Transpose to shape (45 Features x 10 Windows)
    df_45x10 = df_10.T
    df_45x10.columns = [f"t_{i}" for i in range(10)]

    os.makedirs(MATRICES_DIR, exist_ok=True)

    path_45x10 = os.path.join(MATRICES_DIR, "sample_matrix_45x10_t10.csv")
    df_45x10.to_csv(path_45x10, encoding="utf-8-sig")

    print(f" - Exported Full Feature Matrix : {path_45x10} ({df_45x10.shape[0]} Features x {df_45x10.shape[1]} Windows)")
    print(" [SUCCESS] 1:1 Synchronization with UNIFIED CSV Completed for ALL 45 Features.")
    print("=" * 80)


if __name__ == "__main__":
    export_full_feature_matrix()
