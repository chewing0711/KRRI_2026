"""
export_all_sliding_sequence_tensors.py (gearbox_final_suite / data_processing)

Generates 3D Sliding Sequence Tensors & Iteration Matrices for ALL Streaming Steps:
1. Iteration Sequence Tensor Shape: (Total_Iter, 45_Features, Seq_Len=5)
2. Exports all_sliding_sequence_tensors_45x5.npy (Numpy 3D Tensor)
3. Exports sample iteration CSVs (e.g., iter_0_matrix_45x5.csv, iter_1_matrix_45x5.csv, ...)

Rule 3 Compliance:
- Code comments & terminal output in Korean.
"""

import os
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
MATRICES_DIR = os.path.join(RESULTS_DIR, "extracted_matrices")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")


def export_all_sliding_iteration_matrices(seq_len=5):
    print("=" * 80)
    print(f" 🛠 EXPORTING SLIDING SEQUENCE TENSORS FOR ALL ITERATIONS (Seq_Len={seq_len})")
    print("=" * 80)

    if not os.path.exists(UNIFIED_CSV):
        print(f"[ERROR] Unified CSV missing: {UNIFIED_CSV}")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)
    meta_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df_unified.columns if c not in meta_cols]

    X_all = df_unified[feature_cols].values # (452, 45)
    n_rows, n_feats = X_all.shape

    sliding_matrices = []
    
    # Session boundary aware sliding window matrix extraction
    for sess_id, df_sess in df_unified.groupby("session_id", sort=False):
        X_sess = df_sess[feature_cols].values
        n_sess = len(df_sess)
        if n_sess < seq_len:
            continue

        for i in range(n_sess - seq_len + 1):
            # Extract 5 consecutive windows & transpose to (45 Features x 5 Timesteps)
            mat_45x5 = X_sess[i : i + seq_len].T
            sliding_matrices.append(mat_45x5)

    sliding_tensor_3d = np.array(sliding_matrices) # Shape: (Total_Iter, 45_Features, 5_Timesteps)

    os.makedirs(MATRICES_DIR, exist_ok=True)

    # 1. Save Full 3D Numpy Array
    npy_path = os.path.join(MATRICES_DIR, f"all_sliding_sequence_tensors_45x{seq_len}.npy")
    np.save(npy_path, sliding_tensor_3d)

    # 2. Save First 3 Iterations as Sample CSVs for inspection
    for iter_idx in range(min(3, len(sliding_tensor_3d))):
        df_iter = pd.DataFrame(
            sliding_tensor_3d[iter_idx],
            index=feature_cols,
            columns=[f"t_{i}" for i in range(seq_len)]
        )
        iter_csv_path = os.path.join(MATRICES_DIR, f"iter_{iter_idx}_matrix_45x{seq_len}.csv")
        df_iter.to_csv(iter_csv_path, encoding="utf-8-sig")

    print(f" - Total Iteration Steps Extracted : {len(sliding_tensor_3d)} Iterations")
    print(f" - Full 3D Tensor Shape            : {sliding_tensor_3d.shape} (Iter x Features x Seq_Len)")
    print(f" [SUCCESS] Saved Full 3D Array      : {npy_path}")
    print(f" [SUCCESS] Saved Sample CSV (Iter 0): {os.path.join(MATRICES_DIR, f'iter_0_matrix_45x{seq_len}.csv')}")
    print("=" * 80)


if __name__ == "__main__":
    export_all_sliding_iteration_matrices(seq_len=5)
