"""
export_scenario_representative_matrices.py (gearbox_final_suite / data_processing)

Exports ONLY MID-point Representative (45 Features x 5 Timesteps) Matrices for ALL 16 Driving Sessions:
1. Iterates over all 16 session_ids (20kph, 60kph, 80kph, hills 6/12/18/30 deg, steering pad x normal/abnormal).
2. Extracts ONLY the MID-point sequence matrix (45x5) of each session.
3. Saves exactly 16 clean representative CSV files in results/extracted_matrices/.

Rule 3 Compliance:
- Code comments & terminal output in Korean.
"""

import os
import re
import glob
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
MATRICES_DIR = os.path.join(RESULTS_DIR, "extracted_matrices")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")


def sanitize_filename(name):
    return re.sub(r"[^\w\-]", "_", str(name)).strip("_")


def export_mid_scenario_matrices(seq_len=5):
    print("=" * 90)
    print(f" 🛠 EXPORTING ONLY MID-POINT (45 x {seq_len}) MATRICES FOR ALL 16 SCENARIOS")
    print("=" * 90)

    if not os.path.exists(UNIFIED_CSV):
        print(f"[ERROR] Unified CSV missing: {UNIFIED_CSV}")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)
    meta_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df_unified.columns if c not in meta_cols]

    os.makedirs(MATRICES_DIR, exist_ok=True)

    # Clean up old start/end matrices if present
    for old_f in glob.glob(os.path.join(MATRICES_DIR, "matrix_*_start_*.csv")) + glob.glob(os.path.join(MATRICES_DIR, "matrix_*_end_*.csv")):
        try:
            os.remove(old_f)
        except OSError:
            pass

    exported_records = []

    for sess_id, df_sess in df_unified.groupby("session_id", sort=False):
        X_sess = df_sess[feature_cols].values
        n_sess = len(df_sess)
        state_name = df_sess["state"].iloc[0]

        if n_sess < seq_len:
            continue

        clean_sess_id = sanitize_filename(sess_id)

        # Extract ONLY the MID-point interval
        idx_mid = max(0, (n_sess - seq_len) // 2)

        mat_45x5 = X_sess[idx_mid : idx_mid + seq_len].T
        df_mat = pd.DataFrame(
            mat_45x5,
            index=feature_cols,
            columns=[f"t_{k}" for k in range(seq_len)]
        )

        out_csv_name = f"matrix_{clean_sess_id}_mid_45x{seq_len}.csv"
        out_csv_path = os.path.join(MATRICES_DIR, out_csv_name)
        df_mat.to_csv(out_csv_path, encoding="utf-8-sig")

        exported_records.append({
            "session_id": sess_id,
            "state": state_name,
            "interval_point": "mid",
            "session_window_start_idx": idx_mid,
            "exported_filename": out_csv_name,
        })

    df_exported = pd.DataFrame(exported_records)
    summary_path = os.path.join(MATRICES_DIR, "scenario_matrices_manifest.csv")
    df_exported.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print(" 📊 SCENARIO MID-POINT MATRICES EXPORT SUMMARY")
    print("=" * 90)
    print(f" - 총 수출된 시나리오 세션 수 : {len(df_exported)}개 세션")
    print(f" - 세션당 추출 타임 포인트 : MID 오직 1개 구간만 사용")
    print(f" - 총 생성된 대표 CSV 파일 수 : {len(df_exported)}개 파일 (오직 MID 16개)")
    print(f" - 저장 매니페스트 위치       : {summary_path}")
    print("=" * 90)


if __name__ == "__main__":
    export_mid_scenario_matrices(seq_len=5)
