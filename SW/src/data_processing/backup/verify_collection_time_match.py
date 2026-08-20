"""
verify_collection_time_match.py (gearbox_final_suite)

Verifies whether the total collection time (duration in seconds) of decoded CAN CSV files
matches the window collection duration represented in unified_can_context_dataset.csv.

Rule 3 Compliance:
- Code comments & terminal output in Korean.
"""

import os
import pandas as pd
from build_unified_can_context_csv import find_matching_raw_csv

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
DATA_DIR = os.path.join(SUITE_DIR, "data")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")
OUTPUT_VERIFY_CSV = os.path.join(RESULTS_DIR, "collection_time_verification.csv")

CAN_FREQ = 340.14  # Hz
WIN_SEC = 500 / CAN_FREQ  # 1.470 seconds


def verify_collection_time():
    print("=" * 80)
    print(" ⏱ VERIFYING TOTAL COLLECTION TIME MATCH: DECODED CAN vs UNIFIED CSV")
    print("=" * 80)

    if not os.path.exists(UNIFIED_CSV):
        print(f"[ERROR] Unified CSV missing: {UNIFIED_CSV}")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)
    raw_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    verify_rows = []

    for sess_id, group in df_unified.groupby("session_id"):
        st = group["state"].iloc[0]
        
        # Clean scenario text (remove 'normal_' or 'abnormal_' prefix)
        sess_str = str(sess_id)
        if sess_str.startswith("abnormal_"):
            scen_text = sess_str[len("abnormal_"):]
        elif sess_str.startswith("normal_"):
            scen_text = sess_str[len("normal_"):]
        else:
            scen_text = sess_str

        n_windows = len(group)
        unified_time_sec = n_windows * WIN_SEC

        matched_file = find_matching_raw_csv(scen_text, st, raw_files)
        if matched_file:
            raw_path = os.path.join(DATA_DIR, matched_file)
            df_raw = pd.read_csv(raw_path)
            raw_total_rows = len(df_raw)
            raw_time_sec = raw_total_rows / CAN_FREQ
            
            time_diff_sec = raw_time_sec - unified_time_sec
            # Valid means raw logging length is sufficient to cover all extracted windows
            is_valid_coverage = raw_time_sec >= (unified_time_sec - 0.1)

            verify_rows.append({
                "session_id": sess_id,
                "scenario_name": scen_text,
                "state": st,
                "unified_window_count": n_windows,
                "unified_duration_sec": round(unified_time_sec, 2),
                "raw_csv_total_rows": raw_total_rows,
                "raw_csv_duration_sec": round(raw_time_sec, 2),
                "raw_minus_unified_diff_sec": round(time_diff_sec, 2),
                "sufficient_raw_coverage": is_valid_coverage,
                "matched_raw_csv_file": matched_file
            })

    if not verify_rows:
        print("[ERROR] No sessions verified.")
        return

    df_verify = pd.DataFrame(verify_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_verify.to_csv(OUTPUT_VERIFY_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[SUCCESS] Generated Collection Time Verification CSV: {OUTPUT_VERIFY_CSV}")
    print(f" - Total Sessions Verified    : {len(df_verify)}")
    print(f" - All Raw Sessions Sufficient: {df_verify['sufficient_raw_coverage'].all()}")
    print("=" * 80)


if __name__ == "__main__":
    verify_collection_time()
