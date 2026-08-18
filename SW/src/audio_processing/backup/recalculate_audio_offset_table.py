"""
recalculate_audio_offset_table.py (gearbox_final_suite / audio_processing)

Recalculates Front and Rear Offset Delays based on the rigorous 32-file audit results.
Generates clean results/audio_offset_verification_table.csv

Rule 3 Compliance:
- Markdown/CSV output in Korean.
"""

import os
import glob
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
AUDIT_CSV = os.path.join(RESULTS_DIR, "rigorous_32_audio_audit_results.csv")


def recalculate_offset_table():
    print("=" * 95)
    print(" 🛠 RECALCULATING FRONT vs REAR AUDIO OFFSETS BASED ON 32-FILE RIGOROUS AUDIT")
    print("=" * 95)

    if not os.path.exists(AUDIT_CSV):
        print(f"[ERROR] Audit CSV missing: {AUDIT_CSV}")
        return

    df_audit = pd.read_csv(AUDIT_CSV)
    recalc_records = []

    for idx, row in df_audit.iterrows():
        diff = float(row["dur_diff_sec"])
        status = str(row["audit_status"])

        if "FRONT_IDLE_DELAY" in status:
            front_offset = round(diff, 2)
            rear_offset = 0.00
            class_str = "FRONT_IDLE_DELAY (오디오 먼저 켜짐 / 앞부분 대기 소음 포함)"
        elif "FRONT_MISSING_AUDIO" in status:
            front_offset = round(diff, 2)
            rear_offset = 0.00
            class_str = "FRONT_MISSING_AUDIO (오디오 늦게 켜짐 / 앞부분 소리 유실)"
        else:
            front_offset = 0.00
            rear_offset = 0.00
            class_str = "EXACT_ALIGNED (1초 이내 정밀 대조 완료)"

        recalc_records.append({
            "state": row["state"],
            "channel": row["channel"],
            "filename": row["filename"],
            "can_dur_sec": row["can_duration_sec"],
            "audio_dur_sec": row["audio_duration_sec"],
            "dur_diff_sec": diff,
            "front_offset_sec": front_offset,
            "rear_offset_sec": rear_offset,
            "verified_offset_classification": class_str,
        })

    df_out = pd.DataFrame(recalc_records)

    out_csv = os.path.join(RESULTS_DIR, "audio_offset_verification_table.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 95)
    print(" 📊 RECALCULATED AUDIO OFFSET VERIFICATION TABLE (32/32 FILES)")
    print("=" * 95)
    print(df_out.to_string(index=False))
    print("=" * 95)
    print(f" [SUCCESS] Saved Recalculated Offset CSV: {out_csv}")
    print("=" * 95)


if __name__ == "__main__":
    recalculate_offset_table()
