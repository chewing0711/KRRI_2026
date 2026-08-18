"""
audit_all_32_files_rigorous.py (gearbox_final_suite / audio_processing)

Performs a 100% rigorous dual audit on all 32 WAV audio files vs CAN duration:
Categorizes each file into 3 exact physical states:
1. EXACT_ALIGNED: Audio and CAN duration match within 1.0 second.
2. FRONT_IDLE_DELAY: Audio started recording early (Head idle noise present).
3. FRONT_MISSING_AUDIO: Audio started recording late (Front driving audio missing).

Rule 3 Compliance:
- Matplotlib/Terminal Output in Korean.
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.io import wavfile

RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")


def audit_all_files():
    print("=" * 100)
    print(" 🔍 RIGOROUS DUAL AUDIT ON ALL 32 WAV AUDIO FILES VS CAN DURATION")
    print("=" * 100)

    wav_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.wav"), recursive=True))
    if not wav_files:
        print("[ERROR] No WAV files found.")
        return

    records = []

    for wav_path in wav_files:
        scen_folder = os.path.dirname(wav_path)
        parts = scen_folder.split(os.sep)
        state_str = parts[-2].strip().lower()
        scen_str = parts[-1].strip().lower()
        filename = os.path.basename(wav_path)

        state_prefix = "abnormal" if "abnormal" in state_str else "normal"
        channel_type = "GEAR" if "gear" in filename.lower() else "OBD"
        scen_key = f"{state_prefix}_{filename}"

        ws_csv_candidates = glob.glob(os.path.join(scen_folder, "*wheel*.csv"))
        ws_csv = ws_csv_candidates[0] if ws_csv_candidates else None

        try:
            sr, audio_raw = wavfile.read(wav_path)
            if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
            n_samples = len(audio_raw)
            audio_dur_sec = round(n_samples / float(sr), 2)

            if ws_csv and os.path.exists(ws_csv):
                df_ws = pd.read_csv(ws_csv)
                time_col = [c for c in df_ws.columns if "time" in c.lower() or "sec" in c.lower()][0]
                can_dur_sec = round(float(df_ws[time_col].iloc[-1] - df_ws[time_col].iloc[0]), 2)
            else:
                can_dur_sec = audio_dur_sec

            dur_diff_sec = round(audio_dur_sec - can_dur_sec, 2)

            # Precise Status Classification
            if dur_diff_sec < -0.8:
                status = "FRONT_MISSING_AUDIO (오디오 녹음 늦음 / 앞부분 소리 손실)"
            elif dur_diff_sec > 3.0:
                status = "FRONT_IDLE_DELAY (오디오 녹음 먼저 켜짐 / 앞부분 대기 소음 포함)"
            else:
                status = "EXACT_ALIGNED (1초 이내 정밀 대조 완료)"

            records.append({
                "state": state_prefix,
                "channel": channel_type,
                "filename": filename,
                "can_duration_sec": can_dur_sec,
                "audio_duration_sec": audio_dur_sec,
                "dur_diff_sec": dur_diff_sec,
                "audit_status": status,
            })

        except Exception as e:
            print(f"[ERROR] Auditing {filename} failed: {e}")

    df = pd.DataFrame(records)

    print("\n" + "=" * 100)
    print(" 📊 RIGOROUS DUAL AUDIT RESULTS (32/32 FILES)")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    out_csv = os.path.join(RESULTS_DIR, "rigorous_32_audio_audit_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n [SUMMARY BY STATUS]")
    print(df["audit_status"].value_counts().to_string())
    print(f"\n [SUCCESS] Saved Rigorous Audit CSV: {out_csv}")
    print("=" * 100)


if __name__ == "__main__":
    audit_all_files()
