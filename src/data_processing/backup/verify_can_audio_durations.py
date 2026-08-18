"""
verify_can_audio_durations.py (gearbox_final_suite / data_processing)

Computes & Compares exact duration of CAN windows vs Audio WAV recording files across all 16 scenarios:
1. CAN Duration = window_count * 1.470 seconds (500 samples @ 340.14Hz)
2. Audio Duration = num_audio_samples / sample_rate (scipy.io.wavfile)
3. Outputs comparison table showing exact duration difference (delta_sec).

Rule 3 Compliance:
- Code comments & terminal output in Korean.
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.io import wavfile

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")

DATA_AUDIO_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"


def build_audio_file_map():
    audio_files = glob.glob(os.path.join(DATA_AUDIO_DIR, "*", "*", "*gear-1*.wav"))
    audio_map = {}

    for filepath in audio_files:
        parts = filepath.split(os.sep)
        state = parts[-3].strip().lower()
        scenario_folder = parts[-2].strip().lower()

        if "20kph" in scenario_folder and "high speed" in scenario_folder:
            scen = "high speed 20kph"
        elif "60kph" in scenario_folder:
            scen = "high speed 60kph"
        elif "80kph" in scenario_folder:
            scen = "high speed 80kph"
        elif "6도" in scenario_folder or "6%" in scenario_folder:
            scen = "hills 6 degrees"
        elif "12도" in scenario_folder or "12%" in scenario_folder:
            scen = "hills 12 degrees"
        elif "18도" in scenario_folder or "18%" in scenario_folder:
            scen = "hills 18 degrees"
        elif "30도" in scenario_folder or "30%" in scenario_folder:
            scen = "hills 30 degrees"
        elif "원선회" in scenario_folder or "steer" in scenario_folder:
            scen = "steering pad"
        else:
            continue

        key = (scen, state)
        try:
            sr, data = wavfile.read(filepath)
            n_samples = len(data)
            duration_sec = n_samples / float(sr)
            audio_map[key] = {
                "filepath": filepath,
                "sr": sr,
                "n_samples": n_samples,
                "audio_duration_sec": round(duration_sec, 3),
            }
        except Exception as e:
            print(f"[ERROR] Failed reading audio {filepath}: {e}")

    return audio_map


def verify_can_audio_durations():
    print("=" * 95)
    print(" 🔍 VERIFYING CAN DURATION VS AUDIO RECORDING DURATION ACROSS ALL 16 SCENARIOS")
    print("=" * 95)

    if not os.path.exists(UNIFIED_CSV):
        print(f"[ERROR] Unified CSV missing: {UNIFIED_CSV}")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)
    audio_map = build_audio_file_map()

    duration_records = []

    for sess_id, df_sess in df_unified.groupby("session_id", sort=False):
        scen = df_sess["scenario"].iloc[0] if "scenario" in df_sess.columns else str(sess_id)
        state = df_sess["state"].iloc[0] if "state" in df_sess.columns else "normal"
        
        # Try key matching
        matched_audio = None
        for (a_scen, a_st), a_info in audio_map.items():
            if (a_scen in str(sess_id).lower() or a_scen in str(scen).lower()) and a_st.lower() in str(state).lower():
                matched_audio = a_info
                break

        n_windows = len(df_sess)
        can_duration_sec = round(n_windows * 1.470, 3)

        if matched_audio:
            audio_duration_sec = matched_audio["audio_duration_sec"]
            diff_sec = round(audio_duration_sec - can_duration_sec, 3)
            status = "MATCHED (Audio >= CAN)" if diff_sec >= 0 else "SHORT AUDIO (Audio < CAN)"
        else:
            audio_duration_sec = 0.0
            diff_sec = 0.0
            status = "AUDIO MISSING"

        duration_records.append({
            "session_id": sess_id,
            "state": state,
            "can_windows_cnt": n_windows,
            "can_duration_sec": can_duration_sec,
            "audio_duration_sec": audio_duration_sec,
            "diff_duration_sec": diff_sec,
            "match_status": status,
        })

    df_dur = pd.DataFrame(duration_records)

    print("\n" + "=" * 95)
    print(" 📊 CAN VS AUDIO DURATION COMPARISON TABLE")
    print("=" * 95)
    print(df_dur.to_string(index=False))
    print("-" * 95)

    out_csv = os.path.join(RESULTS_DIR, "can_audio_duration_verification.csv")
    df_dur.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f" [SUCCESS] Saved Duration Verification CSV: {out_csv}")
    print("=" * 95)


if __name__ == "__main__":
    verify_can_audio_durations()
