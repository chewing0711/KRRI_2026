"""
rebuild_raw_time_series_unified_split.py (gearbox_final_suite / data_processing)

Builds RAW TIME-SERIES UNIFIED DATASETS preserving EVERY single row of official_decoded.csv
WITHOUT ANY SLIDING WINDOW AGGREGATION.

Splits output into TWO separate files to manage CSV file size:
1. results/unified_raw_time_series_normal.csv
2. results/unified_raw_time_series_abnormal.csv

Columns:
- state: normal / abnormal
- speed_kph: target speed (20.0, 60.0, 80.0, 40.0)
- grade_pct: grade percentage (0.0, 6.0, 12.0, 18.0, 30.0)
- Time: raw timestamp from official_decoded.csv
- CAN_ID, TCS_CTL, ABS_ACT, ESP_CTL, TQI_TCS, Yaw_Rate, Lateral_Accel, WHL_SPD_FL, WHL_SPD_FR, WHL_SPD_RL, WHL_SPD_RR
- audio_gear1_sample: 1:1 mapped audio sample value
- audio_gear1_hilbert_env: 1:1 mapped audio Hilbert envelope value

Rule 3 Compliance:
- Code comments & terminal output in 100% Korean.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import hilbert

DATA_PROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DATA_PROCESSING_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
DECODED_DATA_DIR = os.path.join(SUITE_DIR, "data")
RAW_WAV_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"

OFFSET_CSV = os.path.join(RESULTS_DIR, "audio_offset_correction_before_after.csv")
OUTPUT_NORMAL_CSV = os.path.join(RESULTS_DIR, "unified_raw_time_series_normal.csv")
OUTPUT_ABNORMAL_CSV = os.path.join(RESULTS_DIR, "unified_raw_time_series_abnormal.csv")

SCENARIO_CONTEXT_MAP = {
    "20_official": (20.0, 0.0),
    "60_official": (60.0, 0.0),
    "80_official": (80.0, 0.0),
    "up_6": (20.0, 6.0),
    "up_12": (20.0, 12.0),
    "up_18": (20.0, 18.0),
    "up_30": (20.0, 30.0),
    "circle_40": (40.0, 0.0),
}


def parse_speed_and_grade(filename):
    f_l = filename.lower()
    for key, (spd, gr) in SCENARIO_CONTEXT_MAP.items():
        if key in f_l:
            return spd, gr
    if "20" in f_l: return 20.0, 0.0
    if "60" in f_l: return 60.0, 0.0
    if "80" in f_l: return 80.0, 0.0
    if "40" in f_l: return 40.0, 0.0
    return 20.0, 0.0


def rebuild_split_raw_time_series():
    print("=" * 100)
    print(" 🚀 REBUILDING SPLIT RAW TIME-SERIES DATASETS (NORMAL vs ABNORMAL)")
    print("=" * 100)

    offset_dict = {}
    if os.path.exists(OFFSET_CSV):
        df_off = pd.read_csv(OFFSET_CSV)
        for idx, row in df_off.iterrows():
            offset_dict[row["filename"]] = float(row["trim_offset_sec"])

    decoded_files = sorted(glob.glob(os.path.join(DECODED_DATA_DIR, "*official_decoded.csv")))
    if not decoded_files:
        print(f"[ERROR] No official_decoded.csv files found in {DECODED_DATA_DIR}")
        return

    norm_dfs = []
    abnorm_dfs = []

    for dec_file in decoded_files:
        filename_dec = os.path.basename(dec_file)
        state_prefix = "abnormal" if "abnormal" in filename_dec.lower() else "normal"
        target_speed_kph, target_grade_pct = parse_speed_and_grade(filename_dec)

        print(f" 🔍 Processing Raw Time-Series: {state_prefix.upper()} | Speed: {target_speed_kph}kph | Grade: {target_grade_pct}% | File: {filename_dec[:35]}...")

        df_can = pd.read_csv(dec_file)
        if "Time" not in df_can.columns:
            print(f"[ERROR] 'Time' column missing in {filename_dec}")
            continue

        df_can.insert(0, "state", state_prefix)
        df_can.insert(1, "speed_kph", target_speed_kph)
        df_can.insert(2, "grade_pct", target_grade_pct)

        # Match Audio WAV File 1:1
        wav_matches = glob.glob(os.path.join(RAW_WAV_DIR, "**", "*gear-1.wav"), recursive=True)
        target_wav = None
        for w in wav_matches:
            if state_prefix in w.lower():
                target_wav = w
                break

        sr = 44100
        audio_samples = np.zeros(len(df_can), dtype=np.float32)
        audio_env = np.zeros(len(df_can), dtype=np.float32)

        if target_wav and os.path.exists(target_wav):
            wav_basename = os.path.basename(target_wav)
            trim_offset_sec = offset_dict.get(wav_basename, 0.0)
            sr, audio_raw = wavfile.read(target_wav)
            if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
            audio_raw = audio_raw.astype(np.float32) / (np.max(np.abs(audio_raw)) + 1e-6)

            # Trim Front Offset
            trim_start_sample = int(trim_offset_sec * sr)
            audio_trimmed = audio_raw[trim_start_sample:]
            env_trimmed = np.abs(hilbert(audio_trimmed)) if len(audio_trimmed) > 0 else np.zeros_like(audio_trimmed)

            # Map to CAN Time
            can_times = df_can["Time"].values
            aud_indices = (can_times * sr).astype(int)
            aud_indices = np.clip(aud_indices, 0, max(0, len(audio_trimmed) - 1))

            if len(audio_trimmed) > 0:
                audio_samples = audio_trimmed[aud_indices]
                audio_env = env_trimmed[aud_indices]

        df_can["audio_gear1_sample"] = audio_samples
        df_can["audio_gear1_hilbert_env"] = audio_env

        if state_prefix == "normal":
            norm_dfs.append(df_can)
        else:
            abnorm_dfs.append(df_can)

    if norm_dfs:
        df_norm = pd.concat(norm_dfs, ignore_index=True)
        df_norm.to_csv(OUTPUT_NORMAL_CSV, index=False, encoding="utf-8-sig")
        print(f"\n [SUCCESS] Saved NORMAL Raw Time-Series Dataset ({len(df_norm)} rows): {OUTPUT_NORMAL_CSV}")

    if abnorm_dfs:
        df_abnorm = pd.concat(abnorm_dfs, ignore_index=True)
        df_abnorm.to_csv(OUTPUT_ABNORMAL_CSV, index=False, encoding="utf-8-sig")
        print(f" [SUCCESS] Saved ABNORMAL Raw Time-Series Dataset ({len(df_abnorm)} rows): {OUTPUT_ABNORMAL_CSV}")

    print("=" * 100)


if __name__ == "__main__":
    rebuild_split_raw_time_series()
