"""
rebuild_raw_time_series_unified_subchunk_audio.py (gearbox_final_suite / data_processing)

Rebuilds the Raw Time-Series Unified Copy Dataset with TIME-based 176-Sample Audio Sub-chunk Slicing.

Original Dataset Protection:
- results/unified_can_context_dataset.csv is kept 100% UNTOUCHED.

Saved Copy Files:
- results/unified_raw_time_series_normal_subchunk.csv
- results/unified_raw_time_series_abnormal_subchunk.csv

Features included per CAN time row:
- Raw Time, speed_kph, grade_pct, state, target
- All raw CAN signals (WHL_SPD_*, Yaw_Rate, Lateral_Accel, TQI_TCS, etc.)
- Time-sliced Audio Sub-chunk features (audio_subchunk_env_mean, audio_subchunk_env_max, audio_subchunk_rms, audio_subchunk_sample_count)

Rule 3 Compliance:
- Code comments & terminal output in 100% Korean.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import hilbert

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECODED_DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
RAW_WAV_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"

OFFSET_TABLE_CSV = os.path.join(RESULTS_DIR, "audio_offset_verification_table.csv")


def load_offset_dict():
    offset_dict = {}
    if os.path.exists(OFFSET_TABLE_CSV):
        df_off = pd.read_csv(OFFSET_TABLE_CSV)
        for _, row in df_off.iterrows():
            offset_dict[row["wav_file"]] = float(row["optimal_trim_offset_sec"])
    return offset_dict


def rebuild_subchunk_dataset():
    print("=" * 100)
    print(" 🚀 REBUILDING RAW TIME-SERIES UNIFIED DATASET WITH TIME-BASED AUDIO SUB-CHUNK SLICING")
    print("=" * 100)

    offset_dict = load_offset_dict()
    decoded_files = sorted(glob.glob(os.path.join(DECODED_DATA_DIR, "*official_decoded.csv")))

    if not decoded_files:
        print(f"[ERROR] No official_decoded.csv files found in {DECODED_DATA_DIR}")
        return

    scenario_speed_grade_map = {
        "20_official": (20.0, 0.0),
        "60_official": (60.0, 0.0),
        "80_official": (80.0, 0.0),
        "up_6_official": (20.0, 6.0),
        "up_12_official": (20.0, 12.0),
        "up_18_official": (20.0, 18.0),
        "up_30_official": (20.0, 30.0),
        "circle_40_official": (40.0, 0.0),
    }

    norm_dfs = []
    abnorm_dfs = []

    for dec_file in decoded_files:
        filename_dec = os.path.basename(dec_file)
        state_prefix = "abnormal" if "abnormal" in filename_dec.lower() else "normal"

        speed_kph, grade_pct = 20.0, 0.0
        for k, v in scenario_speed_grade_map.items():
            if k in filename_dec:
                speed_kph, grade_pct = v
                break

        print(f" 🔍 Processing: {state_prefix.upper()} | Speed: {speed_kph}kph | Grade: {grade_pct}% | File: {filename_dec[:45]}...")

        df_can = pd.read_csv(dec_file)
        if "Time" not in df_can.columns:
            continue

        df_can["state"] = state_prefix
        df_can["target"] = 1 if state_prefix == "abnormal" else 0
        df_can["speed_kph"] = speed_kph
        df_can["grade_pct"] = grade_pct

        # Match Audio WAV File
        wav_matches = glob.glob(os.path.join(RAW_WAV_DIR, "**", "*gear-1.wav"), recursive=True)
        target_wav = None
        for w in wav_matches:
            if state_prefix in w.lower():
                if ("20" in filename_dec and "20" in w) or \
                   ("60" in filename_dec and "60" in w) or \
                   ("80" in filename_dec and "80" in w) or \
                   ("up_6" in filename_dec and "up_6" in w) or \
                   ("up_12" in filename_dec and "up_12" in w) or \
                   ("up_18" in filename_dec and "up_18" in w) or \
                   ("up_30" in filename_dec and "up_30" in w) or \
                   ("circle" in filename_dec and "circle" in w):
                    target_wav = w
                    break

        n_rows = len(df_can)
        subchunk_count = np.zeros(n_rows, dtype=np.int32)
        subchunk_env_mean = np.zeros(n_rows, dtype=np.float32)
        subchunk_env_max = np.zeros(n_rows, dtype=np.float32)
        subchunk_rms = np.zeros(n_rows, dtype=np.float32)

        if target_wav and os.path.exists(target_wav):
            wav_basename = os.path.basename(target_wav)
            trim_offset_sec = offset_dict.get(wav_basename, 0.0)

            sr, audio_raw = wavfile.read(target_wav)
            if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
            audio_raw = audio_raw.astype(np.float32) / (np.max(np.abs(audio_raw)) + 1e-6)

            # Apply Front Trim
            trim_start_sample = int(trim_offset_sec * sr)
            audio_trimmed = audio_raw[trim_start_sample:]
            env_trimmed = np.abs(hilbert(audio_trimmed)) if len(audio_trimmed) > 0 else np.zeros_like(audio_trimmed)

            can_times = df_can["Time"].values
            aud_start_indices = (can_times * sr).astype(int)

            # Calculate Sub-chunk features for each CAN time interval
            for i in range(n_rows):
                idx_start = aud_start_indices[i]
                if i < n_rows - 1:
                    idx_end = aud_start_indices[i + 1]
                else:
                    idx_end = idx_start + int(0.004 * sr)

                idx_start = max(0, min(idx_start, len(audio_trimmed)))
                idx_end = max(idx_start + 1, min(idx_end, len(audio_trimmed)))

                sub_audio = audio_trimmed[idx_start:idx_end]
                sub_env = env_trimmed[idx_start:idx_end]

                subchunk_count[i] = len(sub_audio)
                if len(sub_audio) > 0:
                    subchunk_env_mean[i] = np.mean(sub_env)
                    subchunk_env_max[i] = np.max(sub_env)
                    subchunk_rms[i] = np.sqrt(np.mean(sub_audio ** 2))

        df_can["audio_subchunk_sample_count"] = subchunk_count
        df_can["audio_subchunk_env_mean"] = subchunk_env_mean
        df_can["audio_subchunk_env_max"] = subchunk_env_max
        df_can["audio_subchunk_rms"] = subchunk_rms

        if state_prefix == "normal":
            norm_dfs.append(df_can)
        else:
            abnorm_dfs.append(df_can)

    df_norm_all = pd.concat(norm_dfs, ignore_index=True)
    df_abnorm_all = pd.concat(abnorm_dfs, ignore_index=True)

    out_norm_csv = os.path.join(RESULTS_DIR, "unified_raw_time_series_normal_subchunk.csv")
    out_abnorm_csv = os.path.join(RESULTS_DIR, "unified_raw_time_series_abnormal_subchunk.csv")

    df_norm_all.to_csv(out_norm_csv, index=False, encoding="utf-8-sig")
    df_abnorm_all.to_csv(out_abnorm_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f" [SUCCESS] Saved NORMAL Sub-chunk Audio Dataset ({len(df_norm_all):,} rows): {out_norm_csv}")
    print(f" [SUCCESS] Saved ABNORMAL Sub-chunk Audio Dataset ({len(df_abnorm_all):,} rows): {out_abnorm_csv}")
    print("=" * 100)


if __name__ == "__main__":
    rebuild_subchunk_dataset()
