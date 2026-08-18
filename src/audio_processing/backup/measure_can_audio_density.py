"""
measure_can_audio_density.py (gearbox_final_suite / audio_processing)

Empirically measures the EXACT number of audio samples contained within each raw CAN row interval
ACROSS ALL 16 SCENARIOS (Normal & Abnormal).

Rule 3 Compliance:
- Code comments & terminal output in 100% Korean.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECODED_DATA_DIR = os.path.join(SUITE_DIR, "data")
RAW_WAV_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"


def measure_all_scenarios_density():
    print("=" * 100)
    print(" 📏 EMPIRICAL MEASUREMENT: ALL 16 SCENARIOS (CAN INTERVAL vs AUDIO DENSITY)")
    print("=" * 100)

    decoded_files = sorted(glob.glob(os.path.join(DECODED_DATA_DIR, "*official_decoded.csv")))
    if not decoded_files:
        print(f"[ERROR] No official_decoded.csv files found in {DECODED_DATA_DIR}")
        return

    all_time_deltas = []
    all_audio_counts = []
    scenario_summaries = []

    for dec_file in decoded_files:
        filename_dec = os.path.basename(dec_file)
        state_prefix = "abnormal" if "abnormal" in filename_dec.lower() else "normal"

        df_can = pd.read_csv(dec_file)
        if "Time" not in df_can.columns:
            continue

        can_times = df_can["Time"].values
        n_can_rows = len(can_times)

        # Match Corresponding WAV File
        wav_matches = glob.glob(os.path.join(RAW_WAV_DIR, "**", "*gear-1.wav"), recursive=True)
        target_wav = None
        for w in wav_matches:
            if state_prefix in w.lower():
                # match scenario keyword
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

        if not target_wav or not os.path.exists(target_wav):
            continue

        sr, audio_data = wavfile.read(target_wav)
        n_audio_samples = len(audio_data)

        # Calculate time deltas and audio sample counts for this file
        deltas = np.diff(can_times)
        counts = (deltas * sr).astype(int)

        all_time_deltas.extend(deltas)
        all_audio_counts.extend(counts)

        scenario_summaries.append({
            "state": state_prefix.upper(),
            "file": filename_dec[:30],
            "can_rows": n_can_rows,
            "min_ms": np.min(deltas) * 1000,
            "max_ms": np.max(deltas) * 1000,
            "mean_ms": np.mean(deltas) * 1000,
            "mean_audio_samples": np.mean(counts)
        })

    all_time_deltas = np.array(all_time_deltas)
    all_audio_counts = np.array(all_audio_counts)

    print("\n 📋 PER-SCENARIO SUMMARY TABLE (16 SCENARIOS):")
    print("-" * 100)
    print(f" {'STATE':<10} | {'FILE':<30} | {'CAN ROWS':<10} | {'MEAN DELTA (ms)':<15} | {'MEAN AUDIO SAMPLES':<18}")
    print("-" * 100)
    for s in scenario_summaries:
        print(f" {s['state']:<10} | {s['file']:<30} | {s['can_rows']:<10,d} | {s['mean_ms']:<15.3f} | {s['mean_audio_samples']:<18.2f}")
    print("-" * 100)

    print("\n" + "=" * 100)
    print(" 📊 OVERALL AGGREGATED METRICS (ACROSS ALL 16 SCENARIOS)")
    print("=" * 100)
    print(f" 1. 전체 CAN 시간 간격 (Total {len(all_time_deltas):,} intervals):")
    print(f"    - 최소 간격:   {np.min(all_time_deltas) * 1000:.3f} ms")
    print(f"    - 최대 간격:   {np.max(all_time_deltas) * 1000:.3f} ms")
    print(f"    - 평균 간격:   {np.mean(all_time_deltas) * 1000:.3f} ms")
    print(f"    - 중앙값 간격:  {np.median(all_time_deltas) * 1000:.3f} ms")

    print(f"\n 2. 전체 CAN 1개 행 당 오디오 샘플 개수 (Audio Samples per CAN Row):")
    print(f"    - 최소 개수:   {np.min(all_audio_counts):,} 개")
    print(f"    - 최대 개수:   {np.max(all_audio_counts):,} 개")
    print(f"    - 평균 개수:   {np.mean(all_audio_counts):.2f} 개")
    print(f"    - 중앙값 개수:  {np.median(all_audio_counts):.0f} 개")

    print("=" * 100)
    print(" 💡 Overall Sample Distribution Breakdown:")
    percentiles = [10, 25, 50, 75, 90, 99]
    for p in percentiles:
        val = np.percentile(all_audio_counts, p)
        print(f"    - 하위 {p:2d}% 개수: {val:.1f} 개의 오디오 샘플 함유")
    print("=" * 100)


if __name__ == "__main__":
    measure_all_scenarios_density()
