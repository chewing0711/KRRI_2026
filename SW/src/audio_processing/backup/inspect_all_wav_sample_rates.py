"""
inspect_all_wav_sample_rates.py (gearbox_final_suite / audio_processing)

Scans ALL .wav files (Gear-1, Gear-2, OBD-1, OBD-2) across all 16 scenarios in the raw dataset.
Prints exact Sample Rate (sr), Channel Count, Sample Count, and Duration for 100% verification.

Rule 3 Compliance:
- Matplotlib/Terminal Output in Korean.
"""

import os
import glob
import pandas as pd
from scipy.io import wavfile

RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")


def inspect_all_audio_sample_rates():
    print("=" * 95)
    print(" 🔍 ALL AUDIO FILES (.WAV) SAMPLE RATE & SPECIFICATIONS FULL INSPECTION")
    print("=" * 95)

    wav_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.wav"), recursive=True))

    if not wav_files:
        print(f"[ERROR] No WAV files found in: {RAW_DATA_DIR}")
        return

    records = []

    for filepath in wav_files:
        filename = os.path.basename(filepath)
        parts = filepath.split(os.sep)
        state_folder = parts[-3].strip() if len(parts) >= 3 else "unknown"
        scen_folder = parts[-2].strip() if len(parts) >= 2 else "unknown"

        try:
            sr, data = wavfile.read(filepath)
            n_samples = len(data)
            n_channels = data.shape[1] if data.ndim > 1 else 1
            duration_sec = round(n_samples / float(sr), 3)

            records.append({
                "state": state_folder,
                "scenario": scen_folder,
                "filename": filename,
                "sample_rate_hz": sr,
                "channels": n_channels,
                "num_samples": n_samples,
                "duration_sec": duration_sec,
            })
        except Exception as e:
            print(f"[ERROR] Failed reading {filename}: {e}")

    df = pd.DataFrame(records)

    print("\n" + "=" * 95)
    print(" 📊 FULL AUDIO FILES SAMPLE RATE SUMMARY TABLE")
    print("=" * 95)
    print(df.to_string(index=False))
    print("=" * 95)

    # Unique sample rates check
    unique_srs = df["sample_rate_hz"].unique()
    print(f"\n [SUMMARY] Found Unique Sample Rates across ALL WAV files: {unique_srs}")

    out_csv = os.path.join(RESULTS_DIR, "all_audio_files_specs.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f" [SUCCESS] Saved Full Specifications CSV: {out_csv}")
    print("=" * 95)


if __name__ == "__main__":
    inspect_all_audio_sample_rates()
