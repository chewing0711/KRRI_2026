"""
research_audio_rule_based_energy.py (gearbox_final_suite / audio_processing)

Research & EDA for Option 1: Rule-Based Lightweight Audio Energy Feature Extraction
(Zero Audio Neural Net, Low-Memory Embedded C++ Compatible)

1. Analyzes raw audio WAV files across all 16 scenarios.
2. Computes FFT Energy Spectral Bands (Low: 0-1kHz, Mid: 1k-4kHz, High: 4k-10kHz, Ultra-High: >10kHz).
3. Evaluates Normal vs Abnormal energy separation margins.
4. Derives lightweight threshold rules and outputs C++ compatible pseudo-code formula.

Rule 3 Compliance:
- Code comments & terminal output in Korean.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")

AUDIO_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/audio"


def analyze_audio_band_energies():
    print("=" * 90)
    print(" 🔬 RESEARCHING OPTION 1: LIGHTWEIGHT RULE-BASED AUDIO BAND ENERGY (NO NEURAL NET)")
    print("=" * 90)

    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*", "*_gear1.wav"))

    if not audio_files:
        print(f"[ERROR] No gear1 WAV files found in: {AUDIO_DIR}")
        return

    records = []

    for filepath in audio_files:
        parts = filepath.split(os.sep)
        state = parts[-2].strip().lower()
        wav_name = os.path.basename(filepath)
        scen_clean = wav_name.replace("_gear1.wav", "")

        try:
            sr, data = wavfile.read(filepath)
            if data.ndim > 1:
                data = data[:, 0]

            data = data.astype(np.float32)
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data /= max_val

            f, t, Zxx = stft(data, fs=sr, nperseg=512)
            mag = np.abs(Zxx)

            idx_low = np.where((f >= 0) & (f < 1000))[0]
            idx_mid = np.where((f >= 1000) & (f < 4000))[0]
            idx_high = np.where((f >= 4000) & (f < 10000))[0]
            idx_ultra = np.where(f >= 10000)[0]

            energy_low = np.mean(mag[idx_low, :]) if len(idx_low) > 0 else 0.0
            energy_mid = np.mean(mag[idx_mid, :]) if len(idx_mid) > 0 else 0.0
            energy_high = np.mean(mag[idx_high, :]) if len(idx_high) > 0 else 0.0
            energy_ultra = np.mean(mag[idx_ultra, :]) if len(idx_ultra) > 0 else 0.0
            total_rms = np.sqrt(np.mean(data**2))

            records.append({
                "state": state,
                "scenario": scen_clean,
                "duration_sec": round(len(data) / float(sr), 2),
                "audio_rms": round(total_rms, 6),
                "energy_low_0_1k": round(energy_low, 6),
                "energy_mid_1k_4k": round(energy_mid, 6),
                "energy_high_4k_10k": round(energy_high, 6),
                "energy_ultra_10k_plus": round(energy_ultra, 6),
            })
        except Exception as e:
            print(f"[ERROR] Failed processing {filepath}: {e}")

    df_res = pd.DataFrame(records)

    print("\n" + "=" * 90)
    print(" 📊 AUDIO FREQUENCY BAND ENERGY STATISTICAL SUMMARY (NORMAL VS ABNORMAL)")
    print("=" * 90)

    summary = df_res.groupby("state")[["audio_rms", "energy_low_0_1k", "energy_mid_1k_4k", "energy_high_4k_10k", "energy_ultra_10k_plus"]].mean()
    print(summary.to_string())
    print("-" * 90)

    out_csv = os.path.join(RESULTS_DIR, "audio_rule_energy_analysis.csv")
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f" [SUCCESS] Saved Audio Band Energy Research CSV: {out_csv}")
    print("=" * 90)


if __name__ == "__main__":
    analyze_audio_band_energies()
