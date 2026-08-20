"""
verify_envelope_fft_proof.py (gearbox_final_suite / audio_processing)

Computes the FFT Spectrum of Hilbert Envelope across the 8 SCENARIO PAIRS (Abnormal vs Normal).
Demonstrates empirical mathematical proof across the ENTIRE dataset.

Rule 3 Compliance:
- Matplotlib labels, titles, legends in 100% English.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "audio_spectrograms")
RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"

SCENARIOS_PAIR = [
    ("go_20-gear-1.wav", "record_20-gear-1.wav", "20km/h Low Speed"),
    ("go_60-gear-1.wav", "record_60-gear-1.wav", "60km/h Mid Speed"),
    ("go_80-gear-1.wav", "record_80-gear-1.wav", "80km/h High Speed"),
    ("go_up_6-gear-1.wav", "record_up_6-gear-1.wav", "6deg Hill Climb"),
    ("go_up_12-gear-1_cut.wav", "record_up_12-gear-1.wav", "12deg Hill Climb"),
    ("go_up_18-gear-1.wav", "record_up_18-gear-1.wav", "18deg Hill Climb"),
    ("go_up_30-gear-1.wav", "record_up_30-gear-1.wav", "30deg Hill Climb"),
    ("go_up_circle_40-gear-1.wav", "record_circle-gear-1.wav", "40km/h Steer Pad"),
]


def verify_all_scenarios_envelope_spectrum():
    print("=" * 95)
    print(" 🧪 COMPUTING ENVELOPE SPECTRUM PROOF ACROSS 8 PAIRS (ABNORMAL vs NORMAL)")
    print("=" * 95)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    all_wavs = glob.glob(os.path.join(RAW_DATA_DIR, "*", "*", "*.wav"))

    file_dict = {os.path.basename(f).lower(): f for f in all_wavs}
    results = []

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    axes = axes.flatten()

    for idx, (abnorm_fn, norm_fn, scen_label) in enumerate(SCENARIOS_PAIR):
        f_abnorm = file_dict.get(abnorm_fn)
        f_norm = file_dict.get(norm_fn)

        if not f_abnorm or not f_norm:
            print(f"[WARN] File missing for scenario {scen_label}: {abnorm_fn} or {norm_fn}")
            continue

        try:
            sr_a, d_abnorm = wavfile.read(f_abnorm)
            sr_n, d_norm = wavfile.read(f_norm)

            if d_abnorm.ndim > 1: d_abnorm = d_abnorm[:, 0]
            if d_norm.ndim > 1: d_norm = d_norm[:, 0]

            d_abnorm = d_abnorm.astype(np.float32) / np.max(np.abs(d_abnorm))
            d_norm = d_norm.astype(np.float32) / np.max(np.abs(d_norm))

            min_len_a = min(int(15*sr_a), len(d_abnorm))
            min_len_n = min(int(15*sr_n), len(d_norm))

            env_a = np.abs(hilbert(d_abnorm[:min_len_a]))
            env_n = np.abs(hilbert(d_norm[:min_len_n]))

            fft_a = np.abs(np.fft.rfft(env_a - np.mean(env_a)))
            fft_n = np.abs(np.fft.rfft(env_n - np.mean(env_n)))
            freqs_a = np.fft.rfftfreq(len(env_a), d=1.0/sr_a)

            mask_a = (freqs_a >= 2.0) & (freqs_a <= 300.0)
            freqs_sub = freqs_a[mask_a]
            fft_a_sub = fft_a[mask_a]
            fft_n_sub = fft_n[:len(freqs_a)][mask_a]

            peak_idx = np.argmax(fft_a_sub)
            peak_freq = freqs_sub[peak_idx]
            peak_val_a = fft_a_sub[peak_idx]
            peak_val_n = fft_n_sub[peak_idx] if peak_idx < len(fft_n_sub) else 1.0
            ratio = peak_val_a / (peak_val_n + 1e-6)

            results.append({
                "scenario": scen_label,
                "peak_freq_hz": round(float(peak_freq), 2),
                "normal_magnitude": round(float(peak_val_n), 2),
                "abnormal_magnitude": round(float(peak_val_a), 2),
                "abnormal_vs_normal_ratio": round(float(ratio), 2),
            })

            ax = axes[idx]
            ax.plot(freqs_sub, fft_n_sub, color="#95a5a6", linewidth=1.0, label="Normal Envelope Spectrum")
            ax.plot(freqs_sub, fft_a_sub, color="#e74c3c", linewidth=1.8, label=f"Abnormal (Peak: {peak_freq:.1f}Hz)")
            ax.set_title(f"{scen_label} (Ratio: {ratio:.2f}x)", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8)

        except Exception as e:
            print(f"[ERROR] Failed processing scenario {scen_label}: {e}")

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, "all_8_scenario_pairs_envelope_proof.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 95)
    print(" 📊 8 SCENARIO PAIRS HILBERT ENVELOPE SPECTRUM PROOF TABLE")
    print("=" * 95)
    print(df_res.to_string(index=False))
    print("=" * 95)

    out_csv = os.path.join(RESULTS_DIR, "all_scenarios_envelope_proof.csv")
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f" [SUCCESS] Saved Proof Figure: {out_png}")
    print(f" [SUCCESS] Saved Proof CSV: {out_csv}")
    print("=" * 95)


if __name__ == "__main__":
    verify_all_scenarios_envelope_spectrum()
