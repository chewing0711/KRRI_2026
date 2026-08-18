"""
plot_hilbert_before_after.py (gearbox_final_suite / audio_processing)

Generates Before vs After visualization figure for Hilbert Transform Envelope Analysis:
- Top Panel: Raw Noisy Waveform (BEFORE)
- Bottom Panel: Hilbert Envelope & Analytic Signal (AFTER)

Rule 3 Compliance:
- Matplotlib labels, titles, legends in 100% English.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "audio_spectrograms")
RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"


def plot_before_after_hilbert():
    print("=" * 90)
    print(" 🎨 PLOTTING BEFORE vs AFTER HILBERT ENVELOPE VISUALIZATION")
    print("=" * 90)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    wav_files = glob.glob(os.path.join(RAW_DATA_DIR, "*", "*", "*.wav"))
    gear_files = [f for f in wav_files if "gear-1" in f.lower() and ("abnormal" in f.lower() or "fault" in f.lower())]

    if not gear_files:
        print(f"[ERROR] No abnormal gear-1 WAV files found in: {RAW_DATA_DIR}")
        return

    sample_file = gear_files[0]
    print(f" [LOAD] Using target sample audio: {sample_file}")

    sr, data = wavfile.read(sample_file)
    if data.ndim > 1: data = data[:, 0]
    data = data.astype(np.float32)
    max_v = np.max(np.abs(data))
    if max_v > 0: data /= max_v

    # Slice a 3-second segment for clear visual inspection
    start_sec = 20.0
    end_sec = 23.0
    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)

    if end_idx > len(data):
        start_sec = 0.0
        end_sec = min(3.0, len(data)/sr)
        start_idx = 0
        end_idx = int(end_sec * sr)

    t_axis = np.linspace(start_sec, end_sec, end_idx - start_idx)
    raw_segment = data[start_idx:end_idx]

    # Compute Hilbert Transform Envelope
    analytic_segment = hilbert(raw_segment)
    envelope_segment = np.abs(analytic_segment)

    # Create Before vs After Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # --- Top Subplot: BEFORE (Raw Waveform) ---
    ax1.plot(t_axis, raw_segment, color="#3498db", linewidth=0.8, alpha=0.85, label="Raw Audio Signal (BEFORE)")
    ax1.set_title("BEFORE: Raw Audio Signal with High-Frequency Oscillations & Background Noise", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Amplitude", fontsize=10, fontweight="bold")
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # --- Bottom Subplot: AFTER (Hilbert Envelope Extracted) ---
    ax2.plot(t_axis, raw_segment, color="#b0bec5", linewidth=0.6, alpha=0.4, label="Raw Background Noise")
    ax2.plot(t_axis, envelope_segment, color="#e74c3c", linewidth=2.2, label="Hilbert Envelope (AFTER)")
    ax2.plot(t_axis, -envelope_segment, color="#e74c3c", linewidth=2.2, linestyle="--", alpha=0.7)

    ax2.set_title("AFTER: Extracted Hilbert Envelope (Outer Peak Contour minus Noise)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Time (Seconds)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Envelope Amplitude", fontsize=10, fontweight="bold")
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()

    out_png = os.path.join(FIGURES_DIR, "hilbert_envelope_before_after.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()

    print(f" [SUCCESS] Saved Hilbert Before-After Figure: {out_png}")
    print("=" * 90)


if __name__ == "__main__":
    plot_before_after_hilbert()
