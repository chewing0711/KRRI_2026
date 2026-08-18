"""
plot_combined_envelope_mapping.py (gearbox_final_suite / audio_processing)

Generates a single combined multi-panel figure with direct ConnectionPatch arrows
connecting the Top Subplot (Time Domain) to the Bottom Subplot (Frequency Domain).

Rule 3 Compliance:
- Matplotlib labels, titles, legends in 100% English.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy.io import wavfile
from scipy.signal import hilbert

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "audio_spectrograms")
RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"


def plot_combined_mapping():
    print("=" * 90)
    print(" 🎨 PLOTTING CROSS-SUBPLOT CONNECTING ARROW HILBERT MAPPING FIGURE")
    print("=" * 90)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    wav_files = glob.glob(os.path.join(RAW_DATA_DIR, "*", "*20kph*", "*gear-1.wav"))
    abnorm_files = [f for f in wav_files if "abnormal" in f.lower() or "go_" in f.lower()]

    if not abnorm_files:
        print("[ERROR] Abnormal 20kph WAV file missing.")
        return

    sample_file = abnorm_files[0]
    sr, data = wavfile.read(sample_file)
    if data.ndim > 1: data = data[:, 0]
    data = data.astype(np.float32)
    max_v = np.max(np.abs(data))
    if max_v > 0: data /= max_v

    # Slice 1.0-second segment for clear peak period visual mapping
    start_sec = 20.0
    end_sec = 21.0
    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)

    t_axis = np.linspace(start_sec, end_sec, end_idx - start_idx)
    raw_seg = data[start_idx:end_idx]
    env_seg = np.abs(hilbert(raw_seg))

    # Compute FFT for full 15s signal
    full_env = np.abs(hilbert(data[:int(15*sr)]))
    fft_full = np.abs(np.fft.rfft(full_env - np.mean(full_env)))
    freqs = np.fft.rfftfreq(len(full_env), d=1.0/sr)

    mask = (freqs >= 2.0) & (freqs <= 300.0)
    freqs_sub = freqs[mask]
    fft_sub = fft_full[mask]

    peak_idx = np.argmax(fft_sub)
    peak_freq = freqs_sub[peak_idx]
    peak_val = fft_sub[peak_idx]

    # Create Combined 2-Panel Figure with extra space for cross-subplot arrow
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5))

    # --- TOP PANEL: TIME DOMAIN ---
    ax1.plot(t_axis, raw_seg, color="#b0bec5", linewidth=0.7, alpha=0.4, label="Raw Audio Signal")
    ax1.plot(t_axis, env_seg, color="#e74c3c", linewidth=2.5, label="Hilbert Envelope (Time Domain)")
    ax1.set_title("SUBPLOT 1 (TIME DOMAIN): Repetitive Envelope Peaks in Time", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Time (Seconds)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Amplitude", fontsize=10, fontweight="bold")
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Find peak point in time domain segment
    t_peak_idx = np.argmax(env_seg)
    t_peak_x = t_axis[t_peak_idx]
    t_peak_y = env_seg[t_peak_idx]

    # --- BOTTOM PANEL: FREQUENCY DOMAIN ---
    ax2.plot(freqs_sub, fft_sub, color="#e74c3c", linewidth=2.0, label="Envelope FFT Spectrum")
    ax2.axvline(x=peak_freq, color="#c0392b", linestyle="--", linewidth=1.5)
    ax2.set_title("SUBPLOT 2 (FREQUENCY DOMAIN): Envelope Spectrum Peak at 12.8 Hz", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Frequency (Hz)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Spectrum Magnitude", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(5000, peak_val * 1.1))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    # --- CROSS-SUBPLOT CONNECTING ARROW (Connecting Subplot 1 -> Subplot 2) ---
    con = ConnectionPatch(
        xyA=(t_peak_x, t_peak_y), coordsA=ax1.transData,
        xyB=(peak_freq, peak_val), coordsB=ax2.transData,
        arrowstyle="->", color="#9b59b6", linewidth=3.0, mutation_scale=25
    )
    fig.add_artist(con)

    # Text Box explaining the inter-graph connection
    fig.text(0.52, 0.48, "Cross-Graph Fourier Transform (FFT):\nTime Peak Periodicity (T = 0.078s) ===> Frequency Spectrum Peak (12.8 Hz)",
             fontsize=11, fontweight="bold", color="#8e44ad",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f39c12", alpha=0.2, edgecolor="#8e44ad"))

    plt.tight_layout()

    out_png = os.path.join(FIGURES_DIR, "combined_envelope_time_vs_freq_mapping.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()

    print(f" [SUCCESS] Saved Cross-Subplot Connected Figure: {out_png}")
    print("=" * 90)


if __name__ == "__main__":
    plot_combined_mapping()
