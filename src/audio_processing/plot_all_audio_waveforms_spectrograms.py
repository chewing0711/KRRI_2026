"""
plot_all_audio_waveforms_spectrograms.py (gearbox_final_suite / audio_processing)

Ultra-Lightweight RAM-Safe Waveform & Spectrogram Plotting FOR ALL AUDIO CHANNELS (gear-1 and obd-2):
1. Reads directly from raw original program data directory.
2. Plots both gear-1 (Gearbox Channel) and obd-2 (OBD/Cabin Channel) audio files.
3. Saves high-res PNG plots in results/figures/audio_spectrograms/.

Rule 3 Compliance:
- Comments/Output in Korean.
- Matplotlib labels, titles, legends strictly in English.
"""

import os
import glob
import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "audio_spectrograms")

RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"


def sanitize_filename(name):
    return str(name).strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")


def plot_all_audio_visualizations():
    print("=" * 90)
    print(" 🎨 PLOTTING WAVEFORMS & SPECTROGRAMS FOR ALL AUDIO CHANNELS (GEAR-1 & OBD-2)")
    print("=" * 90)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Search ALL .wav files (both gear-1 and obd-2 channels)
    audio_files = glob.glob(os.path.join(RAW_DATA_DIR, "*", "*", "*.wav"))

    if not audio_files:
        print(f"[ERROR] No WAV audio files found in: {RAW_DATA_DIR}")
        return

    plotted_count = 0

    for filepath in audio_files:
        parts = filepath.split(os.sep)
        state = parts[-3].strip().lower()
        scenario_folder = parts[-2].strip().lower()

        state_clean = "abnormal" if "abnormal" in state or "fault" in state else "normal"
        scen_clean = sanitize_filename(scenario_folder)

        wav_basename = os.path.basename(filepath).lower()
        channel_tag = "gear1" if "gear-1" in wav_basename else ("obd2" if "obd-2" in wav_basename or "obd2" in wav_basename else "ch")

        try:
            sr, data = wavfile.read(filepath)
            if data.ndim > 1:
                data = data[:, 0]

            data = data.astype(np.float32)
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data /= max_val

            # Light Waveform Downsampling (Max 20,000 points)
            step = max(1, len(data) // 20000)
            data_sub = data[::step]
            duration = len(data) / float(sr)
            time_arr = np.linspace(0, duration, num=len(data_sub))

            # Spectrogram (Light nperseg=512)
            f, t_spec, Sxx = spectrogram(data, fs=sr, nperseg=512, noverlap=256)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)

            # Limit Frequency Range to 12kHz
            max_f_idx = np.searchsorted(f, 12000)
            f_sub = f[:max_f_idx]
            Sxx_db_sub = Sxx_db[:max_f_idx, :]

            # Plotting Figure (Rule 3: 100% English labels and titles)
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            # Panel 1: Waveform
            axes[0].plot(time_arr, data_sub, color="#1f77b4" if state_clean == "normal" else "#d62728", alpha=0.8, linewidth=0.5)
            axes[0].set_ylabel("Amplitude", fontsize=10, fontweight="bold")
            axes[0].set_title(f"Audio Waveform ({channel_tag.upper()}) - State: {state_clean.upper()} | Scenario: {scen_clean}", fontsize=11, fontweight="bold")
            axes[0].grid(True, linestyle="--", alpha=0.4)

            # Panel 2: Fast imshow Spectrogram (Zero Memory Leak)
            im = axes[1].imshow(
                Sxx_db_sub,
                aspect="auto",
                origin="lower",
                extent=[0, duration, f_sub[0], f_sub[-1]],
                cmap="inferno"
            )
            axes[1].set_ylabel("Frequency (Hz)", fontsize=10, fontweight="bold")
            axes[1].set_xlabel("Time (Seconds)", fontsize=10, fontweight="bold")
            axes[1].set_title(f"STFT Power Spectrogram ({channel_tag.upper()})", fontsize=11, fontweight="bold")

            cbar = fig.colorbar(im, ax=axes[1], orientation="vertical", pad=0.02)
            cbar.set_label("Power (dB)", fontsize=9, fontweight="bold")

            plt.tight_layout()

            out_filename = f"audio_{state_clean}_{scen_clean}_{channel_tag}_waveform_spectrogram.png"
            out_path = os.path.join(FIGURES_DIR, out_filename)
            plt.savefig(out_path, dpi=120, bbox_inches="tight")

            plt.close(fig)
            plt.close('all')
            gc.collect()

            plotted_count += 1
            print(f" [PLOTTED] {out_filename} (Duration: {duration:.2f}s)")

        except Exception as e:
            print(f"[ERROR] Failed plotting {filepath}: {e}")

    print("\n" + "=" * 90)
    print(" 📊 VISUALIZATION GENERATION SUMMARY")
    print("=" * 90)
    print(f" - 총 생성된 Waveform & Spectrogram 그래프 수 : {plotted_count}개 PNG 파일 (Gear-1 + OBD-2 전체)")
    print(f" - 그래프 저장 폴더                           : {FIGURES_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    plot_all_audio_visualizations()
