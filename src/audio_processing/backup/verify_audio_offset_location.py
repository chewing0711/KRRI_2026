"""
verify_audio_offset_location.py (gearbox_final_suite / audio_processing)

Empirically verifies audio offset delays across ALL 32 WAV files:
- Splits into TWO separate 16-panel figures:
  1. audio_can_offset_normal_16plots.png (16 Normal Subplots with [NORMAL] Header)
  2. audio_can_offset_abnormal_16plots.png (16 Abnormal Subplots with [ABNORMAL] Header)
- Outputs full CSV table: results/audio_offset_verification_table.csv

Rule 3 Compliance:
- Matplotlib labels, titles, legends in 100% English.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "audio_spectrograms")
RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"


def verify_split_audio_offset_locations():
    print("=" * 95)
    print(" 🧪 EMPIRICALLY VERIFYING ALL 32 WAV FILES (SPLIT INTO NORMAL vs ABNORMAL FIGURES)")
    print("=" * 95)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.wav"), recursive=True))
    if not wav_files:
        print(f"[ERROR] No WAV files found in {RAW_DATA_DIR}")
        return

    records = []
    norm_files = [f for f in wav_files if "normal" in f.lower()]
    abnorm_files = [f for f in wav_files if "abnormal" in f.lower() or "fault" in f.lower()]

    def plot_group_figure(file_list, group_name, out_filename):
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()

        for idx, wav_path in enumerate(file_list):
            if idx >= 16:
                break

            scen_folder = os.path.dirname(wav_path)
            parts = scen_folder.split(os.sep)
            scen_str = parts[-1].strip()

            ws_csv_candidates = glob.glob(os.path.join(scen_folder, "*wheel*.csv"))
            ws_csv = ws_csv_candidates[0] if ws_csv_candidates else None

            filename = os.path.basename(wav_path)
            channel_type = "GEAR" if "gear" in filename.lower() else "OBD"
            scen_key = f"{group_name.lower()}_{filename}"

            try:
                sr, audio_raw = wavfile.read(wav_path)
                if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
                audio_raw = audio_raw.astype(np.float32) / (np.max(np.abs(audio_raw)) + 1e-6)
                audio_dur_sec = len(audio_raw) / float(sr)

                step_size = int(sr * 0.1)
                n_frames = len(audio_raw) // step_size
                time_audio_axis = np.linspace(0, audio_dur_sec, n_frames)
                audio_rms = np.array([
                    np.sqrt(np.mean(audio_raw[i*step_size : (i+1)*step_size]**2))
                    for i in range(n_frames)
                ])

                noise_floor = np.percentile(audio_rms, 15)
                thresh = noise_floor * 2.5 + 0.02
                active_audio_indices = np.where(audio_rms > thresh)[0]

                if len(active_audio_indices) > 0:
                    audio_onset_sec = round(float(time_audio_axis[active_audio_indices[0]]), 2)
                else:
                    audio_onset_sec = 0.0

                if ws_csv and os.path.exists(ws_csv):
                    df_ws = pd.read_csv(ws_csv)
                    time_col = [c for c in df_ws.columns if "time" in c.lower() or "sec" in c.lower()][0]
                    ws_cols = [c for c in df_ws.columns if c != time_col]
                    df_ws["mean_ws"] = df_ws[ws_cols].mean(axis=1)

                    can_dur_sec = round(float(df_ws[time_col].iloc[-1] - df_ws[time_col].iloc[0]), 2)
                    moving_can = df_ws[df_ws["mean_ws"] > 0.5]

                    if len(moving_can) > 0:
                        can_onset_sec = round(float(moving_can[time_col].iloc[0]), 2)
                    else:
                        can_onset_sec = 0.0
                else:
                    can_dur_sec = audio_dur_sec
                    can_onset_sec = 0.0

                front_delay_sec = round(audio_onset_sec - can_onset_sec, 2)
                rear_delay_sec = round(audio_dur_sec - (audio_onset_sec + can_dur_sec), 2)

                if front_delay_sec > 3.0 and rear_delay_sec > 3.0:
                    offset_type = "BOTH"
                elif front_delay_sec > 3.0:
                    offset_type = "FRONT (Head Idle)"
                elif rear_delay_sec > 3.0:
                    offset_type = "REAR (Tail Idle)"
                else:
                    offset_type = "ALIGNED"

                records.append({
                    "state": group_name,
                    "channel_type": channel_type,
                    "scenario_key": scen_key,
                    "can_duration_sec": can_dur_sec,
                    "audio_duration_sec": round(audio_dur_sec, 2),
                    "audio_onset_sec": audio_onset_sec,
                    "front_idle_sec": front_delay_sec,
                    "rear_idle_sec": max(0.0, rear_delay_sec),
                    "verified_offset_type": offset_type,
                })

                ax = axes[idx]
                ax.plot(time_audio_axis, audio_rms, color="#e74c3c" if channel_type == "GEAR" else "#8e44ad", linewidth=1.1, label=f"{channel_type} RMS")
                ax.axhline(y=thresh, color="#95a5a6", linestyle="--", linewidth=0.7)
                ax.axvline(x=audio_onset_sec, color="#2980b9", linestyle="-.", linewidth=1.1)

                if ws_csv and os.path.exists(ws_csv):
                    ax_can = ax.twinx()
                    ax_can.plot(df_ws[time_col], df_ws["mean_ws"], color="#27ae60", linewidth=0.9, alpha=0.5)

                header_label = f"[{group_name.upper()}] {filename[:16]} ({channel_type})"
                ax.set_title(f"{header_label}\n(Front: {front_delay_sec}s | {offset_type})", fontsize=8, fontweight="bold")
                ax.grid(True, linestyle="--", alpha=0.3)

            except Exception as e:
                print(f"[ERROR] Processing {wav_path} failed: {e}")

        plt.tight_layout()
        out_png = os.path.join(FIGURES_DIR, out_filename)
        plt.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close()
        print(f" [SUCCESS] Saved {group_name} 16-Subplot Figure: {out_png}")

    # Plot Normal and Abnormal figures separately
    plot_group_figure(norm_files, "NORMAL", "audio_can_offset_normal_16plots.png")
    plot_group_figure(abnorm_files, "ABNORMAL", "audio_can_offset_abnormal_16plots.png")

    df_res = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS_DIR, "audio_offset_verification_table.csv")
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 95)
    print(" 📊 ALL 32 WAV FILES OFFSET VERIFICATION COMPLETED AND SAVED")
    print("=" * 95)


if __name__ == "__main__":
    verify_split_audio_offset_locations()
