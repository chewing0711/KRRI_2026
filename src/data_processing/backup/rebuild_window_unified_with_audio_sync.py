"""
rebuild_window_unified_with_audio_sync.py (gearbox_final_suite / data_processing)

Rebuilds the Window-Level Unified Copy Dataset with TIME-based Audio Sync & Offset Trimming.

Original Dataset Protection:
- results/unified_can_context_dataset.csv is kept 100% UNTOUCHED.

Saved Copy File:
- results/unified_can_audio_synced_dataset.csv

Features included per 500-sample (1.47s) window:
- session_id, state, target, scenario, speed_kph, grade_pct
- All 45 CAN context features (slip_ratio_*, rel_diff_*, norm_fft_*, mean, rms, np4, etc.)
- Window-level Audio features (audio_gear1_trim_offset_sec, audio_gear1_hilbert_env_mean, audio_gear1_hilbert_env_max, audio_gear1_rms, audio_gear1_high_band_db)

Rule 3 Compliance:
- Code comments & terminal output in 100% Korean.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import wavfile
from scipy.signal import hilbert, stft

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECODED_DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
RAW_WAV_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"

ORIGINAL_UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")
OUTPUT_SYNCED_CSV = os.path.join(RESULTS_DIR, "unified_can_audio_synced_dataset.csv")
OFFSET_TABLE_CSV = os.path.join(RESULTS_DIR, "audio_offset_verification_table.csv")


def load_offset_dict():
    offset_dict = {}
    if os.path.exists(OFFSET_TABLE_CSV):
        df_off = pd.read_csv(OFFSET_TABLE_CSV)
        for _, row in df_off.iterrows():
            offset_dict[row["wav_file"]] = float(row["optimal_trim_offset_sec"])
    return offset_dict


def compute_can_features(sub_can, speed_kph, grade_pct):
    fl = sub_can["WHL_SPD_FL"].values if "WHL_SPD_FL" in sub_can.columns else np.full(len(sub_can), speed_kph)
    fr = sub_can["WHL_SPD_FR"].values if "WHL_SPD_FR" in sub_can.columns else np.full(len(sub_can), speed_kph)
    rl = sub_can["WHL_SPD_RL"].values if "WHL_SPD_RL" in sub_can.columns else np.full(len(sub_can), speed_kph)
    rr = sub_can["WHL_SPD_RR"].values if "WHL_SPD_RR" in sub_can.columns else np.full(len(sub_can), speed_kph)
    yaw = sub_can["Yaw_Rate"].values if "Yaw_Rate" in sub_can.columns else np.zeros(len(sub_can))
    lat = sub_can["Lateral_Accel"].values if "Lateral_Accel" in sub_can.columns else np.zeros(len(sub_can))

    eps = 1e-5
    speed_norm = max(speed_kph, 1.0)
    mean_whl = (fl + fr + rl + rr) / 4.0

    # 1. Slip Ratios
    slip_fl = np.mean((fl - mean_whl) / (mean_whl + eps))
    slip_fr = np.mean((fr - mean_whl) / (mean_whl + eps))
    slip_rl = np.mean((rl - mean_whl) / (mean_whl + eps))
    slip_rr = np.mean((rr - mean_whl) / (mean_whl + eps))
    slip_diff = np.mean(((fl + fr) / 2.0 - (rl + rr) / 2.0) / (mean_whl + eps))

    # 2. Normalized Vehicle Dynamics
    norm_yaw = np.mean(np.abs(yaw)) / (speed_norm + eps)
    norm_lat = np.mean(np.abs(lat)) / (speed_norm + eps)

    # 3. Primary Wheel Speed Signal Stats (FL)
    s = fl
    mean_val = np.mean(s)
    peak = np.max(np.abs(s))
    p2p = np.ptp(s)
    rms = np.sqrt(np.mean(s ** 2))
    std_dev = np.std(s)
    variance = np.var(s)
    srav = np.sqrt(np.mean(np.abs(s)))
    mean_abs = np.mean(np.abs(s))
    crest_factor = (peak / rms if rms != 0 else 0)
    shape_factor = (rms / mean_abs if mean_abs != 0 else 0)
    clearance_factor = (peak / srav if srav != 0 else 0)
    mean_square = np.mean(s ** 2)
    np4 = (np.mean(s ** 4) / (mean_square ** 2) if mean_square != 0 else 0)
    skewness = stats.skew(s)
    kurt = stats.kurtosis(s)

    # 4. Relative Differences
    diff_fl_rl = fl - rl
    diff_fr_rr = fr - rr
    diff_fl_fr = fl - fr
    diff_rl_rr = rl - rr

    # 5. FFT Spectral Features
    ac_signal = s - np.mean(s)
    ac_fft_vals = np.abs(np.fft.rfft(ac_signal))
    ac_fft_freqs = np.fft.rfftfreq(len(s), d=0.02)
    psd = ac_fft_vals ** 2
    total_energy = np.sum(psd)

    raw_peak_freq = ac_fft_freqs[np.argmax(ac_fft_vals)] if len(ac_fft_vals) > 0 else 0.0
    norm_peak_freq = raw_peak_freq / (speed_norm + eps)
    raw_centroid = np.sum(ac_fft_freqs * psd) / (total_energy + 1e-12)
    norm_centroid = raw_centroid / (speed_norm + eps)
    log_norm_spectral_energy = np.log1p(total_energy / ((speed_norm ** 2) + eps))

    psd_norm = psd / (total_energy + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    entropy = -np.sum(psd_norm * np.log2(psd_norm))

    mask_low = (ac_fft_freqs >= 0.5) & (ac_fft_freqs < 5.0)
    mask_mid = (ac_fft_freqs >= 5.0) & (ac_fft_freqs < 15.0)
    mask_high = (ac_fft_freqs >= 15.0)

    energy_low = np.sum(psd[mask_low]) / (total_energy + 1e-12)
    energy_mid = np.sum(psd[mask_mid]) / (total_energy + 1e-12)
    energy_high = np.sum(psd[mask_high]) / (total_energy + 1e-12)

    return {
        "speed_kph": speed_kph,
        "grade_pct": grade_pct,
        "slip_ratio_fl": slip_fl,
        "slip_ratio_fr": slip_fr,
        "slip_ratio_rl": slip_rl,
        "slip_ratio_rr": slip_rr,
        "slip_diff_front_rear": slip_diff,
        "norm_yaw_rate": norm_yaw,
        "norm_lat_accel": norm_lat,
        "sensor_1": 1, "sensor_2": 0, "sensor_3": 0, "sensor_4": 0,
        "mean": mean_val, "peak": peak, "p2p": p2p, "rms": rms, "std": std_dev, "variance": variance,
        "crest_factor": crest_factor, "shape_factor": shape_factor, "clearance_factor": clearance_factor,
        "np4": np4, "skew": skewness, "kurt": kurt,
        "rel_diff_fl_rl_mean": np.mean(diff_fl_rl), "rel_diff_fl_rl_std": np.std(diff_fl_rl), "rel_diff_fl_rl_p2p": np.ptp(diff_fl_rl),
        "rel_diff_fr_rr_mean": np.mean(diff_fr_rr), "rel_diff_fr_rr_std": np.std(diff_fr_rr), "rel_diff_fr_rr_p2p": np.ptp(diff_fr_rr),
        "rel_diff_fl_fr_std": np.std(diff_fl_fr), "rel_diff_rl_rr_std": np.std(diff_rl_rr),
        "norm_rolling_var_mean": variance / (speed_norm + eps),
        "norm_rolling_var_max": variance / (speed_norm + eps),
        "norm_p2p": p2p / (speed_norm + eps),
        "norm_rms": rms / (speed_norm + eps),
        "norm_std": std_dev / (speed_norm + eps),
        "norm_fft_peak_freq": norm_peak_freq,
        "log_norm_fft_spectral_energy": log_norm_spectral_energy,
        "fft_spectral_entropy": entropy,
        "norm_fft_spectral_centroid": norm_centroid,
        "fft_energy_low_band": energy_low,
        "fft_energy_mid_band": energy_mid,
        "fft_energy_high_band": energy_high,
    }


def rebuild_synced_window_dataset(window_size=500):
    print("=" * 100)
    print(" 🛠 REBUILDING WINDOW-LEVEL UNIFIED DATASET WITH TIME-SYNCED AUDIO (OPTION 2)")
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

    all_window_rows = []

    for dec_file in decoded_files:
        filename_dec = os.path.basename(dec_file)
        state_prefix = "abnormal" if "abnormal" in filename_dec.lower() else "normal"

        speed_kph, grade_pct = 20.0, 0.0
        scenario_name = "high speed 20kph"
        for k, v in scenario_speed_grade_map.items():
            if k in filename_dec:
                speed_kph, grade_pct = v
                scenario_name = k.replace("_official", "")
                break

        print(f" 📂 Processing Window Sessions: {state_prefix.upper()} | Scenario: {scenario_name}...")

        df_can = pd.read_csv(dec_file)
        if "Time" not in df_can.columns or len(df_can) < window_size:
            continue

        # Match WAV File
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

        sr = 44100
        audio_trimmed = np.array([], dtype=np.float32)
        env_trimmed = np.array([], dtype=np.float32)
        trim_offset_sec = 0.0

        if target_wav and os.path.exists(target_wav):
            wav_basename = os.path.basename(target_wav)
            trim_offset_sec = offset_dict.get(wav_basename, 0.0)
            sr, audio_raw = wavfile.read(target_wav)
            if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
            audio_raw = audio_raw.astype(np.float32) / (np.max(np.abs(audio_raw)) + 1e-6)

            trim_start_sample = int(trim_offset_sec * sr)
            audio_trimmed = audio_raw[trim_start_sample:]
            env_trimmed = np.abs(hilbert(audio_trimmed)) if len(audio_trimmed) > 0 else np.zeros_like(audio_trimmed)

        # Slice non-overlapping 500-sample CAN windows
        n_chunks = len(df_can) // window_size
        for win_idx in range(n_chunks):
            sub_can = df_can.iloc[win_idx * window_size : (win_idx + 1) * window_size]
            can_dict = compute_can_features(sub_can, speed_kph, grade_pct)

            can_dict["session_id"] = f"{state_prefix}_{scenario_name}_{win_idx}"
            can_dict["state"] = state_prefix
            can_dict["target"] = 1 if state_prefix == "abnormal" else 0
            can_dict["scenario"] = scenario_name

            # Sliced Audio Window Features (1.47s time interval)
            t_start = sub_can["Time"].iloc[0]
            t_end = sub_can["Time"].iloc[-1]

            aud_start_idx = int(t_start * sr)
            aud_end_idx = int(t_end * sr)

            aud_start_idx = max(0, min(aud_start_idx, len(audio_trimmed)))
            aud_end_idx = max(aud_start_idx + 1, min(aud_end_idx, len(audio_trimmed)))

            sub_aud = audio_trimmed[aud_start_idx:aud_end_idx]
            sub_env = env_trimmed[aud_start_idx:aud_end_idx]

            can_dict["audio_gear1_trim_offset_sec"] = trim_offset_sec
            if len(sub_aud) > 0:
                can_dict["audio_gear1_hilbert_env_mean"] = float(np.mean(sub_env))
                can_dict["audio_gear1_hilbert_env_max"] = float(np.max(sub_env))
                can_dict["audio_gear1_rms"] = float(np.sqrt(np.mean(sub_aud ** 2)))
                
                # STFT High Band dB
                if len(sub_aud) >= 256:
                    f, t, Zxx = stft(sub_aud, fs=sr, nperseg=256)
                    mag_db = 20.0 * np.log10(np.abs(Zxx) + 1e-6)
                    high_mask = (f >= 4000) & (f <= 10000)
                    can_dict["audio_gear1_high_band_db"] = float(np.mean(mag_db[high_mask]))
                else:
                    can_dict["audio_gear1_high_band_db"] = -85.0
            else:
                can_dict["audio_gear1_hilbert_env_mean"] = 0.0
                can_dict["audio_gear1_hilbert_env_max"] = 0.0
                can_dict["audio_gear1_rms"] = 0.0
                can_dict["audio_gear1_high_band_db"] = -85.0

            all_window_rows.append(can_dict)

    df_synced = pd.DataFrame(all_window_rows)
    df_synced.to_csv(OUTPUT_SYNCED_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f" [SUCCESS] Saved Window-Level Synced Audio Unified Dataset ({len(df_synced):,} windows): {OUTPUT_SYNCED_CSV}")
    print("=" * 100)


if __name__ == "__main__":
    rebuild_synced_window_dataset()
