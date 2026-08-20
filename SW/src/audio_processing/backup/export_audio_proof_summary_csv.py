"""
export_audio_proof_summary_csv.py (gearbox_final_suite / audio_processing)

Generates audio_proof_summary.csv containing the exact columns:
- 세션 구분 (Session Key)
- Sampling rate
- 샘플 수 (Samples)
- 재생 시간 (Duration)

Rule 3 Compliance:
- Markdown/CSV formatting in 100% Korean.
"""

import os
import glob
import pandas as pd
from scipy.io import wavfile

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
RAW_DATA_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"


def generate_audio_proof_csv():
    print("=" * 90)
    print(" 🛠 EXPORTING EXACT AUDIO PROOF SUMMARY CSV")
    print("=" * 90)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    wav_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*", "*", "*gear-1.wav")))

    if not wav_files:
        print(f"[ERROR] No gear-1 WAV files found in: {RAW_DATA_DIR}")
        return

    records = []

    for filepath in wav_files:
        parts = filepath.split(os.sep)
        state_folder = parts[-3].strip().lower()
        scen_folder = parts[-2].strip().lower()

        # Build Session Key
        state_key = "abnormal" if "abnormal" in state_folder or "fault" in state_folder else "normal"
        
        # Clean scenario name
        scen_clean = scen_folder
        if "(" in scen_clean and ")" in scen_clean:
            start_idx = scen_clean.find("(") + 1
            end_idx = scen_clean.find(")")
            scen_clean = scen_clean[start_idx:end_idx].strip()
        scen_clean = scen_clean.replace("high speed circut, one round", "high speed").replace("test hills", "hills").replace("% grade", " degrees").replace("steer pad, 1minute", "steering pad")

        session_key = f"{state_key}_{scen_clean}"

        try:
            sr, data = wavfile.read(filepath)
            n_samples = len(data)
            duration_sec = round(n_samples / float(sr), 2)

            records.append({
                "세션 구분 (Session Key)": session_key,
                "Sampling rate": f"{sr:,} Hz",
                "샘플 수 (Samples)": f"{n_samples:,}",
                "재생 시간 (Duration)": f"{duration_sec:.2f}초",
            })
        except Exception as e:
            print(f"[ERROR] Failed reading {filepath}: {e}")

    df = pd.DataFrame(records)

    out_csv = os.path.join(RESULTS_DIR, "audio_proof_summary.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f" [SUCCESS] Saved Audio Proof Summary CSV: {out_csv}")
    print("=" * 90)


if __name__ == "__main__":
    generate_audio_proof_csv()
