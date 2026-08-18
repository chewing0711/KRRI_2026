"""
organize_audio_files.py (gearbox_final_suite / audio_processing)

Organizes raw WAV audio files into data/audio/ structure:
1. Destination directory: /mnt/c/Users/kante/Documents/KWU/etrl_task/data/audio/
2. Subfolders: data/audio/normal/ and data/audio/abnormal/
3. Standardized filenames: e.g., high_speed_20kph_gear1.wav, hills_6_degrees_gear1.wav, etc.

Rule 3 Compliance:
- Code comments & terminal output in Korean.
"""

import os
import glob
import shutil
import re

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
EXPERIMENTS_DIR = os.path.dirname(SUITE_DIR)
TASK_DIR = os.path.dirname(EXPERIMENTS_DIR)
DATA_DIR = os.path.join(TASK_DIR, "data")
AUDIO_TARGET_DIR = os.path.join(DATA_DIR, "audio")

RAW_PROGRAM_DIR = os.path.join(DATA_DIR, "gearbox_fault_diagnosis_program", "gearbox_fault_diagnosis_program", "data")


def sanitize_name(name):
    s = str(name).strip().lower()
    if "20kph" in s and "high speed" in s: return "high_speed_20kph"
    if "60kph" in s and "high speed" in s: return "high_speed_60kph"
    if "80kph" in s and "high speed" in s: return "high_speed_80kph"
    if "6도" in s or "6%" in s: return "hills_6_degrees"
    if "12도" in s or "12%" in s: return "hills_12_degrees"
    if "18도" in s or "18%" in s: return "hills_18_degrees"
    if "30도" in s or "30%" in s: return "hills_30_degrees"
    if "원선회" in s or "steer" in s: return "steering_pad"
    return re.sub(r"[^\w\-]", "_", s)


def organize_audio_files():
    print("=" * 80)
    print(" 🛠 ORGANIZING AUDIO WAV FILES INTO DATA/AUDIO/ DIRECTORY")
    print("=" * 80)

    if not os.path.exists(RAW_PROGRAM_DIR):
        print(f"[ERROR] Raw program data dir missing: {RAW_PROGRAM_DIR}")
        return

    os.makedirs(AUDIO_TARGET_DIR, exist_ok=True)
    os.makedirs(os.path.join(AUDIO_TARGET_DIR, "normal"), exist_ok=True)
    os.makedirs(os.path.join(AUDIO_TARGET_DIR, "abnormal"), exist_ok=True)

    audio_files = glob.glob(os.path.join(RAW_PROGRAM_DIR, "*", "*", "*.wav"))

    copied_count = 0

    for filepath in audio_files:
        parts = filepath.split(os.sep)
        state = parts[-3].strip().lower()
        scenario_folder = parts[-2].strip().lower()

        state_clean = "abnormal" if "abnormal" in state or "fault" in state else "normal"
        scen_clean = sanitize_name(scenario_folder)

        wav_basename = os.path.basename(filepath)
        channel_tag = "gear1" if "gear-1" in wav_basename else ("obd2" if "obd-2" in wav_basename else "audio")

        target_filename = f"{scen_clean}_{channel_tag}.wav"
        target_path = os.path.join(AUDIO_TARGET_DIR, state_clean, target_filename)

        shutil.copy2(filepath, target_path)
        copied_count += 1

    print("\n" + "=" * 80)
    print(" 📊 AUDIO ORGANIZATION SUMMARY")
    print("=" * 80)
    print(f" - 총 복사 및 정돈된 WAV 파일 수 : {copied_count}개 파일")
    print(f" - 정상 (normal) 폴더 파일 수   : {len(os.listdir(os.path.join(AUDIO_TARGET_DIR, 'normal')))}개")
    print(f" - 고장 (abnormal) 폴더 파일 수 : {len(os.listdir(os.path.join(AUDIO_TARGET_DIR, 'abnormal')))}개")
    print(f" - 대상 경로                   : {AUDIO_TARGET_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    organize_audio_files()
