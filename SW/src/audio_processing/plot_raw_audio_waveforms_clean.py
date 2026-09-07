"""
plot_raw_audio_waveforms_clean.py (gearbox_final_suite / audio_processing)

순수 원본 WAV 파일의 시간-진폭 파형(Raw Waveform)을 4x2 그리드로 시각화하는 스크립트.
- Calibration, 시간 축 오프셋 보정, CAN 동기화 지표 등을 전부 배제하고 원본 오디오 파형만 플롯.
- 서브플롯 헤더(타이틀)에는 부가적인 메트릭 텍스트 없이 오직 시나리오 이름만 표기.

규정 준수:
- Rule 3: 주석 100% 한글, Matplotlib 라벨/타이틀 100% 영문(English).
- Rule 7: LaTeX 문법 사용 금지.
- Rule 8: 이모지 사용 금지.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 디렉터리 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))

CANDIDATE_PATHS = [
    os.path.join(PROJECT_ROOT, "data", "gearbox_fault_diagnosis_program", "gearbox_fault_diagnosis_program", "data"),
    os.path.join(SUITE_DIR, "data", "gearbox_fault_diagnosis_program", "gearbox_fault_diagnosis_program", "data"),
    "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data",
    r"C:\Users\kante\Documents\KWU\etrl_task\data\gearbox_fault_diagnosis_program\gearbox_fault_diagnosis_program\data"
]

RAW_DATA_DIR = None
for path in CANDIDATE_PATHS:
    if os.path.isdir(path):
        RAW_DATA_DIR = path
        break

if RAW_DATA_DIR is None:
    RAW_DATA_DIR = CANDIDATE_PATHS[0]

os.makedirs(FIG_DIR, exist_ok=True)

# 시나리오 정의: (식별 키워드 리스트, 서브플롯 타이틀)
SCENARIOS = [
    (["고속주회로", "20kph"], "High Speed 20kph"),
    (["고속주회로", "60kph"], "High Speed 60kph"),
    (["고속주회로", "80kph"], "High Speed 80kph"),
    (["등판로", "6도"], "Hills 6deg Grade"),
    (["등판로", "12도"], "Hills 12deg Grade"),
    (["등판로", "18도"], "Hills 18deg Grade"),
    (["등판로", "30도"], "Hills 30deg Grade"),
    (["원선회로"], "Steering Pad 40kph"),
]

# 스타일 설정
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def find_scenario_folder(state_dir, keywords):
    if not os.path.isdir(state_dir):
        return None
    for folder in os.listdir(state_dir):
        folder_path = os.path.join(state_dir, folder)
        if os.path.isdir(folder_path):
            if all(kw in folder for kw in keywords):
                return folder_path
    return None


def plot_clean_waveforms_grid(target_state="normal", channel_keyword="gear-1"):
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), dpi=200)
    axes_flat = axes.flatten()

    state_dir = os.path.join(RAW_DATA_DIR, target_state)

    for idx, (keywords, scen_title) in enumerate(SCENARIOS):
        ax = axes_flat[idx]
        scen_folder_path = find_scenario_folder(state_dir, keywords)

        wav_files = []
        if scen_folder_path and os.path.isdir(scen_folder_path):
            wav_files = glob.glob(os.path.join(scen_folder_path, f"*{channel_keyword}*.wav"))

        if not wav_files:
            ax.set_title(scen_title, fontsize=11, fontweight="bold", pad=6)
            ax.text(0.5, 0.5, "File Not Found", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
            continue

        wav_path = wav_files[0]
        sr, data = wavfile.read(wav_path)

        # 다채널인 경우 1채널(Mono)로 추출
        if data.ndim > 1:
            data = data[:, 0]

        data = data.astype(np.float32)

        # 시간 축 계산 (초 단위)
        duration_sec = len(data) / float(sr)
        time_arr = np.linspace(0.0, duration_sec, num=len(data))

        # 렌더링 부하 방지를 위한 다운샘플링 (최대 50,000 샘플)
        step = max(1, len(data) // 50000)
        time_sub = time_arr[::step]
        data_sub = data[::step]

        plot_color = "#1f77b4" if target_state == "normal" else "#d62728"
        ax.plot(time_sub, data_sub, color=plot_color, linewidth=0.6, alpha=0.85)

        # 헤더에는 오직 시나리오 이름만 설정 (불필요한 부가 텍스트 제거)
        ax.set_title(scen_title, fontsize=11, fontweight="bold", pad=6)
        ax.set_ylabel("Amplitude", fontsize=9, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

        if idx in [6, 7]:
            ax.set_xlabel("Time (Seconds)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, f"raw_audio_waveform_{target_state}_{channel_keyword.replace('-', '')}.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    plot_clean_waveforms_grid("normal", "gear-1")
    plot_clean_waveforms_grid("abnormal", "gear-1")
    plot_clean_waveforms_grid("normal", "obd-2")
    plot_clean_waveforms_grid("abnormal", "obd-2")


if __name__ == "__main__":
    main()
