"""
demo_raw_can_audio_sample_mapping.py (gearbox_final_suite / src / audio_processing)

실제 CAN Raw 데이터의 행별 가변 타임스탬프(Time)와 44.1kHz 오디오 파형이
어떻게 1:1로 시점 어긋남 없이 매핑되는지 10개 샘플 행으로 직접 시연하고 CSV로 출력하는 스크립트.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_ALIGNED_DIR = os.path.join(DATA_DIR, "audio_aligned")
AUDIO_AUDIT_DIR = os.path.join(RESULTS_DIR, "audio_audit")
os.makedirs(AUDIO_AUDIT_DIR, exist_ok=True)


def demo_sample_mapping():
    # 1. 실제 CAN 파일 및 정제된 오디오 파일 로드
    can_path = os.path.join(RAW_DECODED_DIR, "abnormal_high_speed_20kph_official_decoded.csv")
    audio_path = os.path.join(AUDIO_ALIGNED_DIR, "abnormal", "high_speed_20kph", "go_20-gear-1.wav")

    df_can = pd.read_csv(can_path)
    sr, audio_data = wavfile.read(audio_path)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # 정규화 (16-bit PCM -> -1.0 ~ +1.0)
    audio_norm = audio_data.astype(np.float32) / 32768.0

    print("\n" + "=" * 125)
    print(" 🔍 [실제 CAN 타임스탬프 기반 오디오 1:1 동적 시간 매핑 샘플 10개]")
    print(f" 📂 CAN 파일: {os.path.basename(can_path)} | 📂 오디오 파일: {os.path.basename(audio_path)} ({sr} Hz)")
    print("=" * 125)

    sample_rows = []
    t_arr = df_can["Time"].values

    print(f" {'행':<4} | {'CAN 타임스탬프':<16} | {'CAN 간격(dt)':<12} | {'오디오 샘플 범위':<22} | {'샘플수':<7} | {'Audio RMS (실효치)':<18} | {'Audio Peak (최대치)':<18}")
    print("-" * 125)

    for i in range(12):
        t_cur = t_arr[i] - t_arr[0]
        t_next = t_arr[i + 1] - t_arr[0]
        dt_ms = (t_next - t_cur) * 1000.0

        # 실제 시간(초)에 해당하는 오디오 인덱스 계산
        s_start = int(round(t_cur * sr))
        s_end = int(round(t_next * sr))

        audio_chunk = audio_norm[s_start:s_end]
        if len(audio_chunk) > 0:
            rms_val = float(np.sqrt(np.mean(audio_chunk ** 2)))
            peak_val = float(np.max(np.abs(audio_chunk)))
            n_samples = len(audio_chunk)
        else:
            rms_val = 0.0
            peak_val = 0.0
            n_samples = 0

        sample_rows.append({
            "CAN_Row_Idx": i,
            "CAN_Time_sec": round(t_cur, 5),
            "CAN_Delta_ms": round(dt_ms, 3),
            "WHL_SPD_FL_kph": df_can.loc[i, "WHL_SPD_FL"],
            "Audio_Sample_Start": s_start,
            "Audio_Sample_End": s_end,
            "Audio_Sample_Count": n_samples,
            "Audio_RMS_Energy": round(rms_val, 6),
            "Audio_Peak_Amplitude": round(peak_val, 6),
        })

        sample_range_str = f"[{s_start} ~ {s_end}]"
        print(f" {i:<4} | {t_cur:>10.5f} 초    | {dt_ms:>8.3f} ms   | {sample_range_str:<22} | {n_samples:>5} 개 | {rms_val:>16.6f}   | {peak_val:>16.6f}")

    print("=" * 125)

    df_sample = pd.DataFrame(sample_rows)
    out_csv = os.path.join(AUDIO_AUDIT_DIR, "sample_raw_can_audio_mapped_rows.csv")
    df_sample.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f" 💾 1:1 매핑 샘플 CSV 저장 완료: {out_csv}\n")


if __name__ == "__main__":
    demo_sample_mapping()
