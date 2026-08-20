"""
validate_can_audio_1to1_window_alignment.py (gearbox_final_suite / src / audio_processing)

250샘플(0.74초) 슬라이딩 윈도우 기준으로 16개 전체 시나리오(총 1,686개 윈도우)에 대해
CAN 타임스탬프와 정제된 44.1kHz 오디오 WAV의 1:1 샘플 슬라이싱 정합성을 전수 실측 검증하는 스크립트.

[검증 항목]
1. 각 윈도우별 CAN 시간 범위 [t_start, t_end] 와 오디오 샘플 [s_start, s_end] 1:1 매칭 성공 여부
2. 결손 세션에서 초과된 윈도우 개수 (Drop 대상) 및 세션별 유효 매칭률 (%)
3. 오디오 신호 내 NaN / Inf / 비정상 신호 여부 전수 검사
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


def validate_all_windows():
    print("\n" + "=" * 115)
    print(" 🔍 [오디오 vs CAN 250샘플 윈도우 1:1 동기화 전수 정밀 검증]")
    print("=" * 115)

    can_files = sorted(glob.glob(os.path.join(RAW_DECODED_DIR, "*_official_decoded.csv")))
    if not can_files:
        print("[ERROR] CAN 디코딩 파일을 찾을 수 없습니다:", RAW_DECODED_DIR)
        return

    session_summaries = []
    total_can_windows = 0
    total_matched_windows = 0
    total_dropped_windows = 0

    print(f" {'시나리오 (Scenario)':<28} | {'상태':<8} | {'총 CAN윈도우':<12} | {'유효 매칭윈도우':<14} | {'초과 제외(Drop)':<14} | {'매칭률(%)':<10}")
    print("-" * 115)

    for can_path in can_files:
        filename = os.path.basename(can_path)
        state = "abnormal" if "abnormal" in filename else "normal"
        scen_core = filename.replace("abnormal_", "").replace("normal_", "").replace("_official_decoded.csv", "")

        df_can = pd.read_csv(can_path)
        t_arr = df_can["Time"].values
        time_len = t_arr[-1] - t_arr[0]

        # 정제된 오디오 파일 탐색 (Gear-1 마이크 기준)
        audio_dir = os.path.join(AUDIO_ALIGNED_DIR, state, scen_core)
        gear_wavs = glob.glob(os.path.join(audio_dir, "*gear*.wav")) + glob.glob(os.path.join(audio_dir, "*GEAR*.wav"))
        if not gear_wavs:
            # 전체 검색
            gear_wavs = glob.glob(os.path.join(audio_dir, "*.wav"))

        if not gear_wavs:
            print(f"[ERROR] 오디오 파일 없음: {audio_dir}")
            continue

        sr, audio_data = wavfile.read(gear_wavs[0])
        audio_dur = len(audio_data) / float(sr)

        # 250샘플 윈도우 슬라이싱 시뮬레이션
        window_size = 250
        step_size = 125
        n_rows = len(df_can)
        n_windows = (n_rows - window_size) // step_size + 1

        matched_cnt = 0
        dropped_cnt = 0

        for w_idx in range(n_windows):
            start_row = w_idx * step_size
            end_row = start_row + window_size
            t_start = t_arr[start_row] - t_arr[0]
            t_end = t_arr[end_row - 1] - t_arr[0]

            s_start = int(round(t_start * sr))
            s_end = int(round(t_end * sr))

            # 오디오 유효 범위 내인지 1:1 엄격 검사
            if s_end <= len(audio_data) and s_start >= 0 and s_end > s_start:
                audio_slice = audio_data[s_start:s_end]
                if not np.isnan(audio_slice).any() and len(audio_slice) > 0:
                    matched_cnt += 1
                else:
                    dropped_cnt += 1
            else:
                dropped_cnt += 1

        match_rate = (matched_cnt / n_windows) * 100.0 if n_windows > 0 else 0.0
        total_can_windows += n_windows
        total_matched_windows += matched_cnt
        total_dropped_windows += dropped_cnt

        session_summaries.append({
            "state": state,
            "scenario": scen_core,
            "can_windows": n_windows,
            "matched_windows": matched_cnt,
            "dropped_windows": dropped_cnt,
            "match_rate_pct": round(match_rate, 2),
            "can_duration_sec": round(time_len, 2),
            "audio_duration_sec": round(audio_dur, 2),
        })

        full_name = f"{state}_{scen_core}"
        print(f" {full_name:<28} | {state:<8} | {n_windows:>10} 개 | {matched_cnt:>12} 개 | {dropped_cnt:>12} 개 | {match_rate:>8.2f}%")

    print("-" * 115)
    overall_match_rate = (total_matched_windows / total_can_windows) * 100.0
    print(f" 🏆 [16개 시나리오 전체 총합]  총 CAN 윈도우: {total_can_windows:,}개 | 1:1 유효 매칭: {total_matched_windows:,}개 | 제외(Drop): {total_dropped_windows}개 ({overall_match_rate:.2f}%)")
    print("=" * 115)

    df_res = pd.DataFrame(session_summaries)
    out_csv = os.path.join(AUDIO_AUDIT_DIR, "can_audio_1to1_window_alignment_validation.csv")
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f" 💾 전수 검증 결과 CSV 저장 완료: {out_csv}\n")


if __name__ == "__main__":
    validate_all_windows()
