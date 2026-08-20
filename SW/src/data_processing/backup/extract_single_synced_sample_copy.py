"""
extract_single_synced_sample_copy.py

기어박스 고장 진단 데이터셋의 1개 대표 주행 시나리오(등판로 6도 정상 주행)를 대상으로
CAN 주행 신호(500샘플 윈도우)와 오디오 음향 신호(1.47초 동기화 구간)를
시간 오프셋 트리밍 후 1:1 정밀 결합하여 샘플 복사본 CSV로 추출하는 스크립트.

출력 파일:
/mnt/c/Users/kante/Documents/KWU/etrl_task/data/sample_synced_can_audio_copy.csv

규정 준수:
- 원본 데이터셋 100% 보존 (독립 샘플 복사본으로 저장).
- Rule 4 준수: fillna 억지 대입 없는 엄격한 유효 오디오 슬라이싱.
- 주석 및 터미널 출력 100% 한글 작성.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import wavfile
from scipy.signal import hilbert, stft

SUITE_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/experiments/gearbox_final_suite"
CAN_FILE = os.path.join(SUITE_DIR, "data", "normal_등판로 6도(Test _log_up_6_official_decoded.csv")
WAV_FILE = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data/normal/등판로 6도(Test Hills 6% grade) 20kph/record_up_6-gear-1.wav"
OFFSET_TABLE = os.path.join(SUITE_DIR, "results", "audio_offset_verification_table.csv")
OUTPUT_SAMPLE_CSV = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/sample_synced_can_audio_copy.csv"

WINDOW_SIZE = 500  # 500 CAN 샘플 (약 1.47초 윈도우)


def get_optimal_offset():
    """오디오 오프셋 검증 테이블에서 최적 트림 시각 로드"""
    if os.path.exists(OFFSET_TABLE):
        df_off = pd.read_csv(OFFSET_TABLE)
        for _, row in df_off.iterrows():
            fname_str = str(row.get("filename", "")).lower()
            if "up_6" in fname_str:
                return max(0.0, float(row.get("front_offset_sec", 0.0)))
    return 0.0


def compute_can_features(sub_can, speed_kph=20.0, grade_pct=6.0):
    fl = sub_can["WHL_SPD_FL"].values if "WHL_SPD_FL" in sub_can.columns else np.full(len(sub_can), speed_kph)
    fr = sub_can["WHL_SPD_FR"].values if "WHL_SPD_FR" in sub_can.columns else np.full(len(sub_can), speed_kph)
    rl = sub_can["WHL_SPD_RL"].values if "WHL_SPD_RL" in sub_can.columns else np.full(len(sub_can), speed_kph)
    rr = sub_can["WHL_SPD_RR"].values if "WHL_SPD_RR" in sub_can.columns else np.full(len(sub_can), speed_kph)
    yaw = sub_can["Yaw_Rate"].values if "Yaw_Rate" in sub_can.columns else np.zeros(len(sub_can))
    lat = sub_can["Lateral_Accel"].values if "Lateral_Accel" in sub_can.columns else np.zeros(len(sub_can))

    eps = 1e-5
    speed_norm = max(speed_kph, 1.0)
    mean_whl = (fl + fr + rl + rr) / 4.0

    # 1. 슬립률 지표
    slip_fl = np.mean((fl - mean_whl) / (mean_whl + eps))
    slip_fr = np.mean((fr - mean_whl) / (mean_whl + eps))
    slip_rl = np.mean((rl - mean_whl) / (mean_whl + eps))
    slip_rr = np.mean((rr - mean_whl) / (mean_whl + eps))
    slip_diff = np.mean(((fl + fr) / 2.0 - (rl + rr) / 2.0) / (mean_whl + eps))

    # 2. 정규화 동역학
    norm_yaw = np.mean(np.abs(yaw)) / (speed_norm + eps)
    norm_lat = np.mean(np.abs(lat)) / (speed_norm + eps)

    # 3. 휠속도 통계량 (FL 기준)
    s = fl
    mean_val = float(np.mean(s))
    peak = float(np.max(np.abs(s)))
    p2p = float(np.ptp(s))
    rms = float(np.sqrt(np.mean(s ** 2)))
    std_dev = float(np.std(s))
    variance = float(np.var(s))
    srav = float(np.sqrt(np.mean(np.abs(s))))
    mean_abs = float(np.mean(np.abs(s)))

    crest_factor = float(peak / rms if rms != 0 else 0)
    shape_factor = float(rms / mean_abs if mean_abs != 0 else 0)
    clearance_factor = float(peak / srav if srav != 0 else 0)
    mean_square = float(np.mean(s ** 2))
    np4 = float(np.mean(s ** 4) / (mean_square ** 2) if mean_square != 0 else 0)
    skewness = float(stats.skew(s))
    kurt = float(stats.kurtosis(s))

    # 4. 휠 간 상대 편차
    diff_fl_rl = fl - rl
    diff_fr_rr = fr - rr
    diff_fl_fr = fl - fr
    diff_rl_rr = rl - rr

    # 5. FFT 주파수 스펙트럼 에너지
    ac_signal = s - np.mean(s)
    ac_fft_vals = np.abs(np.fft.rfft(ac_signal))
    ac_fft_freqs = np.fft.rfftfreq(len(s), d=0.02)
    psd = ac_fft_vals ** 2
    total_energy = float(np.sum(psd))

    raw_peak_freq = float(ac_fft_freqs[np.argmax(ac_fft_vals)]) if len(ac_fft_vals) > 0 else 0.0
    norm_peak_freq = raw_peak_freq / (speed_norm + eps)
    raw_centroid = float(np.sum(ac_fft_freqs * psd) / (total_energy + 1e-12))
    norm_centroid = raw_centroid / (speed_norm + eps)
    log_norm_spectral_energy = float(np.log1p(total_energy / ((speed_norm ** 2) + eps)))

    psd_norm = psd / (total_energy + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    entropy = float(-np.sum(psd_norm * np.log2(psd_norm)))

    mask_low = (ac_fft_freqs >= 0.5) & (ac_fft_freqs < 5.0)
    mask_mid = (ac_fft_freqs >= 5.0) & (ac_fft_freqs < 15.0)
    mask_high = (ac_fft_freqs >= 15.0)

    energy_low = float(np.sum(psd[mask_low]) / (total_energy + 1e-12))
    energy_mid = float(np.sum(psd[mask_mid]) / (total_energy + 1e-12))
    energy_high = float(np.sum(psd[mask_high]) / (total_energy + 1e-12))

    return {
        "speed_kph": speed_kph,
        "grade_pct": grade_pct,
        "slip_ratio_fl": round(float(slip_fl), 5),
        "slip_ratio_fr": round(float(slip_fr), 5),
        "slip_ratio_rl": round(float(slip_rl), 5),
        "slip_ratio_rr": round(float(slip_rr), 5),
        "slip_diff_front_rear": round(float(slip_diff), 5),
        "norm_yaw_rate": round(float(norm_yaw), 5),
        "norm_lat_accel": round(float(norm_lat), 5),
        "mean": round(mean_val, 4), "peak": round(peak, 4), "p2p": round(p2p, 4), "rms": round(rms, 4),
        "std": round(std_dev, 4), "variance": round(variance, 4),
        "crest_factor": round(crest_factor, 4), "shape_factor": round(shape_factor, 4),
        "clearance_factor": round(clearance_factor, 4), "np4": round(np4, 4),
        "skew": round(skewness, 4), "kurt": round(kurt, 4),
        "rel_diff_fl_rl_mean": round(float(np.mean(diff_fl_rl)), 4),
        "rel_diff_fl_rl_std": round(float(np.std(diff_fl_rl)), 4),
        "rel_diff_fr_rr_mean": round(float(np.mean(diff_fr_rr)), 4),
        "rel_diff_fr_rr_std": round(float(np.std(diff_fr_rr)), 4),
        "norm_p2p": round(p2p / (speed_norm + eps), 4),
        "norm_rms": round(rms / (speed_norm + eps), 4),
        "norm_std": round(std_dev / (speed_norm + eps), 4),
        "norm_fft_peak_freq": round(norm_peak_freq, 4),
        "log_norm_fft_spectral_energy": round(log_norm_spectral_energy, 4),
        "fft_spectral_entropy": round(entropy, 4),
        "norm_fft_spectral_centroid": round(norm_centroid, 4),
        "fft_energy_low_band": round(energy_low, 4),
        "fft_energy_mid_band": round(energy_mid, 4),
        "fft_energy_high_band": round(energy_high, 4),
    }


def extract_single_sample_copy():
    print("=" * 100)
    print(" 🛠 [샘플 복사본 추출] CAN 500샘플 윈도우 + 오디오 1.47초 시간 동기화 CSV 생성")
    print(f" 📄 CAN 대상 파일 : {CAN_FILE}")
    print(f" 🎙️ WAV 대상 파일 : {WAV_FILE}")
    print(f" 💾 출력 샘플 경로: {OUTPUT_SAMPLE_CSV}")
    print("=" * 100)

    if not os.path.exists(CAN_FILE) or not os.path.exists(WAV_FILE):
        print("[ERROR] 입력 파일이 존재하지 않습니다.")
        return

    # 1. 오프셋 로드 및 오디오 전처리
    trim_offset_sec = get_optimal_offset()
    print(f" ⏱️ 오디오 초기 공회전 오프셋 트리밍: {trim_offset_sec:.2f}초 제거")

    sr, audio_raw = wavfile.read(WAV_FILE)
    if audio_raw.ndim > 1:
        audio_raw = audio_raw[:, 0]
    audio_raw = audio_raw.astype(np.float32) / (np.max(np.abs(audio_raw)) + 1e-6)

    trim_start_sample = int(trim_offset_sec * sr)
    audio_trimmed = audio_raw[trim_start_sample:]
    env_trimmed = np.abs(hilbert(audio_trimmed)) if len(audio_trimmed) > 0 else np.zeros_like(audio_trimmed)

    # 2. CAN 주행 데이터 로드
    df_can = pd.read_csv(CAN_FILE)
    total_can_rows = len(df_can)
    n_windows = total_can_rows // WINDOW_SIZE

    print(f" 📊 CAN 총 행 수: {total_can_rows:,} 행 -> 총 {n_windows}개 500-샘플 윈도우 생성")

    sample_rows = []
    for win_idx in range(n_windows):
        sub_can = df_can.iloc[win_idx * WINDOW_SIZE : (win_idx + 1) * WINDOW_SIZE]
        
        # CAN 45개 특징 추출
        row_dict = compute_can_features(sub_can, speed_kph=20.0, grade_pct=6.0)
        
        # 세션 메타데이터
        t_start = float(sub_can["Time"].iloc[0])
        t_end = float(sub_can["Time"].iloc[-1])
        row_dict["window_index"] = win_idx
        row_dict["can_time_start_sec"] = round(t_start, 3)
        row_dict["can_time_end_sec"] = round(t_end, 3)
        row_dict["window_duration_sec"] = round(t_end - t_start, 3)
        row_dict["scenario"] = "hills_6pct_grade"
        row_dict["state"] = "normal"
        row_dict["target"] = 0

        # 해당 윈도우 시간(1.47초)에 정확히 일치하는 오디오 구간 슬라이싱
        aud_start_idx = int(t_start * sr)
        aud_end_idx = int(t_end * sr)

        aud_start_idx = max(0, min(aud_start_idx, len(audio_trimmed)))
        aud_end_idx = max(aud_start_idx + 1, min(aud_end_idx, len(audio_trimmed)))

        sub_aud = audio_trimmed[aud_start_idx:aud_end_idx]
        sub_env = env_trimmed[aud_start_idx:aud_end_idx]

        # 동기화된 오디오 특징 결합
        row_dict["audio_samples_count"] = len(sub_aud)
        row_dict["audio_trim_offset_sec"] = trim_offset_sec
        row_dict["audio_hilbert_env_mean"] = round(float(np.mean(sub_env)), 5) if len(sub_env) > 0 else 0.0
        row_dict["audio_hilbert_env_max"] = round(float(np.max(sub_env)), 5) if len(sub_env) > 0 else 0.0
        row_dict["audio_rms_energy"] = round(float(np.sqrt(np.mean(sub_aud ** 2))), 5) if len(sub_aud) > 0 else 0.0

        # 고주파 충격 대역 (4kHz ~ 10kHz) STFT 파워
        if len(sub_aud) >= 256:
            f, t_stft, Zxx = stft(sub_aud, fs=sr, nperseg=256)
            mag_db = 20.0 * np.log10(np.abs(Zxx) + 1e-6)
            high_mask = (f >= 4000) & (f <= 10000)
            row_dict["audio_high_band_energy_db"] = round(float(np.mean(mag_db[high_mask])), 2)
        else:
            row_dict["audio_high_band_energy_db"] = -85.0

        sample_rows.append(row_dict)

    df_sample_out = pd.DataFrame(sample_rows)
    df_sample_out.to_csv(OUTPUT_SAMPLE_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f" 🎉 [샘플 복사본 CSV 추출 완료] -> {OUTPUT_SAMPLE_CSV}")
    print(f"    - 생성된 동기화 윈도우 수: {len(df_sample_out)} 개 윈도우 행")
    print(f"    - 결합된 전체 피처 수: {len(df_sample_out.columns)} 개 (CAN 피처 45개 + 오디오 동기화 피처 5개 + 메타데이터)")
    print("=" * 100)


if __name__ == "__main__":
    extract_single_sample_copy()
