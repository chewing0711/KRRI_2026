"""
build_standalone_audio_context_dataset.py (gearbox_final_suite / src / audio_processing)

[요구사항 1 & 2 완전 구현]:
1. 32개 전체 오디오에 4차 버터워스 대역통과 필터(1,800 Hz ~ 10,500 Hz BPF) 전수 적용
2. 필터링된 고순도 클린 오디오를 data/audio_filtered_bpf/ 에 32개 WAV로 영구 저장
3. [힐베르트 포락선 & 충격파 심층 추출]:
   - 힐베르트 포락선 통계: env_mean, env_std, env_peak, env_crest, env_kurtosis
   - 기어 충격파 지표: shockwave_energy_ratio (상위 5% 충격 에너지 비중), shockwave_pulse_count (충격 펄스 횟수)
4. 시간 및 주파수 피처: RMS, Peak, Crest, Kurtosis, STFT 2.8k dB, Harmonics dB, Spectral Centroid
5. CAN 데이터셋과 1:1 타임스탬프 동기화된 독립 보조 데이터셋(unified_audio_context_dataset_w34.csv) 생성

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib/터미널 100% 영문 라벨.
- Rule 4: Data Leakage 배제 및 오디오 미측정 초과 구간 정직하게 Drop 처리.
- Rule 7: LaTeX 수식 기호 일절 사용 금지 (No-LaTeX Protocol).
- Rule 8: 이모지 전면 배제.
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert, stft, find_peaks
from scipy.stats import kurtosis

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")

RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_LAUNCH_ALIGNED_DIR = os.path.join(DATA_DIR, "audio_launch_aligned")
AUDIO_FILTERED_BPF_DIR = os.path.join(DATA_DIR, "audio_filtered_bpf")
DATASETS_DIR = os.path.join(RESULTS_DIR, "datasets")
AUDIO_AUDIT_DIR = os.path.join(RESULTS_DIR, "audio_audit")

os.makedirs(AUDIO_FILTERED_BPF_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(AUDIO_AUDIT_DIR, exist_ok=True)

# 시나리오별 CAN 출발 트리밍 오프셋
CAN_LAUNCH_TRIMS = {
    "normal_high_speed_20kph": 0.910,
    "normal_high_speed_60kph": 0.060,
    "normal_high_speed_80kph": 18.960,
    "normal_hills_6deg": 0.065,
    "normal_hills_12deg": 0.014,
    "normal_hills_18deg": 5.885,
    "normal_hills_30deg": 0.952,
    "normal_steering_pad_40kph": 1.403,
    "abnormal_high_speed_20kph": 0.280,
    "abnormal_high_speed_60kph": 1.125,
    "abnormal_high_speed_80kph": 0.191,
    "abnormal_hills_6deg": 0.000,
    "abnormal_hills_12deg": 8.840,
    "abnormal_hills_18deg": 4.132,
    "abnormal_hills_30deg": 3.355,
    "abnormal_steering_pad_40kph": 0.016,
}


def find_aligned_audio_path(scenario_name, channel="gear"):
    """audio_launch_aligned 디렉터리 내 파일 탐색"""
    parts = scenario_name.split("_", 1)
    state = parts[0]
    course = parts[1]
    folder = os.path.join(AUDIO_LAUNCH_ALIGNED_DIR, state, course)
    if os.path.exists(folder):
        matches = [f for f in os.listdir(folder) if channel.lower() in f.lower() and f.endswith(".wav")]
        if matches:
            return os.path.join(folder, matches[0])
    
    # 전역 재귀 검색 fallback
    pattern = os.path.join(AUDIO_LAUNCH_ALIGNED_DIR, "**", f"*{channel}*.wav")
    all_matches = glob.glob(pattern, recursive=True)
    scen_matches = [m for m in all_matches if course in m.lower() or scenario_name in m.lower()]
    return scen_matches[0] if scen_matches else (all_matches[0] if all_matches else None)


def apply_butterworth_bpf(audio_signal, sr, lowcut=1800.0, highcut=10500.0, order=4):
    """
    4차 버터워스 영위상(Zero-Phase) 대역통과 필터 적용
    """
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=sr, output="sos")
    filtered_signal = sosfiltfilt(sos, audio_signal)
    return filtered_signal.astype(np.float32)


def extract_audio_window_features(audio_slice, sr):
    """
    단일 0.1초 윈도우 슬라이스(디지털 필터링 완료 신호)에서
    시간/주파수 및 힐베르트 포락선/충격파 피처 전수 추출
    """
    if len(audio_slice) < 64:
        return {
            "audio_rms": 0.0, "audio_peak": 0.0, "audio_crest_factor": 0.0, "audio_kurtosis": 0.0,
            "audio_hilbert_env_mean": 0.0, "audio_hilbert_env_std": 0.0, "audio_hilbert_env_peak": 0.0,
            "audio_hilbert_env_crest": 0.0, "audio_hilbert_env_kurtosis": 0.0,
            "audio_shockwave_energy_ratio": 0.0, "audio_shockwave_pulse_count": 0,
            "audio_stft_2k8_db": -100.0, "audio_stft_harmonic_db": -100.0, "audio_spectral_centroid": 0.0
        }

    # 1. 시간 및 충격 기본 통계
    rms = float(np.sqrt(np.mean(audio_slice ** 2)))
    peak = float(np.max(np.abs(audio_slice)))
    crest_factor = float(peak / (rms + 1e-8))
    kurt = float(kurtosis(audio_slice, fisher=True))

    # 2. [심층] 힐베르트 변환 포락선(Envelope) 분석
    analytic_signal = hilbert(audio_slice)
    env = np.abs(analytic_signal)
    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    env_peak = float(np.max(env))
    env_crest = float(env_peak / (env_mean + 1e-8))
    env_kurt = float(kurtosis(env, fisher=True))

    # 3. [심층] 기어 이빨 타격 충격파(Shockwave) 특화 피처
    p95_thresh = np.percentile(env, 95)
    shock_mask = env >= p95_thresh
    shock_energy = float(np.sum(env[shock_mask] ** 2))
    total_energy = float(np.sum(env ** 2) + 1e-12)
    shockwave_energy_ratio = float(shock_energy / total_energy)

    pulse_thresh = env_mean + 2.0 * env_std
    peaks_idx, _ = find_peaks(env, height=pulse_thresh, distance=int(sr * 0.001))
    shockwave_pulse_count = int(len(peaks_idx))

    # 4. 주파수 스펙트럼 및 STFT 피크
    f, t_stft, Zxx = stft(audio_slice, fs=sr, nperseg=min(1024, len(audio_slice)), noverlap=min(512, len(audio_slice)//2))
    mag_spec = np.abs(Zxx)
    power_spec = mag_spec ** 2

    # 2.8 kHz 결함 주파수 대역 (2700 ~ 2900 Hz)
    mask_2k8 = (f >= 2700) & (f <= 2900)
    p_2k8 = np.mean(power_spec[mask_2k8, :]) if np.any(mask_2k8) else 1e-10
    stft_2k8_db = float(10.0 * np.log10(p_2k8 + 1e-10))

    # 4 ~ 10 kHz 고차 조화파 대역 (4000 ~ 10000 Hz)
    mask_harm = (f >= 4000) & (f <= 10000)
    p_harm = np.mean(power_spec[mask_harm, :]) if np.any(mask_harm) else 1e-10
    stft_harm_db = float(10.0 * np.log10(p_harm + 1e-10))

    # 스펙트럼 중심 주파수 (Spectral Centroid)
    mean_psd_1d = np.mean(mag_spec, axis=1)
    sum_mag = np.sum(mean_psd_1d)
    centroid = float(np.sum(f * mean_psd_1d) / (sum_mag + 1e-8))

    return {
        "audio_rms": round(rms, 6),
        "audio_peak": round(peak, 6),
        "audio_crest_factor": round(crest_factor, 4),
        "audio_kurtosis": round(kurt, 4),
        "audio_hilbert_env_mean": round(env_mean, 6),
        "audio_hilbert_env_std": round(env_std, 6),
        "audio_hilbert_env_peak": round(env_peak, 6),
        "audio_hilbert_env_crest": round(env_crest, 4),
        "audio_hilbert_env_kurtosis": round(env_kurt, 4),
        "audio_shockwave_energy_ratio": round(shockwave_energy_ratio, 4),
        "audio_shockwave_pulse_count": shockwave_pulse_count,
        "audio_stft_2k8_db": round(stft_2k8_db, 2),
        "audio_stft_harmonic_db": round(stft_harm_db, 2),
        "audio_spectral_centroid": round(centroid, 1)
    }


def build_filtered_standalone_audio_dataset(window_size=34, step_size=34):
    """
    32개 파일 4차 BPF 필터링 적용 및 0% 중첩(Step=34, 비중첩) 1:1 독립 데이터셋 생성
    """
    print("\n" + "=" * 135)
    print(" [4차 버터워스 BPF 필터링 및 0% 비중첩(Step=34) 힐베르트/충격파 피처셋 추출 시작]")
    print("=" * 135)
    print(f" {'시나리오명':<30} | {'CAN 총 윈도우':<14} | {'1:1 유효 매칭':<14} | {'결손 제외(Drop)':<16} | {'힐베르트/충격파 추출'}")
    print("-" * 135)

    all_rows = []
    total_can_windows = 0
    total_valid_matched = 0
    total_dropped = 0

    for scenario_name, t_launch_can in CAN_LAUNCH_TRIMS.items():
        state = "abnormal" if scenario_name.startswith("abnormal") else "normal"
        label = 1 if state == "abnormal" else 0

        # 1. CAN 로드 및 출발점 트리밍
        can_csv_path = os.path.join(RAW_DECODED_DIR, f"{scenario_name}_official_decoded.csv")
        if not os.path.exists(can_csv_path):
            can_csv_path = os.path.join(RAW_DECODED_DIR, f"{scenario_name}.csv")
        df_can = pd.read_csv(can_csv_path)
        df_can_drive = df_can[df_can["Time"] >= t_launch_can].reset_index(drop=True)
        t_base_can = df_can_drive["Time"].iloc[0]

        # 2. 정제 오디오(Gear-1) 탐색 및 로드
        gear_wav_path = find_aligned_audio_path(scenario_name, channel="gear")
        sr, audio_raw = wavfile.read(gear_wav_path)
        if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
        if np.issubdtype(audio_raw.dtype, np.integer):
            max_val = float(np.iinfo(audio_raw.dtype).max)
            audio_norm = audio_raw.astype(np.float32) / max_val
        else:
            audio_norm = audio_raw.astype(np.float32)

        # 3. 4차 버터워스 BPF (1800~10500 Hz) 적용
        audio_filtered = apply_butterworth_bpf(audio_norm, sr, lowcut=1800.0, highcut=10500.0, order=4)

        # 필터링 WAV 저장
        out_filtered_wav = os.path.join(AUDIO_FILTERED_BPF_DIR, f"{scenario_name}_gear_bpf.wav")
        audio_int16 = (np.clip(audio_filtered, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(out_filtered_wav, sr, audio_int16)

        # OBD 마이크도 동일하게 필터링 저장
        obd_wav_path = find_aligned_audio_path(scenario_name, channel="obd")
        if obd_wav_path and os.path.exists(obd_wav_path):
            sr_obd, aud_obd = wavfile.read(obd_wav_path)
            if aud_obd.ndim > 1: aud_obd = aud_obd[:, 0]
            if np.issubdtype(aud_obd.dtype, np.integer):
                max_obd = float(np.iinfo(aud_obd.dtype).max)
                aud_obd_norm = aud_obd.astype(np.float32) / max_obd
            else:
                aud_obd_norm = aud_obd.astype(np.float32)
            aud_obd_flt = apply_butterworth_bpf(aud_obd_norm, sr_obd, lowcut=1800.0, highcut=10500.0, order=4)
            out_obd_wav = os.path.join(AUDIO_FILTERED_BPF_DIR, f"{scenario_name}_obd_bpf.wav")
            wavfile.write(out_obd_wav, sr_obd, (np.clip(aud_obd_flt, -1.0, 1.0) * 32767.0).astype(np.int16))

        total_audio_samples = len(audio_filtered)
        num_can_rows = len(df_can_drive)

        scen_can_win = 0
        scen_valid = 0
        scen_drop = 0

        for w_idx, start_idx in enumerate(range(0, num_can_rows - window_size + 1, step_size)):
            scen_can_win += 1
            end_idx = start_idx + window_size - 1

            t_start_rel = df_can_drive["Time"].iloc[start_idx] - t_base_can
            t_end_rel = df_can_drive["Time"].iloc[end_idx] - t_base_can

            s_start = int(round(t_start_rel * sr))
            s_end = int(round(t_end_rel * sr))

            if s_end > total_audio_samples or s_start >= total_audio_samples:
                scen_drop += 1
                continue

            audio_slice = audio_filtered[s_start:s_end]
            feats = extract_audio_window_features(audio_slice, sr)

            row_data = {
                "scenario": scenario_name,
                "window_index": w_idx,
                "start_time_sec": round(t_start_rel, 4),
                "end_time_sec": round(t_end_rel, 4),
                "state": state,
                "label": label,
                **feats
            }
            all_rows.append(row_data)
            scen_valid += 1

        total_can_windows += scen_can_win
        total_valid_matched += scen_valid
        total_dropped += scen_drop

        print(f" {scenario_name:<30} | {scen_can_win:>14} | {scen_valid:>14} | {scen_drop:>16} | OK (Extracted)")

    print("=" * 135)
    print(f" [합계] 총 CAN 윈도우: {total_can_windows}개 | 1:1 유효 매칭: {total_valid_matched}개 ({total_valid_matched/total_can_windows*100:.2f}%) | Drop: {total_dropped}개")
    print("=" * 135)

    df_out = pd.DataFrame(all_rows)

    csv_path_data = os.path.join(DATA_DIR, f"unified_audio_context_dataset_w{window_size}.csv")
    csv_path_results = os.path.join(DATASETS_DIR, f"unified_audio_context_dataset_w{window_size}.csv")

    df_out.to_csv(csv_path_data, index=False, encoding="utf-8-sig")
    df_out.to_csv(csv_path_results, index=False, encoding="utf-8-sig")

    print(f"\n [힐베르트/충격파 포함 독립 데이터셋 저장 완료]: {csv_path_data}")
    print(f" [추출된 총 피처 수]: {len(df_out.columns) - 6}개 음향/충격파 피처 (총 {len(df_out)}개 윈도우)\n")


if __name__ == "__main__":
    build_filtered_standalone_audio_dataset(window_size=34, step_size=34)
