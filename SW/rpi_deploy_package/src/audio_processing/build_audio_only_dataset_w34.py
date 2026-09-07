"""
build_audio_only_dataset_w34.py (gearbox_final_suite / src / audio_processing)

audio_filtered_bpf/ 의 BPF 필터링 완료 WAV에서 피처만 추출하여
unified_audio_only_dataset_w34.csv 저장.

오디오 단독 모델 성능 실험용.

규정 준수:
- Rule 3: 코드 주석 100% 한글, Matplotlib/터미널 100% 영문 라벨.
- Rule 4: Data Leakage 배제 및 오디오 미측정 초과 구간 정직하게 Drop 처리.
- Rule 7: LaTeX 수식 기호 일절 사용 금지 (No-LaTeX Protocol).
- Rule 8: 이모지 전면 배제.
"""

import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import hilbert, stft, find_peaks
from scipy.stats import kurtosis
from scipy import stats

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")

RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_FILTERED_BPF_DIR = os.path.join(DATA_DIR, "audio_filtered_bpf")
DATASETS_DIR = os.path.join(RESULTS_DIR, "datasets")

os.makedirs(DATASETS_DIR, exist_ok=True)

# 시나리오별 CAN 출발 트리밍 오프셋 (오디오 슬라이싱 시간 기준 정렬용)
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


def extract_audio_window_features(audio_slice, sr):
    """
    단일 윈도우 슬라이스에서 속도 비의존 무차원(Speed-Invariant) 고급 DSP 피처 추출
    """
    if len(audio_slice) < 64:
        return {
            "audio_crest_factor": 0.0, "audio_shape_factor": 0.0, "audio_impulse_factor": 0.0, "audio_margin_factor": 0.0,
            "audio_kurtosis": 0.0, "audio_skewness": 0.0,
            "audio_hilbert_env_crest": 0.0, "audio_hilbert_env_kurtosis": 0.0, "audio_hilbert_env_skewness": 0.0,
            "audio_env_mod_index": 0.0, "audio_shockwave_energy_ratio": 0.0, "audio_shockwave_pulse_count": 0,
            "audio_spectral_flatness": 0.0, "audio_spectral_entropy": 0.0, "audio_spectral_rolloff_norm": 0.0, "audio_spectral_centroid_norm": 0.0,
            "audio_stft_2k8_ratio": 0.0, "audio_stft_harmonic_ratio": 0.0, "audio_peak_power_ratio": 0.0,
            "audio_subband_ratio_1": 0.0, "audio_subband_ratio_2": 0.0, "audio_subband_ratio_3": 0.0, "audio_subband_ratio_4": 0.0,
            "audio_est_gmf_hz": 0.0
        }

    # 1. 시간 파형 기본 통계 및 무차원 형상 피처 (Scale-Invariant Waveform Ratios)
    abs_audio = np.abs(audio_slice)
    mean_abs = float(np.mean(abs_audio))
    rms = float(np.sqrt(np.mean(audio_slice ** 2)))
    peak = float(np.max(abs_audio))
    mean_sqrt_abs = float(np.mean(np.sqrt(abs_audio)))

    crest_factor = float(peak / (rms + 1e-8))
    shape_factor = float(rms / (mean_abs + 1e-8))
    impulse_factor = float(peak / (mean_abs + 1e-8))
    margin_factor = float(peak / ((mean_sqrt_abs ** 2) + 1e-8))
    kurt = float(kurtosis(audio_slice, fisher=True))
    skew_val = float(stats.skew(audio_slice)) if len(audio_slice) > 1 else 0.0

    # 2. 힐베르트 변환 포락선(Envelope) 및 변조 지수 (Envelope Modulation Index)
    analytic_signal = hilbert(audio_slice)
    env = np.abs(analytic_signal)
    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    env_peak = float(np.max(env))
    env_crest = float(env_peak / (env_mean + 1e-8))
    env_kurt = float(kurtosis(env, fisher=True))
    env_skew = float(stats.skew(env)) if len(env) > 1 else 0.0
    env_mod_index = float(env_std / (env_mean + 1e-8))  # 변조 지수

    # 3. 기어 이빨 타격 충격파(Shockwave) 특화 피처
    p95_thresh = np.percentile(env, 95)
    shock_mask = env >= p95_thresh
    shock_energy = float(np.sum(env[shock_mask] ** 2))
    total_energy = float(np.sum(env ** 2) + 1e-12)
    shockwave_energy_ratio = float(shock_energy / total_energy)

    pulse_thresh = env_mean + 2.0 * env_std
    peaks_idx, _ = find_peaks(env, height=pulse_thresh, distance=int(sr * 0.001))
    shockwave_pulse_count = int(len(peaks_idx))

    # 4. STFT 주파수 무차원 스펙트럼 피처 (Speed-Invariant Spectral Profile)
    f, _, Zxx = stft(audio_slice, fs=sr, nperseg=min(1024, len(audio_slice)), noverlap=min(512, len(audio_slice) // 2))
    mag_spec = np.abs(Zxx)
    power_spec = mag_spec ** 2
    total_power = np.sum(power_spec) + 1e-12

    # 스펙트럼 평탄도 (Spectral Flatness: Geometric Mean / Arithmetic Mean)
    log_mag = np.log(mag_spec + 1e-12)
    geo_mean = np.exp(np.mean(log_mag))
    ari_mean = np.mean(mag_spec) + 1e-12
    spectral_flatness = float(geo_mean / ari_mean)

    # 스펙트럼 엔트로피 (Spectral Entropy: 에너지 분산도/집중도)
    mean_psd_1d = np.mean(power_spec, axis=1)
    norm_psd = mean_psd_1d / (np.sum(mean_psd_1d) + 1e-12)
    spectral_entropy = float(-np.sum(norm_psd * np.log2(norm_psd + 1e-12)))

    # 85% 파워 스펙트럼 롤오프 주파수 (Spectral Rolloff Normalized by Nyquist)
    cum_power = np.cumsum(mean_psd_1d)
    rolloff_idx = np.where(cum_power >= 0.85 * cum_power[-1])[0]
    rolloff_freq = f[rolloff_idx[0]] if len(rolloff_idx) > 0 else f[-1]
    nyquist = sr / 2.0
    spectral_rolloff_norm = float(rolloff_freq / nyquist)

    # 정규화 스펙트럼 센트로이드 (Spectral Centroid Normalized)
    sum_mag = np.sum(np.mean(mag_spec, axis=1)) + 1e-8
    centroid_freq = float(np.sum(f * np.mean(mag_spec, axis=1)) / sum_mag)
    spectral_centroid_norm = float(centroid_freq / nyquist)

    # 피크 주파수 파워 비율 (Peak Power Ratio)
    peak_pwr = np.max(mean_psd_1d)
    peak_power_ratio = float(peak_pwr / (np.sum(mean_psd_1d) + 1e-12))

    # 대역별 정규화 파워 비율 (Volume-Independent Power Ratios)
    mask_2k8 = (f >= 2700) & (f <= 2900)
    p_2k8 = np.sum(power_spec[mask_2k8, :]) if np.any(mask_2k8) else 0.0
    stft_2k8_ratio = float(p_2k8 / total_power)

    mask_harm = (f >= 4000) & (f <= 10000)
    p_harm = np.sum(power_spec[mask_harm, :]) if np.any(mask_harm) else 0.0
    stft_harmonic_ratio = float(p_harm / total_power)

    # 4분할 상대 서브밴드 파워 비율 (Relative Subband Power Distribution: Sum = 1.0)
    n_bins = len(f)
    q1, q2, q3 = n_bins // 4, n_bins // 2, (3 * n_bins) // 4
    sub1 = np.sum(mean_psd_1d[:q1])
    sub2 = np.sum(mean_psd_1d[q1:q2])
    sub3 = np.sum(mean_psd_1d[q2:q3])
    sub4 = np.sum(mean_psd_1d[q3:])
    sub_total = sub1 + sub2 + sub3 + sub4 + 1e-12

    subband_ratio_1 = float(sub1 / sub_total)
    subband_ratio_2 = float(sub2 / sub_total)
    subband_ratio_3 = float(sub3 / sub_total)
    subband_ratio_4 = float(sub4 / sub_total)

    # 오디오 자가 속도/RPM 추적 (Peak Frequency Estimation)
    peak_f_idx = np.argmax(mean_psd_1d)
    est_gmf_hz = float(f[peak_f_idx])

    return {
        "audio_crest_factor": round(crest_factor, 4),
        "audio_shape_factor": round(shape_factor, 4),
        "audio_impulse_factor": round(impulse_factor, 4),
        "audio_margin_factor": round(margin_factor, 4),
        "audio_kurtosis": round(kurt, 4),
        "audio_skewness": round(skew_val, 4),
        "audio_hilbert_env_crest": round(env_crest, 4),
        "audio_hilbert_env_kurtosis": round(env_kurt, 4),
        "audio_hilbert_env_skewness": round(env_skew, 4),
        "audio_env_mod_index": round(env_mod_index, 4),
        "audio_shockwave_energy_ratio": round(shockwave_energy_ratio, 4),
        "audio_shockwave_pulse_count": shockwave_pulse_count,
        "audio_spectral_flatness": round(spectral_flatness, 6),
        "audio_spectral_entropy": round(spectral_entropy, 4),
        "audio_spectral_rolloff_norm": round(spectral_rolloff_norm, 4),
        "audio_spectral_centroid_norm": round(spectral_centroid_norm, 4),
        "audio_stft_2k8_ratio": round(stft_2k8_ratio, 6),
        "audio_stft_harmonic_ratio": round(stft_harmonic_ratio, 6),
        "audio_peak_power_ratio": round(peak_power_ratio, 6),
        "audio_subband_ratio_1": round(subband_ratio_1, 4),
        "audio_subband_ratio_2": round(subband_ratio_2, 4),
        "audio_subband_ratio_3": round(subband_ratio_3, 4),
        "audio_subband_ratio_4": round(subband_ratio_4, 4),
        "audio_est_gmf_hz": round(est_gmf_hz, 1)
    }


def build_audio_only_dataset(window_size=34, step_size=34):
    """
    audio_filtered_bpf/ WAV에서 오디오 피처만 추출 후 CSV 저장
    """
    print("\n" + "=" * 120)
    print(" [Audio-Only Feature Extraction - w34, Step=34, Non-overlapping]")
    print("=" * 120)
    print(f" {'Scenario':<35} | {'CAN Windows':<13} | {'Valid':<10} | {'Dropped':<10}")
    print("-" * 120)

    all_rows = []
    total_can_windows = 0
    total_valid = 0
    total_dropped = 0

    for scenario_name, t_launch_can in CAN_LAUNCH_TRIMS.items():
        state = "abnormal" if scenario_name.startswith("abnormal") else "normal"
        label = 1 if state == "abnormal" else 0

        # CAN 타임스탬프 로드 (윈도우 시간 경계 계산용)
        can_csv_path = os.path.join(RAW_DECODED_DIR, f"{scenario_name}_official_decoded.csv")
        df_can = pd.read_csv(can_csv_path)
        df_can_drive = df_can[df_can["Time"] >= t_launch_can].reset_index(drop=True)
        t_base_can = df_can_drive["Time"].iloc[0]

        # 이미 필터링된 BPF WAV 로드
        bpf_wav_path = os.path.join(AUDIO_FILTERED_BPF_DIR, f"{scenario_name}_gear_bpf.wav")
        sr, audio_raw = wavfile.read(bpf_wav_path)
        if audio_raw.ndim > 1:
            audio_raw = audio_raw[:, 0]
        if np.issubdtype(audio_raw.dtype, np.integer):
            audio_filtered = audio_raw.astype(np.float32) / float(np.iinfo(audio_raw.dtype).max)
        else:
            audio_filtered = audio_raw.astype(np.float32)

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

            all_rows.append({
                "scenario": scenario_name,
                "window_index": w_idx,
                "start_time_sec": round(t_start_rel, 4),
                "end_time_sec": round(t_end_rel, 4),
                "state": state,
                "label": label,
                **feats
            })
            scen_valid += 1

        total_can_windows += scen_can_win
        total_valid += scen_valid
        total_dropped += scen_drop

        print(f" {scenario_name:<35} | {scen_can_win:>13} | {scen_valid:>10} | {scen_drop:>10}")

    print("=" * 120)
    print(f" Total CAN Windows: {total_can_windows} | Valid: {total_valid} ({total_valid/total_can_windows*100:.2f}%) | Dropped: {total_dropped}")
    print("=" * 120)

    df_out = pd.DataFrame(all_rows)

    csv_path_data = os.path.join(DATA_DIR, f"unified_audio_only_dataset_w{window_size}.csv")
    csv_path_results = os.path.join(DATASETS_DIR, f"unified_audio_only_dataset_w{window_size}.csv")

    df_out.to_csv(csv_path_data, index=False, encoding="utf-8-sig")
    df_out.to_csv(csv_path_results, index=False, encoding="utf-8-sig")

    print(f"\n [Saved]: {csv_path_data}")
    print(f" [Total]: {len(df_out.columns) - 6} audio features | {len(df_out)} windows\n")


if __name__ == "__main__":
    build_audio_only_dataset(window_size=34, step_size=34)
