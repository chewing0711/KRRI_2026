"""
sound_main_run.py (git_uploads/sound)

실시간 마이크 입력 기반 오디오 이상 진단 모듈
1. record_audio(): 마이크로부터 1초 동안 48,000 Hz 오디오 음원 수집 (메모리 로드)
2. preprocess_sound(): 4차 Butterworth BPF(1.8kHz~10.5kHz) 필터링 및 24개 DSP 통계량 추출 후 StandardScaler 정규화
3. sound_preprocess(): 사전 학습된 XGBoost 모델(rpi_xgboost_no_gap.joblib) 기반 실시간 고장 예측
4. sound_main(): 위 3단계를 순차적으로 실행하여 최종 이상 진단 결과 반환
"""

import os
import sys
import numpy as np
import sounddevice as sd
import joblib
from scipy.signal import butter, filtfilt, hilbert, stft, find_peaks
from scipy.stats import kurtosis, skew
import warnings

warnings.filterwarnings("ignore")

# =========================================================================
# 경로 및 글로벌 모델/스케일러 캐시 설정
# =========================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "SW", "rpi_deploy_package", "models")

SCALER_PATH = os.path.join(MODELS_DIR, "rpi_scaler_feat.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "rpi_xgboost_no_gap.joblib")

_SCALER_CACHE = None
_MODEL_CACHE = None


def load_audio_model_and_scaler():
    """
    사전 학습된 StandardScaler 및 XGBoost 모델을 메모리에 로드 (Lazy Loading & Caching)
    """
    global _SCALER_CACHE, _MODEL_CACHE

    if _SCALER_CACHE is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {SCALER_PATH}")
        _SCALER_CACHE = joblib.load(SCALER_PATH)

    if _MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        _MODEL_CACHE = joblib.load(MODEL_PATH)

    return _SCALER_CACHE, _MODEL_CACHE


# =========================================================================
# DSP 필터 및 피처 추출 함수
# =========================================================================
def butter_bandpass(lowcut=1800.0, highcut=10500.0, fs=48000, order=4):
    """
    4차 Butterworth Bandpass Filter 계수 생성
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(data, lowcut=1800.0, highcut=10500.0, fs=48000, order=4):
    """
    노면 소음 및 외부 노이즈 제거를 위한 1.8kHz ~ 10.5kHz BPF 적용
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = filtfilt(b, a, data)
    return filtered


def extract_audio_features_from_array(audio_slice, sr=48000):
    """
    1초 오디오 배열로부터 24개 속도 비의존(Speed-Invariant) 통계 및 DSP 피처 추출
    """
    if len(audio_slice) < 64:
        return np.zeros((1, 24), dtype=np.float32)

    # 1. 파형 기본 통계량 및 무차원 형상 지수 (Waveform Shape Ratios)
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
    skew_val = float(skew(audio_slice)) if len(audio_slice) > 1 else 0.0

    # 2. Hilbert Transform 포락선(Envelope) 및 변조 지수 (Envelope Modulation Index)
    analytic_signal = hilbert(audio_slice)
    env = np.abs(analytic_signal)
    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    env_peak = float(np.max(env))
    env_crest = float(env_peak / (env_mean + 1e-8))
    env_kurt = float(kurtosis(env, fisher=True))
    env_skew = float(skew(env)) if len(env) > 1 else 0.0
    env_mod_index = float(env_std / (env_mean + 1e-8))

    # 3. 충격파(Shockwave) 특화 피처
    p95_thresh = np.percentile(env, 95)
    shock_mask = env >= p95_thresh
    shock_energy = float(np.sum(env[shock_mask] ** 2))
    total_energy = float(np.sum(env ** 2) + 1e-12)
    shockwave_energy_ratio = float(shock_energy / total_energy)

    pulse_thresh = env_mean + 2.0 * env_std
    peaks_idx, _ = find_peaks(env, height=pulse_thresh, distance=int(sr * 0.001))
    shockwave_pulse_count = float(len(peaks_idx))

    # 4. STFT 주파수 무차원 스펙트럼 피처 (Spectral Profile)
    f, _, Zxx = stft(audio_slice, fs=sr, nperseg=min(1024, len(audio_slice)), noverlap=min(512, len(audio_slice) // 2))
    mag_spec = np.abs(Zxx)
    power_spec = mag_spec ** 2
    total_power = np.sum(power_spec) + 1e-12

    # Spectral Flatness
    log_mag = np.log(mag_spec + 1e-12)
    geo_mean = np.exp(np.mean(log_mag))
    ari_mean = np.mean(mag_spec) + 1e-12
    spectral_flatness = float(geo_mean / ari_mean)

    # Spectral Entropy
    mean_psd_1d = np.mean(power_spec, axis=1)
    norm_psd = mean_psd_1d / (np.sum(mean_psd_1d) + 1e-12)
    spectral_entropy = float(-np.sum(norm_psd * np.log2(norm_psd + 1e-12)))

    # Spectral Rolloff (85%) & Centroid
    cum_power = np.cumsum(mean_psd_1d)
    rolloff_idx = np.where(cum_power >= 0.85 * cum_power[-1])[0]
    rolloff_freq = f[rolloff_idx[0]] if len(rolloff_idx) > 0 else f[-1]
    nyquist = sr / 2.0
    spectral_rolloff_norm = float(rolloff_freq / nyquist)

    sum_mag = np.sum(np.mean(mag_spec, axis=1)) + 1e-8
    centroid_freq = float(np.sum(f * np.mean(mag_spec, axis=1)) / sum_mag)
    spectral_centroid_norm = float(centroid_freq / nyquist)

    # Peak Power Ratio & Specific Bands
    peak_pwr = np.max(mean_psd_1d)
    peak_power_ratio = float(peak_pwr / (np.sum(mean_psd_1d) + 1e-12))

    mask_2k8 = (f >= 2700) & (f <= 2900)
    p_2k8 = np.sum(power_spec[mask_2k8, :]) if np.any(mask_2k8) else 0.0
    stft_2k8_ratio = float(p_2k8 / total_power)

    mask_harm = (f >= 4000) & (f <= 10000)
    p_harm = np.sum(power_spec[mask_harm, :]) if np.any(mask_harm) else 0.0
    stft_harmonic_ratio = float(p_harm / total_power)

    # Subband Power Ratios (4-Bands)
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

    peak_f_idx = np.argmax(mean_psd_1d)
    est_gmf_hz = float(f[peak_f_idx])

    # 24개 피처 리스트 구성 (학습 데이터셋 피처 순서와 100% 동일)
    feature_vector = [
        crest_factor, shape_factor, impulse_factor, margin_factor, kurt, skew_val,
        env_crest, env_kurt, env_skew, env_mod_index, shockwave_energy_ratio, shockwave_pulse_count,
        spectral_flatness, spectral_entropy, spectral_rolloff_norm, spectral_centroid_norm,
        stft_2k8_ratio, stft_harmonic_ratio, peak_power_ratio,
        subband_ratio_1, subband_ratio_2, subband_ratio_3, subband_ratio_4, est_gmf_hz
    ]

    return np.array(feature_vector, dtype=np.float32).reshape(1, -1)


# =========================================================================
# 메인 파이프라인 함수 정의
# =========================================================================
def get_valid_input_device():
    """
    사용 가능한 마이크 입력 장치 ID를 탐색하여 반환 (기본값이 -1일 경우 자동 검색)
    """
    try:
        devices = sd.query_devices()
        if not devices:
            return None

        # 1. 기본 입력 장치가 유효한지 확인
        default_in = sd.default.device[0]
        if isinstance(default_in, int) and default_in >= 0 and default_in < len(devices):
            if devices[default_in].get("max_input_channels", 0) > 0:
                return default_in

        # 2. 입력 채널(max_input_channels > 0)을 지원하는 첫 번째 장치 검색
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                return idx
    except Exception:
        pass
    return None


def record_audio(duration=1.0, samplerate=48000, device=None):
    """
    1초 동안 마이크 음원을 수집하여 1D Numpy Array로 반환
    """
    if device is None or device < 0:
        device = get_valid_input_device()

    if device is None:
        raise RuntimeError("사용 가능한 오디오 입력 장치(마이크)를 찾을 수 없습니다. (WSL 오디오 설정 또는 마이크 연결 확인 필요)")

    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device
    )
    sd.wait()

    # 1D 평탄화 (Flatten)
    audio_1s = audio.flatten()
    return audio_1s



def preprocess_sound(audio_1s, samplerate=48000):
    """
    마이크 입력 오디오에 BPF 필터를 적용하고 24개 통계 피처 추출 후 StandardScaler 변환 수행
    """
    # 1. 4차 Butterworth BPF 필터 적용 (1.8kHz ~ 10.5kHz)
    audio_filtered = apply_bandpass_filter(audio_1s, lowcut=1800.0, highcut=10500.0, fs=samplerate, order=4)

    # 2. 24개 DSP 통계량 피처 추출 (1, 24)
    raw_features = extract_audio_features_from_array(audio_filtered, sr=samplerate)

    # 3. 사전 학습된 StandardScaler 로드 및 정규화
    scaler, _ = load_audio_model_and_scaler()
    preprocessed_sound_1s = scaler.transform(raw_features)

    return preprocessed_sound_1s


def sound_preprocess(preprocessed_sound_1s):
    """
    사전 학습된 XGBoost 모델을 로드하여 고장 여부(0: 정상, 1: 고장) 및 고장 확률 추론
    """
    _, model = load_audio_model_and_scaler()

    prediction = int(model.predict(preprocessed_sound_1s)[0])
    probabilities = model.predict_proba(preprocessed_sound_1s)[0]
    fault_prob = float(probabilities[1])

    sound_model_output = {
        "prediction": prediction,       # 0: Normal, 1: Abnormal
        "fault_probability": fault_prob, # 고장 발생 확률 (0.0 ~ 1.0)
        "status": "ABNORMAL" if prediction == 1 else "NORMAL"
    }

    return sound_model_output


def sound_main():
    """
    실시간 오디오 수집 -> DSP 통계량 전처리 -> AI 모델 추론 메인 파이프라인
    """
    audio_1s = record_audio()
    preprocessed_sound_1s = preprocess_sound(audio_1s)
    sound_model_output = sound_preprocess(preprocessed_sound_1s)

    return sound_model_output

