import os
import glob
import time
import warnings
import numpy as np
import sounddevice as sd
from scipy.signal import butter, filtfilt, hilbert, stft, find_peaks
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

SR = 48000
DURATION = 0.5
GAIN = 2.0

_wav_cache = None
_wav_idx = 0
_wav_pos = 0


def get_filter(lowcut=1800.0, highcut=10500.0, fs=48000, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype="band")


def filter_audio(data, lowcut=1800.0, highcut=10500.0, fs=48000, order=4):
    b, a = get_filter(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)


def extract_features(audio, sr=48000):
    if len(audio) < 64:
        return np.zeros((1, 24), dtype=np.float32)

    abs_a = np.abs(audio)
    mean_abs = float(np.mean(abs_a))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(abs_a))
    mean_sqrt_abs = float(np.mean(np.sqrt(abs_a)))

    crest = float(peak / (rms + 1e-8))
    shape = float(rms / (mean_abs + 1e-8))
    impulse = float(peak / (mean_abs + 1e-8))
    margin = float(peak / ((mean_sqrt_abs ** 2) + 1e-8))
    kurt = float(kurtosis(audio, fisher=True))
    skw = float(skew(audio)) if len(audio) > 1 else 0.0

    env = np.abs(hilbert(audio))
    env_m = float(np.mean(env))
    env_s = float(np.std(env))
    env_peak = float(np.max(env))
    env_crest = float(env_peak / (env_m + 1e-8))
    env_kurt = float(kurtosis(env, fisher=True))
    env_skw = float(skew(env)) if len(env) > 1 else 0.0
    env_mod = float(env_s / (env_m + 1e-8))

    p95 = np.percentile(env, 95)
    shock_energy = float(np.sum(env[env >= p95] ** 2))
    total_energy = float(np.sum(env ** 2) + 1e-12)
    shock_ratio = float(shock_energy / total_energy)

    peaks_idx, _ = find_peaks(env, height=env_m + 2.0 * env_s, distance=int(sr * 0.001))
    shock_count = float(len(peaks_idx))

    f, _, zxx = stft(audio, fs=sr, nperseg=min(1024, len(audio)), noverlap=min(512, len(audio) // 2))
    mag = np.abs(zxx)
    pwr = mag ** 2
    tot_pwr = np.sum(pwr) + 1e-12

    log_m = np.log(mag + 1e-12)
    flatness = float(np.exp(np.mean(log_m)) / (np.mean(mag) + 1e-12))

    psd = np.mean(pwr, axis=1)
    norm_psd = psd / (np.sum(psd) + 1e-12)
    entropy = float(-np.sum(norm_psd * np.log2(norm_psd + 1e-12)))

    cum_pwr = np.cumsum(psd)
    roll_idx = np.where(cum_pwr >= 0.85 * cum_pwr[-1])[0]
    roll_freq = f[roll_idx[0]] if len(roll_idx) > 0 else f[-1]
    nyq = sr / 2.0
    rolloff = float(roll_freq / nyq)

    mean_mag = np.mean(mag, axis=1)
    centroid = float((np.sum(f * mean_mag) / (np.sum(mean_mag) + 1e-8)) / nyq)
    peak_ratio = float(np.max(psd) / (np.sum(psd) + 1e-12))

    m_2k8 = (f >= 2700) & (f <= 2900)
    p_2k8 = np.sum(pwr[m_2k8, :]) if np.any(m_2k8) else 0.0
    ratio_2k8 = float(p_2k8 / tot_pwr)

    m_harm = (f >= 4000) & (f <= 10000)
    p_harm = np.sum(pwr[m_harm, :]) if np.any(m_harm) else 0.0
    ratio_harm = float(p_harm / tot_pwr)

    nb = len(f)
    q1, q2, q3 = nb // 4, nb // 2, (3 * nb) // 4
    sub = [np.sum(psd[:q1]), np.sum(psd[q1:q2]), np.sum(psd[q2:q3]), np.sum(psd[q3:])]
    sub_tot = sum(sub) + 1e-12
    sub_r = [float(s / sub_tot) for s in sub]

    est_gmf = float(f[np.argmax(psd)])

    feats = [
        crest, shape, impulse, margin, kurt, skw,
        env_crest, env_kurt, env_skw, env_mod, shock_ratio, shock_count,
        flatness, entropy, rolloff, centroid,
        ratio_2k8, ratio_harm, peak_ratio,
        sub_r[0], sub_r[1], sub_r[2], sub_r[3], est_gmf,
    ]
    return np.array(feats, dtype=np.float32).reshape(1, -1)


def get_mic_device():
    try:
        devs = sd.query_devices()
        if not devs:
            return None
        default_in = sd.default.device[0]
        if isinstance(default_in, int) and 0 <= default_in < len(devs):
            if devs[default_in].get("max_input_channels", 0) > 0:
                return default_in
        for idx, dev in enumerate(devs):
            if dev.get("max_input_channels", 0) > 0:
                return idx
    except Exception:
        pass
    return None


def fallback_audio(duration=0.5, samplerate=48000):
    global _wav_cache, _wav_idx, _wav_pos
    if _wav_cache is None:
        import soundfile as sf
        wav_files = (
            sorted(glob.glob(os.path.join(CURRENT_DIR, "*.wav")))
            or sorted(glob.glob(os.path.join(BASE_DIR, "sound", "*.wav")))
            or sorted(glob.glob(os.path.join(BASE_DIR, "*.wav")))
        )
        _wav_cache = []
        for fp in wav_files:
            try:
                data, sr = sf.read(fp, dtype="float32")
                if data.ndim > 1:
                    data = data[:, 0]
                if sr != samplerate:
                    num = int(len(data) * float(samplerate) / sr)
                    idx = np.linspace(0, len(data) - 1, num)
                    data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
                _wav_cache.append(data)
            except Exception:
                pass
        _wav_idx = 0
        _wav_pos = 0

    chunk_size = int(duration * samplerate)
    if not _wav_cache:
        time.sleep(duration)
        return np.zeros(chunk_size, dtype=np.float32)

    cur = _wav_cache[_wav_idx]
    if _wav_pos + chunk_size > len(cur):
        _wav_idx = (_wav_idx + 1) % len(_wav_cache)
        _wav_pos = 0
        cur = _wav_cache[_wav_idx]

    chunk = cur[_wav_pos:_wav_pos + chunk_size]
    _wav_pos += chunk_size
    time.sleep(duration)
    return chunk


def record_audio(duration=0.5, samplerate=48000, device=None):
    if device is None or device < 0:
        device = get_mic_device()
    if device is None:
        return fallback_audio(duration=duration, samplerate=samplerate)

    try:
        gain = GAIN
        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            device=device,
        )
        sd.wait()
        return np.clip(audio.flatten() * gain, -1.0, 1.0)
    except Exception:
        return fallback_audio(duration=duration, samplerate=samplerate)


def sound_main():
    audio = record_audio(duration=DURATION, samplerate=SR)
    filtered = filter_audio(audio, lowcut=1800.0, highcut=10500.0, fs=SR, order=4)
    feats_24d = extract_features(filtered, sr=SR).reshape(-1)
    return audio, feats_24d


# 호환 별칭
butter_bandpass = get_filter
apply_bandpass_filter = filter_audio
extract_audio_features_from_array = extract_features
get_valid_input_device = get_mic_device
get_simulated_audio_from_wav = fallback_audio
