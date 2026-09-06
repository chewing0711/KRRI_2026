import sounddevice as sd
import numpy as np


def check_mic():
    try:
        devices = sd.query_devices()

        for device in devices:
            if device["max_input_channels"] > 0:
                return True

        raise RuntimeError("사용 가능한 마이크가 없습니다.")

    except RuntimeError as e:
        print(f"오디오 장치 확인 실패: {e}")

def check_mic_size():
    # RMS 값이 얼마나 되는지 확인용 함수
    """
    RMS: 0.02808
    RMS: 0.00144
    RMS: 0.00319
    RMS: 0.00609
    RMS: 0.00311
    RMS: 0.00119
    RMS: 0.00084
    RMS: 0.00105
    RMS: 0.00166
    RMS: 0.00056
    """
    for _ in range(10):
        check_sound(
            threshold=0.01,
            duration=1.0
        )


def check_sound(
    threshold=0.01,
    duration=1.0,
    samplerate=48000.0,
    device=None
):
    """
    1초 동안 마이크 입력을 받고,
    RMS가 threshold 이상이면 True 반환
    """

    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device
    )

    sd.wait()

    rms = np.sqrt(np.mean(audio ** 2))

    print(f"RMS: {rms:.5f}")

    return rms >= threshold
