import sounddevice as sd


def check_mic():
    try:
        devices = sd.query_devices()

        for device in devices:
            if device["max_input_channels"] > 0:
                return True

        raise RuntimeError("사용 가능한 마이크가 없습니다.")

    except RuntimeError as e:
        print(f"오디오 장치 확인 실패: {e}")

