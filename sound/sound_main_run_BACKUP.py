import sounddevice as sd


def sound_main():

    audio_1s = record_audio()
    preprocessed_sound_1s = preprocess_sound(audio_1s)
    sound_model_output = sound_preprocess(preprocessed_sound_1s)
    
    return sound_model_output


def record_audio(duration=1.0, device=None):
    if device is None:
        device = sd.default.device[0]

    device_info = sd.query_devices(device, "input")
    samplerate = int(device_info["default_samplerate"])

    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device
    )

    sd.wait()

    return audio