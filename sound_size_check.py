import sounddevice as sd
import numpy as np

while True:
    audio = sd.rec(
        24000,
        samplerate=48000,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    print(f"RMS: {rms:.6f} | Peak: {peak:.6f}", flush=True)