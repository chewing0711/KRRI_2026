import sounddevice as sd
import wave
import datetime

SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = "int16"
BLOCK_SIZE = 1024


def record_until_stop():
    filename = datetime.datetime.now().strftime("record_%Y%m%d_%H%M%S.wav")

    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # int16 = 2 bytes
    wf.setframerate(SAMPLE_RATE)

    def callback(indata, frames, time, status):
        if status:
            print(status)

        wf.writeframes(indata.tobytes())

    print(f"Recording... → {filename}")
    print("Ctrl+C to stop")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=callback,
        ):
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nRecording stopped.")

    finally:
        wf.close()
        print(f"Saved: {filename}")


if __name__ == "__main__":
    record_until_stop()