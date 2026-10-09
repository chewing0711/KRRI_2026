import sounddevice as sd

class SoundNoDevice(Exception):
    pass

class CANNoDevice(Exception):
    pass

def check_mic():
    devices = sd.query_devices()

    for device in devices:
        if device["max_input_channels"] > 0:
            return True

    raise SoundNoDevice("No such device.")


def check_can():
    import os
    if not os.path.exists("/sys/class/net/can0"):
        raise CANNoDevice("No CAN device detected (can0).")
    return True