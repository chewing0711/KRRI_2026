import os
import sounddevice as sd


class SoundNoDevice(Exception):
    pass


class CANNoDevice(Exception):
    pass


def check_mic():
    for dev in sd.query_devices():
        if dev.get("max_input_channels", 0) > 0:
            return True
    raise SoundNoDevice("No such device.")


def check_can():
    if not os.path.exists("/sys/class/net/can0"):
        raise CANNoDevice("No CAN device detected (can0).")
    return True
