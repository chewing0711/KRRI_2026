import sounddevice as sd

class SoundNoDevice(Exception):
    pass

class CANNoDevice(Exception):
    pass

def check_mic():
    # try:
    devices = sd.query_devices()

    for device in devices:
        if device["max_input_channels"] > 0:
            return True

    raise SoundNoDevice("No such device.")


# def check_can():
#     pass