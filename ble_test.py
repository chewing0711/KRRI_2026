import sys
import wave

import objc
import IOBluetooth

from Foundation import NSObject
from Foundation import NSRunLoop
from Foundation import NSDate


PI_MAC = "2C:CF:67:76:9D:6E"

RFCOMM_CHANNEL = 1

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

SAMPLES_PER_PACKET = 320
PACKET_SIZE = 640

OUTPUT_FILE = "received.wav"


# ============================================================
# Pairing
# ============================================================

class PairingDelegate(NSObject):

    def init(self):
        self = objc.super(PairingDelegate, self).init()

        if self is None:
            return None

        self.finished = False
        self.result = None

        return self

    def devicePairingStarted_(self, sender):
        print("Pairing started")

    def devicePairingConnecting_(self, sender):
        print("Pairing connecting")

    def devicePairingConnected_(self, sender):
        print("Pairing connected")

    def devicePairingUserConfirmationRequest_numericValue_(
        self,
        sender,
        numeric_value,
    ):
        print(
            "Pairing confirmation:",
            numeric_value,
        )

        # Just Works / numeric confirmation 자동 승인
        sender.replyUserConfirmation_(True)

    def deviceSimplePairingComplete_status_(
        self,
        sender,
        status,
    ):
        print(
            "Simple pairing status:",
            status,
        )

    def devicePairingFinished_error_(
        self,
        sender,
        error,
    ):
        self.result = error
        self.finished = True

        print(
            "Pairing finished:",
            f"0x{error & 0xFFFFFFFF:08X}",
        )


def pair_device(device):

    if device.isPaired():
        print("Already paired")
        return True

    print("Not paired -> starting pairing")

    delegate = PairingDelegate.alloc().init()

    pair = (
        IOBluetooth.IOBluetoothDevicePair
        .pairWithDevice_(device)
    )

    if pair is None:
        print("Cannot create pairing object")
        return False

    pair.setDelegate_(delegate)

    result = pair.start()

    if result != 0:
        print(
            "Pairing start failed:",
            f"0x{result & 0xFFFFFFFF:08X}",
        )
        return False

    run_loop = NSRunLoop.currentRunLoop()

    while not delegate.finished:

        run_loop.runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(
                0.1
            )
        )

    if delegate.result != 0:
        print("Pairing failed")
        return False

    print("Pairing OK")

    return True


# ============================================================
# RFCOMM Receiver
# ============================================================

class Receiver(NSObject):

    def init(self):
        self = objc.super(Receiver, self).init()

        if self is None:
            return None

        self.buffer = bytearray()
        self.packet_count = 0
        self.closed = False

        self.wav = wave.open(
            OUTPUT_FILE,
            "wb",
        )

        self.wav.setnchannels(CHANNELS)
        self.wav.setsampwidth(SAMPLE_WIDTH)
        self.wav.setframerate(SAMPLE_RATE)

        return self

    def rfcommChannelData_data_length_(
        self,
        channel,
        data_pointer,
        data_length,
    ):

        try:
            data = bytes(
                data_pointer.as_buffer(
                    data_length
                )
            )

        except Exception:
            data = bytes(
                data_pointer[:data_length]
            )

        self.buffer.extend(data)

        while len(self.buffer) >= PACKET_SIZE:

            packet = bytes(
                self.buffer[:PACKET_SIZE]
            )

            del self.buffer[:PACKET_SIZE]

            self.wav.writeframesraw(packet)

            self.packet_count += 1

            if self.packet_count % 50 == 0:

                seconds = (
                    self.packet_count
                    * SAMPLES_PER_PACKET
                    / SAMPLE_RATE
                )

                print(
                    f"received={self.packet_count} "
                    f"time={seconds:.1f}s"
                )

    def rfcommChannelClosed_(
        self,
        channel,
    ):

        print("RFCOMM closed")

        self.closed = True

        try:
            self.wav.close()
        except Exception:
            pass


# ============================================================
# Main
# ============================================================

def main():

    device = (
        IOBluetooth.IOBluetoothDevice
        .withAddressString_(PI_MAC)
    )

    if device is None:
        print("Cannot create device")
        sys.exit(1)

    print(
        "Device:",
        device.getNameOrAddress(),
    )

    print(
        "Paired:",
        bool(device.isPaired()),
    )

    # ------------------------------------------------
    # Pair first
    # ------------------------------------------------

    if not pair_device(device):
        sys.exit(1)

    print(
        "Paired:",
        bool(device.isPaired()),
    )

    # ------------------------------------------------
    # Bluetooth ACL connection
    # ------------------------------------------------

    if not device.isConnected():

        print(
            "Opening Bluetooth connection..."
        )

        result = device.openConnection()

        if result != 0:

            print(
                "Bluetooth connection failed:",
                f"0x{result & 0xFFFFFFFF:08X}",
            )

            sys.exit(1)

    print("Bluetooth connected")

    # ------------------------------------------------
    # RFCOMM
    # ------------------------------------------------

    receiver = Receiver.alloc().init()

    print(
        f"Opening RFCOMM channel "
        f"{RFCOMM_CHANNEL}..."
    )

    ret = (
        device
        .openRFCOMMChannelSync_withChannelID_delegate_(
            None,
            RFCOMM_CHANNEL,
            receiver,
        )
    )

    if isinstance(ret, tuple):
        result = ret[0]
        channel = ret[1]

    else:
        result = ret
        channel = None

    if result != 0:

        print(
            "RFCOMM connection failed:",
            f"0x{result & 0xFFFFFFFF:08X}",
        )

        sys.exit(1)

    print("RFCOMM connected")

    if channel is not None:
        print("MTU:", channel.getMTU())

    print("Receiving...")

    # ------------------------------------------------
    # Run loop
    # ------------------------------------------------

    run_loop = NSRunLoop.currentRunLoop()

    try:

        while not receiver.closed:

            run_loop.runUntilDate_(
                NSDate.dateWithTimeIntervalSinceNow_(
                    0.05
                )
            )

    except KeyboardInterrupt:
        print("Stopping")

    finally:

        if channel is not None:
            try:
                channel.closeChannel()
            except Exception:
                pass

        try:
            receiver.wav.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()