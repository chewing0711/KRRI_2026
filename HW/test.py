import can

bus = can.Bus(
    interface="socketcan",
    channel="can0"
)

print("Waiting for CAN.")

try:
    while True:
        msg = bus.recv()

        if msg is None:
            continue

        print(
            f"Time: {msg.timestamp:.5f} | "
            f"CAN_ID: {hex(msg.arbitration_id)} | "
            f"DLC: {msg.dlc} | "
            f"DATA: {msg.data.hex(' ')}"
        )
        

except KeyboardInterrupt:
    print("Exit.")

finally:
    bus.shutdown()