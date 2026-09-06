import can

bus = can.Bus(
    interface="slcan",
    channel="/dev/ttyACM0",
    bitrate=500000,
)

for msg in bus:
    print(
        hex(msg.arbitration_id),
        "EXT=", msg.is_extended_id,
        "FD=", msg.is_fd,
        "BRS=", msg.bitrate_switch,
        "DLC=", msg.dlc,
        msg.data.hex(),
    )