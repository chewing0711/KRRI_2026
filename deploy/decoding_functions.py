def parse_bits(data, start, length, signed=False):
    val = int.from_bytes(bytes(data), byteorder="little")
    mask = (1 << length) - 1
    raw = (val >> start) & mask
    if signed and (raw & (1 << (length - 1))):
        raw -= (1 << length)
    return raw


def decode_can(frames_by_id):
    if not isinstance(frames_by_id, dict):
        frames_by_id = {}
    return {
        "0x371": decode_pedal(frames_by_id.get("0x371", [])),
        "0x220": decode_motion(frames_by_id.get("0x220", [])),
        "0x2b0": decode_steering(frames_by_id.get("0x2b0", [])),
        "0x386": decode_wheel_speed(frames_by_id.get("0x386", [])),
    }


def decode_pedal(packets):
    out = []
    for pkt in packets:
        d = pkt["data"]
        out.append({
            "time": pkt["time"],
            "accel_pedal": parse_bits(d, start=31, length=8),
            "brake_pedal": parse_bits(d, start=0, length=8),
        })
    return out


def decode_motion(packets):
    out = []
    for pkt in packets:
        d = pkt["data"]
        out.append({
            "time": pkt["time"],
            "lat_accel": parse_bits(d, start=0, length=11) * 0.01 - 10.23,
            "yaw_rate": parse_bits(d, start=40, length=13) * 0.01 - 40.95,
        })
    return out


def decode_steering(packets):
    out = []
    for pkt in packets:
        d = pkt["data"]
        out.append({
            "time": pkt["time"],
            "steering_angle": parse_bits(d, start=0, length=16, signed=True) * 0.1,
            "steering_speed": parse_bits(d, start=16, length=8) * 4.0,
        })
    return out


def decode_wheel_speed(packets):
    out = []
    for pkt in packets:
        d = pkt["data"]
        out.append({
            "time": pkt["time"],
            "wheel_speed_fl": parse_bits(d, start=0, length=14) * 0.03125,
            "wheel_speed_fr": parse_bits(d, start=16, length=14) * 0.03125,
            "wheel_speed_rl": parse_bits(d, start=32, length=14) * 0.03125,
            "wheel_speed_rr": parse_bits(d, start=48, length=14) * 0.03125,
        })
    return out


# 호환 별칭
extract_bits = parse_bits
decoding = decode_can
decode_371 = decode_pedal
decode_220 = decode_motion
decode_2b0 = decode_steering
decode_386 = decode_wheel_speed
