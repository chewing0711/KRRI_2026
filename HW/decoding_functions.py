def extract_bits(data, start_bit, length, signed=False):
    # CAN 데이터 바이트 배열에서 원하는 비트 구간만 잘라서 정수값으로 꺼내는 함수
    
    value = int.from_bytes(bytes(data), byteorder='little')

    mask = (1 << length) - 1
    raw = (value >> start_bit) & mask

    # signed 값이면 2's complement 처리
    if signed and (raw & (1 << (length - 1))):
        raw -= (1 << length)

    return raw

def decoding(results_1s):
    decoded_data = {}

    decoded_data['0x371'] = decode_371(results_1s['0x371'])
    decoded_data['0x220'] = decode_220(results_1s['0x220'])
    decoded_data['0x2b0'] = decode_2b0(results_1s['0x2b0'])
    decoded_data['0x386'] = decode_386(results_1s['0x386'])

    return decoded_data

# 0x371 DBC 정의가 안보임
def decode_371(packets):
    results = []

    for packet in packets:
        results.append({
            'time': packet['time'],
            'data': packet['data']
        })

    return results

# yaw rate, lateral acceleration
def decode_220(packets):
    """
    LAT_ACCEL: bit 0부터 11bit, raw × 0.01 - 10.23
    YAW_RATE: bit 40부터 13bit, raw × 0.01 - 40.95
    """
    results = []

    for packet in packets:
        data = packet['data']

        lat_raw = extract_bits(data, 0, 11)
        yaw_raw = extract_bits(data, 40, 13)

        lat_accel = lat_raw * 0.01 - 10.23
        yaw_rate = yaw_raw * 0.01 - 40.95

        results.append({
            'time': packet['time'],
            'lat_accel': lat_accel,
            'yaw_rate': yaw_rate
        })

    return results

# 조향각 조향속도
def decode_2b0(packets):
    """
    SAS_Angle: bit 0부터 signed 16bit, raw × 0.1
    SAS_Speed: bit 16부터 8bit, raw × 4.0
    """
    results = []

    for packet in packets:
        data = packet['data']

        angle_raw = extract_bits(
            data,
            0,
            16,
            signed=True
        )

        speed_raw = extract_bits(
            data,
            16,
            8
        )

        steering_angle = angle_raw * 0.1
        steering_speed = speed_raw * 4.0

        results.append({
            'time': packet['time'],
            'steering_angle': steering_angle,
            'steering_speed': steering_speed
        })

    return results

# 네 바퀴 속도
def decode_386(packets):
    """
    FL: 0부터 14bit
    FR: 16부터 14bit
    RL: 32부터 14bit
    RR: 48부터 14bit
    """
    results = []

    for packet in packets:
        data = packet['data']

        fl_raw = extract_bits(data, 0, 14)
        fr_raw = extract_bits(data, 16, 14)
        rl_raw = extract_bits(data, 32, 14)
        rr_raw = extract_bits(data, 48, 14)

        results.append({
            'time': packet['time'],
            'wheel_speed_fl': fl_raw * 0.03125,
            'wheel_speed_fr': fr_raw * 0.03125,
            'wheel_speed_rl': rl_raw * 0.03125,
            'wheel_speed_rr': rr_raw * 0.03125
        })

    return results