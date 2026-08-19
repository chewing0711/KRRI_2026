# 가속 브레이크 페달 / 횡가속도 요레이트/ 조향각 조향속도/ 네 바퀴 속도
CAN_ID_LIST = ['0x371', '0x220', '0x2b0', '0x386']

CAN_DATA = {
    '0x371': [],
    '0x220': [],
    '0x2b0': [],
    '0x386': []
}

LOG_INTERVAL = 1
NEXT_INPUT_TIME = 1

def parser(msg):
    received_time = msg.timestamp
    can_id = hex(msg.arbitration_id)
    data = list(msg.data)

    if can_id in CAN_ID_LIST:
        return received_time, can_id, data
    else:
        return received_time, None, None

def collect_data(relative_time, can_id, data):
    global NEXT_INPUT_TIME, CAN_DATA

    results_1s = None

    # 1초 경계를 넘었다면 기존 데이터 반환
    if relative_time >= NEXT_INPUT_TIME:
        results_1s = CAN_DATA

        CAN_DATA = {
            '0x371': [],
            '0x220': [],
            '0x2b0': [],
            '0x386': []
        }

        NEXT_INPUT_TIME += LOG_INTERVAL

    # 현재 패킷은 새로운 구간에 저장
    if can_id in CAN_DATA:
        CAN_DATA[can_id].append({
            "time": relative_time,
            "data": data
        })

    return results_1s

