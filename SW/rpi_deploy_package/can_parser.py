"""Raspberry Pi CAN 수집 모듈 (pi-hw).

기존 0.5초 윈도우 구성과 반환 형식을 유지한다.
연속 2초간 유효한 윈도우를 받지 못하면 (None, None)을 반환한다.
"""
import time

# 가속/브레이크, 횡가속도/요레이트, 조향각/조향속도, 네 바퀴 속도
CAN_ID_LIST = ['0x371', '0x220', '0x2b0', '0x386']
LOG_INTERVAL = 0.5


def _empty_can_data():
    return {can_id: [] for can_id in CAN_ID_LIST}


CAN_DATA = _empty_can_data()
NEXT_INPUT_TIME = LOG_INTERVAL
_start_time = None


def reset_can_collection():
    """연결 중단 시 예전 0.5초 부분 윈도우를 버리고 새 세션에서 시작."""
    global CAN_DATA, NEXT_INPUT_TIME, _start_time
    CAN_DATA = _empty_can_data()
    NEXT_INPUT_TIME = LOG_INTERVAL
    _start_time = None


def collect_can(bus, timeout_seconds=2.0):
    """0.5초 CAN 데이터 수집. timeout 시 (None, None) 반환.

    `bus.recv(timeout=0.1)`의 timeout만으로 전체 수집 시간이 제한되지는
    않으므로 monotonic deadline을 별도로 사용한다.
    """
    global _start_time
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        msg = bus.recv(timeout=min(0.1, max(0.0, remaining)))
        if msg is None:
            continue

        received_time, can_id_str, data = parser(msg)
        if can_id_str is None:
            continue
        if _start_time is None:
            _start_time = received_time
        relative_time = received_time - _start_time
        result = collect_data_from_can(relative_time, can_id_str, data)
        if result is not None:
            return result, relative_time

    reset_can_collection()
    return None, None


def parser(msg):
    received_time = msg.timestamp
    can_id_str = hex(msg.arbitration_id)
    data = list(msg.data)
    if can_id_str in CAN_ID_LIST:
        return received_time, can_id_str, data
    return received_time, None, None


def collect_data_from_can(relative_time, can_id, data):
    global NEXT_INPUT_TIME, CAN_DATA
    result = None

    if relative_time >= NEXT_INPUT_TIME:
        result = CAN_DATA
        CAN_DATA = _empty_can_data()
        # 송신 중단/불규칙한 시간 간격이 있었으면 지난 구간을 건너뛴다.
        NEXT_INPUT_TIME = (int(relative_time / LOG_INTERVAL) + 1) * LOG_INTERVAL

    if can_id in CAN_DATA:
        CAN_DATA[can_id].append({"time": relative_time, "data": data})
    return result
