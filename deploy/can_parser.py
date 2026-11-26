import time

CAN_IDS = ["0x371", "0x220", "0x2b0", "0x386"]
WINDOW_SEC = 0.5


def _empty_buf():
    return {cid: [] for cid in CAN_IDS}


_buf = _empty_buf()
_next_slot = WINDOW_SEC
_base_time = None


def reset_can():
    global _buf, _next_slot, _base_time
    _buf = _empty_buf()
    _next_slot = WINDOW_SEC
    _base_time = None


def collect_can(bus, timeout_seconds=2.0):
    global _base_time
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if bus is None:
        time.sleep(timeout_seconds)
        reset_can()
        return None, None

    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        rem = deadline - time.monotonic()
        msg = bus.recv(timeout=min(0.1, max(0.0, rem)))
        if msg is None:
            continue

        ts, cid, data = parse_msg(msg)
        if cid is None:
            continue
        if _base_time is None:
            _base_time = ts

        rel_time = ts - _base_time
        window = append_can(rel_time, cid, data)
        if window is not None:
            return window, rel_time

    reset_can()
    return None, None


def parse_msg(msg):
    ts = msg.timestamp
    cid = hex(msg.arbitration_id)
    if cid in CAN_IDS:
        return ts, cid, list(msg.data)
    return ts, None, None


def append_can(rel_time, cid, data):
    global _next_slot, _buf
    ready = None

    if rel_time >= _next_slot:
        ready = _buf
        _buf = _empty_buf()
        _next_slot = (int(rel_time / WINDOW_SEC) + 1) * WINDOW_SEC

    if cid in _buf:
        _buf[cid].append({"time": rel_time, "data": data})
    return ready


# 호환 별칭
reset_can_collection = reset_can
parser = parse_msg
collect_data_from_can = append_can
