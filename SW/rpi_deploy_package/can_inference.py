import os
import sys
import time
import queue
import threading
import joblib
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import torch
import can

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, "src"))


import can_parser
import decoding_functions
from can_models import AnomalyAutoEncoder, SingleGRU
from sequence_matrix_builder import RealtimeSequenceBuffer


def compute_direct_window_features(ws_seg, yr_seg, sas_seg, ab_seg, spd_target, grd_target):
    """실시간 디코딩된 윈도우 데이터(DataFrame)에서 41개 피처 추출"""
    fl = ws_seg["wheel_speed_fl"].values
    fr = ws_seg["wheel_speed_fr"].values
    rl = ws_seg["wheel_speed_rl"].values
    rr = ws_seg["wheel_speed_rr"].values

    lat_acc = yr_seg["lat_accel"].values
    yaw_rt = yr_seg["yaw_rate"].values

    sas_ang = sas_seg["steering_angle"].values
    sas_spd = sas_seg["steering_speed"].values

    acc_pos = ab_seg["accel_pedal"].values
    brk_pos = ab_seg["brake_pedal"].values

    whl_all = np.concatenate([fl, fr, rl, rr])

    # 1. 휠속도 기본 통계량
    mean_val = float(np.mean(whl_all))
    std_val = float(np.std(whl_all))
    p2p_val = float(np.ptp(whl_all))
    rms_val = float(np.sqrt(np.mean(whl_all ** 2)))
    var_val = float(np.var(whl_all))
    peak_val = float(np.max(np.abs(whl_all)))
    crest_factor = float(peak_val / (rms_val + 1e-6))
    shape_factor = float(rms_val / (mean_val + 1e-6))
    skew_val = float(stats.skew(whl_all)) if len(whl_all) > 1 else 0.0
    kurt_val = float(stats.kurtosis(whl_all)) if len(whl_all) > 1 else 0.0

    # 2. 4륜 간 상대 속도차
    diff_fl_rl = fl - rl
    diff_fr_rr = fr - rr
    diff_fl_fr = fl - fr
    diff_rl_rr = rl - rr

    diff_fl_rl_mean = float(np.mean(diff_fl_rl))
    diff_fl_rl_std = float(np.std(diff_fl_rl)) if len(diff_fl_rl) > 1 else 0.0
    diff_fl_rl_p2p = float(np.ptp(diff_fl_rl))

    diff_fr_rr_mean = float(np.mean(diff_fr_rr))
    diff_fr_rr_std = float(np.std(diff_fr_rr)) if len(diff_fr_rr) > 1 else 0.0
    diff_fr_rr_p2p = float(np.ptp(diff_fr_rr))

    diff_fl_fr_std = float(np.std(diff_fl_fr)) if len(diff_fl_fr) > 1 else 0.0
    diff_rl_rr_std = float(np.std(diff_rl_rr)) if len(diff_rl_rr) > 1 else 0.0

    # 3. 차체 거동 동역학
    lat_accel_mean = float(np.mean(lat_acc))
    lat_accel_std = float(np.std(lat_acc)) if len(lat_acc) > 1 else 0.0
    yaw_rate_mean = float(np.mean(yaw_rt))
    yaw_rate_std = float(np.std(yaw_rt)) if len(yaw_rt) > 1 else 0.0

    # 4. 조향 시스템 특성
    sas_angle_mean = float(np.mean(sas_ang))
    sas_angle_std = float(np.std(sas_ang)) if len(sas_ang) > 1 else 0.0
    sas_angle_p2p = float(np.ptp(sas_ang))
    sas_speed_mean = float(np.mean(sas_spd))

    # 5. 운전자 조작 특성
    accel_pedal_mean = float(np.mean(acc_pos))
    accel_pedal_max = float(np.max(acc_pos))
    brake_pedal_mean = float(np.mean(brk_pos))
    brake_pedal_max = float(np.max(brk_pos))

    # 6. 무차원 슬립 비율
    fl_mean = float(np.mean(fl))
    fr_mean = float(np.mean(fr))
    rl_mean = float(np.mean(rl))
    rr_mean = float(np.mean(rr))

    base_spd = max(spd_target, 1.0)
    slip_ratio_fl = fl_mean / base_spd
    slip_ratio_fr = fr_mean / base_spd
    slip_ratio_rl = rl_mean / base_spd
    slip_ratio_rr = rr_mean / base_spd

    slip_diff_front_rear = ((fl_mean + fr_mean) / 2.0) - ((rl_mean + rr_mean) / 2.0)
    norm_yaw_rate = yaw_rate_mean / base_spd
    norm_lat_accel = lat_accel_mean / (grd_target + 1.0)

    # 7. 무차원 특징 추가
    dimless_std = std_val / max(mean_val, 1.0)
    dimless_p2p = p2p_val / max(mean_val, 1.0)
    dimless_diff_fr_rr_std = diff_fr_rr_std / max(mean_val, 1.0)
    pedal_slip_response = diff_fr_rr_std / max(accel_pedal_mean, 1.0)

    return [
        slip_ratio_fl, slip_ratio_fr, slip_ratio_rl, slip_ratio_rr, slip_diff_front_rear, norm_yaw_rate, norm_lat_accel,
        mean_val, std_val, p2p_val, rms_val, var_val, peak_val, crest_factor, shape_factor, skew_val, kurt_val,
        diff_fl_rl_mean, diff_fl_rl_std, diff_fl_rl_p2p, diff_fr_rr_mean, diff_fr_rr_std, diff_fr_rr_p2p, diff_fl_fr_std, diff_rl_rr_std,
        lat_accel_mean, lat_accel_std, yaw_rate_mean, yaw_rate_std,
        sas_angle_mean, sas_angle_std, sas_angle_p2p, sas_speed_mean,
        accel_pedal_mean, accel_pedal_max, brake_pedal_mean, brake_pedal_max,
        dimless_std, dimless_p2p, dimless_diff_fr_rr_std, pedal_slip_response
    ]

_CACHED_CAN_MODELS = None

def init_can_models(w_suffix="_w34"):
    """모델 파일, 스케일러, 시계열 버퍼 로드 및 내부 캐시 저장"""
    global _CACHED_CAN_MODELS

    scaler_path = os.path.join(MODELS_DIR, f"scaler_can{w_suffix}.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"[ERROR] 모델 파일 없음: {scaler_path}")
    scaler = joblib.load(scaler_path)

    ae_model = AnomalyAutoEncoder(in_dim=43, latent_dim=8)
    ae_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step1_autoencoder{w_suffix}.pt"), map_location="cpu", weights_only=True))
    ae_model.eval()

    oc_svm = joblib.load(os.path.join(MODELS_DIR, f"step1_oc_svm{w_suffix}.pkl"))
    iforest = joblib.load(os.path.join(MODELS_DIR, f"step1_iforest{w_suffix}.pkl"))

    # 최우수 GRU 모델 로드 (iForest + GRU)
    gru_model = SingleGRU(in_dim=44, hidden=32)
    gru_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_if_gru_can{w_suffix}.pt"), map_location="cpu", weights_only=True))
    gru_model.eval()

    # 시계열 FIFO 버퍼 설정
    seq_len = 10
    buf_if = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)

    _CACHED_CAN_MODELS = {
        "scaler": scaler,
        "ae_model": ae_model,
        "oc_svm": oc_svm,
        "iforest": iforest,
        "gru_model": gru_model,
        "buf_if": buf_if
    }

    return _CACHED_CAN_MODELS


def init_can_bus(channel="can0", interface="socketcan"):
    """CAN 버스 소켓 연결 초기화"""
    try:
        bus = can.Bus(interface=interface, channel=channel)
        print(f" ▶ [{channel}] RAW CAN 포트 청취 중...")
        return bus
    except Exception as e:
        print(f"[ERROR] CAN 버스 연결 실패: {e}")
        raise e


_start_time = None

def init_can_parser_state():
    """can_parser의 내부 버퍼와 타임라인 초기화"""
    global _start_time
    can_parser.NEXT_INPUT_TIME = 1.0
    can_parser.CAN_DATA = {
        '0x371': [],
        '0x220': [],
        '0x2b0': [],
        '0x386': []
    }
    _start_time = None


def collect_can(bus):
    """
    [단독 동기 호출용] CAN 버스로부터 1초 분량의 패킷 데이터를 스니핑하여 반환.
    1초 데이터 획득 완료 시 (results_1s, relative_time) 반환.
    """
    global _start_time

    while True:
        msg = bus.recv(timeout=0.1)
        if msg is None:
            continue

        # HW can_parser를 이용해 ID별로 수신 패킷 수집
        received_time, can_id_str, data = can_parser.parser(msg)

        if can_id_str is not None:
            if _start_time is None:
                _start_time = received_time

            relative_time = received_time - _start_time
            results_1s = can_parser.collect_data(relative_time, can_id_str, data)

            # 1.0초 경계 수집 완료 시 즉시 반환
            if results_1s is not None:
                return results_1s, relative_time


def can_collector_thread(bus, can_queue, stop_event=None):
     """
     [백그라운드 스레드] 1초 단위로 CAN 패킷을 연속 수집
     
       1. CAN 버스에서 메시지를 끊김 없이 계속 수집 (공백 없이)
       2. 1초 분량이 완성될 때마다 frame_id를 1씩 증가
       3. can_queue(maxsize=1)에 최신 1초 데이터 삽입
       4. 만약 큐가 차있으면(오디오가 정상이어서 소비를 안 한 경우),
          이전 오래된 1초 데이터를 버리고(Drop-oldest) '가장 최신 1초'로 덮어씀 (메모리 누수 및 시차 방지)
     """
     global _start_time
     frame_id = 0
 
     print(" [CAN Thread] 백그라운드 CAN 1초 연속 수집 스레드 시작")
 
     while stop_event is None or not stop_event.is_set():
         msg = bus.recv(timeout=0.1)
         if msg is None:
             continue
 
         # 1. 패킷 파싱
         received_time, can_id_str, data = can_parser.parser(msg)
 
         if can_id_str is not None:
             if _start_time is None:
                 _start_time = received_time
 
             relative_time = received_time - _start_time
             results_1s = can_parser.collect_data(relative_time, can_id_str, data)
 
             # 2. 1.0초 버퍼 완성 시
             if results_1s is not None:
                 frame_id += 1
                 item = {
                     "frame_id": frame_id,          # 오디오 1초와 1:1 매칭용 회차 번호
                     "relative_time": relative_time, # 타임스탬프
                     "results_1s": results_1s        # 1초 RAW 데이터
                 }
 
                 # 3. 큐가 꽉 차 있으면(maxsize=1), 과거 데이터를 버리고 최신 데이터로 덮어쓰기
                 try:
                     can_queue.put_nowait(item)
                 except queue.Full:
                     try:
                         _ = can_queue.get_nowait()  # 과거 1초 데이터 폐기 (시차 방지)
                     except queue.Empty:
                         pass
                     can_queue.put_nowait(item)     # 최신 1초 데이터로 갱신

def infer_can(results_1s, relative_time, models, speed=20.0, grade=0.0):
    """
    수집된 1초 CAN 데이터를 HW 디코딩, 피처 추출 후 모델 추론 수행
    반환값: (is_abnormal: bool, abnormal_ratio: float, status_str: str)
    """
    scaler = models["scaler"]
    iforest = models["iforest"]
    gru_model = models["gru_model"]
    buf_if = models["buf_if"]

    # 1. HW 디코더 실행
    dataset = decoding_functions.decoding(results_1s)

    df_ws = pd.DataFrame(dataset['0x386'])
    df_yr = pd.DataFrame(dataset['0x220'])
    df_sas = pd.DataFrame(dataset['0x2b0'])
    df_ab = pd.DataFrame(dataset['0x371'])

    # 1초 데이터를 0.1초 단위 윈도우 10개로 쪼개어 실시간 추론 진행
    t_base = relative_time - 1.0
    abnormal_votes = 0

    for i in range(10):
        w_start = t_base + i * 0.1
        w_end = w_start + 0.1

        ws_seg = df_ws[(df_ws["time"] >= w_start) & (df_ws["time"] < w_end)]
        yr_seg = df_yr[(df_yr["time"] >= w_start) & (df_yr["time"] < w_end)]
        sas_seg = df_sas[(df_sas["time"] >= w_start) & (df_sas["time"] < w_end)]
        ab_seg = df_ab[(df_ab["time"] >= w_start) & (df_ab["time"] < w_end)]

        # 패킷 누락 시 캐리 오버 (ZOH 대체 작동)
        if len(ws_seg) == 0:
            ws_prev = df_ws[df_ws["time"] < w_start]
            ws_seg = ws_prev.tail(1) if len(ws_prev) > 0 else df_ws.head(1)
        if len(yr_seg) == 0:
            yr_prev = df_yr[df_yr["time"] < w_start]
            yr_seg = yr_prev.tail(1) if len(yr_prev) > 0 else df_yr.head(1)
        if len(sas_seg) == 0:
            sas_prev = df_sas[df_sas["time"] < w_start]
            sas_seg = sas_prev.tail(1) if len(sas_prev) > 0 else df_sas.head(1)
        if len(ab_seg) == 0:
            ab_prev = df_ab[df_ab["time"] < w_start]
            ab_seg = ab_prev.tail(1) if len(ab_prev) > 0 else df_ab.head(1)

        if len(ws_seg) == 0 or len(yr_seg) == 0 or len(sas_seg) == 0 or len(ab_seg) == 0:
            continue

        # 41개 피처 추출 및 43개 조합
        raw_feats = compute_direct_window_features(ws_seg, yr_seg, sas_seg, ab_seg, speed, grade)
        features_43 = [speed, grade] + raw_feats

        # 스케일러 및 Step 1 iForest 적용
        X_scaled = scaler.transform([features_43])[0]
        if_score = iforest.decision_function(X_scaled.reshape(1, -1))[0]

        # 44차원 벡터 버퍼 Push
        buf_if.push(np.append(X_scaled, if_score))

        # Step 2 GRU 추론
        seq_if = buf_if.get_sequence_tensor()
        with torch.no_grad():
            out = gru_model(torch.tensor(seq_if, dtype=torch.float32))
            pred = torch.argmax(out, dim=1).item()
            if pred == 1:
                abnormal_votes += 1

    # 10개 윈도우 판정 기반 종합 이상 판정 비율
    abnormal_ratio = (abnormal_votes / 10.0) * 100
    is_abnormal = abnormal_ratio >= 50.0
    status_str = "\033[91m이상(ABNORMAL)\033[0m" if is_abnormal else "\033[92m정상(NORMAL)\033[0m"

    return is_abnormal, abnormal_ratio, status_str


# ==============================================================================
# [main.py 연동용 편의 Wrapper 함수]
# ==============================================================================
def init_can(channel="can0", interface="socketcan", w_suffix="_w34"):
    """
    [1. 초기화 Wrapper] 
    모델 로드(내부 캐싱) + CAN 버스 연결 + 파서 초기화를 한 번에 수행하여 bus 반환
    """
    init_can_models(w_suffix)
    bus = init_can_bus(channel=channel, interface=interface)
    init_can_parser_state()
    return bus


def can_collect(bus):
    """
    [2. 1초 수집 Wrapper] 
    main의 ThreadPoolExecutor에서 병렬 수집할 때 사용
    반환값: (results_1s, relative_time)
    """
    return collect_can(bus)


def can_infer(results_1s, relative_time, speed=20.0, grade=0.0, models=None):
    """
    [3. 추론 Wrapper] 
    오디오가 비정상일 때 수집된 1초 CAN 데이터를 넘겨서 고장 여부 판정 (models 인자 생략 가능)
    반환값: dict {"is_abnormal": bool, "abnormal_ratio": float, "status": str, "status_display": str}
    """
    global _CACHED_CAN_MODELS
    target_models = models if models is not None else _CACHED_CAN_MODELS
    if target_models is None:
        target_models = init_can_models()

    is_abnormal, abnormal_ratio, status_str = infer_can(
        results_1s, relative_time, target_models, speed=speed, grade=grade
    )
    return {
        "is_abnormal": is_abnormal,
        "abnormal_ratio": abnormal_ratio,
        "status": "ABNORMAL" if is_abnormal else "NORMAL",
        "status_display": status_str
    }


# CAN 1s 스니핑해서 메모리에 저장 / 전처리 및 모델 추론
def can_main():
    parser = argparse.ArgumentParser(description="실시간 CAN 고장 진단 노드")
    parser.add_argument("--interface", type=str, default="socketcan", help="python-can 인터페이스 타입")
    parser.add_argument("--channel", type=str, default="can0", help="CAN 채널 이름")
    parser.add_argument("--speed", type=float, default=20.0, help="현재 차량 목표 주행 속도 (kph)")
    parser.add_argument("--grade", type=float, default=0.0, help="현재 주행 도로 경사도 (%)")
    args = parser.parse_args()

    W_SUFFIX = "_w34"

    print("=" * 110)
    print(" [실시간 CAN 버스 고장 진단 시스템 시작]")
    print(f" * CAN 채널: {args.channel} ({args.interface})")
    print(f" * 설정 컨텍스트: 속도 {args.speed} kph, 경사 {args.grade} %")
    print("=" * 110)

    # 1. 모델 리소스 초기화
    try:
        models = init_can_models(W_SUFFIX)
    except Exception as e:
        print(e)
        return

    # 2. CAN 버스 소켓 연결
    try:
        bus = init_can_bus(channel=args.channel, interface=args.interface)
    except Exception:
        return

    # 3. can_parser 내부 타임라인 초기화
    init_can_parser_state()

    try:
        while True:
            # 1. 1초 버퍼 데이터 획득
            results_1s, relative_time = collect_can(bus)
            print(f"\n [수신] 1초 버퍼 데이터 획득 (기준시각: {relative_time:.1f}s)")

            # 2. 전처리 및 모델 추론
            is_abnormal, abnormal_ratio, status_str = infer_can(
                results_1s, relative_time, models, speed=args.speed, grade=args.grade
            )
            print(f" ▶ 고장 판정 비율: {abnormal_ratio:>5.1f}% | 최종 상태: {status_str}")

    except KeyboardInterrupt:
        print("\n [종료] 실시간 CAN 추론 노드 청취 종료.")
    finally:
        bus.shutdown()


    # can_queue = queue.Queue(maxsize=1)
    # stop_event = threading.Event()
    # t_collector = threading.Thread(
    #     target=can_collector_thread,
    #     args=(bus, can_queue, stop_event),
    #     daemon=True
    # )
    # t_collector.start()
    # 
    # try:
    #     while True:
    #         item = can_queue.get()
    #         frame_id = item["frame_id"]
    #         relative_time = item["relative_time"]
    #         results_1s = item["results_1s"]
    # 
    #         print(f"\n [수신] Frame #{frame_id:04d} | 1초 버퍼 획득 (기준시각: {relative_time:.1f}s)")
    #         is_abnormal, abnormal_ratio, status_str = infer_can(
    #             results_1s, relative_time, models, speed=args.speed, grade=args.grade
    #         )
    #         print(f" ▶ [Frame #{frame_id:04d}] 고장 판정 비율: {abnormal_ratio:>5.1f}% | 최종 상태: {status_str}")
    # except KeyboardInterrupt:
    #     stop_event.set()
    # finally:
    #     bus.shutdown()


if __name__ == "__main__":
    can_main()