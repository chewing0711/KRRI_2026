"""
live_can_inference_node_direct_array.py (실시간 CAN 고장 진단 노드 - Pandas DataFrame 배제 버전)

설명:
실제 차량 또는 PC의 CAN 시뮬레이터가 송신하는 RAW CAN 패킷을 실시간 수신하되,
연산 오버헤드와 가비지 컬렉션(GC) 부하를 방지하기 위해 Pandas DataFrame 생성을 완전히 배제하고,
디코딩된 딕셔너리 리스트를 순수 Python 및 NumPy Array만으로 실시간 슬라이싱 및 연산하여 추론합니다.

실행 방법:
python live_can_inference_node_direct_array.py --interface socketcan --channel can0 --speed 20 --grade 0
"""

import os
import sys
import time
import joblib
import argparse
import numpy as np
from scipy import stats
import torch
import can

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_UPLOADS_DIR = os.path.dirname(CURRENT_DIR)
SW_DIR = os.path.join(GIT_UPLOADS_DIR, "SW")
MODELS_DIR = os.path.join(SW_DIR, "models")

sys.path.insert(0, CURRENT_DIR)  # HW 폴더
sys.path.insert(0, os.path.join(SW_DIR, "src", "models_pipelines"))
sys.path.insert(0, os.path.join(SW_DIR, "src", "data_processing"))

import can_parser
import decoding_functions
from train_unified_models import AnomalyAutoEncoder, SingleGRU
from sequence_matrix_builder import RealtimeSequenceBuffer


def dict_list_to_numpy_dict(dict_list, keys):
    """딕셔너리 리스트를 각 키별 NumPy Array를 가진 단일 딕셔너리로 변환"""
    if not dict_list:
        return {k: np.array([], dtype=np.float32) for k in ['time'] + keys}
    
    # 루프를 돌며 NumPy 배열로 변환
    res = {'time': np.array([d['time'] for d in dict_list], dtype=np.float32)}
    for k in keys:
        res[k] = np.array([d[k] for d in dict_list], dtype=np.float32)
    return res


def slice_numpy_dict(np_dict, w_start, w_end, last_known_state=None):
    """0.1초 시간 범위에 속하는 데이터만 NumPy 슬라이싱 (비어있을 시 직전 값으로 ZOH)"""
    times = np_dict['time']
    indices = np.where((times >= w_start) & (times < w_end))[0]

    # 해당 윈도우 구간에 데이터가 유입되지 않은 경우
    if len(indices) == 0:
        if last_known_state is not None:
            # 직전 윈도우의 최종 값 반환
            return last_known_state
        else:
            # 이전 전체 데이터에서 마지막 수신값 탐색
            if len(times) > 0:
                idx = len(times) - 1
                return {k: np.array([np_dict[k][idx]], dtype=np.float32) for k in np_dict.keys()}
            else:
                # 초기화 상태 (0.0 기본값 설정)
                return {k: np.array([0.0], dtype=np.float32) for k in np_dict.keys()}

    sliced = {k: np_dict[k][indices] for k in np_dict.keys()}
    return sliced


def compute_direct_window_features(ws_seg, yr_seg, sas_seg, ab_seg, spd_target, grd_target):
    """NumPy Array 세그먼트 데이터로부터 직접 통계 피처 연산"""
    fl = ws_seg["wheel_speed_fl"]
    fr = ws_seg["wheel_speed_fr"]
    rl = ws_seg["wheel_speed_rl"]
    rr = ws_seg["wheel_speed_rr"]

    lat_acc = yr_seg["lat_accel"]
    yaw_rt = yr_seg["yaw_rate"]

    sas_ang = sas_seg["steering_angle"]
    sas_spd = sas_seg["steering_speed"]

    acc_pos = ab_seg["accel_pedal"]
    brk_pos = ab_seg["brake_pedal"]

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


def run_live_inference():
    parser = argparse.ArgumentParser(description="실시간 CAN 고장 진단 노드 (DataFrame 배제)")
    parser.add_argument("--interface", type=str, default="socketcan", help="python-can 인터페이스 타입")
    parser.add_argument("--channel", type=str, default="can0", help="CAN 채널 이름")
    parser.add_argument("--speed", type=float, default=20.0, help="차량 목표 주행 속도 (kph)")
    parser.add_argument("--grade", type=float, default=0.0, help="주행 도로 경사도 (%)")
    args = parser.parse_args()

    W_SUFFIX = "_w34"

    print("=" * 110)
    print(" [실시간 CAN 고장 진단 노드 - DataFrame 배제 최적화 실행]")
    print(f" * CAN 채널: {args.channel} ({args.interface})")
    print(f" * 설정 컨텍스트: 속도 {args.speed} kph, 경사 {args.grade} %")
    print("=" * 110)

    # 1. 모델 파일 로드
    scaler_path = os.path.join(MODELS_DIR, f"scaler_can{W_SUFFIX}.pkl")
    if not os.path.exists(scaler_path):
        print(f"[ERROR] 모델 파일 없음: {scaler_path}")
        return
    scaler = joblib.load(scaler_path)

    ae_model = AnomalyAutoEncoder(in_dim=43, latent_dim=8)
    ae_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step1_autoencoder{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    ae_model.eval()

    oc_svm = joblib.load(os.path.join(MODELS_DIR, f"step1_oc_svm{W_SUFFIX}.pkl"))
    iforest = joblib.load(os.path.join(MODELS_DIR, f"step1_iforest{W_SUFFIX}.pkl"))

    gru_model = SingleGRU(in_dim=44, hidden=32)
    gru_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_if_gru_can{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    gru_model.eval()

    # FIFO 링버퍼 설정
    seq_len = 10
    buf_if = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)

    # 2. CAN 버스 소켓 연결
    try:
        bus = can.Bus(interface=args.interface, channel=args.channel)
        print(f" ▶ [{args.channel}] RAW CAN 포트 청취 중...")
    except Exception as e:
        print(f"[ERROR] CAN 버스 연결 실패: {e}")
        return

    # can_parser 내부 타임라인 초기화
    can_parser.NEXT_INPUT_TIME = 1.0
    can_parser.CAN_DATA = {
        '0x371': [],
        '0x220': [],
        '0x2b0': [],
        '0x386': []
    }
    start_time = None

    # ZOH 대체용 캐리 오버 변수 초기화
    last_ws_state = None
    last_yr_state = None
    last_sas_state = None
    last_ab_state = None

    try:
        while True:
            msg = bus.recv(timeout=0.1)
            if msg is None:
                continue

            received_time, can_id_str, data = can_parser.parser(msg)

            if can_id_str is not None:
                if start_time is None:
                    start_time = received_time
                
                relative_time = received_time - start_time
                results_1s = can_parser.collect_data(relative_time, can_id_str, data)

                # 1.0초 경계 수집 완료 시
                if results_1s is not None:
                    t0 = time.perf_counter()
                    
                    # 1. HW 디코더 실행
                    dataset = decoding_functions.decoding(results_1s)
                    
                    # 2. 딕셔너리 리스트 ➔ NumPy 딕셔너리로 다이렉트 변환
                    ws_np = dict_list_to_numpy_dict(dataset['0x386'], ['wheel_speed_fl', 'wheel_speed_fr', 'wheel_speed_rl', 'wheel_speed_rr'])
                    yr_np = dict_list_to_numpy_dict(dataset['0x220'], ['lat_accel', 'yaw_rate'])
                    sas_np = dict_list_to_numpy_dict(dataset['0x2b0'], ['steering_angle', 'steering_speed'])
                    ab_np = dict_list_to_numpy_dict(dataset['0x371'], ['accel_pedal', 'brake_pedal'])

                    t_base = relative_time - 1.0
                    abnormal_votes = 0

                    # 3. 1초 구간을 0.1초 단위 10개 윈도우로 분할하여 루프
                    for i in range(10):
                        w_start = t_base + i * 0.1
                        w_end = w_start + 0.1

                        # DataFrame 배제하고 NumPy 슬라이싱 수행
                        ws_seg = slice_numpy_dict(ws_np, w_start, w_end, last_ws_state)
                        yr_seg = slice_numpy_dict(yr_np, w_start, w_end, last_yr_state)
                        sas_seg = slice_numpy_dict(sas_np, w_start, w_end, last_sas_state)
                        ab_seg = slice_numpy_dict(ab_np, w_start, w_end, last_ab_state)

                        # 다음 윈도우 결손 대응을 위해 캐리 오버 변수 최신화
                        last_ws_state = ws_seg
                        last_yr_state = yr_seg
                        last_sas_state = sas_seg
                        last_ab_state = ab_seg

                        # 피처 41개 추출 및 43개 결합
                        raw_feats = compute_direct_window_features(ws_seg, yr_seg, sas_seg, ab_seg, args.speed, args.grade)
                        features_43 = [args.speed, args.grade] + raw_feats

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

                    t1 = time.perf_counter()
                    latency_ms = (t1 - t0) * 1000.0

                    # 최종 결과 출력
                    abnormal_ratio = (abnormal_votes / 10.0) * 100
                    status_str = "\033[91m이상(ABNORMAL)\033[0m" if abnormal_ratio >= 50.0 else "\033[92m정상(NORMAL)\033[0m"
                    print(f" ▶ 고장 판정 비율: {abnormal_ratio:>5.1f}% | 최종 상태: {status_str} | 연산 소요: {latency_ms:.2f} ms")

    except KeyboardInterrupt:
        print("\n [종료] 실시간 CAN 추론 노드 청취 종료.")
    finally:
        bus.shutdown()


if __name__ == "__main__":
    run_live_inference()
