import os
import sys
import time
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

    # 최우수 GRU 모델 로드 (iForest + GRU)
    gru_model = SingleGRU(in_dim=44, hidden=32)
    gru_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_if_gru_can{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    gru_model.eval()

    # 시계열 FIFO 버퍼 설정
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

    try:
        while True:
            msg = bus.recv(timeout=0.1)
            if msg is None:
                continue

            # HW can_parser를 이용해 ID별로 수신 패킷 수집
            received_time, can_id_str, data = can_parser.parser(msg)

            if can_id_str is not None:
                if start_time is None:
                    start_time = received_time
                
                relative_time = received_time - start_time
                results_1s = can_parser.collect_data(relative_time, can_id_str, data)

                # 1.0초 경계 수집 완료 시
                if results_1s is not None:
                    print(f"\n [수신] 1초 버퍼 데이터 획득 (기준시각: {relative_time:.1f}s)")
                    
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

                    # 10개 윈도우 판정 기반 종합 이상 판정 비율 출력
                    abnormal_ratio = (abnormal_votes / 10.0) * 100
                    status_str = "\033[91m이상(ABNORMAL)\033[0m" if abnormal_ratio >= 50.0 else "\033[92m정상(NORMAL)\033[0m"
                    print(f" ▶ 고장 판정 비율: {abnormal_ratio:>5.1f}% | 최종 상태: {status_str}")

    except KeyboardInterrupt:
        print("\n [종료] 실시간 CAN 추론 노드 청취 종료.")
    finally:
        bus.shutdown()


if __name__ == "__main__":
    can_main()
