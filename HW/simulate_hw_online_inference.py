"""
simulate_hw_online_inference.py (HW 디코더 연동 실시간 CAN 추론 시뮬레이터)

설명:
각 시나리오의 4대 물리 CSV 데이터를 역으로 인코딩하여 RAW CAN 패킷(바이트 배열) 스트림을 생성합니다.
이 패킷 스트림을 HW 폴더의 can_parser.py 및 decoding_functions.py에 실시간으로 인입시키고,
1.0초 주기로 수집 및 디코딩된 결과를 수신하여 0.1초 윈도우(10개)로 분할한 뒤 모델 추론을 수행합니다.

규정 준수:
- 100% 한글 주석
- LaTeX 기호 금지 (No-LaTeX Protocol)
- 이모지 일절 배제
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import torch

# 경로 추가 (SW의 모델 및 데이터 처리 코드 임포트용)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_UPLOADS_DIR = os.path.dirname(CURRENT_DIR)
SW_DIR = os.path.join(GIT_UPLOADS_DIR, "SW")
MODELS_DIR = os.path.join(SW_DIR, "models")
DATA_ROOT = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"

sys.path.insert(0, CURRENT_DIR)  # HW 폴더
sys.path.insert(0, os.path.join(SW_DIR, "src", "models_pipelines"))
sys.path.insert(0, os.path.join(SW_DIR, "src", "data_processing"))

import can_parser
import decoding_functions
from train_unified_models import AnomalyAutoEncoder, SingleGRU
from sequence_matrix_builder import RealtimeSequenceBuffer

# CSV 파일 내 컬럼 탐색용 매핑 딕셔너리
MAPPINGS = {
    "wheel speed": {
        "time": "Time",
        "fl": "WHL_SPD_FL",
        "fr": "WHL_SPD_FR",
        "rl": "WHL_SPD_RL",
        "rr": "WHL_SPD_RR"
    },
    "yaw_rat": {
        "time": "Time",
        "lat": "Lateral_Accel",
        "yaw": "Yaw_Rate"
    },
    "steer": {
        "time": "Time",
        "angle": "SAS_Angle",
        "speed": "SAS_Speed"
    },
    "accel_brake": {
        "time": "Time",
        "accel": "Accel_Pedal_Pos",
        "brake": "Brake_Pedal_Pos"
    }
}

# 시나리오 매칭 규칙 (target_speed, grade_pct, clean_name)
SCENARIO_RULES = {
    "80kph": (80.0, 0.0, "high_speed_80kph"),
    "60kph": (60.0, 0.0, "high_speed_60kph"),
    "20kph": (20.0, 0.0, "high_speed_20kph"),
    "12deg": (20.0, 12.0, "hills_12deg"),
    "18deg": (20.0, 18.0, "hills_18deg"),
    "30deg": (20.0, 30.0, "hills_30deg"),
    "6deg": (20.0, 6.0, "hills_6deg"),
    "12도": (20.0, 12.0, "hills_12deg"),
    "18도": (20.0, 18.0, "hills_18deg"),
    "30도": (20.0, 30.0, "hills_30deg"),
    "6도": (20.0, 6.0, "hills_6deg"),
    "steer": (40.0, 0.0, "steering_pad_40kph"),
    "circle": (40.0, 0.0, "steering_pad_40kph"),
    "원선회": (40.0, 0.0, "steering_pad_40kph"),
}


# =========================================================================
# [역인코딩 파트] 물리 피처 ➔ RAW CAN 바이트 배열 생성
# =========================================================================
def encode_0x386(fl, fr, rl, rr):
    """4륜 속도 물리값 ➔ 0x386 CAN 패킷 바이트 데이터"""
    fl_raw = int(round(fl / 0.03125)) & 0x3FFF
    fr_raw = int(round(fr / 0.03125)) & 0x3FFF
    rl_raw = int(round(rl / 0.03125)) & 0x3FFF
    rr_raw = int(round(rr / 0.03125)) & 0x3FFF
    val = fl_raw | (fr_raw << 16) | (rl_raw << 32) | (rr_raw << 48)
    return list(val.to_bytes(8, byteorder='little'))


def encode_0x220(lat_accel, yaw_rate):
    """횡가속도/요레이트 물리값 ➔ 0x220 CAN 패킷 바이트 데이터"""
    lat_raw = int(round((lat_accel + 10.23) / 0.01)) & 0x7FF
    yaw_raw = int(round((yaw_rate + 40.95) / 0.01)) & 0x1FFF
    val = lat_raw | (yaw_raw << 40)
    return list(val.to_bytes(8, byteorder='little'))


def encode_0x2b0(sas_angle, sas_speed):
    """조향각/조향속도 물리값 ➔ 0x2B0 CAN 패킷 바이트 데이터 (signed 16-bit angle)"""
    angle_raw = int(round(sas_angle / 0.1))
    if angle_raw < 0:
        angle_raw = (1 << 16) + angle_raw
    angle_raw = angle_raw & 0xFFFF
    
    speed_raw = int(round(sas_speed / 4.0)) & 0xFF
    val = angle_raw | (speed_raw << 16)
    return list(val.to_bytes(8, byteorder='little'))


def encode_0x371(accel_pedal, brake_pedal):
    """가속/브레이크 페달 물리값 ➔ 0x371 CAN 패킷 바이트 데이터"""
    brake_raw = int(round(brake_pedal)) & 0xFF
    accel_raw = int(round(accel_pedal)) & 0xFF
    val = brake_raw | (accel_raw << 31)
    return list(val.to_bytes(8, byteorder='little'))


def load_and_map_csv(filepath, col_mapping):
    df = pd.read_csv(filepath)
    new_cols = {}
    for col in df.columns:
        col_lower = col.lower()
        matched = False
        for key, val in col_mapping.items():
            if key in col_lower:
                new_cols[col] = val
                matched = True
                break
        if not matched:
            new_cols[col] = col
    return df.rename(columns=new_cols)


def compute_direct_window_features(ws_seg, yr_seg, sas_seg, ab_seg, spd_target, grd_target):
    """독립 리스트로부터 통계 피처 추출"""
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


def simulate_hw_pipeline_scenario(scenario_path, state, scenario_name, scaler, ae_model, oc_svm, iforest, gru_models):
    # 시나리오의 target speed 및 grade 획득
    spd_target, grd_target, clean_scen = 20.0, 0.0, "high_speed_20kph"
    for k, v in SCENARIO_RULES.items():
        if k in scenario_name:
            spd_target, grd_target, clean_scen = v
            break

    # 파일 로드 및 표준화
    try:
        df_ws_phys = load_and_map_csv(os.path.join(scenario_path, "wheel speed.csv"), MAPPINGS["wheel speed"])
        df_yr_phys = load_and_map_csv(os.path.join(scenario_path, "yaw_rat.csv"), MAPPINGS["yaw_rat"])
        df_sas_phys = load_and_map_csv(os.path.join(scenario_path, "steer angle_speed.csv"), MAPPINGS["steer"])
        df_ab_phys = load_and_map_csv(os.path.join(scenario_path, "accel_brake.csv"), MAPPINGS["accel_brake"])
    except Exception as e:
        print(f" [SKIP] 파일 로드 실패 ({scenario_name}): {e}")
        return

    # -------------------------------------------------------------------------
    # [1단계] 물리 피처 ➔ RAW CAN 바이트 배열 생성
    # -------------------------------------------------------------------------
    can_stream = []

    for _, r in df_ws_phys.iterrows():
        t = float(r["Time"])
        data_bytes = encode_0x386(r["WHL_SPD_FL"], r["WHL_SPD_FR"], r["WHL_SPD_RL"], r["WHL_SPD_RR"])
        can_stream.append((t, 0x386, data_bytes))

    for _, r in df_yr_phys.iterrows():
        t = float(r["Time"])
        data_bytes = encode_0x220(r["Lateral_Accel"], r["Yaw_Rate"])
        can_stream.append((t, 0x220, data_bytes))

    for _, r in df_sas_phys.iterrows():
        t = float(r["Time"])
        data_bytes = encode_0x2b0(r["SAS_Angle"], r["SAS_Speed"])
        can_stream.append((t, 0x2B0, data_bytes))

    for _, r in df_ab_phys.iterrows():
        t = float(r["Time"])
        data_bytes = encode_0x371(r["Accel_Pedal_Pos"], r["Brake_Pedal_Pos"])
        can_stream.append((t, 0x371, data_bytes))

    # 시간순 스트림 정렬 (실시간 인입 모사)
    can_stream = sorted(can_stream, key=lambda x: x[0])
    t_start = can_stream[0][0]
    t_end = can_stream[-1][0]

    print(f"\n ▶ [HW 연동] 시나리오 시작: [{state.upper()}] {scenario_name}")
    print(f"   * 시간축 범위: {t_start:.3f}s ~ {t_end:.3f}s (총 시간: {t_end - t_start:.2f}초)")

    # can_parser 내부 변수 초기화
    can_parser.NEXT_INPUT_TIME = 1.0
    can_parser.CAN_DATA = {
        '0x371': [],
        '0x220': [],
        '0x2b0': [],
        '0x386': []
    }

    # 시계열 링버퍼 설정
    seq_len = 10
    buf_ae = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)
    buf_oc = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)
    buf_if = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)

    predictions = {name: [] for name in gru_models.keys()}
    true_label = 1 if state == "abnormal" else 0

    # -------------------------------------------------------------------------
    # [2단계] 실시간 수신 ➔ can_parser ➔ decoding_functions ➔ 모델 추론
    # -------------------------------------------------------------------------
    for t_msg, can_id, data_bytes in can_stream:
        relative_time = t_msg - t_start
        
        # HW can_parser에 주입하여 1.0초 단위 버퍼링 수신
        results_1s = can_parser.collect_data(relative_time, hex(can_id), data_bytes)

        if results_1s is not None:
            # 1.0초 단위 데이터 디코딩 실행 (HW 디코더 통과)
            dataset = decoding_functions.decoding(results_1s)
            
            # 디코딩 완료 데이터를 DataFrame으로 일시 변환하여 0.1초 윈도우 슬라이싱
            df_ws = pd.DataFrame(dataset['0x386'])
            df_yr = pd.DataFrame(dataset['0x220'])
            df_sas = pd.DataFrame(dataset['0x2b0'])
            df_ab = pd.DataFrame(dataset['0x371'])

            # 1초 주기를 10개 서브 윈도우(0.1초 단위)로 쪼개어 실시간 추론 수행
            t_base = relative_time - 1.0
            for i in range(10):
                w_start = t_base + i * 0.1
                w_end = w_start + 0.1

                ws_seg = df_ws[(df_ws["time"] >= w_start) & (df_ws["time"] < w_end)]
                yr_seg = df_yr[(df_yr["time"] >= w_start) & (df_yr["time"] < w_end)]
                sas_seg = df_sas[(df_sas["time"] >= w_start) & (df_sas["time"] < w_end)]
                ab_seg = df_ab[(df_ab["time"] >= w_start) & (df_ab["time"] < w_end)]

                # 패킷 누락 시 캐리 오버
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

                # 통계 피처 41개 추출
                raw_feats = compute_direct_window_features(ws_seg, yr_seg, sas_seg, ab_seg, spd_target, grd_target)
                features_43 = [spd_target, grd_target] + raw_feats

                # 스케일러 적용
                X_scaled = scaler.transform([features_43])[0]

                # Step 1 Anomaly Score
                x_t = torch.tensor(X_scaled.reshape(1, -1), dtype=torch.float32)
                with torch.no_grad():
                    ae_err = torch.mean((ae_model(x_t) - x_t) ** 2).item()
                oc_dist = oc_svm.decision_function(X_scaled.reshape(1, -1))[0]
                if_score = iforest.decision_function(X_scaled.reshape(1, -1))[0]

                # 링버퍼 삽입
                buf_ae.push(np.append(X_scaled, ae_err))
                buf_oc.push(np.append(X_scaled, oc_dist))
                buf_if.push(np.append(X_scaled, if_score))

                # Step 2 GRU 추론
                seq_ae = buf_ae.get_sequence_tensor()
                with torch.no_grad():
                    out = gru_models["AE_GRU"](torch.tensor(seq_ae, dtype=torch.float32))
                    predictions["AE_GRU"].append(torch.argmax(out, dim=1).item())

                seq_oc = buf_oc.get_sequence_tensor()
                with torch.no_grad():
                    out = gru_models["OC_GRU"](torch.tensor(seq_oc, dtype=torch.float32))
                    predictions["OC_GRU"].append(torch.argmax(out, dim=1).item())

                seq_if = buf_if.get_sequence_tensor()
                with torch.no_grad():
                    out = gru_models["iForest_GRU"](torch.tensor(seq_if, dtype=torch.float32))
                    predictions["iForest_GRU"].append(torch.argmax(out, dim=1).item())

    # 시나리오 결과 요약 출력
    print(" ----------------------------------------------------")
    print(f" * 실제 정답(Ground Truth): {'이상(Abnormal)' if true_label == 1 else '정상(Normal)'}")
    
    for name, pred_list in predictions.items():
        if len(pred_list) == 0:
            continue
        abnormal_ratio = np.mean(pred_list) * 100
        final_decision = "이상(Abnormal)" if abnormal_ratio >= 50.0 else "정상(Normal)"
        is_correct = "정답(Pass)" if (final_decision == "이상(Abnormal)") == (true_label == 1) else "오답(Fail)"
        
        print(f"   [{name:<12}] 윈도우별 고장 판정 비율: {abnormal_ratio:>6.2f}% | 최종 세션 판정: {final_decision:<10} | {is_correct}")


def main():
    W_SUFFIX = "_w34"
    
    print("=" * 110)
    print(" [HW 연동 실시간 패킷 디코딩 및 추론 파이프라인 시뮬레이션]")
    print(f" * 로드 모델 경로: {MODELS_DIR}")
    print("=" * 110)

    # 1. 스케일러 및 Step 1 모델 로드
    scaler_can_path = os.path.join(MODELS_DIR, f"scaler_can{W_SUFFIX}.pkl")
    if not os.path.exists(scaler_can_path):
        print(f"[ERROR] 모델 학습 및 저장이 완료되지 않았습니다. ({scaler_can_path} 없음)")
        return
    scaler = joblib.load(scaler_can_path)

    ae_path = os.path.join(MODELS_DIR, f"step1_autoencoder{W_SUFFIX}.pt")
    ae_model = AnomalyAutoEncoder(in_dim=43, latent_dim=8)
    ae_model.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
    ae_model.eval()

    oc_svm = joblib.load(os.path.join(MODELS_DIR, f"step1_oc_svm{W_SUFFIX}.pkl"))
    iforest = joblib.load(os.path.join(MODELS_DIR, f"step1_iforest{W_SUFFIX}.pkl"))

    # 2. Step 2 GRU 모델군 로드 (상위 3개 조합)
    gru_models = {}
    
    gru_ae = SingleGRU(in_dim=44, hidden=32)
    gru_ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_ae_gru_can{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    gru_ae.eval()
    gru_models["AE_GRU"] = gru_ae

    gru_oc = SingleGRU(in_dim=44, hidden=32)
    gru_oc.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_oc_gru_can{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    gru_oc.eval()
    gru_models["OC_GRU"] = gru_oc

    gru_if = SingleGRU(in_dim=44, hidden=32)
    gru_if.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_if_gru_can{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    gru_if.eval()
    gru_models["iForest_GRU"] = gru_if

    # 3. 모든 주행 시나리오 순차 테스트 실행
    states = ["normal", "abnormal"]
    for state in states:
        state_path = os.path.join(DATA_ROOT, state)
        if not os.path.exists(state_path):
            continue
        
        scenarios = sorted(os.listdir(state_path))
        for scen in scenarios:
            scen_path = os.path.join(state_path, scen)
            if os.path.isdir(scen_path):
                simulate_hw_pipeline_scenario(scen_path, state, scen, scaler, ae_model, oc_svm, iforest, gru_models)


if __name__ == "__main__":
    main()
