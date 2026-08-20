"""
simulate_hw_online_inference_direct_array.py (HW 디코더 연동 실시간 CAN 추론 시뮬레이터 - Pandas DataFrame 배제 버전)

설명:
각 시나리오의 4대 물리 CSV 데이터를 RAW CAN 패킷(바이트 배열) 스트림으로 역인코딩하고,
HW 폴더의 can_parser 및 decoding_functions로 실시간 디코딩을 통과시킵니다.
1.0초 주기로 수집 및 디코딩된 물리값 딕셔너리를 Pandas DataFrame 변환 없이 
직접 NumPy Array로 변환하여 0.1초 윈도우 슬라이싱 및 GRU 모델 추론을 시뮬레이션합니다.

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

# 경로 설정 (라즈베리 파이 독립 패키지 전용 로컬 경로 적용)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 패키지/src
PACKAGE_DIR = os.path.dirname(CURRENT_DIR)                # 패키지 루트
MODELS_DIR = os.path.join(PACKAGE_DIR, "models")
DATA_ROOT = os.path.join(PACKAGE_DIR, "data")             # 패키지 내 data 폴더

# 패키지 내부 모듈 검색 경로 최우선 순위 설정
sys.path.insert(0, PACKAGE_DIR)                           # 패키지 루트 (can_parser, decoding_functions 로드용)
sys.path.insert(0, CURRENT_DIR)                           # 패키지/src (train_unified_models, sequence_matrix_builder 로드용)

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

# 시나리오 매칭 규칙
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
    fl_raw = int(round(fl / 0.03125)) & 0x3FFF
    fr_raw = int(round(fr / 0.03125)) & 0x3FFF
    rl_raw = int(round(rl / 0.03125)) & 0x3FFF
    rr_raw = int(round(rr / 0.03125)) & 0x3FFF
    val = fl_raw | (fr_raw << 16) | (rl_raw << 32) | (rr_raw << 48)
    return list(val.to_bytes(8, byteorder='little'))


def encode_0x220(lat_accel, yaw_rate):
    lat_raw = int(round((lat_accel + 10.23) / 0.01)) & 0x7FF
    yaw_raw = int(round((yaw_rate + 40.95) / 0.01)) & 0x1FFF
    val = lat_raw | (yaw_raw << 40)
    return list(val.to_bytes(8, byteorder='little'))


def encode_0x2b0(sas_angle, sas_speed):
    angle_raw = int(round(sas_angle / 0.1))
    if angle_raw < 0:
        angle_raw = (1 << 16) + angle_raw
    angle_raw = angle_raw & 0xFFFF
    
    speed_raw = int(round(sas_speed / 4.0)) & 0xFF
    val = angle_raw | (speed_raw << 16)
    return list(val.to_bytes(8, byteorder='little'))


def encode_0x371(accel_pedal, brake_pedal):
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


# =========================================================================
# [NumPy 최적화 파트] DataFrame 생성을 전면 배제한 고속 연산부
# =========================================================================
def dict_list_to_numpy_dict(dict_list, keys):
    if not dict_list:
        return {k: np.array([], dtype=np.float32) for k in ['time'] + keys}
    
    res = {'time': np.array([d['time'] for d in dict_list], dtype=np.float32)}
    for k in keys:
        res[k] = np.array([d[k] for d in dict_list], dtype=np.float32)
    return res


def slice_numpy_dict(np_dict, w_start, w_end, last_known_state=None):
    times = np_dict['time']
    indices = np.where((times >= w_start) & (times < w_end))[0]

    if len(indices) == 0:
        if last_known_state is not None:
            return last_known_state
        else:
            if len(times) > 0:
                idx = len(times) - 1
                return {k: np.array([np_dict[k][idx]], dtype=np.float32) for k in np_dict.keys()}
            else:
                return {k: np.array([0.0], dtype=np.float32) for k in np_dict.keys()}

    return {k: np_dict[k][indices] for k in np_dict.keys()}


def compute_direct_window_features(ws_seg, yr_seg, sas_seg, ab_seg, spd_target, grd_target):
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


def synthesize_can_stream_from_dataset(df_scenario):
    """통합 데이터셋의 통계 피처 행들로부터 가상의 실시간 RAW CAN 패킷 스트림 합성"""
    can_stream = []
    df_sorted = df_scenario.sort_values(by="start_time_sec")
    
    for _, row in df_sorted.iterrows():
        w_start = float(row["start_time_sec"])
        w_end = float(row["end_time_sec"])
        
        # 1. 휠속도 복원 (4륜 간 물리적 동조성 상관관계 반영 합성)
        mean_spd = float(row["mean"])
        std_spd = float(row["std"])
        for step in range(10):
            t = w_start + step * 0.01
            # 4륜이 공통적으로 공유하는 베이스 속도 트렌드
            common_spd = mean_spd + np.random.normal(0, max(std_spd * 0.9, 0.005))
            # 각 륜의 물리적 미세 편차 노이즈 (동조 관계 붕괴 방지)
            fl = common_spd + np.random.normal(0, max(std_spd * 0.1, 0.001))
            fr = common_spd + np.random.normal(0, max(std_spd * 0.1, 0.001))
            rl = common_spd + np.random.normal(0, max(std_spd * 0.1, 0.001))
            rr = common_spd + np.random.normal(0, max(std_spd * 0.1, 0.001))
            
            data_bytes = encode_0x386(fl, fr, rl, rr)
            can_stream.append((t, 0x386, data_bytes))
            
        # 2. 거동 동역학 복원 (yaw_rate_mean, lat_accel_mean 기반)
        lat_acc = float(row["lat_accel_mean"])
        yaw_rt = float(row["yaw_rate_mean"])
        for step in range(5):
            t = w_start + step * 0.02
            data_bytes = encode_0x220(lat_acc, yaw_rt)
            can_stream.append((t, 0x220, data_bytes))
            
        # 3. 조향각 및 속도 복원 (sas_angle_mean, sas_speed_mean)
        sas_ang = float(row["sas_angle_mean"])
        sas_spd = float(row["sas_speed_mean"])
        for step in range(2):
            t = w_start + step * 0.05
            data_bytes = encode_0x2b0(sas_ang, sas_spd)
            can_stream.append((t, 0x2b0, data_bytes))
            
        # 4. 페달 위치 복원 (accel_pedal_mean, brake_pedal_mean)
        acc_ped = float(row["accel_pedal_mean"])
        brk_ped = float(row["brake_pedal_mean"])
        for step in range(2):
            t = w_start + step * 0.05
            data_bytes = encode_0x371(acc_ped, brk_ped)
            can_stream.append((t, 0x371, data_bytes))
            
    can_stream = sorted(can_stream, key=lambda x: x[0])
    return can_stream


def simulate_hw_pipeline_scenario(df_scenario, state, scenario_name, scaler, ae_model, oc_svm, iforest, gru_models):
    spd_target, grd_target = 20.0, 0.0
    if len(df_scenario) > 0:
        spd_target = float(df_scenario.iloc[0]["speed_kph"])
        grd_target = float(df_scenario.iloc[0]["grade_pct"])

    # 1. RAW CAN 바이트 패킷 스트림 인코딩 생성 (통계치 기반 역합성)
    can_stream = synthesize_can_stream_from_dataset(df_scenario)
    if not can_stream:
        print(f" [SKIP] 시뮬레이션 패킷 생성 실패 ({scenario_name})")
        return

    t_start = can_stream[0][0]
    t_end = can_stream[-1][0]

    print(f"\n ▶ [HW 다이렉트] 시나리오 시작: [{state.upper()}] {scenario_name}")
    print(f"   * 시간축 범위: {t_start:.3f}s ~ {t_end:.3f}s (총 시간: {t_end - t_start:.2f}초)")

    # can_parser 버퍼 초기화
    can_parser.NEXT_INPUT_TIME = 1.0
    can_parser.CAN_DATA = {
        '0x371': [],
        '0x220': [],
        '0x2b0': [],
        '0x386': []
    }

    seq_len = 10
    buf_ae = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)
    buf_oc = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)
    buf_if = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)

    predictions = {name: [] for name in gru_models.keys()}
    true_label = 1 if state == "abnormal" else 0

    # ZOH 캐리 오버 변수 초기화
    last_ws_state = None
    last_yr_state = None
    last_sas_state = None
    last_ab_state = None

    # 2. 스트리밍 인입 및 추론 실행
    for t_msg, can_id, data_bytes in can_stream:
        relative_time = t_msg - t_start
        results_1s = can_parser.collect_data(relative_time, hex(can_id), data_bytes)

        if results_1s is not None:
            # 1. HW 디코더 실행
            dataset = decoding_functions.decoding(results_1s)
            
            # 2. 딕셔너리 리스트 ➔ NumPy 딕셔너리 최적화 변환
            ws_np = dict_list_to_numpy_dict(dataset['0x386'], ['wheel_speed_fl', 'wheel_speed_fr', 'wheel_speed_rl', 'wheel_speed_rr'])
            yr_np = dict_list_to_numpy_dict(dataset['0x220'], ['lat_accel', 'yaw_rate'])
            sas_np = dict_list_to_numpy_dict(dataset['0x2b0'], ['steering_angle', 'steering_speed'])
            ab_np = dict_list_to_numpy_dict(dataset['0x371'], ['accel_pedal', 'brake_pedal'])

            t_base = relative_time - 1.0

            for i in range(10):
                w_start = t_base + i * 0.1
                w_end = w_start + 0.1

                # NumPy 슬라이싱 연산 수행 (Pandas 오버헤드 0%)
                ws_seg = slice_numpy_dict(ws_np, w_start, w_end, last_ws_state)
                yr_seg = slice_numpy_dict(yr_np, w_start, w_end, last_yr_state)
                sas_seg = slice_numpy_dict(sas_np, w_start, w_end, last_sas_state)
                ab_seg = slice_numpy_dict(ab_np, w_start, w_end, last_ab_state)

                last_ws_state = ws_seg
                last_yr_state = yr_seg
                last_sas_state = sas_seg
                last_ab_state = ab_seg

                # 피처 41개 추출 및 43개 구성
                raw_feats = compute_direct_window_features(ws_seg, yr_seg, sas_seg, ab_seg, spd_target, grd_target)
                features_43 = [spd_target, grd_target] + raw_feats

                # 스케일러 및 Step 1 스코어 계산
                X_scaled = scaler.transform([features_43])[0]

                # Step 1 Scores
                x_t = torch.tensor(X_scaled.reshape(1, -1), dtype=torch.float32)
                with torch.no_grad():
                    ae_err = torch.mean((ae_model(x_t) - x_t) ** 2).item()
                oc_dist = oc_svm.decision_function(X_scaled.reshape(1, -1))[0]
                if_score = iforest.decision_function(X_scaled.reshape(1, -1))[0]

                # 44차원 벡터 Push
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
    print(" [HW 디코더 연동 실시간 패킷 디코딩 및 추론 파이프라인 시뮬레이션 - DataFrame 배제 최적화]")
    print(f" * 로드 모델 경로: {MODELS_DIR}")
    print("=" * 110)

    # 1. 스케일러 및 Step 1 모델 로드
    scaler_can_path = os.path.join(MODELS_DIR, f"scaler_can{W_SUFFIX}.pkl")
    if not os.path.exists(scaler_can_path):
        print(f"[ERROR] 모델 파일 없음: {scaler_can_path}")
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

    # 3. 통합 데이터셋 로드
    csv_path = os.path.join(DATA_ROOT, "unified_can_context_dataset_w34.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] 통합 검증 데이터셋 파일 없음: {csv_path}")
        return
    
    print(f" 📂 데이터셋 로드 완료: {csv_path}")
    df_all = pd.read_csv(csv_path)
    
    # 4. 시나리오별 시뮬레이션 순차 실행
    scenarios = df_all["scenario"].unique()
    for scen in scenarios:
        df_scen = df_all[df_all["scenario"] == scen]
        state = df_scen.iloc[0]["state"]
        simulate_hw_pipeline_scenario(df_scen, state, scen, scaler, ae_model, oc_svm, iforest, gru_models)


if __name__ == "__main__":
    main()
