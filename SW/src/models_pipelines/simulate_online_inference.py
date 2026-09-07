"""
simulate_online_inference.py (w34 CAN 전용 온라인 추론 시뮬레이션)

설명:
각 시나리오의 4대 원본 CAN CSV 파일들을 시간순으로 정렬하여 
실시간 스트리밍처럼 받아들이고, 34샘플 버퍼를 구성하여 통계 피처 추출 및 
배포용 모델을 통해 고장 판정을 실시간으로 내리는 과정을 시뮬레이션합니다.

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
import argparse

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(SUITE_DIR, "models")
DATA_ROOT = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/gearbox_fault_diagnosis_program/gearbox_fault_diagnosis_program/data"

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "data_processing"))

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


def load_and_map_csv(filepath, col_mapping):
    """CSV의 난해한 컬럼명을 표준 컬럼명으로 매핑하여 로드"""
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


def compute_can_features_w34(seg, spd_target, grd_target):
    """34샘플 윈도우 데이터에서 41개 통계 피처를 추출"""
    fl = seg["WHL_SPD_FL"].values
    fr = seg["WHL_SPD_FR"].values
    rl = seg["WHL_SPD_RL"].values
    rr = seg["WHL_SPD_RR"].values

    whl_all = np.concatenate([fl, fr, rl, rr])

    # 1. 휠속도 기본 통계량 (10개)
    mean_val = float(np.mean(whl_all))
    std_val = float(np.std(whl_all))
    p2p_val = float(np.ptp(whl_all))
    rms_val = float(np.sqrt(np.mean(whl_all ** 2)))
    var_val = float(np.var(whl_all))
    peak_val = float(np.max(np.abs(whl_all)))
    crest_factor = float(peak_val / (rms_val + 1e-6))
    shape_factor = float(rms_val / (mean_val + 1e-6))
    skew_val = float(stats.skew(whl_all))
    kurt_val = float(stats.kurtosis(whl_all))

    # 2. 4륜 간 상대 속도차 (8개)
    diff_fl_rl = fl - rl
    diff_fr_rr = fr - rr
    diff_fl_fr = fl - fr
    diff_rl_rr = rl - rr

    diff_fl_rl_mean = float(np.mean(diff_fl_rl))
    diff_fl_rl_std = float(np.std(diff_fl_rl))
    diff_fl_rl_p2p = float(np.ptp(diff_fl_rl))

    diff_fr_rr_mean = float(np.mean(diff_fr_rr))
    diff_fr_rr_std = float(np.std(diff_fr_rr))
    diff_fr_rr_p2p = float(np.ptp(diff_fr_rr))

    diff_fl_fr_std = float(np.std(diff_fl_fr))
    diff_rl_rr_std = float(np.std(diff_rl_rr))

    # 3. 차체 거동 동역학 (4개)
    lat_acc = seg["Lateral_Accel"].values if "Lateral_Accel" in seg.columns else np.zeros_like(fl)
    yaw_rt = seg["Yaw_Rate"].values if "Yaw_Rate" in seg.columns else np.zeros_like(fl)

    lat_accel_mean = float(np.mean(lat_acc))
    lat_accel_std = float(np.std(lat_acc))
    yaw_rate_mean = float(np.mean(yaw_rt))
    yaw_rate_std = float(np.std(yaw_rt))

    # 4. 조향 시스템 특성 (4개)
    sas_ang = seg["SAS_Angle"].values if "SAS_Angle" in seg.columns else np.zeros_like(fl)
    sas_spd = seg["SAS_Speed"].values if "SAS_Speed" in seg.columns else np.zeros_like(fl)

    sas_angle_mean = float(np.mean(sas_ang))
    sas_angle_std = float(np.std(sas_ang))
    sas_angle_p2p = float(np.ptp(sas_ang))
    sas_speed_mean = float(np.mean(sas_spd))

    # 5. 운전자 조작 특성 (4개)
    acc_pos = seg["Accel_Pedal_Pos"].values if "Accel_Pedal_Pos" in seg.columns else np.zeros_like(fl)
    brk_pos = seg["Brake_Pedal_Pos"].values if "Brake_Pedal_Pos" in seg.columns else np.zeros_like(fl)

    accel_pedal_mean = float(np.mean(acc_pos))
    accel_pedal_max = float(np.max(acc_pos))
    brake_pedal_mean = float(np.mean(brk_pos))
    brake_pedal_max = float(np.max(brk_pos))

    # 6. 무차원 슬립 비율 (7개)
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

    # 7. 무차원 변동률 추가 (4개)
    dimless_std = std_val / max(mean_val, 1.0)
    dimless_p2p = p2p_val / max(mean_val, 1.0)
    dimless_diff_fr_rr_std = diff_fr_rr_std / max(mean_val, 1.0)
    pedal_slip_response = diff_fr_rr_std / max(accel_pedal_mean, 1.0)

    # 43개 피처 목록 구성 (학습 컬럼 정의와 동일한 인덱스 유지)
    return [
        slip_ratio_fl, slip_ratio_fr, slip_ratio_rl, slip_ratio_rr, slip_diff_front_rear, norm_yaw_rate, norm_lat_accel,
        mean_val, std_val, p2p_val, rms_val, var_val, peak_val, crest_factor, shape_factor, skew_val, kurt_val,
        diff_fl_rl_mean, diff_fl_rl_std, diff_fl_rl_p2p, diff_fr_rr_mean, diff_fr_rr_std, diff_fr_rr_p2p, diff_fl_fr_std, diff_rl_rr_std,
        lat_accel_mean, lat_accel_std, yaw_rate_mean, yaw_rate_std,
        sas_angle_mean, sas_angle_std, sas_angle_p2p, sas_speed_mean,
        accel_pedal_mean, accel_pedal_max, brake_pedal_mean, brake_pedal_max,
        dimless_std, dimless_p2p, dimless_diff_fr_rr_std, pedal_slip_response
    ]


def simulate_scenario(scenario_path, state, scenario_name, scaler, ae_model, oc_svm, iforest, gru_models):
    # 시나리오의 target speed 및 grade 획득
    spd_target, grd_target, clean_scen = 20.0, 0.0, "high_speed_20kph"
    for k, v in SCENARIO_RULES.items():
        if k in scenario_name:
            spd_target, grd_target, clean_scen = v
            break

    # 파일 로드 및 표준화
    try:
        df_ws = load_and_map_csv(os.path.join(scenario_path, "wheel speed.csv"), MAPPINGS["wheel speed"])
        df_yr = load_and_map_csv(os.path.join(scenario_path, "yaw_rat.csv"), MAPPINGS["yaw_rat"])
        df_sas = load_and_map_csv(os.path.join(scenario_path, "steer angle_speed.csv"), MAPPINGS["steer"])
        df_ab = load_and_map_csv(os.path.join(scenario_path, "accel_brake.csv"), MAPPINGS["accel_brake"])
    except Exception as e:
        print(f" [SKIP] 파일 로드 실패 ({scenario_name}): {e}")
        return

    # In-memory ZOH 병합 수행
    df_all = pd.concat([df_ws, df_yr, df_sas, df_ab], ignore_index=True)
    df_all = df_all.sort_values(by="Time").reset_index(drop=True)

    feature_cols = [
        "WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR",
        "Lateral_Accel", "Yaw_Rate",
        "SAS_Angle", "SAS_Speed",
        "Accel_Pedal_Pos", "Brake_Pedal_Pos"
    ]
    df_all[feature_cols] = df_all[feature_cols].ffill().bfill()

    # 슬라이딩 시뮬레이션용 버퍼 설정 (10스텝 시계열)
    seq_len = 10
    buf_ae = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)
    buf_oc = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)
    buf_if = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)

    total_rows = len(df_all)
    window_size = 34
    n_windows = total_rows // window_size

    # 결과 통계 저장용
    predictions = {name: [] for name in gru_models.keys()}
    true_label = 1 if state == "abnormal" else 0

    print(f"\n ▶ 시나리오 시작: [{state.upper()}] {scenario_name} (총 {total_rows}행 -> {n_windows}개 윈도우)")

    for w_idx in range(n_windows):
        start_i = w_idx * window_size
        end_i = start_i + window_size
        seg = df_all.iloc[start_i:end_i]

        # 1. 41개 통계 피처 추출
        raw_feats = compute_can_features_w34(seg, spd_target, grd_target)
        
        # 2. [speed_kph, grade_pct] 결합하여 43개 피처 구성
        features_43 = [spd_target, grd_target] + raw_feats
        
        # 3. 스케일러 적용
        X_scaled = scaler.transform([features_43])[0]

        # 4. Step 1 비지도 이상 탐지 스코어 계산
        # 4-A. AutoEncoder 재구성 에러
        x_t = torch.tensor(X_scaled.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            ae_err = torch.mean((ae_model(x_t) - x_t) ** 2).item()

        # 4-B. One-Class SVM 거리
        oc_dist = oc_svm.decision_function(X_scaled.reshape(1, -1))[0]

        # 4-C. Isolation Forest 스코어
        if_score = iforest.decision_function(X_scaled.reshape(1, -1))[0]

        # 5. Step 2 결합 벡터 구성 [X_scaled(43) + Score(1)] = 44차원
        feat_ae = np.append(X_scaled, ae_err)
        feat_oc = np.append(X_scaled, oc_dist)
        feat_if = np.append(X_scaled, if_score)

        # 6. 시계열 링버퍼에 Push
        buf_ae.push(feat_ae)
        buf_oc.push(feat_oc)
        buf_if.push(feat_if)

        # 7. Step 2 SingleGRU 모델 추론 (버퍼가 덜 찼을 때도 패딩을 이용해 추론)
        # AE + GRU
        seq_ae = buf_ae.get_sequence_tensor()
        with torch.no_grad():
            out = gru_models["AE_GRU"](torch.tensor(seq_ae, dtype=torch.float32))
            pred_ae = torch.argmax(out, dim=1).item()
            predictions["AE_GRU"].append(pred_ae)

        # OC-SVM + GRU
        seq_oc = buf_oc.get_sequence_tensor()
        with torch.no_grad():
            out = gru_models["OC_GRU"](torch.tensor(seq_oc, dtype=torch.float32))
            pred_oc = torch.argmax(out, dim=1).item()
            predictions["OC_GRU"].append(pred_oc)

        # iForest + GRU
        seq_if = buf_if.get_sequence_tensor()
        with torch.no_grad():
            out = gru_models["iForest_GRU"](torch.tensor(seq_if, dtype=torch.float32))
            pred_if = torch.argmax(out, dim=1).item()
            predictions["iForest_GRU"].append(pred_if)

    # 시나리오 종료 후 최종 진단 (가장 많이 출력된 다수결 클래스로 세션의 최종 판정 결정)
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
    
    print("=" * 100)
    print(" [온라인 추론 시뮬레이션 파이프라인 (w34 CAN 전용)]")
    print(f" * 로드 모델 경로: {MODELS_DIR}")
    print("=" * 100)

    # 1. 공통 스케일러 및 Step 1 비지도 모델 로드
    scaler_can_path = os.path.join(MODELS_DIR, f"scaler_can{W_SUFFIX}.pkl")
    if not os.path.exists(scaler_can_path):
        print(f"[ERROR] 모델 학습 및 저장이 완료되지 않았습니다. ({scaler_can_path} 없음)")
        return

    scaler = joblib.load(scaler_can_path)

    # 2. Step 1 비지도 모델군 로드
    # AutoEncoder
    ae_path = os.path.join(MODELS_DIR, f"step1_autoencoder{W_SUFFIX}.pt")
    ae_model = AnomalyAutoEncoder(in_dim=43, latent_dim=8)
    ae_model.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
    ae_model.eval()

    # OneClassSVM
    oc_svm_path = os.path.join(MODELS_DIR, f"step1_oc_svm{W_SUFFIX}.pkl")
    oc_svm = joblib.load(oc_svm_path)

    # IsolationForest
    iforest_path = os.path.join(MODELS_DIR, f"step1_iforest{W_SUFFIX}.pkl")
    iforest = joblib.load(iforest_path)

    # 3. Step 2 GRU 모델군 로드 (상위 3개 조합)
    gru_models = {}
    
    # AE + GRU (44차원 입력)
    gru_ae = SingleGRU(in_dim=44, hidden=32)
    gru_ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_ae_gru_can{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    gru_ae.eval()
    gru_models["AE_GRU"] = gru_ae

    # OC-SVM + GRU (44차원 입력)
    gru_oc = SingleGRU(in_dim=44, hidden=32)
    gru_oc.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_oc_gru_can{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    gru_oc.eval()
    gru_models["OC_GRU"] = gru_oc

    # iForest + GRU (44차원 입력)
    gru_if = SingleGRU(in_dim=44, hidden=32)
    gru_if.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_if_gru_can{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    gru_if.eval()
    gru_models["iForest_GRU"] = gru_if

    # 4. 모든 주행 시나리오 순차 테스트 실행
    states = ["normal", "abnormal"]
    for state in states:
        state_path = os.path.join(DATA_ROOT, state)
        if not os.path.exists(state_path):
            continue
        
        scenarios = sorted(os.listdir(state_path))
        for scen in scenarios:
            scen_path = os.path.join(state_path, scen)
            if os.path.isdir(scen_path):
                simulate_scenario(scen_path, state, scen, scaler, ae_model, oc_svm, iforest, gru_models)


if __name__ == "__main__":
    main()
