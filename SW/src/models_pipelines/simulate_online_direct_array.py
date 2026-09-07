"""
simulate_online_direct_array.py (w34 CAN 전용 온라인 추론 시뮬레이션 - ZOH 병합 배제 버전)

설명:
각 시나리오의 4대 원본 CAN CSV 파일에 대해 오프라인 ZOH DataFrame 병합 과정 없이,
실제 시간 기준 0.1초 윈도우 주기 동안 각 채널(딕셔너리 리스트)에 인입된 로우 데이터만으로 
통계 피처를 독립 연산(NumPy Array)하여 모델에 다이렉트로 전달 및 판정을 시뮬레이션합니다.

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


def compute_direct_window_features(ws_seg, yr_seg, sas_seg, ab_seg, spd_target, grd_target):
    """0.1초 동안 독립 수신된 원본 리스트(DataFrame 슬라이스)로부터 직접 통계 피처 연산"""
    fl = ws_seg["WHL_SPD_FL"].values
    fr = ws_seg["WHL_SPD_FR"].values
    rl = ws_seg["WHL_SPD_RL"].values
    rr = ws_seg["WHL_SPD_RR"].values

    lat_acc = yr_seg["Lateral_Accel"].values
    yaw_rt = yr_seg["Yaw_Rate"].values

    sas_ang = sas_seg["SAS_Angle"].values
    sas_spd = sas_seg["SAS_Speed"].values

    acc_pos = ab_seg["Accel_Pedal_Pos"].values
    brk_pos = ab_seg["Brake_Pedal_Pos"].values

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
    skew_val = float(stats.skew(whl_all)) if len(whl_all) > 1 else 0.0
    kurt_val = float(stats.kurtosis(whl_all)) if len(whl_all) > 1 else 0.0

    # 2. 4륜 간 상대 속도차 (8개)
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

    # 3. 차체 거동 동역학 (4개)
    lat_accel_mean = float(np.mean(lat_acc))
    lat_accel_std = float(np.std(lat_acc)) if len(lat_acc) > 1 else 0.0
    yaw_rate_mean = float(np.mean(yaw_rt))
    yaw_rate_std = float(np.std(yaw_rt)) if len(yaw_rt) > 1 else 0.0

    # 4. 조향 시스템 특성 (4개)
    sas_angle_mean = float(np.mean(sas_ang))
    sas_angle_std = float(np.std(sas_ang)) if len(sas_ang) > 1 else 0.0
    sas_angle_p2p = float(np.ptp(sas_ang))
    sas_speed_mean = float(np.mean(sas_spd))

    # 5. 운전자 조작 특성 (4개)
    accel_pedal_mean = float(np.mean(acc_pos))
    accel_pedal_max = float(np.max(acc_pos))
    brake_pedal_mean = float(np.mean(brk_pos))
    brake_pedal_max = float(np.max(brk_pos))

    # 6. 무차원 슬립 비율 및 정규화 특성 (7개)
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

    # 7. 추가 무차원 특징 (4개)
    dimless_std = std_val / max(mean_val, 1.0)
    dimless_p2p = p2p_val / max(mean_val, 1.0)
    dimless_diff_fr_rr_std = diff_fr_rr_std / max(mean_val, 1.0)
    pedal_slip_response = diff_fr_rr_std / max(accel_pedal_mean, 1.0)

    # 41개 피처 리스트로 최종 반환
    return [
        slip_ratio_fl, slip_ratio_fr, slip_ratio_rl, slip_ratio_rr, slip_diff_front_rear, norm_yaw_rate, norm_lat_accel,
        mean_val, std_val, p2p_val, rms_val, var_val, peak_val, crest_factor, shape_factor, skew_val, kurt_val,
        diff_fl_rl_mean, diff_fl_rl_std, diff_fl_rl_p2p, diff_fr_rr_mean, diff_fr_rr_std, diff_fr_rr_p2p, diff_fl_fr_std, diff_rl_rr_std,
        lat_accel_mean, lat_accel_std, yaw_rate_mean, yaw_rate_std,
        sas_angle_mean, sas_angle_std, sas_angle_p2p, sas_speed_mean,
        accel_pedal_mean, accel_pedal_max, brake_pedal_mean, brake_pedal_max,
        dimless_std, dimless_p2p, dimless_diff_fr_rr_std, pedal_slip_response
    ]


def simulate_direct_scenario(scenario_path, state, scenario_name, scaler, ae_model, oc_svm, iforest, gru_models):
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

    # 시간 탐색 범위 정의
    t_start = min(df_ws["Time"].min(), df_yr["Time"].min(), df_sas["Time"].min(), df_ab["Time"].min())
    t_end = max(df_ws["Time"].max(), df_yr["Time"].max(), df_sas["Time"].max(), df_ab["Time"].max())

    # 시계열 링버퍼 설정 (10스텝 시계열)
    seq_len = 10
    buf_ae = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)
    buf_oc = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)
    buf_if = RealtimeSequenceBuffer(n_features=44, seq_len=seq_len)

    predictions = {name: [] for name in gru_models.keys()}
    true_label = 1 if state == "abnormal" else 0

    print(f"\n ▶ [다이렉트] 시나리오 시작: [{state.upper()}] {scenario_name}")
    print(f"   * 시간축 범위: {t_start:.3f}s ~ {t_end:.3f}s (총 시간: {t_end - t_start:.2f}초)")

    t = t_start
    window_duration = 0.1
    w_idx = 0

    while t + window_duration <= t_end:
        # 시간 범위 슬라이싱
        ws_seg = df_ws[(df_ws["Time"] >= t) & (df_ws["Time"] < t + window_duration)]
        yr_seg = df_yr[(df_yr["Time"] >= t) & (df_yr["Time"] < t + window_duration)]
        sas_seg = df_sas[(df_sas["Time"] >= t) & (df_sas["Time"] < t + window_duration)]
        ab_seg = df_ab[(df_ab["Time"] >= t) & (df_ab["Time"] < t + window_duration)]

        # 윈도우 내 신호가 한 개도 수신되지 않은 경우 캐리 오버 (ZOH 대체 작동)
        if len(ws_seg) == 0:
            ws_prev = df_ws[df_ws["Time"] < t]
            ws_seg = ws_prev.tail(1) if len(ws_prev) > 0 else df_ws.head(1)
        if len(yr_seg) == 0:
            yr_prev = df_yr[df_yr["Time"] < t]
            yr_seg = yr_prev.tail(1) if len(yr_prev) > 0 else df_yr.head(1)
        if len(sas_seg) == 0:
            sas_prev = df_sas[df_sas["Time"] < t]
            sas_seg = sas_prev.tail(1) if len(sas_prev) > 0 else df_sas.head(1)
        if len(ab_seg) == 0:
            ab_prev = df_ab[df_ab["Time"] < t]
            ab_seg = ab_prev.tail(1) if len(ab_prev) > 0 else df_ab.head(1)

        # 독립 수신된 원본 세그먼트로부터 피처 41개 추출
        raw_feats = compute_direct_window_features(ws_seg, yr_seg, sas_seg, ab_seg, spd_target, grd_target)
        features_43 = [spd_target, grd_target] + raw_feats

        # 스케일러 및 Step 1 모델 적용
        X_scaled = scaler.transform([features_43])[0]

        # AutoEncoder Error
        x_t = torch.tensor(X_scaled.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            ae_err = torch.mean((ae_model(x_t) - x_t) ** 2).item()

        # OC-SVM Dist
        oc_dist = oc_svm.decision_function(X_scaled.reshape(1, -1))[0]

        # iForest Score
        if_score = iforest.decision_function(X_scaled.reshape(1, -1))[0]

        # Step 2 피처 구성 [X_scaled(43) + Score(1)] = 44차원
        feat_ae = np.append(X_scaled, ae_err)
        feat_oc = np.append(X_scaled, oc_dist)
        feat_if = np.append(X_scaled, if_score)

        # 링버퍼 Push
        buf_ae.push(feat_ae)
        buf_oc.push(feat_oc)
        buf_if.push(feat_if)

        # Step 2 GRU 추론
        # AE + GRU
        seq_ae = buf_ae.get_sequence_tensor()
        with torch.no_grad():
            out = gru_models["AE_GRU"](torch.tensor(seq_ae, dtype=torch.float32))
            predictions["AE_GRU"].append(torch.argmax(out, dim=1).item())

        # OC-SVM + GRU
        seq_oc = buf_oc.get_sequence_tensor()
        with torch.no_grad():
            out = gru_models["OC_GRU"](torch.tensor(seq_oc, dtype=torch.float32))
            predictions["OC_GRU"].append(torch.argmax(out, dim=1).item())

        # iForest + GRU
        seq_if = buf_if.get_sequence_tensor()
        with torch.no_grad():
            out = gru_models["iForest_GRU"](torch.tensor(seq_if, dtype=torch.float32))
            predictions["iForest_GRU"].append(torch.argmax(out, dim=1).item())

        t += window_duration
        w_idx += 1

    # 최종 결과 종합
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
    print(" [온라인 다이렉트 추론 시뮬레이션 파이프라인 (ZOH 병합 배제)]")
    print(f" * 로드 모델 경로: {MODELS_DIR}")
    print("=" * 100)

    # 1. 공통 스케일러 로드
    scaler_can_path = os.path.join(MODELS_DIR, f"scaler_can{W_SUFFIX}.pkl")
    if not os.path.exists(scaler_can_path):
        print(f"[ERROR] 모델 학습 및 저장이 완료되지 않았습니다. ({scaler_can_path} 없음)")
        return
    scaler = joblib.load(scaler_can_path)

    # 2. Step 1 비지도 모델군 로드
    ae_path = os.path.join(MODELS_DIR, f"step1_autoencoder{W_SUFFIX}.pt")
    ae_model = AnomalyAutoEncoder(in_dim=43, latent_dim=8)
    ae_model.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
    ae_model.eval()

    oc_svm = joblib.load(os.path.join(MODELS_DIR, f"step1_oc_svm{W_SUFFIX}.pkl"))
    iforest = joblib.load(os.path.join(MODELS_DIR, f"step1_iforest{W_SUFFIX}.pkl"))

    # 3. Step 2 GRU 모델군 로드 (상위 3개 조합)
    gru_models = {}
    
    # AE + GRU
    gru_ae = SingleGRU(in_dim=44, hidden=32)
    gru_ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_ae_gru_can{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    gru_ae.eval()
    gru_models["AE_GRU"] = gru_ae

    # OC-SVM + GRU
    gru_oc = SingleGRU(in_dim=44, hidden=32)
    gru_oc.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"step2_oc_gru_can{W_SUFFIX}.pt"), map_location="cpu", weights_only=True))
    gru_oc.eval()
    gru_models["OC_GRU"] = gru_oc

    # iForest + GRU
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
                simulate_direct_scenario(scen_path, state, scen, scaler, ae_model, oc_svm, iforest, gru_models)


if __name__ == "__main__":
    main()
