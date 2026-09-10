import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import torch

from .decoding_functions import decoding
from .src.can_models import SingleGRU

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, "src"))

W_SUFFIX = "_w34"

scaler = joblib.load(
    os.path.join(MODELS_DIR, f"scaler_can{W_SUFFIX}.pkl")
)

iforest = joblib.load(
    os.path.join(MODELS_DIR, f"step1_iforest{W_SUFFIX}.pkl")
)

gru_model = SingleGRU(in_dim=44, hidden=32)

gru_model.load_state_dict(
    torch.load(
        os.path.join(
            MODELS_DIR,
            f"step2_if_gru_can{W_SUFFIX}.pt"
        ),
        map_location="cpu",
        weights_only=True
    )
)

gru_model.eval()


def inference_can(can_frame_1s):
    dataset = decoding(can_frame_1s)
    
    preprocessed_data = preprocess_can(dataset)
    
    is_abnormal = model_run(preprocessed_data)

    return is_abnormal

def preprocess_can(dataset):

    speed = 20.0
    grade = 0.0

    df_ws = pd.DataFrame(dataset["0x386"])
    df_yr = pd.DataFrame(dataset["0x220"])
    df_sas = pd.DataFrame(dataset["0x2b0"])
    df_ab = pd.DataFrame(dataset["0x371"])

    if any(df.empty for df in [df_ws, df_yr, df_sas, df_ab]):
        return []

    all_times = np.concatenate([
        df_ws["time"].values,
        df_yr["time"].values,
        df_sas["time"].values,
        df_ab["time"].values
    ])

    t_base = np.floor(np.min(all_times))

    preprocessed_dataset = []

    for i in range(10):

        w_start = t_base + i * 0.1
        w_end = w_start + 0.1

        ws_seg = df_ws[
            (df_ws["time"] >= w_start) &
            (df_ws["time"] < w_end)
        ]

        yr_seg = df_yr[
            (df_yr["time"] >= w_start) &
            (df_yr["time"] < w_end)
        ]

        sas_seg = df_sas[
            (df_sas["time"] >= w_start) &
            (df_sas["time"] < w_end)
        ]

        ab_seg = df_ab[
            (df_ab["time"] >= w_start) &
            (df_ab["time"] < w_end)
        ]

        # ZOH
        if len(ws_seg) == 0:
            prev = df_ws[df_ws["time"] < w_start]
            ws_seg = prev.tail(1) if len(prev) > 0 else df_ws.head(1)

        if len(yr_seg) == 0:
            prev = df_yr[df_yr["time"] < w_start]
            yr_seg = prev.tail(1) if len(prev) > 0 else df_yr.head(1)

        if len(sas_seg) == 0:
            prev = df_sas[df_sas["time"] < w_start]
            sas_seg = prev.tail(1) if len(prev) > 0 else df_sas.head(1)

        if len(ab_seg) == 0:
            prev = df_ab[df_ab["time"] < w_start]
            ab_seg = prev.tail(1) if len(prev) > 0 else df_ab.head(1)

        # 41개 feature
        raw_feats = compute_direct_window_features(
            ws_seg,
            yr_seg,
            sas_seg,
            ab_seg,
            speed,
            grade
        )

        # 43차원
        features_43 = [speed, grade] + raw_feats

        # Scaling
        X_scaled = scaler.transform([features_43])[0]

        # Isolation Forest
        if_score = iforest.decision_function(
            X_scaled.reshape(1, -1)
        )[0]

        # GRU 입력 44차원
        features_44 = np.append(
            X_scaled,
            if_score
        )

        preprocessed_dataset.append(features_44)

    # (10, 44)
    preprocessed_dataset = np.array(
        preprocessed_dataset,
        dtype=np.float32
    )

    # GRU 입력: (1, 10, 44)
    preprocessed_dataset = np.expand_dims(
        preprocessed_dataset,
        axis=0
    )

    return preprocessed_dataset

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

def model_run(preprocessed_data):

    if len(preprocessed_data) == 0:
        return False

    x = torch.tensor(
        preprocessed_data,
        dtype=torch.float32
    )

    with torch.no_grad():
        output = gru_model(x)
        pred = torch.argmax(output, dim=1).item()

    return pred == 1
