"""
realtime_inference_wrapper.py (git_uploads/HW)

실시간 CAN 수신 데이터(decoding_functions.decoding의 출력)를 
offline ver2 학습 모델의 입력 형식([1, 10, 41] 3D sequence matrix)으로 
실시간 ZOH 병합 및 feature extraction을 수행하는 wrapper 모듈.
"""

import numpy as np
import pandas as pd
from scipy import stats

def extract_online_window_features(seg, spd_target=20.0, grd_target=0.0):
    """
    실시간 수신된 0.1초(34 samples) window segment에서 41개 feature 추출
    """
    fl = seg["WHL_SPD_FL"].values
    fr = seg["WHL_SPD_FR"].values
    rl = seg["WHL_SPD_RL"].values
    rr = seg["WHL_SPD_RR"].values

    whl_all = np.concatenate([fl, fr, rl, rr])

    # 1. wheel speed statistics (10개)
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

    # 2. 4륜 간 relative speed differences (8개)
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

    # 3. vehicle dynamics (4개)
    lat_acc = seg["Lateral_Accel"].values if "Lateral_Accel" in seg.columns else np.zeros_like(fl)
    yaw_rt = seg["Yaw_Rate"].values if "Yaw_Rate" in seg.columns else np.zeros_like(fl)

    lat_accel_mean = float(np.mean(lat_acc))
    lat_accel_std = float(np.std(lat_acc))
    yaw_rate_mean = float(np.mean(yaw_rt))
    yaw_rate_std = float(np.std(yaw_rt))

    # 4. steering angle & speed (4개)
    sas_ang = seg["SAS_Angle"].values if "SAS_Angle" in seg.columns else np.zeros_like(fl)
    sas_spd = seg["SAS_Speed"].values if "SAS_Speed" in seg.columns else np.zeros_like(fl)

    sas_angle_mean = float(np.mean(sas_ang))
    sas_angle_std = float(np.std(sas_ang))
    sas_angle_p2p = float(np.ptp(sas_ang))
    sas_speed_mean = float(np.mean(sas_spd))

    # 5. pedal operations (4개)
    acc_pos = seg["Accel_Pedal_Pos"].values if "Accel_Pedal_Pos" in seg.columns else np.zeros_like(fl)
    brk_pos = seg["Brake_Pedal_Pos"].values if "Brake_Pedal_Pos" in seg.columns else np.zeros_like(fl)

    accel_pedal_mean = float(np.mean(acc_pos))
    accel_pedal_max = float(np.max(acc_pos))
    brake_pedal_mean = float(np.mean(brk_pos))
    brake_pedal_max = float(np.max(brk_pos))

    # 6. dimensionless slip ratio & normalization (7개)
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

    # 7. advanced dimensionless features (4개)
    dimless_std = std_val / max(mean_val, 1.0)
    dimless_p2p = p2p_val / max(mean_val, 1.0)
    dimless_diff_fr_rr_std = diff_fr_rr_std / max(mean_val, 1.0)
    pedal_slip_response = diff_fr_rr_std / max(accel_pedal_mean, 1.0)

    # feature list 순서 맞추기 (Slip 7개 -> stats 10개 -> diff 8개 -> dyn 4개 -> sas 4개 -> pedal 4개 -> adv 4개 = 총 41개)
    feature_vector = [
        # Slip & Norm
        slip_ratio_fl, slip_ratio_fr, slip_ratio_rl, slip_ratio_rr,
        slip_diff_front_rear, norm_yaw_rate, norm_lat_accel,
        # Stats
        mean_val, std_val, p2p_val, rms_val, var_val, peak_val,
        crest_factor, shape_factor, skew_val, kurt_val,
        # Diff
        diff_fl_rl_mean, diff_fl_rl_std, diff_fl_rl_p2p,
        diff_fr_rr_mean, diff_fr_rr_std, diff_fr_rr_p2p,
        diff_fl_fr_std, diff_rl_rr_std,
        # Dyn
        lat_accel_mean, lat_accel_std, yaw_rate_mean, yaw_rate_std,
        # Sas
        sas_angle_mean, sas_angle_std, sas_angle_p2p, sas_speed_mean,
        # Pedal
        accel_pedal_mean, accel_pedal_max, brake_pedal_mean, brake_pedal_max,
        # Adv
        dimless_std, dimless_p2p, dimless_diff_fr_rr_std, pedal_slip_response
    ]

    return feature_vector

def process_decoded_1s(dataset, spd_target=20.0, grd_target=0.0):
    """
    decoding(results_1s)의 출력 결과(1초 단위 raw lists)를 ZOH 병합하고
    10개 sub-windows(각 34 samples = 0.1초)로 slicing하여 [1, 10, 41] sequence matrix 형상으로 변환
    """
    events = []
    
    # 각 CAN ID 채널별 raw list를 event list로 flatten
    for packet in dataset.get('0x371', []):
        events.append({
            "Time": packet['time'],
            "CAN_ID": "0x371",
            "Accel_Pedal_Pos": packet['accel_pedal'],
            "Brake_Pedal_Pos": packet['brake_pedal'],
        })
    for packet in dataset.get('0x220', []):
        events.append({
            "Time": packet['time'],
            "CAN_ID": "0x220",
            "Lateral_Accel": packet['lat_accel'],
            "Yaw_Rate": packet['yaw_rate'],
        })
    for packet in dataset.get('0x2b0', []):
        events.append({
            "Time": packet['time'],
            "CAN_ID": "0x2b0",
            "SAS_Angle": packet['steering_angle'],
            "SAS_Speed": packet['steering_speed'],
        })
    for packet in dataset.get('0x386', []):
        events.append({
            "Time": packet['time'],
            "CAN_ID": "0x386",
            "WHL_SPD_FL": packet['wheel_speed_fl'],
            "WHL_SPD_FR": packet['wheel_speed_fr'],
            "WHL_SPD_RL": packet['wheel_speed_rl'],
            "WHL_SPD_RR": packet['wheel_speed_rr'],
        })

    if not events:
        return None

    # 시간순 sorting 및 ZOH alignment
    df = pd.DataFrame(events)
    df = df.sort_values(by="Time").reset_index(drop=True)
    
    # 기본 flags 채우기
    df["TCS_CTL"] = 0.0
    df["ABS_ACT"] = 0.0
    df["ESP_CTL"] = 1.0
    df["TQI_TCS"] = 6.25

    feature_cols = [
        "WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR",
        "Lateral_Accel", "Yaw_Rate",
        "SAS_Angle", "SAS_Speed",
        "Accel_Pedal_Pos", "Brake_Pedal_Pos"
    ]
    
    # 누락 columns 강제 생성 및 ffill/bfill 적용
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # 1.0초(총 ~340 samples) 데이터를 10개 sub-windows로 slicing (각 34 samples)
    window_size = 34
    seq_features = []
    
    for i in range(10):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        seg = df.iloc[start_idx:end_idx]
        
        # samples 수가 부족한 경우 직전 유효 데이터 혹은 padding 처리
        if len(seg) == 0:
            if len(seq_features) > 0:
                feat_vec = seq_features[-1]
            else:
                feat_vec = [0.0] * 41
        else:
            feat_vec = extract_online_window_features(seg, spd_target, grd_target)
        
        seq_features.append(feat_vec)

    # 3D tensor shape로 변환: [1, 10, 41]
    sequence_matrix = np.array(seq_features, dtype=np.float32)[np.newaxis, :, :]
    return sequence_matrix
