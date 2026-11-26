import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from decoding_functions import decode_can
from can_models import InferenceEngine

engine = InferenceEngine(MODELS_DIR)


def inference(can_frames, audio_feats):
    decoded = decode_can(can_frames)
    can_feats = extract_can_features(decoded)
    return run_model(can_feats, audio_feats)


def extract_can_features(dataset):
    df_ws = pd.DataFrame(dataset.get("0x386", []))
    df_yr = pd.DataFrame(dataset.get("0x220", []))
    df_sas = pd.DataFrame(dataset.get("0x2b0", []))
    df_ab = pd.DataFrame(dataset.get("0x371", []))

    if any(df.empty for df in [df_ws, df_yr, df_sas, df_ab]):
        return np.zeros(41, dtype=np.float32)

    all_times = np.concatenate([
        df_ws["time"].values,
        df_yr["time"].values,
        df_sas["time"].values,
        df_ab["time"].values,
    ])
    t_base = np.floor(np.min(all_times) / 0.5) * 0.5

    window_feats = []
    for i in range(5):
        w_start = t_base + i * 0.1
        w_end = w_start + 0.1

        seg_ws = slice_or_hold(df_ws, w_start, w_end)
        seg_yr = slice_or_hold(df_yr, w_start, w_end)
        seg_sas = slice_or_hold(df_sas, w_start, w_end)
        seg_ab = slice_or_hold(df_ab, w_start, w_end)

        window_feats.append(compute_subwindow(seg_ws, seg_yr, seg_sas, seg_ab))

    return np.mean(window_feats, axis=0).astype(np.float32)


def slice_or_hold(df, start, end):
    seg = df[(df["time"] >= start) & (df["time"] < end)]
    if len(seg) > 0:
        return seg
    prev = df[df["time"] < start]
    return prev.tail(1) if len(prev) > 0 else df.head(1)


def compute_subwindow(seg_ws, seg_yr, seg_sas, seg_ab, spd_target=20.0, grd_target=0.0):
    fl = seg_ws["wheel_speed_fl"].values
    fr = seg_ws["wheel_speed_fr"].values
    rl = seg_ws["wheel_speed_rl"].values
    rr = seg_ws["wheel_speed_rr"].values

    lat_acc = seg_yr["lat_accel"].values
    yaw = seg_yr["yaw_rate"].values

    sas_ang = seg_sas["steering_angle"].values
    sas_spd = seg_sas["steering_speed"].values

    acc = seg_ab["accel_pedal"].values
    brk = seg_ab["brake_pedal"].values

    whl = np.concatenate([fl, fr, rl, rr])

    m_val = float(np.mean(whl))
    s_val = float(np.std(whl))
    p2p_val = float(np.ptp(whl))
    rms_val = float(np.sqrt(np.mean(whl ** 2)))
    var_val = float(np.var(whl))
    pk_val = float(np.max(np.abs(whl)))
    crest = float(pk_val / (rms_val + 1e-6))
    shape = float(rms_val / (m_val + 1e-6))
    skw = float(stats.skew(whl)) if len(whl) > 1 else 0.0
    kurt = float(stats.kurtosis(whl)) if len(whl) > 1 else 0.0

    d_fl_rl = fl - rl
    d_fr_rr = fr - rr
    d_fl_fr = fl - fr
    d_rl_rr = rl - rr

    d_fl_rl_m = float(np.mean(d_fl_rl))
    d_fl_rl_s = float(np.std(d_fl_rl)) if len(d_fl_rl) > 1 else 0.0
    d_fl_rl_p = float(np.ptp(d_fl_rl))

    d_fr_rr_m = float(np.mean(d_fr_rr))
    d_fr_rr_s = float(np.std(d_fr_rr)) if len(d_fr_rr) > 1 else 0.0
    d_fr_rr_p = float(np.ptp(d_fr_rr))

    d_fl_fr_s = float(np.std(d_fl_fr)) if len(d_fl_fr) > 1 else 0.0
    d_rl_rr_s = float(np.std(d_rl_rr)) if len(d_rl_rr) > 1 else 0.0

    lat_m = float(np.mean(lat_acc))
    lat_s = float(np.std(lat_acc)) if len(lat_acc) > 1 else 0.0
    yaw_m = float(np.mean(yaw))
    yaw_s = float(np.std(yaw)) if len(yaw) > 1 else 0.0

    ang_m = float(np.mean(sas_ang))
    ang_s = float(np.std(sas_ang)) if len(sas_ang) > 1 else 0.0
    ang_p = float(np.ptp(sas_ang))
    ang_spd_m = float(np.mean(sas_spd))

    acc_m = float(np.mean(acc))
    acc_max = float(np.max(acc))
    brk_m = float(np.mean(brk))
    brk_max = float(np.max(brk))

    fl_m = float(np.mean(fl))
    fr_m = float(np.mean(fr))
    rl_m = float(np.mean(rl))
    rr_m = float(np.mean(rr))

    base_spd = max(spd_target, 1.0)
    slip_fl = fl_m / base_spd
    slip_fr = fr_m / base_spd
    slip_rl = rl_m / base_spd
    slip_rr = rr_m / base_spd

    slip_diff_fr = ((fl_m + fr_m) / 2.0) - ((rl_m + rr_m) / 2.0)
    norm_yaw = yaw_m / base_spd
    norm_lat = lat_m / (grd_target + 1.0)

    dim_s = s_val / max(m_val, 1.0)
    dim_p = p2p_val / max(m_val, 1.0)
    dim_d_fr_rr = d_fr_rr_s / max(m_val, 1.0)
    pedal_resp = d_fr_rr_s / max(acc_m, 1.0)

    return [
        slip_fl, slip_fr, slip_rl, slip_rr, slip_diff_fr, norm_yaw, norm_lat,
        m_val, s_val, p2p_val, rms_val, var_val, pk_val, crest, shape, skw, kurt,
        d_fl_rl_m, d_fl_rl_s, d_fl_rl_p, d_fr_rr_m, d_fr_rr_s, d_fr_rr_p, d_fl_fr_s, d_rl_rr_s,
        lat_m, lat_s, yaw_m, yaw_s,
        ang_m, ang_s, ang_p, ang_spd_m,
        acc_m, acc_max, brk_m, brk_max,
        dim_s, dim_p, dim_d_fr_rr, pedal_resp,
    ]


def run_model(can_feats, audio_feats):
    aud_vec = np.asarray(audio_feats, dtype=np.float32).reshape(-1)
    if aud_vec.size != 24:
        raise ValueError("Audio 특징은 24차원이어야 합니다.")
    res = engine.predict(can_feats, aud_vec)
    return int(res["pred"])


# 호환 별칭
preprocess_can = extract_can_features
compute_direct_window_features = compute_subwindow
model_run = run_model
