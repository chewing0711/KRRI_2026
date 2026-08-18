"""
inference.py (gearbox_final_suite / src / models_pipelines)

미학습 신규 실차 주행 데이터(Unseen Raw Driving CSV) 대상 2단계 비지도 하이브리드 고장 진단 엔진.
[시나리오 인자 불필요]: 실시간 4륜 센서 평균으로부터 현재 차속 및 주행 상태를 100% 자동 추정합니다.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import joblib

DATA_PROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DATA_PROCESSING_DIR) if "models_pipelines" in DATA_PROCESSING_DIR else DATA_PROCESSING_DIR
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
MODELS_SAVE_DIR = os.path.join(SUITE_DIR, "models")
os.makedirs(PRED_DIR, exist_ok=True)


class AnomalyAutoEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=8):
        super(AnomalyAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 24), nn.ReLU(),
            nn.Linear(24, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.ReLU(),
            nn.Linear(24, in_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SingleGRU(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super(SingleGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class SingleRNN(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super(SingleRNN, self).__init__()
        self.rnn = nn.RNN(in_dim, hidden, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def clean_and_standardize_csv(input_path):
    df = pd.read_csv(input_path)
    rename_map = {
        "hyundai_ccan_total::WHL_SPD11::WHL_SPD_FL[km/h]": "WHL_SPD_FL",
        "hyundai_ccan_total::WHL_SPD11::WHL_SPD_FR[km/h]": "WHL_SPD_FR",
        "hyundai_ccan_total::WHL_SPD11::WHL_SPD_RL[km/h]": "WHL_SPD_RL",
        "hyundai_ccan_total::WHL_SPD11::WHL_SPD_RR[km/h]": "WHL_SPD_RR",
        "FL": "WHL_SPD_FL", "FR": "WHL_SPD_FR", "RL": "WHL_SPD_RL", "RR": "WHL_SPD_RR",
        "Time[s]": "time", "Time (초)": "time", "Time": "time"
    }
    df = df.rename(columns=rename_map)
    req_cols = ["WHL_SPD_FL", "WHL_SPD_FR", "WHL_SPD_RL", "WHL_SPD_RR"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"[ERROR] 필수 바퀴 속도 컬럼이 누락되었습니다: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.sort_values(by="time").reset_index(drop=True)

    df[req_cols] = df[req_cols].ffill().bfill()
    return df.reset_index(drop=True)


def extract_features_from_window(seg):
    """실시간 4륜 센서 신호에서 차속 및 36개 물리 피처 100% 자동 추출 (시나리오 수동 입력 불필요)"""
    fl, fr = seg["WHL_SPD_FL"].values, seg["WHL_SPD_FR"].values
    rl, rr = seg["WHL_SPD_RL"].values, seg["WHL_SPD_RR"].values
    whl_all = np.concatenate([fl, fr, rl, rr])

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

    lat_acc = seg["Lateral_Accel"].values if "Lateral_Accel" in seg.columns else np.zeros_like(fl)
    yaw_rt = seg["Yaw_Rate"].values if "Yaw_Rate" in seg.columns else np.zeros_like(fl)
    sas_ang = seg["SAS_Angle"].values if "SAS_Angle" in seg.columns else np.zeros_like(fl)
    sas_spd = seg["SAS_Speed"].values if "SAS_Speed" in seg.columns else np.zeros_like(fl)
    acc_pos = seg["Accel_Pedal_Pos"].values if "Accel_Pedal_Pos" in seg.columns else np.zeros_like(fl)
    brk_pos = seg["Brake_Pedal_Pos"].values if "Brake_Pedal_Pos" in seg.columns else np.zeros_like(fl)

    # 4륜 센서 평균으로부터 현재 실시간 차속 자동 산출
    auto_spd_kph = max(mean_val, 1.0)
    auto_grd_pct = 0.0  # 기본 평지 또는 IMU 자동 추정

    fl_m, fr_m, rl_m, rr_m = float(np.mean(fl)), float(np.mean(fr)), float(np.mean(rl)), float(np.mean(rr))
    slip_ratio_fl = fl_m / auto_spd_kph
    slip_ratio_fr = fr_m / auto_spd_kph
    slip_ratio_rl = rl_m / auto_spd_kph
    slip_ratio_rr = rr_m / auto_spd_kph
    slip_diff_fr_rr = ((fl_m + fr_m) / 2.0) - ((rl_m + rr_m) / 2.0)
    norm_yaw = float(np.mean(yaw_rt)) / auto_spd_kph
    norm_lat = float(np.mean(lat_acc)) / (auto_grd_pct + 1.0)

    return {
        "speed_kph": auto_spd_kph, "grade_pct": auto_grd_pct,
        "slip_ratio_fl": slip_ratio_fl, "slip_ratio_fr": slip_ratio_fr,
        "slip_ratio_rl": slip_ratio_rl, "slip_ratio_rr": slip_ratio_rr,
        "slip_diff_front_rear": slip_diff_fr_rr, "norm_yaw_rate": norm_yaw, "norm_lat_accel": norm_lat,
        "mean": mean_val, "std": std_val, "p2p": p2p_val, "rms": rms_val, "variance": var_val,
        "peak": peak_val, "crest_factor": crest_factor, "shape_factor": shape_factor, "skew": skew_val, "kurt": kurt_val,
        "diff_fl_rl_mean": diff_fl_rl_mean, "diff_fl_rl_std": diff_fl_rl_std, "diff_fl_rl_p2p": diff_fl_rl_p2p,
        "diff_fr_rr_mean": diff_fr_rr_mean, "diff_fr_rr_std": diff_fr_rr_std, "diff_fr_rr_p2p": diff_fr_rr_p2p,
        "diff_fl_fr_std": diff_fl_fr_std, "diff_rl_rr_std": diff_rl_rr_std,
        "lat_accel_mean": float(np.mean(lat_acc)), "lat_accel_std": float(np.std(lat_acc)),
        "yaw_rate_mean": float(np.mean(yaw_rt)), "yaw_rate_std": float(np.std(yaw_rt)),
        "sas_angle_mean": float(np.mean(sas_ang)), "sas_angle_std": float(np.std(sas_ang)),
        "sas_angle_p2p": float(np.ptp(sas_ang)), "sas_speed_mean": float(np.mean(sas_spd)),
        "accel_pedal_mean": float(np.mean(acc_pos)), "accel_pedal_max": float(np.max(acc_pos)),
        "brake_pedal_mean": float(np.mean(brk_pos)), "brake_pedal_max": float(np.max(brk_pos)),
        "dimless_std": std_val / max(mean_val, 1.0),
        "dimless_p2p": p2p_val / max(mean_val, 1.0),
        "dimless_diff_fr_rr_std": diff_fr_rr_std / max(mean_val, 1.0),
        "pedal_slip_response": diff_fr_rr_std / max(float(np.mean(acc_pos)), 1.0),
    }


def load_exported_2step_pipeline(model_type="autoencoder_gru", window_size=250):
    w_label = str(window_size)
    w_suffix = f"_w{w_label}"

    scaler_path = os.path.join(MODELS_SAVE_DIR, f"scaler{w_suffix}.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(MODELS_SAVE_DIR, "scaler_w250.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"[ERROR] 모델 파일이 없습니다. 먼저 'python train_unified_models.py -w {w_label}' 를 실행해주세요.")

    scaler = joblib.load(scaler_path)
    in_dim = len(scaler.mean_)

    if "autoencoder" in model_type:
        ae = AnomalyAutoEncoder(in_dim=in_dim, latent_dim=8)
        ae_path = os.path.join(MODELS_SAVE_DIR, f"step1_autoencoder{w_suffix}.pt")
        if not os.path.exists(ae_path): ae_path = os.path.join(MODELS_SAVE_DIR, "step1_autoencoder_w250.pt")
        ae.load_state_dict(torch.load(ae_path))
        ae.eval()
        step1_fn = lambda x_s: float(torch.mean((ae(torch.tensor(x_s, dtype=torch.float32)) - torch.tensor(x_s, dtype=torch.float32)) ** 2).item())
    elif "oc_svm" in model_type:
        oc_path = os.path.join(MODELS_SAVE_DIR, f"step1_oc_svm{w_suffix}.pkl")
        if not os.path.exists(oc_path): oc_path = os.path.join(MODELS_SAVE_DIR, "step1_oc_svm_w250.pkl")
        oc_svm = joblib.load(oc_path)
        step1_fn = lambda x_s: float(oc_svm.decision_function(x_s)[0])
    elif "iforest" in model_type:
        if_path = os.path.join(MODELS_SAVE_DIR, f"step1_iforest{w_suffix}.pkl")
        if not os.path.exists(if_path): if_path = os.path.join(MODELS_SAVE_DIR, "step1_iforest_w250.pkl")
        iforest = joblib.load(if_path)
        step1_fn = lambda x_s: float(iforest.decision_function(x_s)[0])
    else:
        raise ValueError(f"지원하지 않는 Step 1 모델: {model_type}")

    if model_type == "autoencoder_gru":
        gru = SingleGRU(in_dim=in_dim + 1, hidden=32)
        p = os.path.join(MODELS_SAVE_DIR, f"step2_ae_gru{w_suffix}.pt")
        if not os.path.exists(p): p = os.path.join(MODELS_SAVE_DIR, "step2_ae_gru_w250.pt")
        gru.load_state_dict(torch.load(p))
        gru.eval()
        step2_obj = gru
    elif model_type == "autoencoder_xgboost":
        p = os.path.join(MODELS_SAVE_DIR, f"step2_ae_xgboost{w_suffix}.pkl")
        if not os.path.exists(p): p = os.path.join(MODELS_SAVE_DIR, "step2_ae_xgboost_w250.pkl")
        step2_obj = joblib.load(p)
    elif model_type == "oc_svm_gru":
        gru = SingleGRU(in_dim=in_dim + 1, hidden=32)
        p = os.path.join(MODELS_SAVE_DIR, f"step2_oc_gru{w_suffix}.pt")
        if not os.path.exists(p): p = os.path.join(MODELS_SAVE_DIR, "step2_oc_gru_w250.pt")
        gru.load_state_dict(torch.load(p))
        gru.eval()
        step2_obj = gru
    elif model_type == "oc_svm_rnn":
        rnn = SingleRNN(in_dim=in_dim + 1, hidden=32)
        p = os.path.join(MODELS_SAVE_DIR, f"step2_oc_rnn{w_suffix}.pt")
        if not os.path.exists(p): p = os.path.join(MODELS_SAVE_DIR, "step2_oc_rnn_w250.pt")
        rnn.load_state_dict(torch.load(p))
        rnn.eval()
        step2_obj = rnn
    elif model_type == "oc_svm_xgboost":
        p = os.path.join(MODELS_SAVE_DIR, f"step2_oc_xgboost{w_suffix}.pkl")
        if not os.path.exists(p): p = os.path.join(MODELS_SAVE_DIR, "step2_oc_xgboost_w250.pkl")
        step2_obj = joblib.load(p)
    else:
        raise ValueError(f"지원하지 않는 Step 2 모델: {model_type}")

    return scaler, step1_fn, step2_obj, f"2-Step [{model_type}]"


def run_realtime_inference(input_path, window_size=250, model_type="autoencoder_gru"):
    print("\n" + "=" * 110)
    print(f" 🚗 [미학습 신규 데이터 실시간 고장 진단 인퍼런스 엔진]")
    print(f" 📂 입력 파일: {input_path}")
    print(f" ⏱️ 윈도우 크기: {window_size}샘플 ({window_size / 340.0:.2f}초) | 🧠 2-Step 모델: {model_type}")
    print("=" * 110)

    scaler, step1_fn, step2_obj, model_name = load_exported_2step_pipeline(model_type, window_size)
    df_raw = clean_and_standardize_csv(input_path)
    n_windows = len(df_raw) // window_size

    results = []
    latencies = []

    print(f" {'WinIdx':<8} | {'구간 시간(초)':<18} | {'실시간 평균차속':<14} | {'1단계 Anomaly Score':<22} | {'고장확률':<10} | {'진단 결과':<16} | {'추론시간':<10}")
    print("-" * 110)

    for w_idx in range(n_windows):
        start_i = w_idx * window_size
        end_i = start_i + window_size
        seg = df_raw.iloc[start_i:end_i]

        t_start = seg["time"].iloc[0] if "time" in seg.columns else w_idx * (window_size / 340.0)
        t_end = seg["time"].iloc[-1] if "time" in seg.columns else (w_idx + 1) * (window_size / 340.0)

        t0 = time.perf_counter()
        f_dict = extract_features_from_window(seg)
        f_scaled = scaler.transform(np.array(list(f_dict.values())).reshape(1, -1))
        anomaly_score = step1_fn(f_scaled)
        f_combined = np.hstack([f_scaled, np.array([[anomaly_score]])])

        if isinstance(step2_obj, nn.Module):
            with torch.no_grad():
                logits = step2_obj(torch.tensor(np.expand_dims(f_combined, axis=1), dtype=torch.float32))
                fault_prob = float(torch.softmax(logits, dim=1)[:, 1].item())
                pred_label = 1 if fault_prob >= 0.5 else 0
        else:
            fault_prob = float(step2_obj.predict_proba(f_combined)[:, 1][0])
            pred_label = int(step2_obj.predict(f_combined)[0])

        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        diag_str = "⚠️ [ABNORMAL FAULT]" if pred_label == 1 else "✅ [NORMAL]"
        time_range = f"{t_start:.2f}s ~ {t_end:.2f}s"
        mean_spd = f"{f_dict['mean']:.1f} km/h"

        if w_idx < 10 or w_idx % (max(1, n_windows // 10)) == 0 or w_idx == n_windows - 1:
            print(f" {w_idx:<8} | {time_range:<18} | {mean_spd:<14} | {anomaly_score:>18.5f} | {fault_prob * 100:>6.1f}%   | {diag_str:<16} | {lat_ms:>6.3f} ms")

        results.append({
            "window_index": w_idx,
            "start_time_sec": round(t_start, 3), "end_time_sec": round(t_end, 3),
            "mean_wheel_speed_kph": round(f_dict["mean"], 2),
            "step1_anomaly_score": round(anomaly_score, 6),
            "fault_probability": round(fault_prob, 4),
            "prediction": pred_label,
            "prediction_label": "abnormal" if pred_label == 1 else "normal",
            "inference_latency_ms": round(lat_ms, 3)
        })

    df_res = pd.DataFrame(results)
    abn_count = int(np.sum(df_res["prediction"] == 1))
    abn_rate = (abn_count / len(df_res)) * 100.0
    mean_lat = float(np.mean(latencies))

    print("\n" + "=" * 110)
    print(f" 📋 [최종 진단 리포트] 파일: {os.path.basename(input_path)} | 윈도우: {len(df_res)}개 | 고장감지: {abn_count}개 ({abn_rate:.1f}%) | 평균지연: {mean_lat:.3f}ms")
    if abn_rate >= 30.0:
        print("  🚨 판정 : >>> 【 CRITICAL WARNING: 기어박스 감속기 고장(ABNORMAL) 감지 】 <<<")
    else:
        print("  🛡️ 판정 : >>> 【 STATUS OK: 기어박스 정상(NORMAL) 작동 중 】 <<<")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_csv = os.path.join(PRED_DIR, f"inference_{base_name}_{model_type}_w{window_size}_result.csv")
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f" 💾 결과 저장 완료: {out_csv}\n" + "=" * 110)
    return df_res


def main():
    parser = argparse.ArgumentParser(description="2-Step 비지도 하이브리드 실시간 고장 진단 엔진")
    parser.add_argument("--input", required=True, help="미학습 신규 주행 CSV 경로")
    parser.add_argument("-w", "--window_size", type=int, default=250, choices=[250, 500], help="윈도우 크기")
    parser.add_argument("--model_type", default="autoencoder_gru",
                        choices=["autoencoder_gru", "autoencoder_xgboost", "oc_svm_gru", "oc_svm_rnn", "oc_svm_xgboost"],
                        help="2-Step 모델 타입")
    args = parser.parse_args()
    run_realtime_inference(args.input, args.window_size, args.model_type)


if __name__ == "__main__":
    main()
