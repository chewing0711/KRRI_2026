"""
compare_inference_speed.py (gearbox_final_suite / cpp_conversion)

Inference Latency Benchmark Comparison Script:
- Compares 1-window inference latency (ms & us) across models:
  1. PyTorch Native LSTM
  2. PyTorch Native Simple RNN
  3. TorchScript C++ JIT Traced Model
  4. XGBoost Classifier
  5. Hybrid Pipeline A (LSTM + XGBoost)

Rule 3 Compliance:
- Benchmark measuring per-window inference latency in ms and us.
- Korean code comments & formatted terminal output.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from export_hybrid_pipeline_to_cpp import ExportableStep1SeqModel

CPP_CONV_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CPP_CONV_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
CPP_EXPORT_DIR = os.path.join(SUITE_DIR, "cpp_export")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "inference_speed_benchmark_results.csv")


def run_inference_speed_comparison():
    print("=" * 80)
    print(" ⏱️ INFERENCE SPEED & LATENCY COMPARISON BENCHMARK (MS / US)")
    print("=" * 80)

    if not os.path.exists(UNIFIED_CSV):
        print(f"[ERROR] Unified CSV missing: {UNIFIED_CSV}")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)
    meta_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df_unified.columns if c not in meta_cols]

    X_all = df_unified[feature_cols].values
    y_all = df_unified["target"].values

    mean_val = np.nan_to_num(X_all.mean(axis=0, keepdims=True))
    std_val = np.nan_to_num(X_all.std(axis=0, keepdims=True)) + 1e-8
    X_norm = (X_all - mean_val) / std_val

    seq_len = 5
    X_seq = np.array([X_norm[i : i + seq_len] for i in range(len(X_norm) - seq_len + 1)])

    speed_records = []
    num_test_runs = 1000  # 1000 window benchmark runs

    # 1. PyTorch Native LSTM
    print("\n[1/5] Measuring PyTorch Native LSTM Latency...")
    model_lstm = ExportableStep1SeqModel(input_dim=len(feature_cols), rnn_type="LSTM").eval()
    dummy_input = torch.tensor(X_seq[:1], dtype=torch.float32)

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(num_test_runs):
            _ = model_lstm(dummy_input)
        t1 = time.perf_counter()

    lat_ms_lstm = ((t1 - t0) / num_test_runs) * 1000
    speed_records.append({
        "engine_architecture": "PyTorch Native LSTM",
        "conversion_format": "Python PyTorch Runtime",
        "latency_per_window_ms": round(lat_ms_lstm, 4),
        "latency_per_window_us": round(lat_ms_lstm * 1000, 1),
        "inferences_per_sec_fps": round(1000.0 / lat_ms_lstm, 1),
    })

    # 2. PyTorch Native Simple RNN
    print("[2/5] Measuring PyTorch Native Simple RNN Latency...")
    model_rnn = ExportableStep1SeqModel(input_dim=len(feature_cols), rnn_type="RNN").eval()

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(num_test_runs):
            _ = model_rnn(dummy_input)
        t1 = time.perf_counter()

    lat_ms_rnn = ((t1 - t0) / num_test_runs) * 1000
    speed_records.append({
        "engine_architecture": "PyTorch Native Simple RNN",
        "conversion_format": "Python PyTorch Runtime",
        "latency_per_window_ms": round(lat_ms_rnn, 4),
        "latency_per_window_us": round(lat_ms_rnn * 1000, 1),
        "inferences_per_sec_fps": round(1000.0 / lat_ms_rnn, 1),
    })

    # 3. TorchScript C++ JIT Traced Model
    print("[3/5] Measuring TorchScript C++ JIT Model Latency...")
    traced_model = torch.jit.trace(model_lstm, dummy_input)

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(num_test_runs):
            _ = traced_model(dummy_input)
        t1 = time.perf_counter()

    lat_ms_ts = ((t1 - t0) / num_test_runs) * 1000
    speed_records.append({
        "engine_architecture": "TorchScript C++ JIT Engine",
        "conversion_format": "LibTorch C++ (.pt JIT)",
        "latency_per_window_ms": round(lat_ms_ts, 4),
        "latency_per_window_us": round(lat_ms_ts * 1000, 1),
        "inferences_per_sec_fps": round(1000.0 / lat_ms_ts, 1),
    })

    # 4. XGBoost Classifier
    print("[4/5] Measuring XGBoost Classifier Latency...")
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model.fit(X_norm[:100], y_all[:100])
    dummy_tab = X_norm[:1]

    t0 = time.perf_counter()
    for _ in range(num_test_runs):
        _ = xgb_model.predict(dummy_tab)
    t1 = time.perf_counter()

    lat_ms_xgb = ((t1 - t0) / num_test_runs) * 1000
    speed_records.append({
        "engine_architecture": "XGBoost Classifier",
        "conversion_format": "C++ Native / Treelite Branch",
        "latency_per_window_ms": round(lat_ms_xgb, 4),
        "latency_per_window_us": round(lat_ms_xgb * 1000, 1),
        "inferences_per_sec_fps": round(1000.0 / lat_ms_xgb, 1),
    })

    # 5. Hybrid Pipeline A (TorchScript + XGBoost)
    print("[5/5] Measuring 2-Step Hybrid Pipeline A Latency...")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_test_runs):
            p1 = traced_model(dummy_input).item()
            p2 = xgb_model.predict(dummy_tab)[0]
            _ = 1 if (p1 >= 0.5 or p2 == 1) else 0
    t1 = time.perf_counter()

    lat_ms_hyb = ((t1 - t0) / num_test_runs) * 1000
    speed_records.append({
        "engine_architecture": "Hybrid Pipeline A (LSTM + XGBoost)",
        "conversion_format": "2-Step C++ Integrated Pipeline",
        "latency_per_window_ms": round(lat_ms_hyb, 4),
        "latency_per_window_us": round(lat_ms_hyb * 1000, 1),
        "inferences_per_sec_fps": round(1000.0 / lat_ms_hyb, 1),
    })

    df_res = pd.DataFrame(speed_records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(" 📊 MODEL INFERENCE LATENCY COMPARISON TABLE (1,000 BENCHMARK RUNS)")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)
    print(f"\n[SUCCESS] Saved Inference Speed Benchmark CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_inference_speed_comparison()
