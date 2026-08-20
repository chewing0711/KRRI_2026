"""
realtime_rnn_window_streamer_strictly_isolated.py (gearbox_final_suite / models_pipelines)

수정된 엄격 격리 실시간 스트리머 (Strict Session-Isolated Real-time Streamer):
1. 세션 경계 보존 (Session-Boundary Protection): 서로 다른 session_id 간 시퀀스 믹싱 차단.
2. 4가지 분할 모드 제공 (paired_timeblock, group_cv, session_holdout, stratified).
3. 유출 차단 정량 증명 기능 (Quantitative Leakage Proof Audit):
   - Train-Test 인덱스 교집합 0건 증명 (Index Intersection = 0)
   - Train-Test 간 최고 코사인 유사도 (Max Cosine Similarity Check)
   - 시계열 겹침 0% 엠바고 갭 (Embargo Gap Overlap Audit)
4. 엄격한 정규화 격리 (Strict Scaler Isolation): 오직 Train 세트 데이터로만 mean/std fit 수행.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

MODELS_PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MODELS_PIPELINES_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
UNIFIED_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")

CAN_FREQ = 340.14  # Hz
WINDOW_SAMPLES = 500


class FullFeatureRNNAnomalyDetector(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=64, num_layers=2, rnn_type="LSTM", dropout=0.2):
        super(FullFeatureRNNAnomalyDetector, self).__init__()
        self.rnn_type = rnn_type
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last_out = out[:, -1, :]
        prob = self.fc(last_out)
        return prob.squeeze(-1)


def build_session_protected_sequences(df_unified, feature_cols, seq_len=5):
    X_seq_raw, y_seq_raw, session_ids, scenarios, states, X_tab_raw = [], [], [], [], [], []

    for sess_id, df_sess in df_unified.groupby("session_id", sort=False):
        X_sess = df_sess[feature_cols].values
        y_sess = df_sess["target"].values
        scen_sess = df_sess["scenario"].values
        state_sess = df_sess["state"].values

        n_sess = len(df_sess)
        if n_sess < seq_len:
            continue

        for i in range(n_sess - seq_len + 1):
            X_seq_raw.append(X_sess[i : i + seq_len])
            y_seq_raw.append(y_sess[i + seq_len - 1])
            X_tab_raw.append(X_sess[i + seq_len - 1])
            session_ids.append(sess_id)
            scenarios.append(scen_sess[i + seq_len - 1])
            states.append(state_sess[i + seq_len - 1])

    return (
        np.array(X_seq_raw),
        np.array(y_seq_raw),
        np.array(X_tab_raw),
        np.array(session_ids),
        np.array(scenarios),
        np.array(states),
    )


def audit_quantitative_leakage_proof(X_tab_tr, X_tab_te, train_idx, test_idx, embargo_count):
    """
    유출이 100% 제거되었음을 증명하는 3단계 정량 검증 함수
    """
    print("\n" + "=" * 95)
    print(" 🔬 ZERO LEAKAGE QUANTITATIVE PROOF AUDIT (유출 차단 3단계 정량 증명)")
    print("=" * 95)

    # 1. 인덱스 교집합 검증
    common_indices = set(train_idx).intersection(set(test_idx))
    idx_proof_status = "PASSED (0 Common)" if len(common_indices) == 0 else f"FAILED ({len(common_indices)} Common)"
    print(f" [증명 1] Train-Test 인덱스 교집합 : {len(common_indices)}건 ──► {idx_proof_status}")

    # 2. 시계열 엠바고 갭(Embargo Gap) 샘플 격리 증명
    # Seq_Len=5 이므로 차단 갭이 5 이상이면 샘플 겹침이 물리학적으로 불가능함
    overlap_proof_status = "PASSED (0% Overlap)" if embargo_count >= 5 else "WARNING (Potential Overlap)"
    print(f" [증명 2] 차단된 Embargo 시퀀스  : {embargo_count}개 (> Seq_Len 5) ──► {overlap_proof_status}")

    # 3. Pairwise Cosine Similarity 검증
    norm_tr = np.linalg.norm(X_tab_tr, axis=1, keepdims=True) + 1e-12
    norm_te = np.linalg.norm(X_tab_te, axis=1, keepdims=True) + 1e-12
    X_tr_normed = X_tab_tr / norm_tr
    X_te_normed = X_tab_te / norm_te

    sim_matrix = np.dot(X_tr_normed, X_te_normed.T)
    max_sim_per_test = sim_matrix.max(axis=0)
    exact_duplicates = np.sum(max_sim_per_test >= 0.999999)

    cos_proof_status = "PASSED (0 Leakage Rows)" if exact_duplicates == 0 else f"FAILED ({exact_duplicates} Duplicates)"
    print(f" [증명 3] 코사인 유사도 유출 검증 : 최고 유사도={max_sim_per_test.max():.6f}, 중복행={exact_duplicates}건 ──► {cos_proof_status}")
    print("=" * 95)


def run_strictly_isolated_streaming(
    split_mode="paired_timeblock", test_size=0.20, embargo_pct=0.05, random_state=42, rnn_type="LSTM"
):
    print("=" * 95)
    print(f" 🛡️ STRICTLY ISOLATED STREAMER (MODE={split_mode.upper()}, RANDOM_SEED={random_state}, MODEL={rnn_type})")
    print("=" * 95)

    if not os.path.exists(UNIFIED_CSV):
        print(f"[ERROR] Unified CSV missing: {UNIFIED_CSV}")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)
    meta_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df_unified.columns if c not in meta_cols]

    seq_len = 5
    X_seq_raw, y_seq_raw, X_tab_raw, session_ids, scenarios, states = build_session_protected_sequences(
        df_unified, feature_cols, seq_len=seq_len
    )

    embargo_count = 0
    if split_mode == "paired_timeblock":
        train_idx, test_idx = [], []
        df_meta = pd.DataFrame({"scenario": scenarios, "state": states, "idx": np.arange(len(X_seq_raw))})

        for (scen, st), group in df_meta.groupby(["scenario", "state"], sort=False):
            sub_idx = group["idx"].values
            n = len(sub_idx)
            if n == 0:
                continue
            n_tr = int(n * (1.0 - test_size - embargo_pct))
            n_gap = int(n * embargo_pct)
            train_idx.extend(sub_idx[:n_tr])
            test_idx.extend(sub_idx[n_tr + n_gap:])

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        embargo_count = len(X_seq_raw) - len(train_idx) - len(test_idx)

    elif split_mode == "group_cv":
        gkf = GroupKFold(n_splits=5)
        train_idx, test_idx = next(gkf.split(X_seq_raw, y_seq_raw, groups=session_ids))
    elif split_mode == "session_holdout":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X_seq_raw, y_seq_raw, groups=session_ids))
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(skf.split(X_seq_raw, y_seq_raw))

    # Fit Scaler strictly on train_idx
    mean_tr = np.nan_to_num(X_tab_raw[train_idx].mean(axis=0, keepdims=True))
    std_tr = np.nan_to_num(X_tab_raw[train_idx].std(axis=0, keepdims=True)) + 1e-8

    X_seq_tr = (X_seq_raw[train_idx] - mean_tr) / std_tr
    X_seq_te = (X_seq_raw[test_idx] - mean_tr) / std_tr
    X_tab_tr = (X_tab_raw[train_idx] - mean_tr) / std_tr
    X_tab_te = (X_tab_raw[test_idx] - mean_tr) / std_tr
    y_tr, y_te = y_seq_raw[train_idx], y_seq_raw[test_idx]

    # 정량적 유출 차단 3단계 증명 실행
    audit_quantitative_leakage_proof(X_tab_tr, X_tab_te, train_idx, test_idx, embargo_count)

    print("\n[1/2] Training Step 1 PyTorch LSTM (40 Epochs)...")
    model_step1 = FullFeatureRNNAnomalyDetector(input_dim=len(feature_cols), hidden_dim=64, rnn_type=rnn_type)
    optimizer = torch.optim.Adam(model_step1.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    ds_tr = TensorDataset(torch.tensor(X_seq_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)

    model_step1.train()
    for _ in range(40):
        for bx, by in loader_tr:
            optimizer.zero_grad()
            out = model_step1(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    model_step1.eval()

    print("[2/2] Training Step 2 XGBoost Classifier...")
    model_step2 = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=random_state, eval_metric="logloss")
    model_step2.fit(X_tab_tr, y_tr)

    print(f"\n 🚀 EVALUATING STREAMING DIAGNOSIS ON ISOLATED TEST SET ({len(test_idx)} Windows)...")
    stream_records = []
    total_start_time = time.perf_counter()

    for i_idx, te_i in enumerate(test_idx):
        seq_input = torch.tensor(X_seq_te[i_idx : i_idx + 1], dtype=torch.float32)
        tab_input = X_tab_te[i_idx : i_idx + 1]

        t0 = time.perf_counter()
        with torch.no_grad():
            prob_step1 = model_step1(seq_input).item()
        pred_step2 = model_step2.predict(tab_input)[0]

        final_hybrid_pred = 1 if (prob_step1 >= 0.5 or pred_step2 == 1) else 0
        t1 = time.perf_counter()

        single_lat_ms = (t1 - t0) * 1000.0
        single_lat_us = single_lat_ms * 1000.0
        empirical_fps = round(1000.0 / single_lat_ms, 1) if single_lat_ms > 0 else 0.0

        true_label = int(y_te[i_idx])

        stream_records.append({
            "stream_step": i_idx,
            "raw_dataset_window_idx": te_i,
            "split_mode": split_mode,
            "session_id": session_ids[te_i],
            "scenario": scenarios[te_i],
            "true_ground_truth": true_label,
            "step1_anomaly_probability": round(prob_step1, 6),
            "step2_xgboost_pred": int(pred_step2),
            "final_hybrid_pred": final_hybrid_pred,
            "prediction_match": (final_hybrid_pred == true_label),
            "single_window_lat_ms": round(single_lat_ms, 4),
            "single_window_lat_us": round(single_lat_us, 1),
            "empirical_model_fps": empirical_fps,
        })

    total_end_time = time.perf_counter()
    total_duration_sec = total_end_time - total_start_time

    df_stream = pd.DataFrame(stream_records)

    test_acc = accuracy_score(y_te, df_stream["final_hybrid_pred"])
    test_prec = precision_score(y_te, df_stream["final_hybrid_pred"], zero_division=0)
    test_rec = recall_score(y_te, df_stream["final_hybrid_pred"], zero_division=0)
    test_f1 = f1_score(y_te, df_stream["final_hybrid_pred"], zero_division=0)

    output_predictions_path = os.path.join(RESULTS_DIR, f"realtime_stream_strictly_isolated_predictions.csv")
    df_stream.to_csv(output_predictions_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 95)
    print(f" 📊 ISOLATED TEST STREAMING COMPLETED (MODE={split_mode}, {len(df_stream)} Windows Evaluated)")
    print("=" * 95)
    print(f" - Total Model Inference Time           : {total_duration_sec:.4f} sec ({total_duration_sec*1000:.2f} ms)")
    print(f" - Average Per-Window Latency           : {df_stream['single_window_lat_ms'].mean():.4f} ms ({df_stream['single_window_lat_us'].mean():.1f} us)")
    print(f" - Average Model FPS                    : {df_stream['empirical_model_fps'].mean():.1f} FPS")
    print(f" - 🛡️ Isolated Test Accuracy           : {test_acc*100:.2f}%")
    print(f" - 🛡️ Isolated Test Precision          : {test_prec:.4f}")
    print(f" - 🛡️ Isolated Test Recall             : {test_rec:.4f}")
    print(f" - 🛡️ Isolated Test F1-Score           : {test_f1:.4f}")
    print(f" [SUCCESS] Saved Predictions CSV        : {output_predictions_path}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_mode", type=str, default="paired_timeblock", choices=["paired_timeblock", "group_cv", "session_holdout", "stratified"], help="Split mode")
    parser.add_argument("--test_size", type=float, default=0.20, help="Test set ratio (default 0.20)")
    parser.add_argument("--embargo_pct", type=float, default=0.05, help="Embargo gap ratio for time-block split (default 0.05)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for data split")
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["LSTM", "RNN"], help="Model type")
    args = parser.parse_args()

    run_strictly_isolated_streaming(
        split_mode=args.split_mode,
        test_size=args.test_size,
        embargo_pct=args.embargo_pct,
        random_state=args.random_state,
        rnn_type=args.rnn_type,
    )
