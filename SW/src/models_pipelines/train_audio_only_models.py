"""
train_audio_only_models.py (gearbox_final_suite / src / models_pipelines)

오디오 단독 이상 탐지 실험 (Audio-Only Anomaly Detection Experiment)

입력: unified_audio_only_dataset_w34.csv (14개 오디오 피처)

3가지 패러다임 시나리오 내 시간축 분할 평가 (앞 75% Train / gap 20 / 나머지 Test):
  - Unsupervised  : AutoEncoder (정상 데이터만 학습 -> MSE 임계값 예측)
  - Semi-Super    : OC-SVM, Isolation Forest (정상 데이터만 학습 -> 결정 경계 임계값 예측)
  - Supervised    : XGBoost, SingleGRU (레이블 사용 -> 직접 분류)

참조: train_unified_models.py Step 1 파이프라인

규정 준수:
- Rule 3: 코드 주석 100% 한글, 터미널 100% 영문 라벨.
- Rule 4: Data Leakage 배제 (Train fold만 fit, Test fold은 transform만).
- Rule 7: LaTeX 수식 기호 일절 사용 금지.
- Rule 8: 이모지 전면 배제.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

os.makedirs(METRICS_DIR, exist_ok=True)

AUDIO_ONLY_CSV = os.path.join(DATA_DIR, "unified_audio_only_dataset_w34.csv")


# =========================================================================
# [Step 1 비지도 신경망: AutoEncoder]
# =========================================================================
class AnomalyAutoEncoder(nn.Module):
    """정상 오디오 데이터로만 학습하는 비지도 복원 신경망"""
    def __init__(self, in_dim, latent_dim=8):
        super().__init__()
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


def train_autoencoder(X_norm, epochs=30, batch_size=32, lr=0.01):
    """정상 데이터(y==0)로만 AutoEncoder 학습"""
    in_dim = X_norm.shape[1]
    model = AnomalyAutoEncoder(in_dim, latent_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.tensor(X_norm, dtype=torch.float32)), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (bx,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), bx)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def get_ae_recon_error(model, X):
    """복원 오차(MSE) 계산 -> 이상치 점수"""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32)
        recon = model(x_t)
        err = torch.mean((recon - x_t) ** 2, dim=1).numpy()
    return err


# =========================================================================
# [Step 2 시계열 신경망: SingleGRU]
# =========================================================================
class SingleGRU(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def build_sequence_tensor(X, y, seq_len=10):
    """비중첩 seq_len 윈도우 시퀀스 행렬 생성 [N, seq_len, features]"""
    n_seq = len(X) // seq_len
    if n_seq == 0:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0, dtype=int)
    X_seq = X[:n_seq * seq_len].reshape(n_seq, seq_len, X.shape[1])
    # 시퀀스 레이블: 해당 구간 내 이상 윈도우가 하나라도 있으면 1
    y_seq = (y[:n_seq * seq_len].reshape(n_seq, seq_len).max(axis=1))
    return X_seq.astype(np.float32), y_seq.astype(int)


def train_gru(X_train_3d, y_train, X_test_3d, epochs=30, batch_size=32, lr=0.005):
    """GRU 학습 및 테스트 예측"""
    in_dim = X_train_3d.shape[2]
    model = SingleGRU(in_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_3d), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test_3d))
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = torch.argmax(logits, dim=1).numpy()
    return preds, probs


# =========================================================================
# [결과 출력 헬퍼]
# =========================================================================
def record(results, name, y_true, y_pred, y_prob=None):
    results[name]["acc"].append(accuracy_score(y_true, y_pred))
    results[name]["prec"].append(precision_score(y_true, y_pred, zero_division=0))
    results[name]["rec"].append(recall_score(y_true, y_pred, zero_division=0))
    results[name]["f1"].append(f1_score(y_true, y_pred, zero_division=0))
    if y_prob is not None and len(np.unique(y_true)) > 1:
        results[name]["auc"].append(roc_auc_score(y_true, y_prob))
    else:
        results[name]["auc"].append(float("nan"))


def print_results(results, model_names):
    print(f"\n {'Model':<55} | {'Acc':>8} | {'Prec':>8} | {'Rec':>8} | {'F1':>8} | {'AUC':>8}")
    print("-" * 110)
    for name in model_names:
        r = results[name]
        acc  = np.nanmean(r["acc"])  * 100
        prec = np.nanmean(r["prec"]) * 100
        rec  = np.nanmean(r["rec"])  * 100
        f1   = np.nanmean(r["f1"])   * 100
        auc  = np.nanmean(r["auc"])
        print(f" {name:<55} | {acc:>7.2f}% | {prec:>7.2f}% | {rec:>7.2f}% | {f1:>7.2f}% | {auc:>8.4f}")


# =========================================================================
# [메인: 5-Fold CV 평가]
# =========================================================================
def evaluate_audio_only_splits(splits, X, y, split_name):
    """지정된 분할 방식(Temporal 또는 GroupKFold)으로 모델 평가 및 결과 기록"""
    print("\n" + "=" * 110)
    print(f" [Audio-Only Model Evaluation] Strategy: {split_name}")
    print("=" * 110)

    model_names = [
        "[UNSUPER] AutoEncoder (MSE Threshold)",
        "[SEMISUPER] OC-SVM (Decision Boundary)",
        "[SEMISUPER] Isolation Forest (Anomaly Score)",
        "[SUPERVISED] XGBoost (Window)",
        "[SUPERVISED] XGBoost (Seq=10 Lagged)",
        "[SUPERVISED] SingleGRU (Seq=10)",
    ]
    results = {m: {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []} for m in model_names}

    for fold_idx, (tr_idx, te_idx) in enumerate(splits, start=1):
        print(f"\n --- Fold {fold_idx}/{len(splits)} (Train: {len(tr_idx):,}, Test: {len(te_idx):,}) ---")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 스케일링 (Train fold만 fit)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # 정상 데이터만 추출 (비지도/반지도 학습용)
        X_tr_norm = X_tr_s[y_tr == 0]

        # [1] UNSUPERVISED: AutoEncoder
        ae = train_autoencoder(X_tr_norm, epochs=25)
        tr_err = get_ae_recon_error(ae, X_tr_s)
        te_err = get_ae_recon_error(ae, X_te_s)
        ae_thresh = np.percentile(tr_err[y_tr == 0], 95)
        pred_ae = (te_err > ae_thresh).astype(int)
        record(results, "[UNSUPER] AutoEncoder (MSE Threshold)", y_te, pred_ae, te_err)

        # [2] SEMI-SUPERVISED: OC-SVM
        oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_tr_norm)
        tr_oc = oc_svm.decision_function(X_tr_s)
        te_oc = oc_svm.decision_function(X_te_s)
        oc_thresh = np.percentile(tr_oc[y_tr == 0], 5)
        pred_oc = (te_oc < oc_thresh).astype(int)
        record(results, "[SEMISUPER] OC-SVM (Decision Boundary)", y_te, pred_oc, -te_oc)

        # [3] SEMI-SUPERVISED: Isolation Forest
        iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42).fit(X_tr_norm)
        tr_if = iforest.decision_function(X_tr_s)
        te_if = iforest.decision_function(X_te_s)
        if_thresh = np.percentile(tr_if[y_tr == 0], 5)
        pred_if = (te_if < if_thresh).astype(int)
        record(results, "[SEMISUPER] Isolation Forest (Anomaly Score)", y_te, pred_if, -te_if)

        # [4] SUPERVISED: XGBoost (Window)
        xgb_w = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                    random_state=42, eval_metric="logloss", n_jobs=-1)
        xgb_w.fit(X_tr_s, y_tr)
        pred_xgb_w = xgb_w.predict(X_te_s)
        prob_xgb_w = xgb_w.predict_proba(X_te_s)[:, 1]
        record(results, "[SUPERVISED] XGBoost (Window)", y_te, pred_xgb_w, prob_xgb_w)

        # [5] SUPERVISED: XGBoost (Seq=10 Lagged)
        X_tr_seq, y_tr_seq = build_sequence_tensor(X_tr_s, y_tr, seq_len=10)
        X_te_seq, y_te_seq = build_sequence_tensor(X_te_s, y_te, seq_len=10)

        if len(y_te_seq) > 0:
            X_tr_lag = X_tr_seq.reshape(len(X_tr_seq), -1)
            X_te_lag = X_te_seq.reshape(len(X_te_seq), -1)
            xgb_seq = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                          random_state=42, eval_metric="logloss", n_jobs=-1)
            xgb_seq.fit(X_tr_lag, y_tr_seq)
            pred_xgb_seq = xgb_seq.predict(X_te_lag)
            prob_xgb_seq = xgb_seq.predict_proba(X_te_lag)[:, 1]
            record(results, "[SUPERVISED] XGBoost (Seq=10 Lagged)", y_te_seq, pred_xgb_seq, prob_xgb_seq)

            # [6] SUPERVISED: SingleGRU (Seq=10)
            pred_gru, prob_gru = train_gru(X_tr_seq, y_tr_seq, X_te_seq, epochs=25)
            record(results, "[SUPERVISED] SingleGRU (Seq=10)", y_te_seq, pred_gru, prob_gru)

    print("\n" + "=" * 110)
    print(f" [Results Summary] - Strategy: {split_name}")
    print("=" * 110)
    print_results(results, model_names)

    return {
        name: {
            "accuracy":  float(np.nanmean(v["acc"])),
            "precision": float(np.nanmean(v["prec"])),
            "recall":    float(np.nanmean(v["rec"])),
            "f1_score":  float(np.nanmean(v["f1"])),
            "roc_auc":   float(np.nanmean(v["auc"])),
        }
        for name, v in results.items()
    }


def run_audio_only_experiment():
    import argparse
    parser = argparse.ArgumentParser(description="Audio-Only Anomaly Detection Experiment")
    parser.add_argument("--split_mode", type=str, default="all", choices=["temporal", "group_kfold", "all"],
                        help="분할 방식: temporal(1번 Causal Temporal Split + Gap), group_kfold(2번 Leave-One-Scenario-Out), all(둘 다)")
    args = parser.parse_args()

    if not os.path.exists(AUDIO_ONLY_CSV):
        print(f"[ERROR] 데이터셋 없음: {AUDIO_ONLY_CSV}")
        return

    df = pd.read_csv(AUDIO_ONLY_CSV)
    print(f" Dataset: {len(df):,} rows x {len(df.columns)} cols")

    audio_cols = [c for c in df.columns if c.startswith("audio_")]
    X = df[audio_cols].fillna(0.0).values
    y = df["label"].values
    scenarios = df["scenario"].values

    summary_out = {}

    # [1번 방식] Causal Chronological Split + 40% Embargo Gap (시간 순서 전후 분할: 50% Train / 40% Gap 차단 / 10% Test)
    TRAIN_RATIO = 0.50
    GAP_RATIO = 0.40
    train_idx, test_idx = [], []
    session_col = df["session_id"] if "session_id" in df.columns else (df["state"].astype(str) + "_" + df["scenario"].astype(str))
    for sess in session_col.unique():
        idx = np.where(session_col.values == sess)[0]
        cut_train = int(len(idx) * TRAIN_RATIO)
        gap_len = int(len(idx) * GAP_RATIO)
        train_idx.extend(idx[:cut_train])
        test_idx.extend(idx[cut_train + gap_len:])
    temporal_splits = [(np.array(train_idx), np.array(test_idx))]
    summary_out["1_causal_temporal_split"] = evaluate_audio_only_splits(
        temporal_splits, X, y, "1. Causal Chronological Split (50% Train / 40% Embargo Gap Blocked / 10% Test)"
    )

    # [2번 방식] Leave-One-Group-Out / GroupKFold Split (사용자 지시로 주석 처리)
    # if args.split_mode in ["group_kfold", "all"]:
    #     from sklearn.model_selection import GroupKFold
    #     gkf = GroupKFold(n_splits=min(5, len(np.unique(scenarios))))
    #     group_splits = list(gkf.split(X, y, groups=scenarios))
    #     summary_out["2_leave_one_group_out"] = evaluate_audio_only_splits(
    #         group_splits, X, y, f"2. Leave-One-Group-Out ({len(group_splits)}-Fold Scenario Hold-Out)"
    #     )

    out_path = os.path.join(METRICS_DIR, "audio_only_experiment_w34.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    print(f"\n [Saved Summary]: {out_path}\n")


if __name__ == "__main__":
    run_audio_only_experiment()

