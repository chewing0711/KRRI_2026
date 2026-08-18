"""
train_holdout_test_models.py (gearbox_final_suite / src / models_pipelines)

특정 주행 시나리오 1쌍(정상+고장 파일)을 학습에서 100% 완전히 배제(Hold-Out)하여
'미학습 정답 검증 전용 2-Step 모델'을 별도 빌드하고,
제외된 미학습 정상 파일 및 고장 파일에 대해 실시간 추론(inference) 정답률을 직접 검증하는 전용 스크립트.
"""

import os
import time
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import xgboost as xgb

DATA_PROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DATA_PROCESSING_DIR) if "models_pipelines" in DATA_PROCESSING_DIR else DATA_PROCESSING_DIR
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
MODELS_DIR = os.path.join(SUITE_DIR, "models")
RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
HOLDOUT_MODELS_DIR = os.path.join(MODELS_DIR, "holdout_test")
os.makedirs(HOLDOUT_MODELS_DIR, exist_ok=True)


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


def build_and_evaluate_holdout(scenario_to_exclude="hills_12deg", window_size=250):
    print("\n" + "=" * 110)
    print(f" 🎯 [진정한 미학습 정답 검증] Hold-Out 격리 모델 빌드 및 검증 파이프라인")
    print(f" 🛡️ 제외 대상 시나리오 : '{scenario_to_exclude}' (학습에 0% 반영, 오직 평가에만 사용)")
    print("=" * 110)

    # 1. 통합 데이터셋 로드
    csv_path = os.path.join(RESULTS_DIR, f"unified_can_context_dataset_w{window_size}.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(DATA_DIR, f"unified_can_context_dataset_w{window_size}.csv")

    df = pd.read_csv(csv_path)
    drop_cols = ["state", "target", "scenario", "window_idx", "session_id", "session_file", "raw_scenario"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # 2. Hold-Out 분리 (학습 데이터 vs 검증 데이터)
    train_mask = df["scenario"] != scenario_to_exclude
    test_mask = df["scenario"] == scenario_to_exclude

    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    print(f" 📊 전체 데이터 : {len(df)} 개 윈도우")
    print(f" 📚 학습 데이터 : {len(df_train)} 개 윈도우 (7개 시나리오)")
    print(f" 🧪 미학습 테스트 : {len(df_test)} 개 윈도우 (제외된 '{scenario_to_exclude}' 시나리오 1쌍)")

    X_tr = df_train[feature_cols].copy().fillna(0.0).values
    y_tr = df_train["target"].values
    X_te = df_test[feature_cols].copy().fillna(0.0).values
    y_te = df_test["target"].values

    # 3. [Step 1 비지도 학습: 학습용 정상 데이터 y_tr == 0 만 사용]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    X_tr_norm = X_tr_s[y_tr == 0]

    ae = AnomalyAutoEncoder(in_dim=X_tr.shape[1], latent_dim=8)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=0.01)
    crit_ae = nn.MSELoss()
    ae.train()
    norm_tensor = torch.tensor(X_tr_norm, dtype=torch.float32)
    for _ in range(35):
        opt_ae.zero_grad()
        loss = crit_ae(ae(norm_tensor), norm_tensor)
        loss.backward()
        opt_ae.step()
    ae.eval()

    with torch.no_grad():
        tr_ae_err = torch.mean((ae(torch.tensor(X_tr_s, dtype=torch.float32)) - torch.tensor(X_tr_s, dtype=torch.float32)) ** 2, dim=1, keepdim=True).numpy()
        te_ae_err = torch.mean((ae(torch.tensor(X_te_s, dtype=torch.float32)) - torch.tensor(X_te_s, dtype=torch.float32)) ** 2, dim=1, keepdim=True).numpy()

    X_tr_comb = np.hstack([X_tr_s, tr_ae_err])
    X_te_comb = np.hstack([X_te_s, te_ae_err])

    # 4. [Step 2 지도학습 GRU 학습]
    gru = SingleGRU(in_dim=X_tr_comb.shape[1], hidden=32)
    opt_gru = torch.optim.AdamW(gru.parameters(), lr=0.005, weight_decay=1e-4)
    crit_gru = nn.CrossEntropyLoss()
    gru.train()
    tr_seq = torch.tensor(np.expand_dims(X_tr_comb, axis=1), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    for _ in range(45):
        opt_gru.zero_grad()
        logits = gru(tr_seq)
        loss = crit_gru(logits, y_tr_t)
        loss.backward()
        opt_gru.step()
    gru.eval()

    # 5. [미학습 테스트 데이터셋 판별 검증]
    with torch.no_grad():
        te_seq = torch.tensor(np.expand_dims(X_te_comb, axis=1), dtype=torch.float32)
        te_logits = gru(te_seq)
        te_probs = torch.softmax(te_logits, dim=1)[:, 1].numpy()
        te_preds = (te_probs >= 0.5).astype(int)

    # 6. 미학습 정상 파일 vs 미학습 고장 파일 각각의 정답률 리포트
    norm_idx = np.where(y_te == 0)[0]
    abn_idx = np.where(y_te == 1)[0]

    norm_acc = (np.sum(te_preds[norm_idx] == 0) / len(norm_idx)) * 100.0 if len(norm_idx) > 0 else 0.0
    abn_acc = (np.sum(te_preds[abn_idx] == 1) / len(abn_idx)) * 100.0 if len(abn_idx) > 0 else 0.0
    total_acc = (np.sum(te_preds == y_te) / len(y_te)) * 100.0

    print("\n" + "-" * 110)
    print(f" 🏆 [미학습(Hold-Out) '{scenario_to_exclude}' 실제 정답 검증 결과]")
    print("-" * 110)
    print(f" 1. 미학습 정상 파일(Normal) 정답률 : {norm_acc:>6.2f}% ({np.sum(te_preds[norm_idx] == 0)} / {len(norm_idx)} 윈도우 정상 판별)")
    print(f" 2. 미학습 고장 파일(Abnormal) 탐지율 : {abn_acc:>6.2f}% ({np.sum(te_preds[abn_idx] == 1)} / {len(abn_idx)} 윈도우 고장 탐지)")
    print(f" 3. 미학습 세트 전체 종합 정확도   : {total_acc:>6.2f}%")
    print("-" * 110)

    # 모델 저장
    joblib.dump(scaler, os.path.join(HOLDOUT_MODELS_DIR, f"scaler_holdout_{scenario_to_exclude}.pkl"))
    torch.save(ae.state_dict(), os.path.join(HOLDOUT_MODELS_DIR, f"step1_ae_holdout_{scenario_to_exclude}.pt"))
    torch.save(gru.state_dict(), os.path.join(HOLDOUT_MODELS_DIR, f"step2_gru_holdout_{scenario_to_exclude}.pt"))
    print(f" 💾 미학습 검증 모델 저장 완료: {HOLDOUT_MODELS_DIR}")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="미학습 Hold-Out 정답 검증 스크립트")
    parser.add_argument("--exclude", default="hills_12deg", help="학습에서 제외할 시나리오명 (예: hills_12deg, high_speed_60kph 등)")
    parser.add_argument("-w", "--window_size", type=int, default=250, help="윈도우 크기")
    args = parser.parse_args()
    build_and_evaluate_holdout(args.exclude, args.window_size)


if __name__ == "__main__":
    main()
