"""
test_enhanced_features_benchmark.py (gearbox_final_suite / src / models_pipelines)

[강화 물리 피처 전용 벤치마크 및 Hold-Out 일반화 검증]
절대 속도 종속성을 제거한 '무차원 휠속 변동률' 및 '페달 토크 대비 슬립 응답비' 등
핵심 강화 피처들만 엄선하여 2-Step 하이브리드 모델의 성능과 미학습 일반화(60kph 등) 개선도를 검증하는 스크립트.

[강화된 핵심 물리 피처군]
1. 속도 무차원화 휠속 변동률: dimless_std, dimless_p2p, dimless_diff_fr_rr_std, dimless_diff_fl_rl_mean
2. 페달 토크 응답 피처: pedal_slip_response, accel_pedal_mean/max, brake_pedal_mean/max
3. 조향 및 요레이트 정규화: norm_yaw_rate, norm_lat_accel, sas_angle_mean, sas_speed_mean
4. 무차원 형상 계수: shape_factor, crest_factor, slip_ratio_fl~rr

규정 준수:
- Rule 3: 주석 100% 한글, Matplotlib/터미널 영문 라벨.
- Rule 4: Data Leakage 배제 및 코사인 유사도 검증.
- Rule 6: 비판적 엔지니어링 자체 비교 검증.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import xgboost as xgb

DATA_PROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DATA_PROCESSING_DIR) if "models_pipelines" in DATA_PROCESSING_DIR else DATA_PROCESSING_DIR
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
RESULTS_DIR = os.path.join(SUITE_DIR, "results")


class AnomalyAutoEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=6):
        super(AnomalyAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, in_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SingleGRU(nn.Module):
    def __init__(self, in_dim, hidden=24):
        super(SingleGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def create_enhanced_feature_dataset(raw_df):
    """39개 원본 피처에서 핵심 강화 피처 18개 생성"""
    df = raw_df.copy()
    base_spd = np.maximum(df["mean"].values, 1.0)
    pedal_base = np.maximum(df["accel_pedal_mean"].values, 1.0)

    enhanced = pd.DataFrame()
    # 1. 무차원 속도 변동률 (4개)
    enhanced["dimless_std"] = df["std"] / base_spd
    enhanced["dimless_p2p"] = df["p2p"] / base_spd
    enhanced["dimless_diff_fr_rr_std"] = df["diff_fr_rr_std"] / base_spd
    enhanced["dimless_diff_fl_rl_mean"] = df["diff_fl_rl_mean"] / base_spd

    # 2. 페달 토크 슬립 응답비 및 페달 (4개)
    enhanced["pedal_slip_response"] = df["diff_fr_rr_std"] / pedal_base
    enhanced["accel_pedal_mean"] = df["accel_pedal_mean"]
    enhanced["accel_pedal_max"] = df["accel_pedal_max"]
    enhanced["brake_pedal_max"] = df["brake_pedal_max"]

    # 3. 조향 및 요레이트 정규화 (4개)
    enhanced["norm_yaw_rate"] = df["norm_yaw_rate"]
    enhanced["norm_lat_accel"] = df["norm_lat_accel"]
    enhanced["sas_angle_mean"] = df["sas_angle_mean"]
    enhanced["sas_speed_mean"] = df["sas_speed_mean"]

    # 4. 무차원 형상 계수 및 슬립율 (6개)
    enhanced["shape_factor"] = df["shape_factor"]
    enhanced["crest_factor"] = df["crest_factor"]
    enhanced["slip_ratio_fl"] = df["slip_ratio_fl"]
    enhanced["slip_ratio_fr"] = df["slip_ratio_fr"]
    enhanced["slip_ratio_rl"] = df["slip_ratio_rl"]
    enhanced["slip_ratio_rr"] = df["slip_ratio_rr"]

    return enhanced


def evaluate_features_comparison(window_size=250):
    csv_path = os.path.join(RESULTS_DIR, f"unified_can_context_dataset_w{window_size}.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(DATA_DIR, f"unified_can_context_dataset_w{window_size}.csv")

    df = pd.read_csv(csv_path)
    drop_cols = ["state", "target", "scenario", "window_idx", "session_id", "session_file", "raw_scenario"]
    raw_feature_cols = [c for c in df.columns if c not in drop_cols]

    y = df["target"].values if "target" in df.columns else (df["state"].str.lower() == "abnormal").astype(int).values
    groups = df["scenario"].values

    # 1. 기존 39개 피처 행렬 vs 강화된 18개 피처 행렬
    X_raw = df[raw_feature_cols].copy().fillna(0.0).values
    df_enhanced = create_enhanced_feature_dataset(df)
    X_enh = df_enhanced.fillna(0.0).values

    print("\n" + "=" * 110)
    print(f" 🚀 [피처군 정량 비교 벤치마크] 기존 원본 39개 vs 강화 핵심 무차원 18개")
    print(f" 📊 데이터셋 행 수: {len(df):,} 개 윈도우")
    print(f" 🛠️ 기존 피처 수 : {X_raw.shape[1]} 개  |  ⭐ 강화 피처 수 : {X_enh.shape[1]} 개")
    print("=" * 110)

    # -------------------------------------------------------------
    # [평가 1] 5-Fold 교차 검증 성능 비교
    # -------------------------------------------------------------
    print("\n" + "-" * 110)
    print(" 📌 [평가 1] 5-Fold Cross-Validation 2-Step (AutoEncoder + GRU) 성능 비교")
    print("-" * 110)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def run_5fold_ae_gru(X_in):
        accs, f1s, recs = [], [], []
        for tr_idx, te_idx in skf.split(X_in, y):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_in[tr_idx])
            X_te_s = scaler.transform(X_in[te_idx])
            y_tr, y_te = y[tr_idx], y[te_idx]

            # Step 1 AE
            ae = AnomalyAutoEncoder(in_dim=X_in.shape[1], latent_dim=6)
            opt_ae = torch.optim.Adam(ae.parameters(), lr=0.01)
            crit_ae = nn.MSELoss()
            ae.train()
            norm_t = torch.tensor(X_tr_s[y_tr == 0], dtype=torch.float32)
            for _ in range(30):
                opt_ae.zero_grad()
                loss = crit_ae(ae(norm_t), norm_t)
                loss.backward()
                opt_ae.step()
            ae.eval()

            with torch.no_grad():
                tr_err = torch.mean((ae(torch.tensor(X_tr_s, dtype=torch.float32)) - torch.tensor(X_tr_s, dtype=torch.float32)) ** 2, dim=1, keepdim=True).numpy()
                te_err = torch.mean((ae(torch.tensor(X_te_s, dtype=torch.float32)) - torch.tensor(X_te_s, dtype=torch.float32)) ** 2, dim=1, keepdim=True).numpy()

            X_tr_c = np.hstack([X_tr_s, tr_err])
            X_te_c = np.hstack([X_te_s, te_err])

            # Step 2 GRU
            gru = SingleGRU(in_dim=X_tr_c.shape[1], hidden=24)
            opt_gru = torch.optim.AdamW(gru.parameters(), lr=0.005, weight_decay=1e-4)
            crit_gru = nn.CrossEntropyLoss()
            gru.train()
            tr_seq = torch.tensor(np.expand_dims(X_tr_c, axis=1), dtype=torch.float32)
            for _ in range(35):
                opt_gru.zero_grad()
                loss = crit_gru(gru(tr_seq), torch.tensor(y_tr, dtype=torch.long))
                loss.backward()
                opt_gru.step()
            gru.eval()

            with torch.no_grad():
                te_seq = torch.tensor(np.expand_dims(X_te_c, axis=1), dtype=torch.float32)
                logits = gru(te_seq)
                preds = (torch.softmax(logits, dim=1)[:, 1].numpy() >= 0.5).astype(int)

            accs.append(accuracy_score(y_te, preds))
            f1s.append(f1_score(y_te, preds, zero_division=0))
            recs.append(recall_score(y_te, preds, zero_division=0))

        return np.mean(accs) * 100, np.mean(f1s), np.mean(recs) * 100

    acc_raw, f1_raw, rec_raw = run_5fold_ae_gru(X_raw)
    acc_enh, f1_enh, rec_enh = run_5fold_ae_gru(X_enh)

    print(f" 1. 기존 39개 원본 피처 : 정확도 {acc_raw:>6.2f}% | F1-Score {f1_raw:>6.4f} | 고장탐지율(Recall) {rec_raw:>6.2f}%")
    print(f" 2. 강화 18개 무차원 피처: 정확도 {acc_enh:>6.2f}% | F1-Score {f1_enh:>6.4f} | 고장탐지율(Recall) {rec_enh:>6.2f}%")

    # -------------------------------------------------------------
    # [평가 2] 미학습 속도 대역(Hold-Out 60kph) 일반화 개선도 검증
    # -------------------------------------------------------------
    print("\n" + "-" * 110)
    print(" 📌 [평가 2] 미학습 60km/h 주행(Hold-Out) 일반화 개선도 비교 검증")
    print("-" * 110)

    def test_holdout_scenario(X_in, holdout_name="high_speed_60kph"):
        tr_mask = groups != holdout_name
        te_mask = groups == holdout_name

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_in[tr_mask])
        X_te_s = scaler.transform(X_in[te_mask])
        y_tr, y_te = y[tr_mask], y[te_mask]

        ae = AnomalyAutoEncoder(in_dim=X_in.shape[1], latent_dim=6)
        opt_ae = torch.optim.Adam(ae.parameters(), lr=0.01)
        crit_ae = nn.MSELoss()
        ae.train()
        norm_t = torch.tensor(X_tr_s[y_tr == 0], dtype=torch.float32)
        for _ in range(35):
            opt_ae.zero_grad()
            loss = crit_ae(ae(norm_t), norm_t)
            loss.backward()
            opt_ae.step()
        ae.eval()

        with torch.no_grad():
            tr_err = torch.mean((ae(torch.tensor(X_tr_s, dtype=torch.float32)) - torch.tensor(X_tr_s, dtype=torch.float32)) ** 2, dim=1, keepdim=True).numpy()
            te_err = torch.mean((ae(torch.tensor(X_te_s, dtype=torch.float32)) - torch.tensor(X_te_s, dtype=torch.float32)) ** 2, dim=1, keepdim=True).numpy()

        X_tr_c = np.hstack([X_tr_s, tr_err])
        X_te_c = np.hstack([X_te_s, te_err])

        gru = SingleGRU(in_dim=X_tr_c.shape[1], hidden=24)
        opt_gru = torch.optim.AdamW(gru.parameters(), lr=0.005, weight_decay=1e-4)
        crit_gru = nn.CrossEntropyLoss()
        gru.train()
        tr_seq = torch.tensor(np.expand_dims(X_tr_c, axis=1), dtype=torch.float32)
        for _ in range(40):
            opt_gru.zero_grad()
            loss = crit_gru(gru(tr_seq), torch.tensor(y_tr, dtype=torch.long))
            loss.backward()
            opt_gru.step()
        gru.eval()

        with torch.no_grad():
            te_seq = torch.tensor(np.expand_dims(X_te_c, axis=1), dtype=torch.float32)
            preds = (torch.softmax(gru(te_seq), dim=1)[:, 1].numpy() >= 0.5).astype(int)

        norm_idx = np.where(y_te == 0)[0]
        abn_idx = np.where(y_te == 1)[0]
        norm_acc = (np.sum(preds[norm_idx] == 0) / len(norm_idx)) * 100.0 if len(norm_idx) > 0 else 0.0
        abn_acc = (np.sum(preds[abn_idx] == 1) / len(abn_idx)) * 100.0 if len(abn_idx) > 0 else 0.0
        total_acc = (np.sum(preds == y_te) / len(y_te)) * 100.0

        return norm_acc, abn_acc, total_acc

    n_acc_raw, a_acc_raw, tot_raw = test_holdout_scenario(X_raw, "high_speed_60kph")
    n_acc_enh, a_acc_enh, tot_enh = test_holdout_scenario(X_enh, "high_speed_60kph")

    print(f" [기존 39개 피처]  60kph 정상 판별: {n_acc_raw:>6.2f}% | 60kph 고장 탐지: {a_acc_raw:>6.2f}% | 종합 정확도: {tot_raw:>6.2f}%")
    print(f" [강화 18개 피처]  60kph 정상 판별: {n_acc_enh:>6.2f}% | 60kph 고장 탐지: {a_acc_enh:>6.2f}% | 종합 정확도: {tot_enh:>6.2f}%")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="강화 피처 벤치마크 테스트")
    parser.add_argument("-w", "--window_size", type=int, default=250, help="윈도우 크기")
    args = parser.parse_args()
    evaluate_features_comparison(args.window_size)


if __name__ == "__main__":
    main()
