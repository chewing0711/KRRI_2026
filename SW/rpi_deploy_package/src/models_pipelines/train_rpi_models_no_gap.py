import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.spatial.distance import cosine
from scipy.signal import stft
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import OneClassSVM
import xgboost as xgb

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SUITE_DIR, "data")
METRICS_DIR = os.path.join(SUITE_DIR, "results", "metrics")
RPI_PKG_DIR = os.path.join(SUITE_DIR, "rpi_deploy_package")
RPI_MODEL_DIR = os.path.join(RPI_PKG_DIR, "models")
RPI_RESULT_DIR = os.path.join(RPI_PKG_DIR, "results")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(RPI_MODEL_DIR, exist_ok=True)
os.makedirs(RPI_RESULT_DIR, exist_ok=True)

sys.path.append(SRC_DIR)
from audio_processing.build_audio_only_dataset_w34 import extract_audio_window_features, CAN_LAUNCH_TRIMS

RAW_DECODED_DIR = os.path.join(DATA_DIR, "raw_decoded")
AUDIO_BPF_DIR = os.path.join(DATA_DIR, "audio_filtered_bpf")

NPERSEG = 256
NOVERLAP = 128
N_FREQ = 129

# =========================================================
# 모델 정의
# =========================================================

# 1. 1단계 이상 탐지
class AnomalyAutoEncoder(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=32, bottleneck_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, bottleneck_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(bottleneck_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

# 2. 2단계 단독 모델 (Supervised Only)
class AudioFeatureMLP(nn.Module):
    def __init__(self, in_dim=24, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 2)
        )
    def forward(self, x):
        return self.net(x)

class AudioCRNN(nn.Module):
    def __init__(self, in_freq=129, hidden_dim=64):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_freq, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.gru = nn.GRU(32, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        x_trans = x.transpose(1, 2)
        conv_out = self.conv1d(x_trans)
        gru_out, _ = self.gru(conv_out.transpose(1, 2))
        return self.fc(gru_out[:, -1, :])

class Audio2DSpectrogramCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, 2)
    def forward(self, x):
        feat = self.features(x)
        return self.fc(feat.view(feat.size(0), -1))

# 3. 2단계 하이브리드 모델 (Hybrid)
class AudioHybridFeatureMLP(nn.Module):
    def __init__(self, in_dim=25, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 2)
        )
    def forward(self, x):
        return self.net(x)

class AudioHybridCRNN(nn.Module):
    def __init__(self, in_freq=129, hidden_dim=64):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_freq, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.gru = nn.GRU(32, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim + 1, 2)
    def forward(self, x_spec, x_score):
        conv_out = self.conv1d(x_spec.transpose(1, 2))
        gru_out, _ = self.gru(conv_out.transpose(1, 2))
        return self.fc(torch.cat([gru_out[:, -1, :], x_score], dim=1))

class AudioHybrid2DSpectrogramCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(33, 2)
    def forward(self, x_spec, x_score):
        feat = self.features(x_spec)
        return self.fc(torch.cat([feat.view(feat.size(0), -1), x_score], dim=1))

# =========================================================
# 데이터 로딩
# =========================================================
def compute_window_stft(audio_slice, sr):
    if len(audio_slice) < NPERSEG:
        audio_slice = np.pad(audio_slice, (0, NPERSEG - len(audio_slice)))
    _, _, Zxx = stft(audio_slice, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    power = np.abs(Zxx) ** 2
    return power.T.astype(np.float32)

def build_rpi_no_gap_hybrid_dataset():
    X_feat_tr_list, X_feat_te_list = [], []
    X_spec_tr_list, X_spec_te_list = [], []
    y_tr_list, y_te_list = [], []

    for scenario_name, t_launch_can in tqdm(CAN_LAUNCH_TRIMS.items(), desc="[Extracting Dataset (1D Feat + 2D Spec)]"):
        label = 1 if scenario_name.startswith("abnormal") else 0
        can_csv = os.path.join(RAW_DECODED_DIR, f"{scenario_name}_official_decoded.csv")
        bpf_wav = os.path.join(AUDIO_BPF_DIR, f"{scenario_name}_gear_bpf.wav")
        if not os.path.exists(can_csv) or not os.path.exists(bpf_wav): continue

        df_can = pd.read_csv(can_csv)
        df_can = df_can[df_can["Time"] >= t_launch_can].reset_index(drop=True)
        t_base = df_can["Time"].iloc[0]

        sr, audio_raw = wavfile.read(bpf_wav)
        if audio_raw.ndim > 1: audio_raw = audio_raw[:, 0]
        audio = audio_raw.astype(np.float32) / float(np.iinfo(audio_raw.dtype).max) if np.issubdtype(audio_raw.dtype, np.integer) else audio_raw.astype(np.float32)

        n_audio, n_can = len(audio), len(df_can)
        cut = int(n_can * 0.75)

        for start_idx in range(0, n_can - 34 + 1, 34):
            end_idx = start_idx + 33
            s_start, s_end = int(round((df_can["Time"].iloc[start_idx]-t_base)*sr)), int(round((df_can["Time"].iloc[end_idx]-t_base)*sr))
            if s_end > n_audio or s_start >= n_audio or (s_end - s_start) < NPERSEG: continue

            audio_slice = audio[s_start:s_end]
            feats = list(extract_audio_window_features(audio_slice, sr).values())
            stft_frames = compute_window_stft(audio_slice, sr)

            if start_idx < cut:
                X_feat_tr_list.append(feats)
                X_spec_tr_list.append(stft_frames)
                y_tr_list.append(label)
            else:
                X_feat_te_list.append(feats)
                X_spec_te_list.append(stft_frames)
                y_te_list.append(label)

    def pad_seq(seq_list, max_l):
        return np.array([s[:max_l, :] if s.shape[0] >= max_l else np.vstack([s, np.zeros((max_l - s.shape[0], s.shape[1]), dtype=np.float32)]) for s in seq_list], dtype=np.float32)

    max_len = max(s.shape[0] for s in X_spec_tr_list + X_spec_te_list)
    return (np.array(X_feat_tr_list, dtype=np.float32), np.array(X_feat_te_list, dtype=np.float32),
            pad_seq(X_spec_tr_list, max_len), pad_seq(X_spec_te_list, max_len),
            np.array(y_tr_list, dtype=int), np.array(y_te_list, dtype=int))

def benchmark_latency(infer_fn):
    for _ in range(10): infer_fn()
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        infer_fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    mean_lat = float(np.mean(latencies))
    return mean_lat, (1000.0 / mean_lat if mean_lat > 0 else 0.0)

# =========================================================
# 메인
# =========================================================
def main():
    print("=" * 100)
    print(" [RPi Multi-Stage Evaluation: Supervised 4 -> Hybrid 8]")
    print("=" * 100)

    X_f_tr, X_f_te, X_s_tr, X_s_te, y_tr, y_te = build_rpi_no_gap_hybrid_dataset()
    scaler_feat = StandardScaler().fit(X_f_tr)
    X_f_tr_s, X_f_te_s = scaler_feat.transform(X_f_tr), scaler_feat.transform(X_f_te)

    # 2D Spectrogram STFT Log 변환 및 StandardScaler 정규화 적용
    X_s_tr_log = np.log1p(X_s_tr)
    X_s_te_log = np.log1p(X_s_te)
    N_tr, T_f, F_dim = X_s_tr_log.shape
    N_te, _, _ = X_s_te_log.shape
    scaler_2d = StandardScaler()
    X_s_tr = scaler_2d.fit_transform(X_s_tr_log.reshape(N_tr, -1)).reshape(N_tr, T_f, F_dim)
    X_s_te = scaler_2d.transform(X_s_te_log.reshape(N_te, -1)).reshape(N_te, T_f, F_dim)

    c_w = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    w_t = torch.tensor(c_w, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=w_t)

    summary_out = {"supervised": {}, "hybrid": {}}
    
    # ---------------------------------------------------------
    # PART 1: Supervised Only 4종
    # ---------------------------------------------------------
    print("\n[PART 1] Supervised Only 4 Models Evaluation")
    print(f" {'Model Name':<35} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'AUC':>7} | {'Lat(ms)':>7} | {'FPS':>7}")
    print("-" * 100)

    def train_and_eval_pt(model, ds_tr, ds_te, infer_wrap, name, epochs=30):
        opt = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
        loader = DataLoader(ds_tr, batch_size=32, shuffle=True)
        model.train()
        for _ in tqdm(range(epochs), desc=f"Training {name}"):
            for bx, by in loader:
                opt.zero_grad()
                if isinstance(bx, list): loss = criterion(model(bx[0], bx[1]), by)
                else: loss = criterion(model(bx), by)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            if isinstance(ds_te.tensors[0], tuple): 
                logits = model(ds_te.tensors[0][0], ds_te.tensors[0][1])
            else:
                logits = model(ds_te.tensors[0])
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = torch.argmax(logits, dim=1).numpy()
        
        lat, fps = benchmark_latency(infer_wrap)
        acc, prec, rec, f1, auc = accuracy_score(y_te, preds), precision_score(y_te, preds, zero_division=0), recall_score(y_te, preds, zero_division=0), f1_score(y_te, preds, zero_division=0), roc_auc_score(y_te, probs)
        print(f" {name:<35} | {acc*100:>6.2f}% | {prec*100:>6.2f}% | {rec*100:>6.2f}% | {f1*100:>6.2f}% | {auc:>7.4f} | {lat:>7.2f} | {fps:>7.1f}")
        return {"accuracy": float(acc), "f1_score": float(f1), "auc": float(auc), "latency_ms": lat, "fps": fps}

    '''
    # 1. MLP
    model_mlp = AudioFeatureMLP(in_dim=24)
    ds_mlp = TensorDataset(torch.tensor(X_f_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    ds_mlp_te = TensorDataset(torch.tensor(X_f_te_s, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))
    def inf_mlp():
        with torch.no_grad(): model_mlp(torch.tensor(X_f_te_s[:1], dtype=torch.float32))
    summary_out["supervised"]["MLP"] = train_and_eval_pt(model_mlp, ds_mlp, ds_mlp_te, inf_mlp, "AudioFeatureMLP")

    # 2. XGBoost
    model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    model_xgb.fit(X_f_tr_s, y_tr)
    preds_x, probs_x = model_xgb.predict(X_f_te_s), model_xgb.predict_proba(X_f_te_s)[:, 1]
    def inf_xgb(): model_xgb.predict(X_f_te_s[:1])
    lat_x, fps_x = benchmark_latency(inf_xgb)
    acc_x, prec_x, rec_x, f1_x, auc_x = accuracy_score(y_te, preds_x), precision_score(y_te, preds_x, zero_division=0), recall_score(y_te, preds_x, zero_division=0), f1_score(y_te, preds_x, zero_division=0), roc_auc_score(y_te, probs_x)
    print(f" {'XGBoost':<35} | {acc_x*100:>6.2f}% | {prec_x*100:>6.2f}% | {rec_x*100:>6.2f}% | {f1_x*100:>6.2f}% | {auc_x:>7.4f} | {lat_x:>7.2f} | {fps_x:>7.1f}")
    summary_out["supervised"]["XGBoost"] = {"accuracy": float(acc_x), "f1_score": float(f1_x), "auc": float(auc_x), "latency_ms": lat_x, "fps": fps_x}
    '''

    # 3. CRNN
    model_crnn = AudioCRNN(in_freq=129)
    ds_crnn = TensorDataset(torch.tensor(X_s_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    ds_crnn_te = TensorDataset(torch.tensor(X_s_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))
    def inf_crnn():
        with torch.no_grad(): model_crnn(torch.tensor(X_s_te[:1], dtype=torch.float32))
    summary_out["supervised"]["CRNN"] = train_and_eval_pt(model_crnn, ds_crnn, ds_crnn_te, inf_crnn, "AudioCRNN")

    '''
    # 4. CNN2D
    model_cnn = Audio2DSpectrogramCNN(in_channels=1)
    ds_cnn = TensorDataset(torch.tensor(np.expand_dims(X_s_tr, 1), dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    ds_cnn_te = TensorDataset(torch.tensor(np.expand_dims(X_s_te, 1), dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))
    def inf_cnn():
        with torch.no_grad(): model_cnn(torch.tensor(np.expand_dims(X_s_te[:1], 1), dtype=torch.float32))
    summary_out["supervised"]["CNN2D"] = train_and_eval_pt(model_cnn, ds_cnn, ds_cnn_te, inf_cnn, "Audio2DSpectrogramCNN")
    '''
    
    # [SAVE] 스케일러 및 단독 CRNN 모델 저장
    os.makedirs(RPI_MODEL_DIR, exist_ok=True)
    joblib.dump(scaler_feat, os.path.join(RPI_MODEL_DIR, "rpi_scaler_feat.joblib"))
    joblib.dump(scaler_2d, os.path.join(RPI_MODEL_DIR, "rpi_scaler_stft.joblib"))
    torch.save(model_crnn.state_dict(), os.path.join(RPI_MODEL_DIR, "rpi_supervised_crnn.pt"))

    # ---------------------------------------------------------
    # PART 2: Step 1 (AE / OCSVM)
    # ---------------------------------------------------------
    print("\n[PART 2] Step 1 Anomaly Extraction...")
    X_f_tr_norm = X_f_tr_s[y_tr == 0]
    
    # AE
    model_ae = AnomalyAutoEncoder()
    opt_ae = optim.Adam(model_ae.parameters(), lr=0.002)
    loader_ae = DataLoader(TensorDataset(torch.tensor(X_f_tr_norm, dtype=torch.float32)), batch_size=32, shuffle=True)
    model_ae.train()
    for _ in tqdm(range(25), desc="Training AnomalyAutoEncoder"):
        for (bx,) in loader_ae:
            opt_ae.zero_grad()
            nn.MSELoss()(model_ae(bx), bx).backward()
            opt_ae.step()
    model_ae.eval()
    with torch.no_grad():
        t_tr, t_te = torch.tensor(X_f_tr_s, dtype=torch.float32), torch.tensor(X_f_te_s, dtype=torch.float32)
        ae_tr = torch.mean((model_ae(t_tr) - t_tr)**2, dim=1, keepdim=True).numpy()
        ae_te = torch.mean((model_ae(t_te) - t_te)**2, dim=1, keepdim=True).numpy()
    
    torch.save(model_ae.state_dict(), os.path.join(RPI_MODEL_DIR, "rpi_step1_ae.pt"))

    # OCSVM
    model_oc = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    model_oc.fit(X_f_tr_norm)
    oc_tr = (-model_oc.decision_function(X_f_tr_s)).reshape(-1, 1)
    oc_te = (-model_oc.decision_function(X_f_te_s)).reshape(-1, 1)
    
    joblib.dump(model_oc, os.path.join(RPI_MODEL_DIR, "rpi_step1_ocsvm.joblib"))
    # ---------------------------------------------------------
    # PART 3: Hybrid 8종
    # ---------------------------------------------------------
    print("\n[PART 3] Hybrid Pipeline 8 Models Evaluation")
    print(f" {'Hybrid Pipeline Name':<35} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'AUC':>7} | {'Lat(ms)':>7} | {'FPS':>7}")
    print("-" * 100)

    for step1_name, s1_tr, s1_te in [("AE", ae_tr, ae_te), ("OCSVM", oc_tr, oc_te)]:
        # 1D Hybrid features (25-dim)
        h_tr = StandardScaler().fit_transform(np.hstack([X_f_tr_s, s1_tr]))
        h_te = StandardScaler().fit_transform(np.hstack([X_f_te_s, s1_te]))
        
        s1_tr_sc, s1_te_sc = StandardScaler().fit_transform(s1_tr), StandardScaler().fit_transform(s1_te)

        '''
        # 1. Hybrid MLP
        model_hmlp = AudioHybridFeatureMLP(in_dim=25)
        ds_hmlp = TensorDataset(torch.tensor(h_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
        ds_hmlp_te = TensorDataset(torch.tensor(h_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))
        def inf_hmlp():
            with torch.no_grad(): model_hmlp(torch.tensor(h_te[:1], dtype=torch.float32))
        summary_out["hybrid"][f"{step1_name}+MLP"] = train_and_eval_pt(model_hmlp, ds_hmlp, ds_hmlp_te, inf_hmlp, f"[{step1_name}] + MLP")

        # 2. Hybrid XGBoost
        model_hx = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        model_hx.fit(h_tr, y_tr)
        preds_hx, probs_hx = model_hx.predict(h_te), model_hx.predict_proba(h_te)[:, 1]
        def inf_hx(): model_hx.predict(h_te[:1])
        lat_hx, fps_hx = benchmark_latency(inf_hx)
        acc_hx, prec_hx, rec_hx, f1_hx, auc_hx = accuracy_score(y_te, preds_hx), precision_score(y_te, preds_hx, zero_division=0), recall_score(y_te, preds_hx, zero_division=0), f1_score(y_te, preds_hx, zero_division=0), roc_auc_score(y_te, probs_hx)
        print(f" {f'[{step1_name}] + XGBoost':<35} | {acc_hx*100:>6.2f}% | {prec_hx*100:>6.2f}% | {rec_hx*100:>6.2f}% | {f1_hx*100:>6.2f}% | {auc_hx:>7.4f} | {lat_hx:>7.2f} | {fps_hx:>7.1f}")
        summary_out["hybrid"][f"{step1_name}+XGB"] = {"accuracy": float(acc_hx), "f1_score": float(f1_hx), "auc": float(auc_hx), "latency_ms": lat_hx, "fps": fps_hx}
        '''

        # 3. Hybrid CRNN
        model_hcrnn = AudioHybridCRNN(in_freq=129)
        ds_hcrnn = TensorDataset(torch.tensor(X_s_tr, dtype=torch.float32), torch.tensor(s1_tr_sc, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
        ds_hcrnn_te = TensorDataset(torch.tensor(X_s_te, dtype=torch.float32), torch.tensor(s1_te_sc, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))
        def train_and_eval_hybrid(model, ds, ds_te, name):
            opt = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
            loader = DataLoader(ds, batch_size=32, shuffle=True)
            model.train()
            for _ in tqdm(range(30), desc=f"Training {name}"):
                for b_spec, b_score, by in loader:
                    opt.zero_grad()
                    criterion(model(b_spec, b_score), by).backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                logits = model(ds_te.tensors[0], ds_te.tensors[1])
                probs = torch.softmax(logits, dim=1)[:, 1].numpy()
                preds = torch.argmax(logits, dim=1).numpy()
            
            def inf_h():
                with torch.no_grad(): model(ds_te.tensors[0][:1], ds_te.tensors[1][:1])
            lat, fps = benchmark_latency(inf_h)
            acc, prec, rec, f1, auc = accuracy_score(y_te, preds), precision_score(y_te, preds, zero_division=0), recall_score(y_te, preds, zero_division=0), f1_score(y_te, preds, zero_division=0), roc_auc_score(y_te, probs)
            print(f" {name:<35} | {acc*100:>6.2f}% | {prec*100:>6.2f}% | {rec*100:>6.2f}% | {f1*100:>6.2f}% | {auc:>7.4f} | {lat:>7.2f} | {fps:>7.1f}")
            return {"accuracy": float(acc), "f1_score": float(f1), "auc": float(auc), "latency_ms": lat, "fps": fps}
        
        summary_out["hybrid"][f"{step1_name}+CRNN"] = train_and_eval_hybrid(model_hcrnn, ds_hcrnn, ds_hcrnn_te, f"[{step1_name}] + CRNN")

        '''
        # 4. Hybrid CNN2D
        model_hcnn = AudioHybrid2DSpectrogramCNN(in_channels=1)
        ds_hcnn = TensorDataset(torch.tensor(np.expand_dims(X_s_tr, 1), dtype=torch.float32), torch.tensor(s1_tr_sc, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
        ds_hcnn_te = TensorDataset(torch.tensor(np.expand_dims(X_s_te, 1), dtype=torch.float32), torch.tensor(s1_te_sc, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))
        summary_out["hybrid"][f"{step1_name}+CNN2D"] = train_and_eval_hybrid(model_hcnn, ds_hcnn, ds_hcnn_te, f"[{step1_name}] + CNN2D")
        '''
        
        # [SAVE] Hybrid 스케일러 및 Hybrid CRNN 모델 저장
        joblib.dump(StandardScaler().fit(s1_tr), os.path.join(RPI_MODEL_DIR, f"rpi_scaler_hybrid_{step1_name.lower()}.joblib"))
        torch.save(model_hcrnn.state_dict(), os.path.join(RPI_MODEL_DIR, f"rpi_hybrid_crnn_{step1_name.lower()}.pt"))
    with open(os.path.join(RPI_RESULT_DIR, "rpi_no_gap_evaluation_summary.json"), "w") as f:
        json.dump(summary_out, f, indent=2)
    print("\n[Done] All evaluations completed.")

if __name__ == "__main__":
    main()
