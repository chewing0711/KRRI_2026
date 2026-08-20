"""
evaluate_unified_can_context.py

Evaluates All Baseline & Stress Test Models on the UNIFIED CAN + CONTEXT CSV (32 Dims: Pure CAN 30 + speed_kph + grade_pct).
Runs 4 Evaluation Split Modes:
  Mode 1: Raw Random Split (80:20)
  Mode 2: Moderate Purged Split (75:5:20)
  Mode 3: Harsh Purged Split (40:20:40)
  Mode 4: Leave-One-Scenario-Out (LOSO OOD Stress Test)
"""

import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "results", "unified_can_context_dataset.csv")
RESULT_DIR = os.path.join(SCRIPT_DIR, "results", "metrics")
os.makedirs(RESULT_DIR, exist_ok=True)


class GearboxDeepMLP(nn.Module):
    def __init__(self, input_dim):
        super(GearboxDeepMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


class OmniAnomalyVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super(OmniAnomalyVAE, self).__init__()
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_recon = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        out, _ = self.encoder_gru(x)
        h_last = out[:, -1, :]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        z = self.reparameterize(mu, logvar)

        seq_len = x.size(1)
        dec_in = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder_gru(dec_in)
        recon_x = self.fc_recon(dec_out)
        return recon_x, mu, logvar


def fit_and_evaluate_all_models(X_train, X_test, y_train, y_test):
    input_dim = X_train.shape[1]
    results = {}

    # 1. XGBoost
    model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric="logloss")
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    results["XGBoost"] = {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, zero_division=0),
        "rec": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    # 2. Decision Tree
    model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_dt.fit(X_train, y_train)
    y_pred = model_dt.predict(X_test)
    results["Decision Tree"] = {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, zero_division=0),
        "rec": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    # 3. Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    results["Random Forest"] = {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, zero_division=0),
        "rec": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    # 4. PyTorch Deep MLP
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    train_ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    mlp = GearboxDeepMLP(input_dim)
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    mlp.train()
    for _ in range(40):
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = mlp(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    mlp.eval()
    with torch.no_grad():
        test_logits = mlp(torch.tensor(X_te_s, dtype=torch.float32))
        y_pred = torch.argmax(test_logits, dim=1).numpy()

    results["PyTorch Deep MLP"] = {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, zero_division=0),
        "rec": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    # 5. One-Class SVM
    X_tr_norm = X_tr_s[y_train == 0]
    if len(X_tr_norm) > 0:
        oc_svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        oc_svm.fit(X_tr_norm)
        y_pred_oc = oc_svm.predict(X_te_s)
        y_pred_binary = np.where(y_pred_oc == -1, 1, 0)
        results["One-Class SVM"] = {
            "acc": accuracy_score(y_test, y_pred_binary),
            "prec": precision_score(y_test, y_pred_binary, zero_division=0),
            "rec": recall_score(y_test, y_pred_binary, zero_division=0),
            "f1": f1_score(y_test, y_pred_binary, zero_division=0)
        }

    # 6. Isolation Forest
    if len(X_tr_norm) > 0:
        iso_forest = IsolationForest(n_estimators=100, random_state=42)
        iso_forest.fit(X_tr_norm)
        y_pred_iso = iso_forest.predict(X_te_s)
        y_pred_binary_iso = np.where(y_pred_iso == -1, 1, 0)
        results["Isolation Forest"] = {
            "acc": accuracy_score(y_test, y_pred_binary_iso),
            "prec": precision_score(y_test, y_pred_binary_iso, zero_division=0),
            "rec": recall_score(y_test, y_pred_binary_iso, zero_division=0),
            "f1": f1_score(y_test, y_pred_binary_iso, zero_division=0)
        }

    # 7. OmniAnomaly (PyTorch GRU-VAE)
    if len(X_tr_norm) > 0:
        X_tr_norm_seq = torch.tensor(X_tr_norm, dtype=torch.float32).unsqueeze(1)
        X_te_seq = torch.tensor(X_te_s, dtype=torch.float32).unsqueeze(1)

        vae = OmniAnomalyVAE(input_dim)
        optimizer_vae = optim.Adam(vae.parameters(), lr=0.001)

        vae.train()
        for _ in range(30):
            for i in range(0, len(X_tr_norm_seq), 32):
                bx = X_tr_norm_seq[i:i+32]
                if len(bx) == 0:
                    continue
                optimizer_vae.zero_grad()
                recon_x, mu, logvar = vae(bx)
                recon_loss = nn.MSELoss()(recon_x, bx)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.01 * kl_loss
                loss.backward()
                optimizer_vae.step()

        vae.eval()
        with torch.no_grad():
            recon_x, _, _ = vae(X_te_seq)
            mse = torch.mean((recon_x - X_te_seq) ** 2, dim=(1, 2)).numpy()
            threshold = np.percentile(mse, 80)
            y_pred_omni = np.where(mse > threshold, 1, 0)

        results["OmniAnomaly (PyTorch)"] = {
            "acc": accuracy_score(y_test, y_pred_omni),
            "prec": precision_score(y_test, y_pred_omni, zero_division=0),
            "rec": recall_score(y_test, y_pred_omni, zero_division=0),
            "f1": f1_score(y_test, y_pred_omni, zero_division=0)
        }

    return results


def print_summary_table(title, results):
    print(f"\n [{title}]")
    print("-" * 90)
    print(f" {'Model Name':<26} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} ")
    print("-" * 90)
    for model_name, m in results.items():
        print(f" {model_name:<26} | {m['acc']*100:8.2f}%  | {m['prec']*100:8.2f}%  | {m['rec']*100:8.2f}%  | {m['f1']*100:8.2f}% ")
    print("=" * 90)


def run_unified_evaluations():
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Unified CSV dataset not found: {INPUT_CSV}")
        print(" Please run build_unified_can_context_csv.py first!")
        return

    df = pd.read_csv(INPUT_CSV)
    # EXCLUDE shortcut environment columns (speed_kph, grade_pct) to prevent tree shortcut learning
    shortcut_cols = ["session_id", "state", "target", "scenario", "speed_kph", "grade_pct"]
    feature_cols = [c for c in df.columns if c not in shortcut_cols]

    X = df[feature_cols].to_numpy()
    y = df["target"].to_numpy()

    print("\n" + "=" * 90)
    print(f" 🚀 EVALUATING PURE PHYSICAL SLIP RATIO DATASET (Excluding Shortcut Columns: speed_kph, grade_pct)")
    print(f" Dataset Scale: {len(df)} rows x {len(feature_cols)} pure physical feature columns")
    print(f" Feature Columns: {feature_cols[:8]} ...")
    print("=" * 90)

    # Mode 1: Raw Random Split (80:20)
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    res1 = fit_and_evaluate_all_models(X_tr1, X_te1, y_tr1, y_te1)
    print_summary_table("MODE 1: RAW RANDOM SPLIT (80:20, 32 Dims)", res1)

    # Mode 2: Moderate Purged Split (75:5:20)
    tr_idx2, te_idx2 = [], []
    for scen in df["scenario"].unique():
        s_idx = df[df["scenario"] == scen].index.to_numpy()
        n = len(s_idx)
        n_tr = int(n * 0.75)
        n_gap = max(1, int(n * 0.05))
        tr_idx2.extend(s_idx[:n_tr])
        te_idx2.extend(s_idx[n_tr + n_gap:])
    res2 = fit_and_evaluate_all_models(X[np.array(tr_idx2)], X[np.array(te_idx2)], y[np.array(tr_idx2)], y[np.array(te_idx2)])
    print_summary_table("MODE 2: MODERATE PURGED SPLIT (75:5:20, 32 Dims)", res2)

    # Mode 3: Harsh Purged Split (40:20:40)
    tr_idx3, te_idx3 = [], []
    for scen in df["scenario"].unique():
        s_idx = df[df["scenario"] == scen].index.to_numpy()
        n = len(s_idx)
        n_tr = int(n * 0.40)
        n_gap = max(1, int(n * 0.20))
        tr_idx3.extend(s_idx[:n_tr])
        te_idx3.extend(s_idx[n_tr + n_gap:])
    res3 = fit_and_evaluate_all_models(X[np.array(tr_idx3)], X[np.array(te_idx3)], y[np.array(tr_idx3)], y[np.array(te_idx3)])
    print_summary_table("MODE 3: HARSH PURGED SPLIT (40:20:40, 32 Dims)", res3)

    # Mode 4: Leave-One-Scenario-Out (LOSO OOD)
    groups = df["scenario"].to_numpy()
    logo = LeaveOneGroupOut()
    model_names = ["XGBoost", "Decision Tree", "Random Forest", "PyTorch Deep MLP", "One-Class SVM", "Isolation Forest", "OmniAnomaly (PyTorch)"]
    accum_res = {m: {"acc": [], "prec": [], "rec": [], "f1": []} for m in model_names}

    for tr_i, te_i in logo.split(X, y, groups=groups):
        res = fit_and_evaluate_all_models(X[tr_i], X[te_i], y[tr_i], y[te_i])
        for m_name, metrics in res.items():
            for k in ["acc", "prec", "rec", "f1"]:
                accum_res[m_name][k].append(metrics[k])

    avg_res4 = {}
    for m_name, metrics in accum_res.items():
        avg_res4[m_name] = {
            "acc": float(np.mean(metrics["acc"])),
            "prec": float(np.mean(metrics["prec"])),
            "rec": float(np.mean(metrics["rec"])),
            "f1": float(np.mean(metrics["f1"]))
        }
    print_summary_table("MODE 4: LEAVE-ONE-SCENARIO-OUT (LOSO) OOD STRESS TEST (32 Dims)", avg_res4)

    # Mode 5: Session-Isolated GroupShuffleSplit (Balanced Unseen Session Log Test)
    from sklearn.model_selection import GroupShuffleSplit
    groups_sess = df["session_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    accum_res5 = {m: {"acc": [], "prec": [], "rec": [], "f1": []} for m in model_names}

    for tr_i, te_i in gss.split(X, y, groups=groups_sess):
        if len(np.unique(y[tr_i])) < 2 or len(np.unique(y[te_i])) < 2:
            continue
        res = fit_and_evaluate_all_models(X[tr_i], X[te_i], y[tr_i], y[te_i])
        for m_name, metrics in res.items():
            for k in ["acc", "prec", "rec", "f1"]:
                accum_res5[m_name][k].append(metrics[k])

    avg_res5 = {}
    for m_name, metrics in accum_res5.items():
        if len(metrics["acc"]) > 0:
            avg_res5[m_name] = {
                "acc": float(np.mean(metrics["acc"])),
                "prec": float(np.mean(metrics["prec"])),
                "rec": float(np.mean(metrics["rec"])),
                "f1": float(np.mean(metrics["f1"]))
            }
    print_summary_table("MODE 5: SESSION-ISOLATED GROUP SHUFFLE SPLIT (Unseen Session Log Test)", avg_res5)

    # Save results
    all_res = {
        "Mode 1 (Raw Random Split 32D)": res1,
        "Mode 2 (Moderate Purged Split 32D)": res2,
        "Mode 3 (Harsh Purged Split 32D)": res3,
        "Mode 4 (LOSO OOD Stress Test 32D)": avg_res4,
    }
    out_json = os.path.join(RESULT_DIR, "unified_can_context_32d_results.json")
    with open(out_json, "w") as f:
        json.dump(all_res, f, indent=4)
    print(f"\n[SUCCESS] Unified CAN + Context evaluation complete! Results saved to: {out_json}\n")


if __name__ == "__main__":
    run_unified_evaluations()
