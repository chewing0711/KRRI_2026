"""
run_master_integrated_test.py (gearbox_final_suite)

Integrated Master Test Script incorporating all improvements:
- Pure CAN 30-dim features + Speed/Grade Context + Speed-Normalized Physical Slip Ratios
- Scenario Z-Score Normalization (Domain Shift Mitigation)
- Paired-Session 75:5:20 Class-Balanced Time-Block Split

Evaluates 5 Split Modes:
  Mode 1: Raw Random Split (80:20)
  Mode 2: Purged Time-Block Split (75:5:20)
  Mode 3: Extreme Purged Split (40:20:40)
  Mode 4: Leave-One-Scenario-Out (LOSO OOD with Z-Score Normalization)
  Mode 5: Paired-Session Time-Block Split (Primary Class-Balanced Test)
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")

if not os.path.exists(INPUT_CSV):
    INPUT_CSV = os.path.join(os.path.dirname(SUITE_DIR), "gearbox_fault_diagnosis_fixed", "results", "unified_can_context_dataset.csv")


def fit_and_eval_models(X_tr, X_te, y_tr, y_te):
    results = {}
    models = {
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric="logloss"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)
        f1 = f1_score(y_te, pred, zero_division=0)
        cm = confusion_matrix(y_te, pred, labels=[0, 1])

        results[name] = {
            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "cm": f"TN:{cm[0,0]} FP:{cm[0,1]} FN:{cm[1,0]} TP:{cm[1,1]}"
        }
    return results


def print_table(title, results):
    print(f"\n [{title}]")
    print("-" * 95)
    print(f" {'Model Name':<20} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} | {'Confusion Matrix'}")
    print("-" * 95)
    for name, m in results.items():
        print(f" {name:<20} | {m['acc']*100:8.2f}%  | {m['prec']*100:8.2f}%  | {m['rec']*100:8.2f}%  | {m['f1']*100:8.2f}%  | {m['cm']}")
    print("=" * 95)


def run_master_test():
    print("=" * 95)
    print(" 🚀 RUNNING MASTER INTEGRATED TEST SUITE (gearbox_final_suite)")
    print("=" * 95)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] CSV missing: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    metadata_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X = df[feature_cols].to_numpy()
    y = df["target"].to_numpy()

    # Feature matrix excluding raw speed/grade for Z-Score normalization
    pure_cols = [c for c in feature_cols if c not in ["speed_kph", "grade_pct"]]
    X_pure = df[pure_cols].to_numpy()

    # Per-Scenario Z-Score Normalization for Mode 4
    X_norm = np.zeros_like(X_pure)
    for scen in df["scenario"].unique():
        mask = (df["scenario"] == scen)
        scaler = StandardScaler()
        X_norm[mask] = scaler.fit_transform(X_pure[mask])

    all_results = {}

    # Mode 1: Raw Random Split (80:20)
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    res1 = fit_and_eval_models(X_tr1, X_te1, y_tr1, y_te1)
    print_table("MODE 1: RAW RANDOM SPLIT (80:20 Benchmark)", res1)
    all_results["Mode 1 (Random Split)"] = res1

    # Mode 2: Purged Time-Block Split (75:5:20)
    tr_idx2, te_idx2 = [], []
    for scen in df["scenario"].unique():
        s_idx = df[df["scenario"] == scen].index.to_numpy()
        n = len(s_idx)
        n_tr = int(n * 0.75)
        n_gap = max(1, int(n * 0.05))
        tr_idx2.extend(s_idx[:n_tr])
        te_idx2.extend(s_idx[n_tr + n_gap:])
    res2 = fit_and_eval_models(X[np.array(tr_idx2)], X[np.array(te_idx2)], y[np.array(tr_idx2)], y[np.array(te_idx2)])
    print_table("MODE 2: PURGED TIME-BLOCK SPLIT (75:5:20)", res2)
    all_results["Mode 2 (Purged 75:5:20)"] = res2

    # Mode 3: Extreme Purged Split (40:20:40)
    tr_idx3, te_idx3 = [], []
    for scen in df["scenario"].unique():
        s_idx = df[df["scenario"] == scen].index.to_numpy()
        n = len(s_idx)
        n_tr = int(n * 0.40)
        n_gap = max(1, int(n * 0.20))
        tr_idx3.extend(s_idx[:n_tr])
        te_idx3.extend(s_idx[n_tr + n_gap:])
    res3 = fit_and_eval_models(X[np.array(tr_idx3)], X[np.array(te_idx3)], y[np.array(tr_idx3)], y[np.array(te_idx3)])
    print_table("MODE 3: EXTREME PURGED SPLIT (40:20:40 Data Reduction)", res3)
    all_results["Mode 3 (Extreme Purged 40:20:40)"] = res3

    # Mode 4: Leave-One-Scenario-Out (LOSO OOD with Z-Score Normalization)
    groups_scen = df["scenario"].to_numpy()
    logo = LeaveOneGroupOut()
    accum_res4 = {m: {"acc": [], "prec": [], "rec": [], "f1": []} for m in ["XGBoost", "Decision Tree", "Random Forest"]}

    for tr_i, te_i in logo.split(X_norm, y, groups=groups_scen):
        r = fit_and_eval_models(X_norm[tr_i], X_norm[te_i], y[tr_i], y[te_i])
        for name in accum_res4:
            for k in ["acc", "prec", "rec", "f1"]:
                accum_res4[name][k].append(r[name][k])

    avg_res4 = {}
    for name in accum_res4:
        avg_res4[name] = {
            "acc": float(np.mean(accum_res4[name]["acc"])),
            "prec": float(np.mean(accum_res4[name]["prec"])),
            "rec": float(np.mean(accum_res4[name]["rec"])),
            "f1": float(np.mean(accum_res4[name]["f1"])),
            "cm": "OOD Cross-Val Average"
        }
    print_table("MODE 4: LEAVE-ONE-SCENARIO-OUT (LOSO OOD with Scenario Z-Score Norm)", avg_res4)
    all_results["Mode 4 (LOSO OOD Normalized)"] = avg_res4

    # Mode 5: Paired-Session Time-Block Split (Primary Test)
    tr_indices5, te_indices5 = [], []
    for scen in df["scenario"].unique():
        for st in ["normal", "abnormal"]:
            sub_idx = df[(df["scenario"] == scen) & (df["state"].str.lower() == st)].index.to_numpy()
            n = len(sub_idx)
            if n == 0:
                continue
            n_tr = int(n * 0.75)
            n_gap = max(1, int(n * 0.05)) if n >= 20 else 0
            tr_indices5.extend(sub_idx[:n_tr])
            te_indices5.extend(sub_idx[n_tr + n_gap:])
    res5 = fit_and_eval_models(X[np.array(tr_indices5)], X[np.array(te_indices5)], y[np.array(tr_indices5)], y[np.array(te_indices5)])
    print_table("MODE 5: PAIRED-SESSION TIME-BLOCK SPLIT (Primary Class-Balanced Test)", res5)
    all_results["Mode 5 (Paired-Session Primary)"] = res5

    # Export JSON
    out_json = os.path.join(RESULTS_DIR, "master_integrated_test_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[SUCCESS] Master Integrated Test Complete! Results exported to: {out_json}")


if __name__ == "__main__":
    run_master_test()
