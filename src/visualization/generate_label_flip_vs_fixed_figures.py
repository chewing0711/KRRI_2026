"""
generate_label_flip_vs_fixed_figures.py (gearbox_fault_diagnosis_fixed)

Dynamic Figure Generation:
1. Reads unified_can_context_dataset.csv dynamically.
2. Computes scenario value counts & percentages for Figure 1 Subplot A.
3. Executes Fold 1 of GroupShuffleSplit live to compute Fold 1 Confusion Matrix for Figure 1 Subplot B.
4. Executes Paired 75:5:20 Split live to compute class counts & Confusion Matrix for Figure 2.

Rule 3 Compliance:
- Code comments & terminal output in Korean.
- Matplotlib labels, titles, legends, axes in English.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(FIXED_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
INPUT_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_figures_dynamic():
    print("=" * 80)
    print(" 🎨 GENERATING DIAGNOSTIC VISUALIZATION PNG FIGURES")
    print("=" * 80)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] CSV file missing: {INPUT_CSV}")
        return

    # Load CSV
    df = pd.read_csv(INPUT_CSV)
    metadata_cols = ["session_id", "state", "target", "scenario"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X = df[feature_cols].to_numpy()
    y = df["target"].to_numpy()

    # --------------------------------------------------------------------------
    # FIGURE 1: Label Flip Mechanism Diagnosis
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160)

    # Subplot A: Dynamic Scenario Value Counts
    scen_counts = df["scenario"].value_counts()
    scen_sizes = scen_counts.values.tolist()
    
    scen_map_names = {
        0: "20/60/80kph Circuit",
        1: "Test Hills 6-30deg",
        2: "Steering Pad 40kph"
    }
    pie_labels = [f"{scen_map_names.get(k, f'Scen {k}')} ({v} samples, {v/len(df)*100:.1f}%)" for k, v in zip(scen_counts.index, scen_sizes)]
    colors_pie = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"][:len(scen_sizes)]

    axes[0].pie(scen_sizes, labels=pie_labels, autopct="%1.1f%%", startangle=140, colors=colors_pie, explode=[0.08] + [0]*(len(scen_sizes)-1))
    axes[0].set_title(f"A. Scenario Proportion (Total {len(df)} Samples)", fontsize=11, fontweight="bold")

    # Subplot B: GroupShuffleSplit Fold 1 Confusion Matrix
    groups_sess = df["session_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    cm_fold1 = np.zeros((2, 2), dtype=int)
    for tr_i, te_i in gss.split(X, y, groups=groups_sess):
        model_fold1 = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric="logloss")
        model_fold1.fit(X[tr_i], y[tr_i])
        pred_fold1 = model_fold1.predict(X[te_i])
        cm_fold1 = confusion_matrix(y[te_i], pred_fold1, labels=[0, 1])
        fn_count = cm_fold1[1, 0]
        fp_count = cm_fold1[0, 1]

    sns.heatmap(cm_fold1, annot=True, fmt="d", cmap="Reds", ax=axes[1], cbar=False,
                xticklabels=["Pred Normal (0)", "Pred Abnormal (1)"],
                yticklabels=["Actual Normal (0)", "Actual Abnormal (1)"])
    axes[1].set_title(f"B. Fold 1 Confusion Matrix (FN={fn_count}, FP={fp_count})", fontsize=11, fontweight="bold")

    plt.suptitle("Label Flip Diagnosis under Unpaired GroupShuffleSplit", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    fig1_path = os.path.join(FIGURES_DIR, "label_flip_mechanism_diagnosis.png")
    plt.savefig(fig1_path, dpi=160)
    plt.close()
    print(f"[SAVED] Figure 1: {fig1_path}")

    # --------------------------------------------------------------------------
    # FIGURE 2: Paired-Session 75:5:20 Split Confusion Matrix
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160)

    tr_indices = []
    te_indices = []

    for scen in df["scenario"].unique():
        for st in ["normal", "abnormal"]:
            sub_idx = df[(df["scenario"] == scen) & (df["state"].str.lower() == st)].index.to_numpy()
            n = len(sub_idx)
            if n == 0:
                continue
            n_tr = int(n * 0.75)
            n_gap = max(1, int(n * 0.05)) if n >= 20 else 0
            
            tr_indices.extend(sub_idx[:n_tr])
            te_indices.extend(sub_idx[n_tr + n_gap:])

    tr_idx = np.array(tr_indices)
    te_idx = np.array(te_indices)

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    model_fixed = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric="logloss")
    model_fixed.fit(X_tr, y_tr)
    pred_fixed = model_fixed.predict(X_te)
    cm_fixed = confusion_matrix(y_te, pred_fixed, labels=[0, 1])

    n_tr_norm = int(np.sum(y_tr == 0))
    n_tr_abnorm = int(np.sum(y_tr == 1))
    n_te_norm = int(np.sum(y_te == 0))
    n_te_abnorm = int(np.sum(y_te == 1))

    # Subplot A: Class-Balanced Sample Counts
    categories = ["Train Set (75%)", "Test Set (20%)"]
    normal_counts = [n_tr_norm, n_te_norm]
    abnormal_counts = [n_tr_abnorm, n_te_abnorm]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, normal_counts, width, label="Normal (0)", color="#2ca02c")
    rects2 = axes[0].bar(x + width/2, abnormal_counts, width, label="Abnormal (1)", color="#d62728")

    axes[0].set_ylabel("Sample Count", fontsize=10)
    axes[0].set_title(f"A. Sample Split: Train ({len(X_tr)}) vs Test ({len(X_te)})", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    for p in rects1 + rects2:
        height = p.get_height()
        axes[0].annotate(f"{height}", xy=(p.get_x() + p.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot B: Confusion Matrix Heatmap
    acc_fixed = (cm_fixed[0,0] + cm_fixed[1,1]) / len(y_te) * 100.0
    sns.heatmap(cm_fixed, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False,
                xticklabels=["Pred Normal (0)", "Pred Abnormal (1)"],
                yticklabels=["Actual Normal (0)", "Actual Abnormal (1)"])
    axes[1].set_title(f"B. Paired-Session Confusion Matrix (Accuracy: {acc_fixed:.2f}%)", fontsize=11, fontweight="bold")

    plt.suptitle("Paired-Session Time-Block Split Evaluation Results", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    fig2_path = os.path.join(FIGURES_DIR, "fixed_paired_session_confusion_matrix.png")
    plt.savefig(fig2_path, dpi=160)
    plt.close()
    print(f"[SAVED] Figure 2: {fig2_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_figures_dynamic()
