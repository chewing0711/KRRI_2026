"""
evaluate_advanced_rnn_kona_ev.py (gearbox_final_suite / models_pipelines)

KONA EV 250-샘플(5.0초) 슬라이딩 3D Matched Matrix 텐서 데이터셋(1,036개 윈도우 x 250스텝 x 73피처)을 기반으로
5대 Advanced RNN 심층 순환신경망 모델 아키텍처를 학습 및 5-Fold 교차 검증 평가하는 복사본 스크립트.

5대 모델 아키텍처:
1. Standard Single GRU
2. Bidirectional GRU (BiGRU)
3. Temporal Self-Attention GRU (Attn-GRU)
4. 1D CNN + GRU Hybrid (CRNN)
5. LayerNorm Residual GRU (Res-GRU)

입력 데이터:
- /mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_rnn_sliding_window_dataset.pt

저장 결과:
- /mnt/c/Users/kante/Documents/KWU/etrl_task/experiments/gearbox_final_suite/results/advanced_rnn_kona_ev_benchmark_results.json
- /mnt/c/Users/kante/Documents/KWU/etrl_task/experiments/gearbox_final_suite/results/advanced_rnn_kona_ev_benchmark_results.csv

규정 준수:
- 원본 evaluate_advanced_rnn_architectures.py 100% 보존.
- 주석 및 터미널 출력 100% 한글 작성.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MODELS_DIR)
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
PT_DATASET_PATH = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_rnn_sliding_window_dataset.pt"


# =========================================================================
# 5대 Advanced RNN 모델 아키텍처 정의 (Input Dim = 73, Seq Len = 250)
# =========================================================================

# 1. Standard Single GRU
class SingleGRU(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=6):
        super(SingleGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# 2. Bidirectional GRU (BiGRU)
class BiGRU(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=6):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# 3. Temporal Self-Attention GRU (Attn-GRU)
class AttnGRU(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=6):
        super(AttnGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)  # (B, T, H)
        scores = self.attn(out)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * out, dim=1)  # (B, H)
        return self.fc(context)


# 4. 1D CNN + GRU Hybrid (CRNN)
class CRNN(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=6):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU()
        )
        self.gru = nn.GRU(48, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B, T, in_dim) -> (B, in_dim, T)
        x_perm = x.permute(0, 2, 1)
        feat = self.conv(x_perm).permute(0, 2, 1)  # (B, T, 48)
        out, _ = self.gru(feat)
        return self.fc(out[:, -1, :])


# 5. LayerNorm Residual GRU (Res-GRU)
class ResGRU(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=6):
        super(ResGRU, self).__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        proj = self.in_proj(x)
        out, _ = self.gru(proj)
        res = self.ln(proj + out)
        return self.fc(res[:, -1, :])


# =========================================================================
# 5-Fold 교차 검증 벤치마크 실행 함수
# =========================================================================

def train_eval_model(model_cls, X_train, y_train, X_test, y_test, in_dim, num_classes, epochs=20, lr=0.003, batch_size=32):
    model = model_cls(in_dim=in_dim, hidden=64, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32)
        logits = model(test_x)
        preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    prec_macro = precision_score(y_test, preds, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, preds, average="macro", zero_division=0)

    return acc, f1_macro, prec_macro, rec_macro


def run_kona_ev_advanced_rnn_benchmark():
    print("=" * 100)
    print(" 🚀 KONA EV 250-샘플 3D 텐서 기반 5대 ADVANCED RNN 아키텍처 벤치마크")
    print(f" 📂 데이터셋 경로: {PT_DATASET_PATH}")
    print("=" * 100)

    if not os.path.exists(PT_DATASET_PATH):
        print(f"[ERROR] KONA EV PyTorch 데이터셋이 없습니다: {PT_DATASET_PATH}")
        return

    pt_data = torch.load(PT_DATASET_PATH, weights_only=False)
    X_tensor = pt_data["X_common_tensor"].numpy()  # (1036, 250, 73)
    y_tensor = pt_data["y_common_tensor"].numpy()  # (1036,)

    n_samples, seq_len, in_dim = X_tensor.shape
    num_classes = len(np.unique(y_tensor))

    print(f" 📊 로드된 3D 텐서 형상 (Shape): {X_tensor.shape}")
    print(f"    - 총 윈도우 수: {n_samples:,} 개 (250 샘플 = 5.0초 윈도우)")
    print(f"    - 공통 피처 차원: {in_dim} 개")
    print(f"    - 주행 시나리오 클래스 수: {num_classes} 개")

    models_dict = {
        "1. Standard Single GRU": SingleGRU,
        "2. Bidirectional GRU (BiGRU)": BiGRU,
        "3. Temporal Self-Attention GRU (Attn-GRU)": AttnGRU,
        "4. 1D CNN + GRU Hybrid (CRNN)": CRNN,
        "5. LayerNorm Residual GRU (Res-GRU)": ResGRU
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = {}
    summary_rows = []

    for mname, mcls in models_dict.items():
        print(f"\n ⚙️ [{mname}] 5-Fold 교차 검증 평가 시작...")
        fold_accs, fold_f1s, fold_precs, fold_recs = [], [], [], []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X_tensor, y_tensor)):
            X_tr, y_tr = X_tensor[tr_idx], y_tensor[tr_idx]
            X_te, y_te = X_tensor[te_idx], y_tensor[te_idx]

            acc, f1_m, prec_m, rec_m = train_eval_model(
                mcls, X_tr, y_tr, X_te, y_te,
                in_dim=in_dim, num_classes=num_classes,
                epochs=15, lr=0.003, batch_size=32
            )

            fold_accs.append(acc)
            fold_f1s.append(f1_m)
            fold_precs.append(prec_m)
            fold_recs.append(rec_m)

        mean_acc = np.mean(fold_accs)
        mean_f1 = np.mean(fold_f1s)
        mean_prec = np.mean(fold_precs)
        mean_rec = np.mean(fold_recs)

        benchmark_results[mname] = {
            "mean_accuracy": round(float(mean_acc), 4),
            "mean_f1_score": round(float(mean_f1), 4),
            "mean_precision": round(float(mean_prec), 4),
            "mean_recall": round(float(mean_rec), 4)
        }

        summary_rows.append({
            "Model Architecture": mname,
            "Accuracy (%)": f"{mean_acc * 100:.2f}%",
            "F1-Score": round(float(mean_f1), 4),
            "Precision": round(float(mean_prec), 4),
            "Recall": round(float(mean_rec), 4)
        })

        print(f"    -> 평균 정확도: {mean_acc * 100:.2f}% | F1-Score: {mean_f1:.4f} | Recall: {mean_rec:.4f}")

    # 결과 표 출력
    df_res = pd.DataFrame(summary_rows)
    print("\n" + "=" * 100)
    print(" 🏆 KONA EV 5대 ADVANCED RNN 아키텍처 5-FOLD 교차 검증 최종 결과 표")
    print("=" * 100)
    print(df_res.to_string(index=False))
    print("=" * 100)

    # 결과 파일 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "advanced_rnn_kona_ev_benchmark_results.json")
    csv_path = os.path.join(RESULTS_DIR, "advanced_rnn_kona_ev_benchmark_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4, ensure_ascii=False)

    df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n 💾 [결과 JSON 저장] -> {json_path}")
    print(f" 💾 [결과 CSV 저장]  -> {csv_path}")


if __name__ == "__main__":
    run_kona_ev_advanced_rnn_benchmark()
