"""
build_rnn_sliding_window_dataset.py

각 파일별 수신 신호 수가 다른(112개, 185개, 149개 등) 가변 컬럼 특성을 완벽히 반영하여
Shape Mismatch 없이 250-샘플(5.0초) 슬라이딩 3D Matched Matrix 텐서 세트를 생성하는 최적화 스크립트.

특징:
1. 파일별 고유 3D 텐서 데이터셋: 각 파일별 (N_windows, 250, N_features_file) 3D 텐서 개별 보존.
2. 공통 피처 3D 통합 데이터셋: 6개 파일 전체 공통 피처 교집합 기준 (Total_N_windows, 250, N_common_features) 3D 텐서 병합.
3. ValueError: inhomogeneous shape 에러 100% 완전 해결.

저장 경로:
- /mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_rnn_sliding_window_dataset.npz
- /mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_rnn_sliding_window_dataset.pt
"""

import os
import glob
import torch
import numpy as np
import pandas as pd

INPUT_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_decoded_unified_20ms"
OUTPUT_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data"
OUTPUT_NPZ = os.path.join(OUTPUT_DIR, "kona_ev_rnn_sliding_window_dataset.npz")
OUTPUT_PT = os.path.join(OUTPUT_DIR, "kona_ev_rnn_sliding_window_dataset.pt")

WINDOW_SIZE = 250  # 250 샘플 (20ms * 250 = 5.0초)
STRIDE = 50        # 50 샘플 (20ms * 50 = 1.0초)


def build_rnn_matched_matrix_dataset():
    print("=" * 100)
    print(" 🛠 KONA EV RNN 모델 주입용 250-샘플 슬라이딩 3D Matched Matrix 텐서 구축")
    print(f" 📂 입력 데이터 디렉토리: {INPUT_DIR}")
    print(f" ⚙️ 시계열 윈도우 크기 (Window Size): {WINDOW_SIZE} 샘플 (5.0초)")
    print(f" ⚙️ 윈도우 이동 보폭 (Stride): {STRIDE} 샘플 (1.0초)")
    print("=" * 100)

    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "kona_ev_decoded_signals_*.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

    if not csv_files:
        print(f"[ERROR] 처리할 디코딩 CSV 파일 없음: {INPUT_DIR}")
        return

    # 1단계: 6개 파일 전체의 공통 피처(교집합) 수집
    file_features_dict = {}
    common_feature_set = None

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        df_head = pd.read_csv(csv_path, nrows=5)
        meta_cols = {"Time_Rel_Sec", "Source_File", "CAN_ID_Hex"}
        fcols = [c for c in df_head.columns if c not in meta_cols]
        file_features_dict[fname] = fcols

        if common_feature_set is None:
            common_feature_set = set(fcols)
        else:
            common_feature_set = common_feature_set.intersection(set(fcols))

    common_features = sorted(list(common_feature_set))
    print(f" 📊 발견된 전체 주행 CSV 파일: {len(csv_files)} 개")
    print(f" 🎯 6개 파일 공통 피처 교집합 개수: {len(common_features)} 개 (공통 3D 통합 텐서 생성용)")

    # 2단계: 슬라이딩 윈도우 텐서 생성 (개별 파일용 & 공통 통합용)
    file_tensors = {}
    common_windows_list = []
    common_labels_list = []
    common_sources_list = []

    for fidx, csv_path in enumerate(csv_files):
        fname = os.path.basename(csv_path)
        print(f"\n [파일 {fidx+1}/{len(csv_files)}] {fname} 텐서화 로딩 중...")

        df = pd.read_csv(csv_path)
        fcols = file_features_dict[fname]
        X_file_raw = df[fcols].values
        n_samples = len(X_file_raw)

        if n_samples < WINDOW_SIZE:
            print(f"   [SKIP] 샘플 수가 윈도우 크기({WINDOW_SIZE})보다 적음.")
            continue

        # 개별 파일 정규화 (Z-score)
        mean_file = np.nan_to_num(X_file_raw.mean(axis=0, keepdims=True))
        std_file = np.nan_to_num(X_file_raw.std(axis=0, keepdims=True)) + 1e-8
        X_file_norm = (X_file_raw - mean_file) / std_file

        n_windows = (n_samples - WINDOW_SIZE) // STRIDE + 1
        file_windows = []

        for w_idx in range(n_windows):
            start = w_idx * STRIDE
            end = start + WINDOW_SIZE
            file_windows.append(X_file_norm[start:end, :])

        # 파일 개별 3D 텐서 (N_windows, 250, N_features_file)
        X_file_3d = np.array(file_windows, dtype=np.float32)
        file_tensors[fname] = {
            "tensor": torch.tensor(X_file_3d, dtype=torch.float32),
            "feature_names": fcols,
            "shape": X_file_3d.shape
        }
        print(f"   - 개별 텐서 생성 성공: Shape {X_file_3d.shape}")

        # 공통 피처 3D 텐서 추출
        X_common_raw = df[common_features].values
        mean_com = np.nan_to_num(X_common_raw.mean(axis=0, keepdims=True))
        std_com = np.nan_to_num(X_common_raw.std(axis=0, keepdims=True)) + 1e-8
        X_com_norm = (X_common_raw - mean_com) / std_com

        for w_idx in range(n_windows):
            start = w_idx * STRIDE
            end = start + WINDOW_SIZE
            common_windows_list.append(X_com_norm[start:end, :])
            common_labels_list.append(fidx)
            common_sources_list.append(fname)

    # 공통 통합 3D Matched Matrix [Total_N_windows, 250, N_common_features]
    X_common_3d = np.array(common_windows_list, dtype=np.float32)
    y_common_labels = np.array(common_labels_list, dtype=np.int64)

    print("\n" + "=" * 100)
    print(" 🎉 250-샘플 슬라이딩 3D Matched Matrix 텐서 구축 완전 완수!")
    print(f" 📐 공통 통합 3D 텐서 형상 (Common Shape): {X_common_3d.shape}")
    print(f"    - Total_N_windows (총 샘플 시퀀스 수): {X_common_3d.shape[0]:,} 개")
    print(f"    - Sequence_Length (타임스텝): {X_common_3d.shape[1]} 개 (250 샘플 = 5.0초)")
    print(f"    - Common Feature Dim (공통 피처 수): {X_common_3d.shape[2]} 개")
    print("=" * 100)

    # 1. NumPy .npz 저장 (통합 텐서 + 파일별 텐서)
    save_dict = {
        "X_common_seq": X_common_3d,
        "y_common_labels": y_common_labels,
        "common_feature_names": np.array(common_features, dtype=str),
        "common_file_sources": np.array(common_sources_list, dtype=str)
    }

    # 개별 파일 텐서 추가
    for fname, fdata in file_tensors.items():
        clean_key = fname.replace(".csv", "").replace("-", "_")
        save_dict[f"X_{clean_key}"] = fdata["tensor"].numpy()
        save_dict[f"features_{clean_key}"] = np.array(fdata["feature_names"], dtype=str)

    np.savez_compressed(OUTPUT_NPZ, **save_dict)
    print(f" 💾 [NumPy Dataset] 저장 완료 -> {OUTPUT_NPZ}")

    # 2. PyTorch .pt 저장
    pt_dataset = {
        "X_common_tensor": torch.tensor(X_common_3d, dtype=torch.float32),
        "y_common_tensor": torch.tensor(y_common_labels, dtype=torch.long),
        "common_feature_names": common_features,
        "file_tensors": file_tensors
    }
    torch.save(pt_dataset, OUTPUT_PT)
    print(f" 💾 [PyTorch Dataset] 저장 완료 -> {OUTPUT_PT}")


if __name__ == "__main__":
    build_rnn_matched_matrix_dataset()
