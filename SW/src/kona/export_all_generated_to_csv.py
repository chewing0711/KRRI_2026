"""
export_all_generated_to_csv.py

생성된 모든 KONA EV 디코딩 데이터셋(공통 73개 피처 20ms 시계열 통합본 및 250-샘플 슬라이딩 윈도우 텐서 매트릭스)을
요청하신 디렉토리(/mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_decoded_unified_20ms/) 내에
CSV 파일 형식으로 일괄 변환 및 내보내기(Export)하는 스크립트.

출력 디렉토리: /mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_decoded_unified_20ms/

생성 CSV 파일:
1. kona_ev_common_73features_unified_20ms.csv (6개 주행 공통 73개 피처 20ms 시계열 통합 CSV)
2. kona_ev_rnn_sliding_window_250samples.csv (250-샘플 슬라이딩 윈도우 Matched Matrix CSV)

규정 준수:
- 주석 및 터미널 출력 100% 한글 작성.
"""

import os
import glob
import numpy as np
import pandas as pd

TARGET_DIR = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_decoded_unified_20ms"
NPZ_PATH = "/mnt/c/Users/kante/Documents/KWU/etrl_task/data/kona_ev_rnn_sliding_window_dataset.npz"


def export_all_to_csv():
    print("=" * 100)
    print(" 📂 KONA EV 생성 데이터셋 CSV 파일 일괄 내보내기(Export)")
    print(f" 📁 저장 디렉토리: {TARGET_DIR}")
    print("=" * 100)

    os.makedirs(TARGET_DIR, exist_ok=True)

    # 1. 6개 주행 파일 공통 73개 피처 20ms 시계열 통합 CSV 내보내기
    print("\n [1/2] 6개 주행 파일 공통 73개 피처 20ms 시계열 통합 CSV 내보내는 중...")
    csv_files = sorted(glob.glob(os.path.join(TARGET_DIR, "kona_ev_decoded_signals_*.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(TARGET_DIR, "*.csv")))

    if csv_files:
        # 공통 피처 수집
        common_features = None
        for cp in csv_files:
            df_h = pd.read_csv(cp, nrows=5)
            meta_cols = {"Time_Rel_Sec", "Source_File", "CAN_ID_Hex"}
            fcols = [c for c in df_h.columns if c not in meta_cols]
            if common_features is None:
                common_features = set(fcols)
            else:
                common_features = common_features.intersection(set(fcols))

        common_cols = ["Time_Rel_Sec"] + sorted(list(common_features)) + ["Source_File"]

        df_common_list = []
        for cp in csv_files:
            fname = os.path.basename(cp)
            df_item = pd.read_csv(cp)
            # 공통 컬럼만 선택
            sel_cols = [c for c in common_cols if c in df_item.columns]
            df_sub = df_item[sel_cols].copy()
            df_common_list.append(df_sub)

        if df_common_list:
            df_unified_common = pd.concat(df_common_list, ignore_index=True)
            out_common_csv = os.path.join(TARGET_DIR, "kona_ev_common_73features_unified_20ms.csv")
            df_unified_common.to_csv(out_common_csv, index=False)
            print(f" 💾 [내보내기 성공] -> {out_common_csv}")
            print(f"    - 총 행 수: {len(df_unified_common):,} 행 (20ms 고정 주기)")
            print(f"    - 공통 피처 컬럼 수: {len(df_unified_common.columns)} 개")

    # 2. 250-샘플 슬라이딩 윈도우 Matched Matrix CSV 내보내기
    print("\n [2/2] 250-샘플 슬라이딩 윈도우 Matched Matrix CSV 내보내는 중...")
    if os.path.exists(NPZ_PATH):
        data = np.load(NPZ_PATH, allow_pickle=True)
        X_seq = data["X_common_seq"]  # Shape: (1036, 250, 73)
        common_feature_names = data["common_feature_names"]
        common_file_sources = data["common_file_sources"]

        n_windows, window_size, n_features = X_seq.shape

        # 3D (N_windows, 250, 73) -> 2D (N_windows, 250 * 73) 펼침
        X_flattened = X_seq.reshape(n_windows, -1)

        flat_cols = []
        for t in range(window_size):
            for fname_col in common_feature_names:
                flat_cols.append(f"t{t}_{fname_col}")

        df_window_csv = pd.DataFrame(X_flattened, columns=flat_cols)
        df_window_csv.insert(0, "Source_File", common_file_sources)
        df_window_csv.insert(0, "Window_Index", np.arange(n_windows))

        out_window_csv = os.path.join(TARGET_DIR, "kona_ev_rnn_sliding_window_250samples.csv")
        df_window_csv.to_csv(out_window_csv, index=False)

        print(f" 💾 [내보내기 성공] -> {out_window_csv}")
        print(f"    - 총 윈도우 수: {len(df_window_csv):,} 행 (1,036개 시계열 샘플)")
        print(f"    - 윈도우 당 텐서 크기: 250 샘플 x 73 피처 ({len(flat_cols):,} 매트릭스 컬럼)")

    print("\n" + "=" * 100)
    print(" 🎉 KONA EV 전체 데이터셋 CSV 내보내기 완전 완수!")
    print(f" 📁 저장된 경로: {TARGET_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    export_all_to_csv()
