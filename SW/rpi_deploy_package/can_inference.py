import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import torch

from . import decoding
from .src.can_models import SingleGRU

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, "src"))

W_SUFFIX = "_w34"

scaler = joblib.load(
    os.path.join(MODELS_DIR, f"scaler_can{W_SUFFIX}.pkl")
)

iforest = joblib.load(
    os.path.join(MODELS_DIR, f"step1_iforest{W_SUFFIX}.pkl")
)

gru_model = SingleGRU(in_dim=44, hidden=32)

gru_model.load_state_dict(
    torch.load(
        os.path.join(
            MODELS_DIR,
            f"step2_if_gru_can{W_SUFFIX}.pt"
        ),
        map_location="cpu",
        weights_only=True
    )
)

gru_model.eval()


def inference_can(can_frame_1s):
    dataset = decoding_functions.decoding(can_frame_1s)
    
    preprocessed_data = preprocess_can(dataset)
    
    is_abnormal = model_run(preprocessed_data)

    return is_abnormal

def preprocess_can(dataset):

    speed = 20.0
    grade = 0.0

    df_ws = pd.DataFrame(dataset["0x386"])
    df_yr = pd.DataFrame(dataset["0x220"])
    df_sas = pd.DataFrame(dataset["0x2b0"])
    df_ab = pd.DataFrame(dataset["0x371"])

    if any(df.empty for df in [df_ws, df_yr, df_sas, df_ab]):
        return []

    all_times = np.concatenate([
        df_ws["time"].values,
        df_yr["time"].values,
        df_sas["time"].values,
        df_ab["time"].values
    ])

    t_base = np.floor(np.min(all_times))

    preprocessed_dataset = []

    for i in range(10):

        w_start = t_base + i * 0.1
        w_end = w_start + 0.1

        ws_seg = df_ws[
            (df_ws["time"] >= w_start) &
            (df_ws["time"] < w_end)
        ]

        yr_seg = df_yr[
            (df_yr["time"] >= w_start) &
            (df_yr["time"] < w_end)
        ]

        sas_seg = df_sas[
            (df_sas["time"] >= w_start) &
            (df_sas["time"] < w_end)
        ]

        ab_seg = df_ab[
            (df_ab["time"] >= w_start) &
            (df_ab["time"] < w_end)
        ]

        # ZOH
        if len(ws_seg) == 0:
            prev = df_ws[df_ws["time"] < w_start]
            ws_seg = prev.tail(1) if len(prev) > 0 else df_ws.head(1)

        if len(yr_seg) == 0:
            prev = df_yr[df_yr["time"] < w_start]
            yr_seg = prev.tail(1) if len(prev) > 0 else df_yr.head(1)

        if len(sas_seg) == 0:
            prev = df_sas[df_sas["time"] < w_start]
            sas_seg = prev.tail(1) if len(prev) > 0 else df_sas.head(1)

        if len(ab_seg) == 0:
            prev = df_ab[df_ab["time"] < w_start]
            ab_seg = prev.tail(1) if len(prev) > 0 else df_ab.head(1)

        # 41개 feature
        raw_feats = compute_direct_window_features(
            ws_seg,
            yr_seg,
            sas_seg,
            ab_seg,
            speed,
            grade
        )

        # 43차원
        features_43 = [speed, grade] + raw_feats

        # Scaling
        X_scaled = scaler.transform([features_43])[0]

        # Isolation Forest
        if_score = iforest.decision_function(
            X_scaled.reshape(1, -1)
        )[0]

        # GRU 입력 44차원
        features_44 = np.append(
            X_scaled,
            if_score
        )

        preprocessed_dataset.append(features_44)

    # (10, 44)
    preprocessed_dataset = np.array(
        preprocessed_dataset,
        dtype=np.float32
    )

    # GRU 입력: (1, 10, 44)
    preprocessed_dataset = np.expand_dims(
        preprocessed_dataset,
        axis=0
    )

    return preprocessed_dataset


def model_run(preprocessed_data):

    if len(preprocessed_data) == 0:
        return False

    x = torch.tensor(
        preprocessed_data,
        dtype=torch.float32
    )

    with torch.no_grad():
        output = gru_model(x)
        pred = torch.argmax(output, dim=1).item()

    return pred == 1
