"""
sequence_matrix_builder.py (gearbox_final_suite / data_processing)

시계열 행렬(Sequence Matrix) 및 3D 텐서 실시간 변환 전용 모듈:
1. 세션 경계 격리(Session Boundary Aware) 슬라이딩 윈도우 3D 텐서 생성 (PyTorch RNN/GRU/LSTM용)
2. 시계열 지연 피처(Flattened Lag Features) 2D 행렬 생성 (XGBoost/RandomForest용)
3. 실시간 추론용 링버퍼(RealtimeSequenceBuffer) 스트리밍 FIFO 제공

규정 준수:
- 100% 한글 주석 및 독스트링
- 데이터 유출(Data Leakage) 100% 방지 (In-Memory 슬라이싱)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def create_sliding_sequence_tensor(X, y=None, session_ids=None, seq_len=5):
    """
    2D 피처 행렬을 세션 경계를 보존하며 3D 시계열 텐서 [N_samples, seq_len, n_features]로 변환합니다.

    Args:
        X (np.ndarray): 2D 피처 행렬 (N, n_features)
        y (np.ndarray, optional): 1D 라벨 배열 (N,)
        session_ids (pd.Series or np.ndarray, optional): 세션 식별자 배열 (N,)
        seq_len (int): 시계열 시퀀스 길이 (기본값: 5)

    Returns:
        X_seq (np.ndarray): 3D 시계열 텐서 (N_seq, seq_len, n_features)
        y_seq (np.ndarray, optional): 시퀀스 마지막 시점 기준 라벨 배열 (N_seq,)
    """
    if seq_len <= 1:
        if y is not None:
            return X[:, np.newaxis, :], y
        return X[:, np.newaxis, :]

    seq_list = []
    y_list = []

    if session_ids is not None:
        unique_sessions = pd.Series(session_ids).unique()
        session_series = pd.Series(session_ids)

        for sess in unique_sessions:
            idx = np.where(session_series == sess)[0]
            X_sess = X[idx]
            y_sess = y[idx] if y is not None else None
            n_sess = len(X_sess)

            if n_sess < seq_len:
                continue

            for i in range(n_sess - seq_len + 1):
                # [seq_len, n_features] 슬라이스
                seq_list.append(X_sess[i : i + seq_len])
                if y_sess is not None:
                    # 시퀀스의 마지막 윈도우 라벨을 해당 시퀀스의 타겟으로 지정
                    y_list.append(y_sess[i + seq_len - 1])
    else:
        n_samples = len(X)
        for i in range(n_samples - seq_len + 1):
            seq_list.append(X[i : i + seq_len])
            if y is not None:
                y_list.append(y[i + seq_len - 1])

    if len(seq_list) == 0:
        empty_x = np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
        if y is not None:
            return empty_x, np.empty((0,), dtype=np.int64)
        return empty_x

    X_seq = np.array(seq_list, dtype=np.float32)

    if y is not None:
        y_seq = np.array(y_list, dtype=np.int64)
        return X_seq, y_seq

    return X_seq


def create_flattened_lag_matrix(X, y=None, session_ids=None, seq_len=5):
    """
    2D 피처 행렬을 5스텝 지연 피처가 결합된 2D 행렬 [N_samples, seq_len * n_features]로 변환합니다.
    XGBoost 등 트리 기반 모델에서 시계열 맥락을 흡수하기 위해 사용됩니다.

    Args:
        X (np.ndarray): 2D 피처 행렬 (N, n_features)
        y (np.ndarray, optional): 1D 라벨 배열 (N,)
        session_ids (pd.Series or np.ndarray, optional): 세션 식별자 배열 (N,)
        seq_len (int): 시계열 시퀀스 길이 (기본값: 5)

    Returns:
        X_flat (np.ndarray): 2D 지연 피처 행렬 (N_seq, seq_len * n_features)
        y_seq (np.ndarray, optional): 라벨 배열 (N_seq,)
    """
    if y is not None:
        X_seq, y_seq = create_sliding_sequence_tensor(X, y=y, session_ids=session_ids, seq_len=seq_len)
        n_samples = len(X_seq)
        X_flat = X_seq.reshape(n_samples, -1)
        return X_flat, y_seq
    else:
        X_seq = create_sliding_sequence_tensor(X, y=None, session_ids=session_ids, seq_len=seq_len)
        n_samples = len(X_seq)
        X_flat = X_seq.reshape(n_samples, -1)
        return X_flat


class SlidingSequenceDataset(Dataset):
    """
    PyTorch 학습/검증용 시계열 텐서 데이터셋 클래스
    """
    def __init__(self, X_seq, y_seq=None):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1) if y_seq is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class RealtimeSequenceBuffer:
    """
    실차 제어기(ECU, TCU, Raspberry Pi) 실시간 배포용 FIFO 링버퍼
    최근 seq_len개의 윈도우 피처를 메모리에 유지하며 모델 추론용 3D 텐서를 즉시 반환합니다.
    """
    def __init__(self, n_features, seq_len=5):
        self.n_features = n_features
        self.seq_len = seq_len
        self.buffer = []

    def push(self, feature_vector):
        """
        단일 윈도우 피처 벡터(1D, n_features)를 버퍼에 추가합니다.
        """
        feat = np.array(feature_vector, dtype=np.float32).ravel()
        self.buffer.append(feat)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def is_ready(self):
        """
        버퍼에 seq_len개의 윈도우가 모두 차서 추론 가능한지 여부를 반환합니다.
        """
        return len(self.buffer) == self.seq_len

    def get_sequence_tensor(self):
        """
        PyTorch 시계열 모델 입력용 3D 텐서 [1, seq_len, n_features]를 반환합니다.
        """
        if not self.is_ready():
            # 버퍼가 덜 찬 초기에는 첫 윈도우로 패딩
            pad_count = self.seq_len - len(self.buffer)
            pad_feats = [self.buffer[0]] * pad_count if len(self.buffer) > 0 else [np.zeros(self.n_features, dtype=np.float32)] * pad_count
            full_seq = pad_feats + self.buffer
        else:
            full_seq = self.buffer

        return np.array(full_seq, dtype=np.float32)[np.newaxis, :, :] # (1, seq_len, n_features)

    def get_flattened_vector(self):
        """
        XGBoost 입력용 2D 지연 벡터 [1, seq_len * n_features]를 반환합니다.
        """
        tensor = self.get_sequence_tensor()
        return tensor.reshape(1, -1)

    def clear(self):
        """
        새 주행 세션 시작 시 버퍼를 초기화합니다.
        """
        self.buffer.clear()
