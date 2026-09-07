"""
train_unified_models.py (SW/rpi_deploy_package/src)

실시간 추론 및 배포 모델 평가용 PyTorch 신경망 클래스 및 유틸리티 모듈:
1. AnomalyAutoEncoder: Step 1 CAN 비지도 이상치 탐지 오토인코더
2. SingleGRU: Step 2 10-Step 시계열 고장 판정 GRU 신경망
3. SingleRNN: Step 2 바닐라 RNN 신경망
4. get_ae_error_feature: 오토인코더 복원 오차 계산 함수
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sequence_matrix_builder import (
    create_sliding_sequence_tensor,
    create_flattened_lag_matrix,
    SlidingSequenceDataset,
    RealtimeSequenceBuffer
)


class AnomalyAutoEncoder(nn.Module):
    """정상 주행 데이터로만 학습하는 1단계 비지도 복원 신경망"""
    def __init__(self, in_dim=43, latent_dim=8):
        super(AnomalyAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 24),
            nn.ReLU(),
            nn.Linear(24, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.ReLU(),
            nn.Linear(24, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def get_ae_error_feature(model, X_s):
    """AutoEncoder 복원 오차(Reconstruction Error)를 1차원 피처로 추출"""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_s, dtype=torch.float32)
        recon = model(x_t)
        err = torch.mean((recon - x_t) ** 2, dim=1, keepdim=True).numpy()
    return err


class SingleGRU(nn.Module):
    """2단계 10-Step 시계열 GRU 고장 판정 신경망"""
    def __init__(self, in_dim=44, hidden=32):
        super(SingleGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        if len(out.shape) == 3:
            out = out[:, -1, :]
        return self.fc(out)


class SingleRNN(nn.Module):
    """2단계 시계열 Vanilla RNN 신경망"""
    def __init__(self, in_dim=44, hidden=32):
        super(SingleRNN, self).__init__()
        self.rnn = nn.RNN(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        if len(out.shape) == 3:
            out = out[:, -1, :]
        return self.fc(out)
