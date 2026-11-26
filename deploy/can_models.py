import os
import time
import joblib
import numpy as np
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        # state_dict 가중치 키('shortcut.weight', 'shortcut.bias')와 반드시 일치해야 함
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(h))
        return nn.functional.gelu(h + self.shortcut(x))


class CrossAttention(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.q_can = nn.Linear(dim, dim)
        self.k_aud = nn.Linear(dim, dim)
        self.v_aud = nn.Linear(dim, dim)
        self.q_aud = nn.Linear(dim, dim)
        self.k_can = nn.Linear(dim, dim)
        self.v_can = nn.Linear(dim, dim)
        self.scale = np.sqrt(dim)

    def forward(self, z_c, z_a):
        a_c2a = torch.sigmoid((self.q_can(z_c) * self.k_aud(z_a)).sum(dim=1, keepdim=True) / self.scale)
        a_a2c = torch.sigmoid((self.q_aud(z_a) * self.k_can(z_c)).sum(dim=1, keepdim=True) / self.scale)
        return z_c + a_a2c * self.v_can(z_c), z_a + a_c2a * self.v_aud(z_a)


class DualAutoEncoder(nn.Module):
    def __init__(self, can_dim=41, audio_dim=24, latent_dim=16):
        super().__init__()
        self.can_enc = nn.Sequential(ResBlock(can_dim, 32), ResBlock(32, latent_dim))
        self.audio_enc = nn.Sequential(ResBlock(audio_dim, 32), ResBlock(32, latent_dim))
        self.cross_attn = CrossAttention(latent_dim)
        self.can_dec = nn.Sequential(nn.Linear(latent_dim, 32), nn.GELU(), nn.Linear(32, can_dim))
        self.audio_dec = nn.Sequential(nn.Linear(latent_dim, 32), nn.GELU(), nn.Linear(32, audio_dim))

    def forward(self, x_can, x_aud):
        zc = nn.functional.normalize(self.can_enc(x_can), p=2, dim=1)
        za = nn.functional.normalize(self.audio_enc(x_aud), p=2, dim=1)
        zc_att, za_att = self.cross_attn(zc, za)
        return zc, za, zc_att, za_att


class InferenceEngine:
    def __init__(self, models_dir=None):
        if models_dir is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            for cand in [
                os.path.join(cur_dir, "models"),
                os.path.join(cur_dir, "..", "models"),
                os.path.join(os.getcwd(), "models"),
            ]:
                if os.path.exists(os.path.join(cand, "scaler_can_latent_w170.pkl")):
                    models_dir = os.path.abspath(cand)
                    break
            if models_dir is None:
                models_dir = os.path.join(cur_dir, "models")

        self.models_dir = models_dir
        self.device = torch.device("cpu")

        can_scaler_path = os.path.join(models_dir, "scaler_can_latent_w170.pkl")
        aud_scaler_path = os.path.join(models_dir, "scaler_audio_latent_w170.pkl")
        if not (os.path.exists(can_scaler_path) and os.path.exists(aud_scaler_path)):
            raise FileNotFoundError(f"Scaler files not found in {models_dir}")

        can_scaler = joblib.load(can_scaler_path)
        aud_scaler = joblib.load(aud_scaler_path)
        self.can_center = getattr(can_scaler, "center_", np.zeros(41)).astype(np.float32)
        self.can_scale = getattr(can_scaler, "scale_", np.ones(41)).astype(np.float32)
        self.aud_center = getattr(aud_scaler, "center_", np.zeros(24)).astype(np.float32)
        self.aud_scale = getattr(aud_scaler, "scale_", np.ones(24)).astype(np.float32)

        self.dae = DualAutoEncoder(41, 24, 16).to(self.device)
        dae_path = os.path.join(models_dir, "dual_autoencoder_latent_w170.pt")
        if os.path.exists(dae_path):
            self.dae.load_state_dict(torch.load(dae_path, map_location=self.device), strict=False)
        self.dae.eval()

        fw_path = os.path.join(models_dir, "latent_fusion_fast_weights.pkl")
        if os.path.exists(fw_path):
            fw = joblib.load(fw_path)
            self.coef = fw["coef"].astype(np.float32).reshape(-1)
            self.intercept = float(np.array(fw["intercept"]).ravel()[0])
            self.use_fast_head = True
        else:
            self.use_fast_head = False

        self._warmup()

    def _warmup(self, n=5):
        c = np.zeros(41, dtype=np.float32)
        a = np.zeros(24, dtype=np.float32)
        for _ in range(n):
            self.predict(c, a)

    def predict(self, can_feats, aud_feats, energy_6d=None):
        t0 = time.perf_counter()
        x_c = np.ascontiguousarray(can_feats, dtype=np.float32).reshape(1, -1)
        x_a = np.ascontiguousarray(aud_feats, dtype=np.float32).reshape(1, -1)
        x_e = x_a[:, :6] if energy_6d is None else np.ascontiguousarray(energy_6d, dtype=np.float32).reshape(1, -1)

        x_c_norm = (x_c - self.can_center) / (self.can_scale + 1e-8)
        x_a_norm = (x_a - self.aud_center) / (self.aud_scale + 1e-8)

        with torch.inference_mode():
            _, _, zc_att, za_att = self.dae(torch.from_numpy(x_c_norm), torch.from_numpy(x_a_norm))
            zc = zc_att.numpy()
            za = za_att.numpy()

        cos_sim = float(np.clip(np.sum(zc * za, axis=1, keepdims=True)[0, 0], -1.0, 1.0))
        dist = float(np.sqrt(max(0.0, 2.0 * (1.0 - cos_sim))))

        fused = np.hstack([
            zc,
            za,
            np.array([[cos_sim]], dtype=np.float32),
            np.array([[dist]], dtype=np.float32),
            x_e,
        ])

        if self.use_fast_head:
            logit = float(np.dot(fused[0], self.coef) + self.intercept)
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))
            pred = 1 if prob >= 0.5 else 0
        else:
            pred = 1 if dist >= 1.0 else 0
            prob = 1.0 if pred == 1 else 0.0

        return {"pred": pred, "prob": prob, "latency_ms": (time.perf_counter() - t0) * 1000.0}

    # 호환 메서드 별칭
    infer_latent_2branch = predict


# 호환 클래스 별칭
GatedResidualBlock = ResBlock
CrossModalAttention = CrossAttention
MultimodalRealtimeInferenceEngine = InferenceEngine
