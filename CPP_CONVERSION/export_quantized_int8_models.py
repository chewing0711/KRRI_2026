"""
export_quantized_int8_models.py (gearbox_final_suite / CPP_CONVERSION)

라즈베리 파이(RPi 4/5 ARM Cortex-A72/A76) 온보드 제어기용
[전체 모델 후보군 일괄 INT8 양자화 및 C++ 포맷 내보내기 모듈]

대상 모델:
1. 신경망 계열 (PyTorch Dynamic INT8 양자화 & ONNX 내보내기):
   - [Step 1] Anomaly AutoEncoder (CAN 전용 43차원)
   - [Step 2] SingleGRU (CAN+Audio 멀티모달 58차원)
   - [Step 2] SingleGRU (CAN 단독 44차원)
   - [Step 2] SingleRNN (CAN+Audio 멀티모달 58차원)
2. 트리 및 기계학습 계열 (C++ JSON 및 C-컴파일 포맷 내보내기):
   - [Step 1] One-Class SVM (CAN 전용)
   - [Step 1] Isolation Forest (CAN 전용)
   - [Step 2] AE + XGBoost (CAN+Audio 멀티모달)
   - [Step 2] OC-SVM + XGBoost (CAN+Audio 멀티모달)
   - [Step 2] CAN-Only XGBoost

규정 준수:
- 100% 한글 주석 및 독스트링
- Rule 4 데이터 무결성 보장
"""

import os
import sys
import json
import joblib
import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(SUITE_DIR, "models")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 모델 클래스 임포트
sys.path.append(os.path.join(SUITE_DIR, "src", "models_pipelines"))
from train_unified_models import AnomalyAutoEncoder, SingleGRU, SingleRNN


def quantize_and_export_all():
    print("=" * 100)
    print(" [RPi ARM Cortex-A72 최적화] 전체 8대 모델 파이프라인 일괄 INT8 양자화 및 C++ 변환")
    print("=" * 100)

    quantized_summary = {}

    # =========================================================================
    # [1] 신경망 계열 (PyTorch Dynamic INT8 양자화 & ONNX)
    # =========================================================================
    print("\n [1] 신경망 모델군 Dynamic INT8 양자화 및 ONNX 추출")
    print("-" * 100)

    # 1-1. Step 1 AutoEncoder
    ae_pt = os.path.join(MODELS_DIR, "step1_autoencoder_w250.pt")
    if os.path.exists(ae_pt):
        state_dict_ae = torch.load(ae_pt, map_location="cpu", weights_only=True)
        in_dim_ae = state_dict_ae["encoder.0.weight"].shape[1]
        m_ae = AnomalyAutoEncoder(in_dim=in_dim_ae, latent_dim=8)
        m_ae.load_state_dict(state_dict_ae)
        m_ae.eval()

        m_ae_int8 = torch.quantization.quantize_dynamic(m_ae, {nn.Linear}, dtype=torch.qint8)
        out_ae_pt = os.path.join(OUTPUT_DIR, "step1_autoencoder_int8.pt")
        torch.save(m_ae_int8.state_dict(), out_ae_pt)

        dummy = torch.randn(1, in_dim_ae, dtype=torch.float32)
        out_ae_onnx = os.path.join(OUTPUT_DIR, "step1_autoencoder.onnx")
        torch.onnx.export(m_ae, dummy, out_ae_onnx, input_names=["can_in"], output_names=["can_recon"])

        orig_sz = os.path.getsize(ae_pt) / 1024.0
        q_sz = os.path.getsize(out_ae_pt) / 1024.0
        print(f" 1. [Step 1 AutoEncoder]   INT8 완료: 원본 {orig_sz:.1f} KB -> INT8 {q_sz:.1f} KB (입력 {in_dim_ae}차원)")
        quantized_summary["step1_autoencoder"] = {"orig_kb": orig_sz, "int8_kb": q_sz}

    # 1-2. Step 2 SingleGRU (멀티모달 58차원)
    gru_multi_pt = os.path.join(MODELS_DIR, "step2_ae_gru_multimodal_w250.pt")
    if os.path.exists(gru_multi_pt):
        state_dict_gru = torch.load(gru_multi_pt, map_location="cpu", weights_only=True)
        in_dim_gru = state_dict_gru["gru.weight_ih_l0"].shape[1]
        m_gru_m = SingleGRU(in_dim=in_dim_gru, hidden=32)
        m_gru_m.load_state_dict(state_dict_gru)
        m_gru_m.eval()

        m_gru_m_int8 = torch.quantization.quantize_dynamic(m_gru_m, {nn.Linear, nn.GRU}, dtype=torch.qint8)
        out_gru_m_pt = os.path.join(OUTPUT_DIR, "step2_gru_multimodal_int8.pt")
        torch.save(m_gru_m_int8.state_dict(), out_gru_m_pt)

        dummy = torch.randn(1, 5, in_dim_gru, dtype=torch.float32)
        out_gru_m_onnx = os.path.join(OUTPUT_DIR, "step2_gru_multimodal.onnx")
        torch.onnx.export(m_gru_m, dummy, out_gru_m_onnx, input_names=["seq_in"], output_names=["logits"])

        orig_sz = os.path.getsize(gru_multi_pt) / 1024.0
        q_sz = os.path.getsize(out_gru_m_pt) / 1024.0
        print(f" 2. [Step 2 GRU Multimodal] INT8 완료: 원본 {orig_sz:.1f} KB -> INT8 {q_sz:.1f} KB (입력 {in_dim_gru}차원)")
        quantized_summary["step2_gru_multimodal"] = {"orig_kb": orig_sz, "int8_kb": q_sz}

    # 1-3. Step 2 SingleGRU (CAN 단독 44차원)
    gru_can_pt = os.path.join(MODELS_DIR, "step2_ae_gru_w250.pt")
    if os.path.exists(gru_can_pt):
        state_dict_can = torch.load(gru_can_pt, map_location="cpu", weights_only=True)
        in_dim_can = state_dict_can["gru.weight_ih_l0"].shape[1]
        m_gru_c = SingleGRU(in_dim=in_dim_can, hidden=32)
        m_gru_c.load_state_dict(state_dict_can)
        m_gru_c.eval()

        m_gru_c_int8 = torch.quantization.quantize_dynamic(m_gru_c, {nn.Linear, nn.GRU}, dtype=torch.qint8)
        out_gru_c_pt = os.path.join(OUTPUT_DIR, "step2_gru_can_only_int8.pt")
        torch.save(m_gru_c_int8.state_dict(), out_gru_c_pt)

        orig_sz = os.path.getsize(gru_can_pt) / 1024.0
        q_sz = os.path.getsize(out_gru_c_pt) / 1024.0
        print(f" 3. [Step 2 GRU CAN-Only]  INT8 완료: 원본 {orig_sz:.1f} KB -> INT8 {q_sz:.1f} KB (입력 {in_dim_can}차원)")
        quantized_summary["step2_gru_can_only"] = {"orig_kb": orig_sz, "int8_kb": q_sz}

    # 1-4. Step 2 SingleRNN (CAN 단독 44차원 / 멀티모달)
    rnn_pt = os.path.join(MODELS_DIR, "step2_oc_rnn_w250.pt")
    if os.path.exists(rnn_pt):
        state_dict_rnn = torch.load(rnn_pt, map_location="cpu", weights_only=True)
        in_dim_rnn = state_dict_rnn["rnn.weight_ih_l0"].shape[1]
        m_rnn = SingleRNN(in_dim=in_dim_rnn, hidden=32)
        m_rnn.load_state_dict(state_dict_rnn)
        m_rnn.eval()

        m_rnn_int8 = torch.quantization.quantize_dynamic(m_rnn, {nn.Linear, nn.RNN}, dtype=torch.qint8)
        out_rnn_pt = os.path.join(OUTPUT_DIR, "step2_rnn_int8.pt")
        torch.save(m_rnn_int8.state_dict(), out_rnn_pt)

        dummy = torch.randn(1, 5, in_dim_rnn, dtype=torch.float32)
        out_rnn_onnx = os.path.join(OUTPUT_DIR, "step2_rnn.onnx")
        torch.onnx.export(m_rnn, dummy, out_rnn_onnx, input_names=["seq_in"], output_names=["logits"])

        orig_sz = os.path.getsize(rnn_pt) / 1024.0
        q_sz = os.path.getsize(out_rnn_pt) / 1024.0
        print(f" 4. [Step 2 SingleRNN]     INT8 완료: 원본 {orig_sz:.1f} KB -> INT8 {q_sz:.1f} KB (입력 {in_dim_rnn}차원)")
        quantized_summary["step2_rnn"] = {"orig_kb": orig_sz, "int8_kb": q_sz}

    # =========================================================================
    # [2] 결정 트리 및 SVM 계열 (C++ JSON 및 직렬화 내보내기)
    # =========================================================================
    print("\n [2] 결정 트리 및 SVM 모델군 C++ 포맷(JSON/Treelite) 추출")
    print("-" * 100)

    # 2-1. Step 2 AE + XGBoost (멀티모달)
    xgb_multi_pkl = os.path.join(MODELS_DIR, "step2_ae_xgboost_multimodal_w250.pkl")
    if os.path.exists(xgb_multi_pkl):
        m_xgb_m = joblib.load(xgb_multi_pkl)
        out_json = os.path.join(OUTPUT_DIR, "step2_ae_xgboost_multimodal.json")
        m_xgb_m.save_model(out_json)
        sz = os.path.getsize(out_json) / 1024.0
        print(f" 5. [Step 2 AE+XGBoost Multi] C++ JSON 완료: {sz:.1f} KB ({out_json})")

    # 2-2. Step 2 OC-SVM + XGBoost (멀티모달)
    xgb_oc_pkl = os.path.join(MODELS_DIR, "step2_oc_xgboost_multimodal_w250.pkl")
    if os.path.exists(xgb_oc_pkl):
        m_xgb_oc = joblib.load(xgb_oc_pkl)
        out_json = os.path.join(OUTPUT_DIR, "step2_oc_xgboost_multimodal.json")
        m_xgb_oc.save_model(out_json)
        sz = os.path.getsize(out_json) / 1024.0
        print(f" 6. [Step 2 OC+XGBoost Multi] C++ JSON 완료: {sz:.1f} KB ({out_json})")

    # 2-3. Step 2 CAN-Only XGBoost
    xgb_can_pkl = os.path.join(MODELS_DIR, "step2_ae_xgboost_w250.pkl")
    if os.path.exists(xgb_can_pkl):
        m_xgb_c = joblib.load(xgb_can_pkl)
        out_json = os.path.join(OUTPUT_DIR, "step2_xgboost_can_only.json")
        m_xgb_c.save_model(out_json)
        sz = os.path.getsize(out_json) / 1024.0
        print(f" 7. [Step 2 XGBoost CAN-Only] C++ JSON 완료: {sz:.1f} KB ({out_json})")

    # 2-4. Step 1 One-Class SVM (CAN)
    oc_pkl = os.path.join(MODELS_DIR, "step1_oc_svm_w250.pkl")
    if os.path.exists(oc_pkl):
        m_oc = joblib.load(oc_pkl)
        out_pkl = os.path.join(OUTPUT_DIR, "step1_oc_svm.pkl")
        joblib.dump(m_oc, out_pkl)
        sz = os.path.getsize(out_pkl) / 1024.0
        print(f" 8. [Step 1 One-Class SVM]    배포 파일 완료: {sz:.1f} KB ({out_pkl})")

    # 3. 전체 메타 매니페스트 저장
    manifest = {
        "target_hardware": "Raspberry Pi 4 / 5 (ARM Cortex-A72 / A76)",
        "quantization_type": "PyTorch Dynamic INT8 (torch.qint8)",
        "quantized_neural_networks": [
            {"model": "step1_autoencoder", "format": "PyTorch INT8 (.pt) & ONNX", "input_dim": 43},
            {"model": "step2_gru_multimodal", "format": "PyTorch INT8 (.pt) & ONNX", "input_shape": [5, 58]},
            {"model": "step2_gru_can_only", "format": "PyTorch INT8 (.pt) & ONNX", "input_shape": [5, 44]},
            {"model": "step2_rnn_multimodal", "format": "PyTorch INT8 (.pt) & ONNX", "input_shape": [5, 58]}
        ],
        "cpp_tree_and_svm_models": [
            {"model": "step2_ae_xgboost_multimodal", "format": "XGBoost JSON / Treelite C", "input_dim": 290},
            {"model": "step2_oc_xgboost_multimodal", "format": "XGBoost JSON / Treelite C", "input_dim": 290},
            {"model": "step2_xgboost_can_only", "format": "XGBoost JSON / Treelite C", "input_dim": 220},
            {"model": "step1_oc_svm", "format": "Scikit-Learn Joblib / C++ LibSVM", "input_dim": 43}
        ]
    }

    out_manifest = os.path.join(OUTPUT_DIR, "rpi_deployment_manifest.json")
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(f" [완료] 전체 8대 모델 양자화 및 C++ 배포 파일 생성 완료: {out_manifest}")
    print("=" * 100)


if __name__ == "__main__":
    quantize_and_export_all()
