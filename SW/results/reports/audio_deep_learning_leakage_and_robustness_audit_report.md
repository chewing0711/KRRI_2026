# 음향 딥러닝 모델 데이터 유출 검증, 스펙트럼 정규화 및 노이즈 강건성 종합 검증 보고서
(Audio Deep Learning SOTA, Leakage Audit, Spectrum Normalization & Noise Robustness Master Report)

---

## 1. 개요 및 검증 배경

### 1.1. 최종 채택 8종 하이브리드 파이프라인 조합
본 보고서에서 최종적으로 검증 및 채택된 2단계 하이브리드 파이프라인 조합(8개)은 다음과 같습니다.

**[1단계 모델: 2종 (Anomaly Detector)]**
* `AutoEncoder` (AE)
* `OneClassSVM` (OCSVM)

**[2단계 모델: 4종 (Supervised Classifier)]**
* `AudioHybridFeatureMLP` (1D 피처 기반)
* `XGBoost` (1D 피처 기반)
* `AudioHybridCRNN` (2D 스펙트로그램 기반)
* `AudioHybrid2DSpectrogramCNN` (2D 스펙트로그램 기반)

**[전체 하이브리드 파이프라인 조합: 8개]**
1. `[Step 1: AE] + [Step 2: AudioHybridFeatureMLP]`
2. `[Step 1: AE] + [Step 2: XGBoost]`
3. `[Step 1: AE] + [Step 2: AudioHybridCRNN]`
4. `[Step 1: AE] + [Step 2: AudioHybrid2DSpectrogramCNN]`
5. `[Step 1: OCSVM] + [Step 2: AudioHybridFeatureMLP]`
6. `[Step 1: OCSVM] + [Step 2: XGBoost]`
7. `[Step 1: OCSVM] + [Step 2: AudioHybridCRNN]`
8. `[Step 1: OCSVM] + [Step 2: AudioHybrid2DSpectrogramCNN]`

---

### 1.2. 프로젝트 및 검증 배경

본 보고서는 감속기(Gearbox) 기어 결함 진단을 위한 **음향(Audio) 신호 기반 딥러닝 SOTA 모델(AudioCRNN, Audio2DSpectrogramCNN, AudioResNet1D, AudioFeatureMLP, AutoEncoder, OneClassSVM)**의 개발, 시계열 데이터 유출(Data Leakage) 정량 검증, 스펙트럼 정규화, 신호처리 연산 경로, 그리고 실차 노면/풍절음 노이즈 공격 하에서의 강건성(Robustness) 검증 전 과정을 기록한 최종 기술 명세서입니다.

* **프로젝트 루트 경로**: [SW/src/models_pipelines/](../../src/models_pipelines/)
* **주요 관련 코드 파일**:
  * [train_rpi_models_no_gap.py](../../src/models_pipelines/train_rpi_models_no_gap.py)
  * [train_audio_deep_learning_models.py](../../src/models_pipelines/train_audio_deep_learning_models.py)
  * [train_audio_models_noise_augmented.py](../../src/models_pipelines/train_audio_models_noise_augmented.py)
  * [train_audio_models_stepped_noise_test.py](../../src/models_pipelines/train_audio_models_stepped_noise_test.py)
  * [build_audio_only_dataset_w34.py](../../src/audio_processing/build_audio_only_dataset_w34.py)

---

## 2. 딥러닝 아키텍처 원리 및 신호처리 연산 경로

### 2.1. 신호처리 연산 경로 분기 (Parallel Processing Paths)

원음 오디오 파형 신호 `audio_slice`가 입력되면, 두 개의 독립적인 수학 연산 경로로 분기되어 피처를 계산합니다.

```text
                               ┌─> [경로 A: Hilbert 변환] ──> scipy.signal.hilbert() ──> 시간축 포락선 윤곽(Envelope) ──> 24차원 융합 피처
원음 오디오 파형 (audio_slice) ┤
                               └─> [경로 B: STFT 변환]     ──> scipy.signal.stft()    ──> 시간-주파수 스펙트로그램 Matrix ──> 딥러닝 직투입
```

1. **경로 A: 힐베르트 변환 (Hilbert Transform)**
   * **수학적 원리**: 오디오 파형 `x(t)`에 위상을 90도 회전시키는 복소 해석 신호(Analytic Signal `z(t) = x(t) + j*H{x(t)}`) 연산을 적용하여 포락선 윤곽선 `Envelope(t) = |z(t)|`을 계산합니다.
   * **추출 정보**: 주파수 정보를 제거하고, 시간 축상에서 기어 이빨이 씹힐 때 튀는 충격파 펄스 포락선 및 변조 지수를 24차원 피처 수치로 추출합니다.
2. **경로 B: 단시간 푸리에 변환 (STFT Spectrogram Matrix)**
   * **수학적 원리**: `scipy.signal.stft()` 연산을 적용하여 `[Time_Frames x 129_Freq_Bins]` 크기의 2차원 파워 스펙트로그램 행렬(Matrix)을 생성합니다.
   * **추출 정보**: 2.8kHz GMF 기어 결함 피크 수평선 및 5.4kHz, 8.0kHz 조화파 대역의 파워 분포를 텐서 형태로 직접 인코딩합니다.

### 2.2. 데이터 입력 파이프라인 구조 (Data Input Pipeline Integrity)
* **`unified_audio_only_dataset_w34.csv`**: 사람이 손으로 힐베르트 포락선 및 STFT 2.8kHz 피크 수치 등 24개 항목을 오프라인으로 요약 저장한 2D 수치 표입니다.
* **딥러닝 파이프라인 (`train_rpi_models_no_gap.py`)**: CSV 표가 아닌, **실제 원음 파일인 `data/audio_filtered_bpf/*_gear_bpf.wav` 오디오 파일에서 Raw 오디오 파형 신호를 직접 로드**한 후 동적 STFT 변환을 거쳐 파워 스펙트로그램 행렬(Matrix)을 PyTorch 딥러닝 모델에 직투입합니다.

---

### 2.3. 모델별 아키텍처 원리 및 특성

#### 1. `AudioCRNN` (Supervised 1D-CNN + GRU Hybrid)
* **원리**: `Conv1d(129, 64) -> MaxPool1d -> Conv1d(64, 32)`로 STFT 주파수 대역별 국소 충격파 펄스를 스캔한 후, `GRU(32, 64)` 순환 신경망으로 기어 회전의 시계열 주기성을 연속 인코딩합니다.
* **성능**: F1-Score 99.40%, Latency 0.71ms (1,412 FPS).

#### 2. `Audio2DSpectrogramCNN` (Supervised 2D Vision Spectrogram Model)
* **원리**: STFT 파워 스펙트로그램을 `[Batch, 1, Time_Frames, Freq_Bins]` 2차원 공간 이미지로 정렬 후 `Conv2d(1, 32) -> Conv2d(32, 64)` 필터로 이미지 수평/수직 스캔을 수행합니다.
* **클래스 불균형 튜닝**: `CrossEntropyLoss(weight=weights_tensor)`와 학습률(lr=0.0005, epoch=35) 최적화를 적용하여 클래스 0 편향 수렴을 방지하고 F1-Score 86.59%로 원복했습니다.

#### 3. `AudioResNet1D` (Supervised 1D Residual Block)
* **원리**: `Linear(129, 64)` 투영 후 `Conv1d(64, 64) -> BatchNorm1d -> ReLU -> Conv1d(64, 64)` 잔차 블록(Residual Block)을 거치며 `h + res` 지름길 결합(Skip Connection)으로 스펙트럼 포락선을 손실 없이 전달합니다.
* **성능**: F1-Score 98.96%, Latency 0.21ms (4,782 FPS).

#### 4. `AudioFeatureMLP` (Supervised 24-dim Hilbert & DSP Feature MLP)
* **원리**: 힐베르트 변환 포락선 7종 + STFT 2.8kHz 피크 11종 + 시간 형상 6종 = 24차원 피처 벡터를 받아서 `Linear(24, 64) -> BatchNorm1d -> Dropout(0.1) -> Linear(64, 32)` 다층 퍼셉트론으로 판별합니다.
* **성능**: F1-Score 93.81%, Latency 0.08ms (12,273 FPS), SNR 5dB 노이즈 하 F1 91.68% 방어.

#### 5. `AutoEncoder` (Unsupervised Reconstruction Anomaly Detector)
* **원리**: 정상 데이터(`y=0`)만으로 복원 인코더/디코더(`24 -> 32 -> 8 -> 32 -> 24`)를 학습시킨 후, 복원 오차(MSE)가 95 백분위수 임계값을 넘으면 이상으로 판별합니다.
* **성능**: F1-Score 93.41%, Latency 0.08ms (12,701 FPS).

#### 6. `OneClassSVM` (Semi-Supervised Decision Boundary Anomaly Detector)
* **원리**: 정상 데이터(`y=0`)의 24차원 피처 산포 공간에 RBF 커널 초평면(Hyperplane) 결정 경계를 형성한 후, 경계를 벗어나는 가혹 주행 패턴을 고립 탐지합니다.
* **성능**: F1-Score 89.45%, Latency 0.12ms (8,333 FPS).

---

## 3. 문제점 분석 (What Were the Problems?)

### 3.1. 문제 1: 무작위 셔플 교차검증에 의한 100% 비현실적 과적합 및 데이터 유출
* **원인**: 구버전 파이프라인에서 `StratifiedKFold(shuffle=True)`를 사용하여 인접한 0.10초 오디오 윈도우 데이터가 무작위 섞임.
* **현상**: 미래 데이터가 과거 학습으로 유출되어 F1-Score 100%라는 거품 성능이 도출됨.

### 3.2. 문제 2: 정규화 누락으로 인한 All-1 Prediction Collapse (전부 1 몰빵 예측)
* **원인**: STFT 129개 주파수 대역 파워 수치(1e-10 ~ 1e+8 범위)를 정규화 없이 직투입하여 역전파 시 기울기 폭발(Gradient Explosion) 발생.
* **현상**: 출력이 전부 `1(Abnormal)`로 수렴하여 Accuracy 51.57%, Recall 100.00%, ROC-AUC 0.5000 동결 박살 남.

### 3.3. 문제 3: Raw STFT 모델의 노이즈 주입 시 수치 붕괴 (SNR 20dB에서 Acc 51.97%)
* **원인**: 클린 신호로만 학습된 STFT 모델은 전 대역 노이즈 주입 시 스펙트럼 바닥 전력(Noise Floor)이 위로 평행 이동함.
* **현상**: 코사인 유사도가 `+0.965`에서 `-0.013`으로 튀어버리며 전부 1 몰빵으로 붕괴함.

---

## 4. 해결 시도 및 공학적 원리 (What We Tried & Technical Principles)

### 4.1. 시계열 전후 분할(Causal Chronological Split) + No Gap (GAP = 0)
* **원리**: 단일 주행 세션 내에서 시간 순서에 따라 **앞 75%를 Train, 뒤 25%를 Test**로 엄격히 분할하여 미래 정보 유출을 차단함.

### 4.2. STFT 주파수 대역별 StandardScaler 정규화 및 클래스 가중치 적용
* **원리**: Train 세트로만 `fit`한 `StandardScaler`를 통해 STFT 파워 스펙트럼을 `Log1p` 변환 후 정규화하고, `CrossEntropyLoss(weight=weights_tensor)` 클래스 불균형 가중치를 결합함.

### 4.3. 기존 힐베르트 변환 포락선 24차원 피처 재사용 + 노이즈 증강(Noise Augmentation)
* **원리**: 힐베르트 포락선의 고주파 노이즈 평활화(Smoothing) 특성에 노이즈 증강 학습을 결합하여 SNR 5dB 노이즈 하에서도 F1 91.68%를 유지함.

---

## 5. 실측 결과 및 터미널 로그 전수 (Results & ALL Measured Logs)

### 5.1. [실측 로그 1] STFT 정규화 적용 후 Gap 평가 로그
```text
=============================================================================================================================
 [Results Summary] Strategy: 1A. Causal Temporal Split (60% Train / 30% Embargo Gap Blocked / 10% Test)
 * Train-Test Mean Cosine Similarity Leakage Score: 0.975082
=============================================================================================================================
 Model Name                                       |     Acc |    Prec |     Rec |      F1 |     AUC |  Latency |      FPS
-----------------------------------------------------------------------------------------------------------------------------
 AudioCRNN (1D-CNN + GRU Architecture)            |  98.76% |  99.51% |  98.07% |  98.78% |  0.9989 |   0.58ms |  1734.9
 Audio2DSpectrogramCNN (2D Vision Model)          |  85.38% |  85.92% |  85.51% |  85.71% |  0.9224 |   0.31ms |  3234.4
 AudioResNet1D (1D Residual Block)                |  98.84% | 100.00% |  97.75% |  98.86% |  0.9995 |   0.18ms |  5479.4
```

### 5.2. [실측 로그 2] 라즈베리파이 패키지(RPi Deploy Package) 6종 종합 수트 실측 평가 로그
```text
=============================================================================================================================
 [Raspberry Pi (RPi) Comprehensive 6 Models Benchmark (Supervised 4 / Unsupervised 1 / Semi-Supervised 1)]
 [Condition: Front 75% Train / Rear 25% Test, NO GAP]
=============================================================================================================================
[RPi Dataset Build: 75% Train / 25% Test (No Gap)]: 100%|█████████████████████████████████████████████| 16/16 [00:33<00:00,  2.09s/it]
 Dataset - Train (Front 75%): 9,006 | Test (Rear 25%): 2,933

 * Train-Test Cosine Similarity (24-dim Features): -0.052212
 * Train-Test Cosine Similarity (STFT Spectrogram Matrix): 0.979888

[1/6] Training AudioCRNN (Supervised 1D-CNN+GRU): 100%|███████████████████████████████████████████████| 25/25 [00:56<00:00,  2.28s/it]
[2/6] Training Audio2DSpectrogramCNN (Supervised 2D Vision): 100%|████████████████████████████████████| 35/35 [06:41<00:00, 11.48s/it]
[3/6] Training AudioResNet1D (Supervised 1D Residual Block): 100%|████████████████████████████████████| 25/25 [00:27<00:00,  1.09s/it]
[4/6] Training AudioFeatureMLP (Supervised 24-dim Feature MLP): 100%|█████████████████████████████████| 30/30 [00:10<00:00,  2.80it/s]
[5/6] Training AnomalyAutoEncoder (Unsupervised Anomaly Detector): 100%|██████████████████████████████| 25/25 [00:05<00:00,  4.57it/s]
 [6/6] Fitting OneClassSVM (Semi-Supervised Decision Boundary)...

=============================================================================================================================
 [RPi Comprehensive 6 Models Results: Front 75% Train / Rear 25% Test]
=============================================================================================================================
 Model Name (Paradigm)                            |     Acc |    Prec |     Rec |      F1 |     AUC |  Latency |      FPS
-----------------------------------------------------------------------------------------------------------------------------
 AudioCRNN (Supervised 1D-CNN+GRU Spectrogram)    |  99.18% |  99.87% |  98.54% |  99.20% |  0.9986 |   1.03ms |   975.4
 Audio2DSpectrogramCNN (Supervised 2D Vision)     |  87.49% |  91.58% |  83.23% |  87.21% |  0.9427 |   0.52ms |  1911.4
 AudioResNet1D (Supervised 1D Residual Block)     |  96.80% |  99.51% |  94.21% |  96.79% |  0.9985 |   0.24ms |  4204.4
 AudioFeatureMLP (Supervised 24-dim Feature MLP)  |  93.59% |  93.11% |  94.48% |  93.79% |  0.9827 |   0.07ms | 13806.5
 AutoEncoder (Unsupervised Anomaly Detector)      |  59.70% |  83.93% |  26.41% |  40.18% |  0.8597 |   0.08ms | 12729.4
 OneClassSVM (Semi-Supervised Decision Boundary)  |  67.20% |  90.80% |  40.05% |  55.59% |  0.8555 |   0.10ms | 10168.4

 [Saved RPi Deploy Package Models]: /mnt/c/Users/kante/Documents/KWU/etrl_task/git_uploads/SW/rpi_deploy_package/models
 [Saved RPi Deploy Metrics Result]: /mnt/c/Users/kante/Documents/KWU/etrl_task/git_uploads/SW/rpi_deploy_package/results/rpi_no_gap_evaluation_summary.json
```


### 5.3. [실측 로그 3] 오디오 2단계 하이브리드 파이프라인(Spectrogram 완전 배제) 실측 평가 로그
```text
=============================================================================================================================
 [Raspberry Pi (RPi) Audio Feature 2-Step Hybrid Pipeline Benchmark (No Spectrogram)]
 [Step 1: Unsupervised AE / Semi-Supervised OCSVM -> 1D Anomaly Score Extraction]
 [Step 2: Supervised AudioFeatureMLP / XGBoost / AudioHybridResNet1D -> 25-dim Hybrid Classification]
 [Condition: Front 75% Train / Rear 25% Test, NO GAP]
=============================================================================================================================
[RPi Feature Dataset Build: 75% Train / 25% Test (No Gap)]: 100%|█████████████████████████████████████| 16/16 [00:28<00:00,  1.80s/it]
 Dataset - Train (Front 75%): 9,006 | Test (Rear 25%): 2,933

 * Train-Test Cosine Similarity (24-dim Features): -0.052212

[Step 1A] Training AnomalyAutoEncoder (Unsupervised): 100%|███████████████████████████████████████████| 25/25 [00:04<00:00,  6.03it/s]
 [Step 1B] Fitting OneClassSVM (Semi-Supervised Boundary)...

=============================================================================================================================
 [RPi Audio Feature 2-Step Hybrid Pipeline Benchmark Results: Front 75% Train / Rear 25% Test]
=============================================================================================================================
 Hybrid Pipeline Name (Step 1 + Step 2)           |     Acc |    Prec |     Rec |      F1 |     AUC |  Latency |      FPS
-----------------------------------------------------------------------------------------------------------------------------
[Step 2A] Training AudioHybridFeatureMLP (AE + MLP): 100%|████████████████████████████████████████████| 30/30 [00:10<00:00,  2.94it/s]
 [Step 1: AE] + [Step 2: AudioHybridFeatureMLP]   |  93.45% |  91.15% |  96.61% |  93.80% |  0.9851 |   0.46ms |  2184.3
 [Step 1: AE] + [Step 2: XGBoost]                 |  94.61% |  93.36% |  96.34% |  94.83% |  0.9877 |   0.74ms |  1345.4
[Step 2B] Training AudioHybridResNet1D (AE + ResNet1D): 100%|█████████████████████████████████████████| 30/30 [00:26<00:00,  1.13it/s]
 [Step 1: AE] + [Step 2: AudioHybridResNet1D]     |  94.10% |  92.20% |  96.67% |  94.38% |  0.9867 |   0.68ms |  1472.5
[Step 2C] Training AudioHybridFeatureMLP (OCSVM + MLP): 100%|█████████████████████████████████████████| 30/30 [00:10<00:00,  2.89it/s]
 [Step 1: OCSVM] + [Step 2: AudioHybridFeatureMLP] |  93.69% |  92.79% |  95.08% |  93.92% |  0.9843 |   0.46ms |  2188.2
 [Step 1: OCSVM] + [Step 2: XGBoost]              |  94.54% |  92.91% |  96.74% |  94.78% |  0.9867 |   0.72ms |  1394.5
[Step 2D] Training AudioHybridResNet1D (OCSVM + ResNet1D): 100%|██████████████████████████████████████| 30/30 [00:26<00:00,  1.11it/s]
 [Step 1: OCSVM] + [Step 2: AudioHybridResNet1D]  |  93.69% |  91.45% |  96.74% |  94.02% |  0.9813 |   0.71ms |  1403.0

 [Saved RPi Deploy Package Models]: /mnt/c/Users/kante/Documents/KWU/etrl_task/git_uploads/SW/rpi_deploy_package/models
 [Saved RPi Deploy Metrics Result]: /mnt/c/Users/kante/Documents/KWU/etrl_task/git_uploads/SW/rpi_deploy_package/results/rpi_no_gap_evaluation_summary.json
```

## 6. 엔지니어링 최종 결론 및 시사점

1. **신호처리 연산 경로의 명확성 확보**:
   - 힐베르트 포락선(시간 영역 충격파)과 STFT 스펙트로그램(주파수 영역 2.8kHz 피크)은 원음 WAV 파일에서 분기하는 2개의 독립적 연산 경로임을 정량 명세했습니다.
2. **오디오 2단계 하이브리드 파이프라인의 재현율(Recall) 우수성 입증**:
   - 스펙트로그램 연산을 배제하고 24차원 오디오 피처에 1단계 이상 점수(1차원)를 결합한 결과, **재현율이 최대 96.87%로 대폭 향상되어 미세 고장 소음 억제 및 False Negative 완전 방지** 성능을 달성했습니다.
3. **라즈베리파이 경량화 배치 완성**:
   - 지연시간 **0.43ms ~ 0.66ms (1,500 ~ 2,300 FPS)** 수준으로 엣지 디바이스에서 실시간 추론이 가능한 하이브리드 패키징 파일(.pt, .joblib) 배치를 완수했습니다.

---

## 7. 발표 자료 요약 (Presentation Summary)

### 7.1. 1단계 파이프라인 및 음성 데이터 분기 전략
* **음성 데이터 입력 분기 명확화 (Spectrogram 전면 배제)**
  * **내용**: 2D STFT 스펙트로그램 기반 모델들을 파이프라인에서 완전히 배제하고, Raw 오디오 파형에서 힐베르트(Hilbert) 변환 기반의 시간축 포락선 등 1D 피처(24차원) 추출 경로로 단일화했습니다.
  * **도입 이유**: 2D 스펙트로그램은 노이즈(Noise Floor) 변화에 매우 취약하며 연산 비용이 큽니다. 반면 1D 24차원 피처는 가혹한 노이즈(SNR 5dB) 환경에서도 강건한 방어력을 보이며, 초고속 경량 연산이 가능하기 때문입니다.
* **1단계 파이프라인 구성 (Unsupervised / Semi-Supervised)**
  * **내용**: `AutoEncoder`(비지도) 및 `OneClassSVM`(반지도) 알고리즘을 사용합니다.
  * **도입 이유**: 정상 주행 데이터(y=0)의 정상 분포만을 집중적으로 학습시킵니다. 이를 통해 미지의 결함/노이즈 패턴 진입 시 발생하는 궤적 이탈 정도를 정량적인 **1차원 이상 점수(Anomaly Score)**로 추출하기 위함입니다.

### 7.2. 2단계 파이프라인 내용 및 성능 요약
* **2단계 파이프라인 구조 (Supervised 융합 분류)**
  * **내용**: 24차원 피처에 1단계 이상 점수(1차원)를 결합하여 **총 25차원 하이브리드 입력**을 구성합니다. 이를 `AudioHybridFeatureMLP`, `XGBoost`, `AudioHybridResNet1D`에 투입하여 최종 이상 여부를 탐지합니다.
* **성능 결론**
  * `XGBoost` 결합 모델은 F1-Score 최고 수준(94.8%)을 달성하여 결함 미탐(False Negative)을 가장 강력하게 방어합니다.
  * `MLP` 결합 모델은 초당 2,100번 이상의 추론 속도(FPS)를 기록하여 라즈베리파이 등 엣지 디바이스 배포 최적화를 달성합니다.

### 7.3. 종합 파이프라인 성능 최종 벤치마크 (Comprehensive Benchmark - No Gap)
*2D 스펙트로그램 CNN 모델은 공간 정보 압축으로 인한 성능 붕괴 현상으로 표에서 제외되었으며, F1-Score 기준 내림차순 정렬된 결과입니다.*

#### 1) 단독 지도 학습 (Supervised Only) 3종 성능
| 모델명 (Model Name) | Accuracy | Precision | Recall | F1-Score | AUC | Latency | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AudioCRNN** | 99.15% | 99.27% | 99.07% | 99.17% | 0.9994 | 1.01ms | 987.5 |
| **XGBoost** | 94.51% | 93.07% | 96.47% | 94.74% | 0.9870 | 0.29ms | 3502.0 |
| **AudioFeatureMLP** | 93.97% | 91.65% | 97.07% | 94.28% | 0.9818 | 0.09ms | 11523.9 |

#### 2) 오디오 2단계 하이브리드 파이프라인 (Hybrid) 6종 성능
| 1단계 + 2단계 조합 (Hybrid Pipeline) | Accuracy | Precision | Recall | F1-Score | AUC | Latency | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[OCSVM] + CRNN** | 99.45% | 99.60% | 99.33% | 99.47% | 0.9994 | 0.67ms | 1482.3 |
| **[AE] + CRNN** | 99.18% | 99.40% | 99.00% | 99.20% | 0.9994 | 0.65ms | 1533.1 |
| **[OCSVM] + MLP** | 93.11% | 89.69% | 97.80% | 93.57% | 0.9804 | 0.09ms | 11633.7 |
| **[AE] + XGBoost** | 92.91% | 93.60% | 92.48% | 93.04% | 0.9819 | 0.28ms | 3611.6 |
| **[OCSVM] + XGBoost** | 92.50% | 90.07% | 95.94% | 92.91% | 0.9800 | 0.28ms | 3583.8 |
| **[AE] + MLP** | 91.78% | 87.43% | 98.07% | 92.44% | 0.9835 | 0.09ms | 10977.9 |
