# 라즈베리 파이(RPi 4/5) 감속기 고장 진단 독립 배포 패키지
(Raspberry Pi Standalone Deployment Package)

---

## 1. 패키지 개요

본 패키지(`rpi_deploy_package/`)는 외부 추가 의존성이나 재학습 없이, **라즈베리 파이 4 / 5 보드에서 즉시 2시간 무중단 실시간 진단, INT8 양자화 추론, 및 실시간 발열 감시를 수행할 수 있는 독립 실행 패키지**입니다.

```
rpi_deploy_package/
├── data/
│   └── unified_multimodal_dataset_w250.csv  # 1,617개 윈도우 융합 검증 데이터셋
├── models/
│   ├── scaler_can_w250.pkl                  # CAN 43개 피처 정규화 스케일러
│   ├── scaler_audio_w250.pkl                # Audio 14개 피처 정규화 스케일러
│   ├── step1_autoencoder_int8.pt            # Step 1 비지도 AutoEncoder INT8 양자화 모델
│   ├── step1_autoencoder_w250.pt            # Step 1 비지도 AutoEncoder FP32 원본 모델
│   ├── step1_oc_svm_w250.pkl                # Step 1 One-Class SVM 모델
│   ├── step2_ae_xgboost_multimodal_w250.pkl # Step 2 AE + XGBoost 멀티모달 최종 모델 (F1 100%)
│   ├── step2_gru_multimodal_int8.pt         # Step 2 AE + SingleGRU INT8 양자화 모델 (F1 100%)
│   ├── step2_ae_gru_multimodal_w250.pt      # Step 2 AE + SingleGRU FP32 원본 모델
│   └── step2_oc_xgboost_multimodal_w250.pkl # Step 2 OC-SVM + XGBoost 멀티모달 모델
├── src/
│   ├── sequence_matrix_builder.py           # 5스텝 실시간 링버퍼 및 Matrix 생성기
│   └── train_unified_models.py              # 모델 클래스 정의
├── run_2hours_thermal_stress_test.py        # 2시간(9,796회 추론) 내구 시험 및 써멀 로거
├── thermal_live_dashboard.py                # 4코어 실시간 Heatmap 및 2시간 롤링 UI 대시보드
├── evaluate_deployed_models.py              # 배포 모델 순수 추론 즉시 검증기
├── rpi_quickstart.sh                        # 원클릭 대화형 실행 런처
├── requirements_rpi.txt                     # 라즈베리 파이 파이썬 패키지 목록
└── README.md                                # 본 설명서
```

---

## 2. 라즈베리 파이 원클릭 실행 방법

### [방법 1] 대화형 원클릭 런처 사용 (가장 간단함)
```bash
chmod +x rpi_quickstart.sh
./rpi_quickstart.sh
```

### [방법 2] 개별 명령어 직접 실행

#### 1. 배포 모델 성능 및 지연시간 즉시 검증 (약 3초 소요)
```bash
python evaluate_deployed_models.py
```
* **출력**: 전체 1,617개 윈도우 정확도(F1 100%), 미학습 60kph 일반화(100%), 지연시간(0.27ms, 3,700 FPS).

#### 2. 1분 사전 모의 검증 (Dry-Run)
```bash
python run_2hours_thermal_stress_test.py --duration 60 --interval 1.0
```

#### 3. 2시간 무중단 연속 내구 스트레스 시험 개시 (총 9,796회 실시간 추론)
```bash
python run_2hours_thermal_stress_test.py --duration 7200 --interval 1.0
```
* **결과 저장**: `results/metrics/thermal_profile_2hours.csv`

#### 4. 실시간 Thermal Heatmap 대시보드 UI 실행
```bash
python thermal_live_dashboard.py --refresh 1000
```
* 4개 CPU 코어 2x2 Heatmap 및 2시간 온도/주파수 롤링 그래프를 실시간 모니터링.

---

## 3. 라즈베리 파이 전송(배포) 방법

PC에서 라즈베리 파이로 폴더를 전송할 때:
```bash
# SCP 복사 명령 예시 (RPi IP 주소 입력)
scp -r rpi_deploy_package pi@172.100.7.176:~/Documents/etrl
```
