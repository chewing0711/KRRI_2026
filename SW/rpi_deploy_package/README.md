# 라즈베리 파이(RPi 4/5) 감속기 고장 진단 w34 CAN 전용 배포 패키지
(Raspberry Pi w34 CAN-only Standalone Deployment Package)

---

## 1. 패키지 개요

본 패키지(`rpi_deploy_package/`)는 외부 추가 의존성이나 재학습 없이, **라즈베리 파이 4 / 5 보드에서 즉시 0.1초 윈도우(w34) CAN 전용 모델의 배포 검증, 온라인 스트리밍 시뮬레이션, 및 실시간 차량 SocketCAN 추론 노드를 구동할 수 있는 독립 실행 패키지**입니다.

```
rpi_deploy_package/
├── data/
│   └── unified_can_context_dataset_w34.csv  # w34 CAN 전용 검증 데이터셋
├── models/
│   ├── scaler_can_w34.pkl                   # CAN 43개 피처 정규화 스케일러
│   ├── step1_autoencoder_w34.pt             # Step 1 비지도 AutoEncoder 모델
│   ├── step1_oc_svm_w34.pkl                 # Step 1 One-Class SVM 모델
│   ├── step1_iforest_w34.pkl                # Step 1 Isolation Forest 모델
│   ├── step2_ae_xgboost_can_w34.pkl         # Step 2 AE + XGBoost 모델
│   ├── step2_oc_xgboost_can_w34.pkl         # Step 2 OC-SVM + XGBoost 모델
│   ├── step2_if_xgboost_can_w34.pkl         # Step 2 iForest + XGBoost 모델
│   ├── step2_ae_gru_can_w34.pt              # Step 2 AE + SingleGRU 모델
│   ├── step2_oc_gru_can_w34.pt              # Step 2 OC-SVM + SingleGRU 모델
│   ├── step2_if_gru_can_w34.pt              # Step 2 iForest + SingleGRU 모델
│   └── can_only_manifest_w34.json           # 배포 메타 매니페스트 설정
├── src/
│   ├── sequence_matrix_builder.py           # 10스텝 실시간 링버퍼 및 Matrix 생성기
│   ├── train_unified_models.py              # 모델 구조 정의 정의부
│   └── simulate_online_direct_array.py      # 온라인 다이렉트 NumPy array 시뮬레이터
├── can_parser.py                            # HW CAN 패킷 수집 모듈
├── decoding_functions.py                    # HW CAN 데이터 물리 디코더
├── live_can_inference_node.py               # 실시간 실차/시뮬레이터 추론 노드
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

#### 1. 배포 모델 성능 및 지연시간 즉시 검증
```bash
python evaluate_deployed_models.py
```

#### 2. ZOH 시간축 정렬 병합 시뮬레이터 구동
```bash
python src/simulate_online_inference.py
```

#### 3. 다이렉트 NumPy Array 온라인 추론 시뮬레이터 구동 (ZOH 배제 고효율 버전)
```bash
python src/simulate_online_direct_array.py
```

#### 4. 실시간 CAN 수신 추론 노드 개시 (실시간 SocketCAN 또는 시뮬레이터 연동)
```bash
python live_can_inference_node.py --interface socketcan --channel can0 --speed 20.0 --grade 0.0
```

---

## 3. 라즈베리 파이 전송(배포) 방법

PC에서 라즈베리 파이로 폴더를 전송할 때:
```bash
# SCP 복사 명령 예시 (RPi IP 주소 및 타겟 폴더 입력)
scp -r rpi_deploy_package pi@<라즈베리파이_IP>:~/Documents/etrl
```
