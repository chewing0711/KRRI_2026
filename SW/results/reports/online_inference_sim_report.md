# w34 고장 진단 온라인 추론 시뮬레이션 결과 보고서

본 보고서는 윈도우 크기 34샘플(0.10초) 및 시퀀스 길이 10스텝으로 모델 학습 설정을 수정한 내용과, 이를 HW 온라인 환경에 맞추어 ZOH DataFrame 병합 오버헤드 없이 NumPy Array로 다이렉트 변환하여 추론을 수행한 시뮬레이션 설계 및 결과 요약입니다.

---

## 1. 모델 학습용 CSV 제작 (w34 변환)
* **기존 사양**: 0.74초 윈도우 크기 (250개 샘플)
* **변경 사양**: 0.10초 윈도우 크기 (34개 샘플)
* **조치 내용**: 16개 주행 시나리오의 디코딩 완료 시계열 파일로부터 34샘플 단위로 43개 CAN 특징점을 비중첩 추출하여 최종 학습용 데이터셋인 [unified_can_context_dataset_w34.csv](../../data/unified_can_context_dataset_w34.csv)를 제작 완료했습니다.
* **관련 소스 코드**: [build_unified_can_context_csv.py](../../src/data_processing/build_unified_can_context_csv.py)

---

## 2. 3D Matrix 및 2D Lag Matrix 변환
* **조치 내용**: 추출된 43개 CAN 피처 행렬을 1단계 비지도 이상치 스코어(1차원)와 결합하여 44차원 피처를 생성한 후, 시계열 특징을 학습할 수 있도록 아래와 같이 행렬 형상을 제어기 친화적으로 변환합니다.
  - **SingleGRU 모델**: 10스텝 시퀀스를 결합하여 3D Tensor `[N_seq, 10, 44]` 형태로 입력
  - **XGBoost 모델**: 10스텝을 직렬 롤링(Flatten)하여 2D Lag Matrix `[N_seq, 440]` 형태로 입력
* **관련 소스 코드**: [sequence_matrix_builder.py](../../src/data_processing/sequence_matrix_builder.py)

---

## 3. 학습 및 검증 주기 축소 (0.74초 x 5스텝 → 0.10초 x 10스텝)
* **조치 내용**: 2단계 하이브리드 고장 진단 메인 루프인 [train_unified_models.py](../../src/models_pipelines/train_unified_models.py)에서 검증 주기를 w34(0.10초), 시퀀스 10스텝으로 변경했습니다. 5-Fold 교차 검증 평가 후 최종 성능이 우수했던 상위 3개 조합(AE+GRU, OC-SVM+GRU, iForest+GRU)의 모델 가중치와 스케일러를 `models/` 디렉토리에 내보내기(Export) 완료했습니다.

---

## 4. HW 코드 연동 시뮬레이션 (ZOH Time-Series 병합 버전)
* **조치 내용**: 실제 `gearbox_fault_diagnosis_program/data/` 아래에 있는 각 시나리오의 4개 raw CSV 파일들(`wheel speed.csv`, `yaw_rat.csv`, `steer angle_speed.csv`, `accel_brake.csv`)을 개별 로드하여, 오프라인 전처리 방식과 동일하게 타임스탬프 기준으로 ZOH 병합한 뒤 34샘플씩 스트리밍하여 판정하는 시뮬레이터를 구현했습니다.
* **관련 소스 코드**: [simulate_online_inference.py](../../src/models_pipelines/simulate_online_inference.py)

---

## 5. ZOH 병합 배제 및 다이렉트 NumPy Array 추론 시뮬레이션
* **조치 내용**: HW 코드의 `can_parser.collect_data`가 1.0초 단위(`results_1s`)로 데이터를 반환하는 특징에 맞추어, **1.0초 범위의 데이터를 0.1초 시간 구간 10개로 분할**하고, 각 채널의 데이터를 딕셔너리로부터 직접 `mean()`, `std()` 처리하여 NumPy Array로 변환 후 모델에 입력하는 ZOH 병합 배제 버전의 시뮬레이터를 독립 구현했습니다.
* **관련 소스 코드**: [simulate_online_direct_array.py](../../src/models_pipelines/simulate_online_direct_array.py)
* **주요 결과 요약 (16개 시나리오 순차 실행)**:
  - **정상 주행 시나리오 (Normal)**: 8개 세션 전체 정상 진단 완료 (Pass)
    - 고속주회로 20/60/80kph, 등판로 6/12/18/30도, 원선회로 40kph 모두 정상 판정.
    - 단, 등판로 12/30도 정상 시나리오에서는 경사 하중에 따른 휠 슬립으로 인해 `iForest_GRU`가 일시적으로 약 20%~25% 수준의 고장 판정 펄스를 생성했으나 다수결로 최종 정상 판정함.
  - **이상 주행 시나리오 (Abnormal)**: 8개 세션 전체 이상 진단 완료 (Pass)
    - 8개 고장 주행 세션 모두 윈도우별 판정 비율 **95% 이상**으로 확실하게 기어박스 고장을 진단해 냄.
