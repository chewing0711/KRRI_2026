# Gearbox Final Suite Source Code Structure (`src/`)

`experiments/gearbox_final_suite/src/` 폴더 내 27개 스크립트의 기능별 4대 서브폴더 구조화 정리 명세서입니다.

---

## 📂 1. `src/data_processing/` (데이터 전처리, 검증, 변환)

* [`build_unified_can_context_csv.py`](./data_processing/build_unified_can_context_csv.py): 49개 피처 UNIFIED CSV 데이터셋 구축
* [`build_unified_can_context_with_waveforms.py`](./data_processing/build_unified_can_context_with_waveforms.py): 파형 특성 통합 구축
* [`export_pkl_to_csv_json.py`](./data_processing/export_pkl_to_csv_json.py): PKL 캐시 데이터의 CSV/JSON 변환 덤프
* [`export_raw_feature_matrices.py`](./data_processing/export_raw_feature_matrices.py): 45개 특성 $\times$ 10개 윈도우 시계열 Matrix 추출
* [`verify_zero_duplication.py`](./data_processing/verify_zero_duplication.py): 실수 부동소수점 오차 기반 중복 행 0개 검증
* [`verify_collection_time_match.py`](./data_processing/verify_collection_time_match.py): 원본 CAN CSV와 Unified 간 1:1 수집 시간 검증
* [`generate_raw_to_unified_verification.py`](./data_processing/generate_raw_to_unified_verification.py): 500개 RAW 행 $\rightarrow$ 1행 수치 대조 보고서 생성

---

## 📂 2. `src/models_pipelines/` (모델 학습, 이상 탐지, 벤치마크)

* [`hybrid_anomaly_diagnosis_pipeline.py`](./models_pipelines/hybrid_anomaly_diagnosis_pipeline.py): 2단계 융합 파이프라인 (Step 1 이상 탐지 + Step 2 정밀 진단)
* [`run_master_benchmark.py`](./models_pipelines/run_master_benchmark.py): 모든 모델, Stride, FPS, Latency, 정확도 종합 마스터 벤치마크
* [`run_stride_ablation_rnn.py`](./models_pipelines/run_stride_ablation_rnn.py): PyTorch LSTM/RNN Stride Ablation (500, 250, 100) 실험
* [`compare_step1_models.py`](./models_pipelines/compare_step1_models.py): 1단계 이상 탐지 4개 모델 (IsolationForest, OCSVM, RNN, LSTM) 대조 비교
* [`realtime_rnn_window_streamer.py`](./models_pipelines/realtime_rnn_window_streamer.py): 윈도우 Iteration별 실시간 추론 확률 스트리머
* [`evaluate_unified_can_context.py`](./models_pipelines/evaluate_unified_can_context.py): 통합 데이터셋 단독 모델 성능 평가

---

## 📂 3. `src/visualization/` (시각화 및 다이어그램)

* [`generate_window_concept_matplotlib.py`](./visualization/generate_window_concept_matplotlib.py): 500개 RAW 행 $\rightarrow$ 1행 슬라이싱 표 다이어그램
* [`generate_2d_linechart_before_after.py`](./visualization/generate_2d_linechart_before_after.py): 전처리 전/후 2D 파형 비교 차트
* [`generate_3d_matrix_surface_graph.py`](./visualization/generate_3d_matrix_surface_graph.py): 3D 특성 매트릭스 서피스 시각화
* [`generate_before_after_3d_comparison.py`](./visualization/generate_before_after_3d_comparison.py): 3D 분포 공간 비교 차트

---

## 📂 4. `src/archive/` (보조 및 기타 분석 스크립트)

* `ablation_window_stride_rnn.py`
* `diagnose_loso_slip.py`
* `improve_mode4_mode5.py`
