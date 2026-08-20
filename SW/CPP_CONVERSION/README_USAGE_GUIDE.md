# 라즈베리 파이 배포 및 2시간 연속 내구 시험 사용 가이드
(Raspberry Pi Deployment & 2-Hour Stress Test User Guide)

---

## 1. 개요 및 패키지 구성

본 디렉토리(`CPP_CONVERSION/`)는 감속기 기어박스 2단계 하이브리드 고장 진단 시스템을 **라즈베리 파이(RPi 4/5, ARM Cortex-A72/A76)**에서 **2시간 연속 무중단 실시간 진단 및 발열(Thermal Throttling) 감시**를 수행하기 위한 전용 실행 패키지입니다.

```
CPP_CONVERSION/
├── export_quantized_int8_models.py       # [1단계] PyTorch Dynamic INT8 양자화 및 모델 추출 스크립트
├── run_2hours_thermal_stress_test.py     # [2단계] 2시간(9,796회) 순환 스트리밍 내구 시험 & 써멀 로거
├── thermal_live_dashboard.py             # [3단계] 4코어 실시간 Heatmap & 2시간 온도 롤링 UI 대시보드
├── gearbox_realtime_diagnosis_engine.cpp # [선택] Pure C++17 고성능 실시간 진단 엔진 소스코드
├── Makefile                              # [선택] C++ 엔진 ARM Cortex-A72 최적화 빌드 스크립트
├── embedded_optimization_and_thermal_engineering_guide.md # 상세 엔지니어링 기술 이론 및 학술 출처 보고서
└── README_USAGE_GUIDE.md                 # 본 사용자 매뉴얼
```

---

## 2. 단계별 실행 방법 (Step-by-Step Guide)

### [Step 1] INT8 양자화 모델 추출 및 C++ 포맷 변환

학습 완료된 원본 모델 가중치를 RPi ARM NEON SIMD 연산에 최적화된 **INT8 동적 양자화 모델(`.pt`) 및 XGBoost 모델(`.json`)로 변환**합니다.

```bash
cd /mnt/c/Users/kante/Documents/KWU/etrl_task/experiments/gearbox_final_suite/CPP_CONVERSION

python export_quantized_int8_models.py
```

* **출력 생성 파일**:
  * `step1_autoencoder_int8.pt`: Step 1 비지도 AutoEncoder INT8 가중치
  * `step2_gru_int8.pt`: Step 2 시계열 SingleGRU INT8 가중치
  * `step2_xgboost_model.json`: Step 2 C++ 호환 XGBoost 트리 모델
  * `rpi_deployment_manifest.json`: 하드웨어 배포 메타 매니페스트

---

### [Step 2] 2시간 연속 내구 스트레스 시험 실행

보유한 1,617개 윈도우를 **총 6.06회 순환 스트리밍(총 9,796회 실시간 추론)**하며, **0.735초 주기 Idle Sleep(발열 억제)** 및 **1초 단위 써멀 로깅**을 수행합니다.

#### 1. 사전 1분 모의 검증 (Dry-Run Test)
시스템 동작과 로깅 무결성을 60초 동안 빠르게 사전 점검합니다:
```bash
python run_2hours_thermal_stress_test.py --duration 60
```

#### 2. 2시간(7,200초) 본 시험 가동 (Full 2-Hour Stress Test)
실제 2시간 동안 무중단 연속 내구 시험을 개시합니다:
```bash
python run_2hours_thermal_stress_test.py --duration 7200 --interval 1.0
```

* **실시간 터미널 출력 항목**:
  * `Elapsed`: 경과 시간 (예: `01h 15m 30s`)
  * `Inference`: 현재 누적 추론 횟수 / 목표 횟수 (`6,120 / 9,796`)
  * `Temp`: CPU 다이 온도 (`46.2 C`)
  * `CPU Freq`: 동작 클록 주파수 (`1.50 GHz`)
  * `CPU Usage`: 총 CPU 점유율 (`0.3 %`)
  * `Latency`: 1회 진단 연산 지연시간 (`0.858 ms`)
  * `Status`: 정상 상태 (`NORMAL`) 또는 쓰로틀링 페일세이프 (`FALLBACK`)
* **누적 로그 파일**: `results/metrics/thermal_profile_2hours.csv`

---

### [Step 3] 실시간 Thermal Heatmap & 모니터링 UI 대시보드 실행

시험이 진행되는 동안 별도 터미널 또는 동일 화면에서 **4개 CPU 코어 Heatmap 및 2시간 시계열 추이 그래프를 실시간 렌더링**합니다.

```bash
python thermal_live_dashboard.py --refresh 1000
```

* **대시보드 4대 화면 구성**:
  1. **Top-Left (2x2 Core Heatmap)**: Core 0 ~ Core 3 개별 실시간 로드(%) 및 평균 온도 표출.
  2. **Top-Right (Temperature Series)**: 실시간 CPU 온도(°C)와 80°C 쓰로틀링 한계선 대조.
  3. **Bottom-Left (Frequency & Usage)**: DVFS 클록 주파수(GHz) 및 전체 CPU 점유율(%).
  4. **Bottom-Right (Health Gauge)**: 실시간 고장 진단 결과(`NORMAL` / `ABNORMAL`) 및 Latency(ms).

---

### [Step 4] [선택 사항] Pure C++ Standalone 네이티브 엔진 빌드

파이썬 런타임 없이 C++ 단독 바이너리로 구동할 경우:

```bash
# 1. ARM Cortex-A72 최적화 C++ 컴파일
make

# 2. 2시간(7,200초) C++ 네이티브 진단 엔진 실행
./gearbox_realtime_diagnosis_engine 7200
```

---

## 3. 라즈베리 파이 실차 탑재 시 운영 팁

1. **방열 대책**:
   * 기본 알루미늄 방열판을 장착하면 0.74초 주기 Idle Sleep 설계 덕분에 온도가 **45 ~ 50도 이내로 유지**됩니다.
2. **Headless CLI 구동**:
   * 실차 탑재 시 불필요한 GUI 렌더링 발열을 막기 위해 리눅스 콘솔(CLI) 모드로 구동을 권장합니다.
3. **권한 확인**:
   * 리눅스 커널 써멀 파일(`/sys/class/thermal/thermal_zone0/temp`)은 기본 읽기 권한(`r--r--r--`)이 부여되어 있어 일반 사용자 계정으로 즉시 계측 가능합니다.
