# KRRI_2026 실시간 다중 센서 고장 진단 및 저지연 BLE 스트리밍 시스템

본 저장소는 라즈베리파이 5(Raspberry Pi 5) 환경에서 마이크(Sound) 및 CAN 버스 데이터를 실시간으로 병렬 수집하고, 2단계 하이브리드 AI 모델(XGBoost + GRU)로 고장 상태를 진단한 뒤, **82바이트 실시간 DSP/판정 바이너리 패킷을 BLE GATT로 초저지연 스트리밍**하는 임베디드 진단 시스템입니다.

---

## 1. 전체 파이프라인 아키텍처

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                        [ 송신기 : Raspberry Pi 5 ]                           │
│                                                                              │
│  [ main_ble.py : Python 3.10 (etrl Conda Env) ]                              │
│   ├── 1. 마이크 1초 녹음 + CAN 데이터 병렬 수집 (ThreadPoolExecutor)           │
│   ├── 2. BPF 필터링 및 XGBoost 1차 음향 고장 진단                            │
│   ├── 3. 음향 이상 감지 시 CAN 2차 정밀 추론 (iForest + SingleGRU)            │
│   ├── 4. ble_server.py의 ble_send() 함수 호출                                │
│   └── 5. 백그라운드 BLE 서버(ble_server.py) 자동 기동 및 모니터링              │
│                                │                                             │
│                                ▼                                             │
│  [ ble_server.py : System Python 3.13 (BlueZ D-Bus Daemon) ]                 │
│   ├── 1. ble_send(): 10개 윈도우 RMSE & Bandwidth DSP 연산 후 /ble_shm 기록  │
│   ├── 2. POSIX 공유 메모리 (/ble_shm) 기반 Zero-Copy IPC                     │
│   ├── 3. D-Bus ObjectManager & BlueZ GattManager1 네이티브 GATT 서버 등록    │
│   ├── 4. GATT Service (`20260001-...`) / Characteristic (`20260002-...`)     │
│   └── 5. 공유 메모리 변경 감지 시 GATT `PropertiesChanged` (Notify) 즉시 방출│
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ (BLE 5.0 LE GATT Notifications / Read)
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       [ 수신기 : remote PC / team4 ]                         │
│                                                                              │
│  [ ble_receiver.py : Python Bleak Client ]                                   │
│   ├── 1. BlueZ D-Bus AutoAgent (`NoInputNoOutput`) 등록으로 무인 자동 승인   │
│   ├── 2. 타겟 MAC (`2C:CF:67:63:37:AA`) 캐시 자동 정리 및 GATT 접속           │
│   ├── 3. 타겟 Characteristic 구독 (`start_notify` + Fast Direct Read 폴백)    │
│   └── 4. 수신된 82바이트 실시간 바이너리 언패킹 및 테이블 시각화 출력         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 모듈별 역할 및 파일 구조

| 파일명 | 실행 환경 | 주요 역할 |
| :--- | :--- | :--- |
| **`main.py`** | Conda `etrl` | **원본 메인 파이프라인 (수정 없이 100% 보존)** |
| **`main_ble.py`** | Conda `etrl` | BLE 지원 메인 실행 파일. 센서 수집/추론 후 `ble_send` 호출 및 서버 관리 |
| **`ble_server.py`** | System Python | **DSP 연산 (`ble_send`) 모듈화** + BlueZ 5.82 D-Bus GATT/LE 광고 데몬 |
| **`ble_receiver.py`** | Client PC | 타겟 장치 무인 자동 연결, 82B 바이너리 패킷 파싱 및 실시간 테이블 출력 |
| **`backup_ble_stable/`**| Local / Remote | 송수신 안정화 버전 코드 백업 디렉토리 |

---

## 3. 실시간 BLE 통신 핵심 동작 원리

### ① `ble_send` 모듈화 및 POSIX 공유 메모리 (`/ble_shm`)
* **프로세스 분리 배경**:
  - 인공지능 모델(PyTorch, XGBoost, Scikit-learn) 구동 가상환경(`conda etrl`)과 시스템 D-Bus 라이브러리(`python-dbus`, `PyGObject`) 간의 의존성 충돌 및 GIL 병목을 방지하기 위해 분리 설계되었습니다.
* **전처리 모듈 캡슐화**:
  - 10개 서브윈도우별 **RMSE(Root Mean Square Error)** 및 **중심 주파수 대역폭(Spectral Bandwidth)** 계산 로직이 `ble_server.py`의 `ble_send()` 함수로 캡슐화되어 있어, `main_ble.py`에서는 단순 호출만으로 82B 패킷이 공유 메모리에 원자적(Atomic)으로 기록됩니다.
* **초저지연 데이터 전달**:
  - `ble_server.py`는 GLib 타임아웃(50ms)을 통해 공유 메모리의 변경을 실시간 감지하여 연결된 수신기로 즉각 Notify 시그널을 방출합니다.

---

### ② 82바이트 바이너리 패킷 규격 (`Little-Endian`)

패킷은 1바이트 헤더, 1바이트 고장 판정 코드, 10개의 100ms 서브윈도우별 DSP 연산 결과(각 8바이트)로 구성되어 정확히 82바이트를 유지합니다.

| 오프셋 (Byte) | 필드명 | 데이터 타입 | 설명 |
| :---: | :---: | :---: | :--- |
| `0` | **Header** | `uint8` (`0xAA`) | 패킷 시작 식별자 매직 넘버 |
| `1` | **Status** | `uint8` | `0`: 정상 (NORMAL), `1`: 고장/이상 (ABNORMAL) |
| `2 ~ 9` | **Window #1** | `float32[2]` (`<ff`) | 0~100ms 구간 `[RMSE, Bandwidth(Hz)]` |
| `10 ~ 17` | **Window #2** | `float32[2]` (`<ff`) | 100~200ms 구간 `[RMSE, Bandwidth(Hz)]` |
| `18 ~ 25` | **Window #3** | `float32[2]` (`<ff`) | 200~300ms 구간 `[RMSE, Bandwidth(Hz)]` |
| `26 ~ 33` | **Window #4** | `float32[2]` (`<ff`) | 300~400ms 구간 `[RMSE, Bandwidth(Hz)]` |
| `34 ~ 41` | **Window #5** | `float32[2]` (`<ff`) | 400~500ms 구간 `[RMSE, Bandwidth(Hz)]` |
| `42 ~ 49` | **Window #6** | `float32[2]` (`<ff`) | 500~600ms 구간 `[RMSE, Bandwidth(Hz)]` |
| `50 ~ 57` | **Window #7** | `float32[2]` (`<ff`) | 600~700ms 구간 `[RMSE, Bandwidth(Hz)]` |
| `58 ~ 65` | **Window #8** | `float32[2]` (`<ff`) | 700~800ms 구간 `[RMSE, Bandwidth(Hz)]` |
| `66 ~ 73` | **Window #9** | `float32[2]` (`<ff`) | 800~900ms 구간 `[RMSE, Bandwidth(Hz)]` |
| `74 ~ 81` | **Window #10** | `float32[2]` (`<ff`) | 900~1000ms 구간 `[RMSE, Bandwidth(Hz)]` |

---

### ③ BlueZ D-Bus GATT 서버 (`ble_server.py`)
* **Service UUID**: `20260001-cafe-1234-5678-0123456789ab`
* **Characteristic UUID**: `20260002-cafe-1234-5678-0123456789ab` (속성: `read`, `notify`)
* **BlueZ 5.82 표준 연동**:
  - `org.bluez.GattManager1`을 통해 계층 구조 Application 및 Characteristic을 D-Bus에 등록합니다.
  - `btmgmt`를 통해 Low Energy 및 Advertising 설정을 자동으로 보장합니다.

---

### ④ 무인 자동 승인 수신기 (`ble_receiver.py`)
* **AutoAgent 탑재**: `org.bluez.AgentManager1`에 `NoInputNoOutput` 에이전트를 자동 등록하여 비밀번호/패스키 팝업 없이 연결 즉시 자동 승인합니다.
* **이중 스트리밍 안정화**: GATT Notification 구독을 기본으로 하되, 커널/BlueZ 레벨의 핸들 불일치 시 Direct Read 스트리밍으로 즉각 전환하여 통신 끊김을 완벽하게 방지합니다.

---

## 4. 필수 환경 설정 (트러블슈팅 가이드)

### ⚠️ BlueZ MIDI 충돌 비활성화 (권장)
리눅스 BlueZ 기본 설정에서는 BLE 접속 시 내장 MIDI 프로파일이 활성화되어 `MIDI I/O: Failed to read initial request` 에러와 함께 연결을 끊는 경우가 있습니다. 아래 설정을 적용하면 안정성이 대폭 향상됩니다.

```bash
# systemd override 설정
sudo mkdir -p /etc/systemd/system/bluetooth.service.d
sudo tee /etc/systemd/system/bluetooth.service.d/override.conf << 'EOF'
[Service]
ExecStart=
ExecStart=/usr/libexec/bluetooth/bluetoothd -P midi
EOF

sudo systemctl daemon-reload
sudo systemctl restart bluetooth
```

---

## 5. 실행 및 테스트 방법

### [1] 송신기 실행 (Raspberry Pi 5)
`main_ble.py`를 실행하면 센서 초기화 및 추론 루프가 시작되고, 백그라운드 BLE 서버(`ble_server.py`)가 자동으로 연동됩니다.

```bash
cd /home/pi/Documents/KRRI_2026
conda activate etrl
python main_ble.py
```

### [2] 수신기 실행 (원격 PC / team4)
수신기 PC에서 `ble_receiver.py`를 실행하면 타겟 라즈베리파이를 자동으로 탐색하여 실시간 10개 윈도우 수치 테이블을 스트리밍합니다.

```bash
cd /home/team4/Documents/etrl/src
python3 ble_receiver.py
```

#### 🖥️ 수신기 실시간 출력 화면 예시
```text
========================================================
 [BLE RECV] Status: ABNORMAL (1) | Size: 82B
--------------------------------------------------------
  Win# | Time(ms) |       RMSE       |   Bandwidth(Hz)  
--------------------------------------------------------
  01   |    0~100 |           0.1012 |        6927.6230
  02   |  100~200 |           0.1006 |        6919.8872
  03   |  200~300 |           0.1010 |        6966.8379
  04   |  300~400 |           0.0992 |        6915.6079
  05   |  400~500 |           0.0988 |        6936.4468
  06   |  500~600 |           0.0994 |        6906.1787
  07   |  600~700 |           0.0986 |        6911.3896
  08   |  700~800 |           0.1022 |        6872.8110
  09   |  800~900 |           0.1011 |        6894.1582
  10   |  900~1000 |           0.1013 |        6951.3652
========================================================
```