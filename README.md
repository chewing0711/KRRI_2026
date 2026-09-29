# 블루투스 통신 아키텍쳐

---

## 1. 시스템 아키텍처

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                        [ 송신기 : Raspberry Pi 5 ]                           │
│                                                                              │
│  [ main.py : AI 진단 파이프라인 (Conda etrl) ]                               │
│   ├── 1. 마이크 1초 음원 수집 (미연결 시 WAV 자동 시뮬레이션)                 │
│   ├── 2. BPF 필터링(1.8k~10.5kHz) 및 XGBoost 음향 고장 예측                  │
│   └── 3. ble_send(): 20개 서브구간 DSP 연산 후 공유메모리(/ble_shm) 기록     │
│                                │                                             │
│                                ▼                                             │
│  [ ble_server.py : GATT 서버 데몬 (BlueZ D-Bus) ]                            │
│   ├── 1. POSIX 공유 메모리 (/ble_shm, 164B) 기반 Zero-Copy IPC               │
│   ├── 2. GATT Service (`20260001-...`) / Characteristic (`20260002-...`)     │
│   └── 3. 공유 메모리 변경 감지 시 GATT Notification 즉시 방출                │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ (BLE 5.0 LE GATT Notifications)
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       [ 수신기 : Remote PC / team4 ]                         │
│                                                                              │
│  [ ble_receiver.py : Bleak Client ]                                          │
│   ├── 1. BlueZ AutoAgent 무인 자동 승인 (0초 연결)                           │
│   ├── 2. 타겟 MAC (`2C:CF:67:63:37:AA`) 탐색 및 GATT 자동 접속               │
│   └── 3. 164B 패킷 파싱 및 실시간 상태/데시벨(dB) 출력                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 164바이트 바이너리 패킷 규격 (`Little-Endian`)

패킷은 1바이트 헤더, 1바이트 상태 코드, 20개의 50ms 서브윈도우별 DSP 연산 결과(각 8바이트), 2바이트 시퀀스 번호/패딩으로 구성됩니다.

| 오프셋 (Byte) | 필드명 | 데이터 타입 | 설명 |
| :---: | :---: | :---: | :--- |
| **`0`** | **Header** | `uint8` (`0xAA`) | 패킷 시작 식별자 매직 넘버 |
| **`1`** | **Status** | `uint8` | `0`: 정상 (NORMAL), `1`: 고장/이상 (ABNORMAL) |
| **`2 ~ 9`** | **Subwindow #01** | `float32[2]` (`<ff`) | 0~50ms 구간 `[RMSE, Bandwidth(Hz)]` |
| **`10 ~ 17`** | **Subwindow #02** | `float32[2]` (`<ff`) | 50~100ms 구간 `[RMSE, Bandwidth(Hz)]` |
| **`...`** | **Subwindow #03 ~ #19** | `float32[2]` (`<ff`) | 50ms 단위 연속 서브구간 DSP 수치 |
| **`154 ~ 161`** | **Subwindow #20** | `float32[2]` (`<ff`) | 950~1000ms 구간 `[RMSE, Bandwidth(Hz)]` |
| **`162 ~ 163`** | **Seq / Padding** | `uint16` (`<H`) | 시퀀스 번호 및 패딩 |

---

## 3. 통신 규격 및 UUID

* **Service UUID**: `20260001-cafe-1234-5678-0123456789ab`
* **Characteristic UUID**: `20260002-cafe-1234-5678-0123456789ab` (속성: `read`, `notify`)
* **타겟 송신기 MAC**: `2C:CF:67:63:37:AA`

---

## 4. 실행 방법

### ① 송신기 실행 (라즈베리파이)

```bash
cd /home/pi/Documents/KRRI_2026
conda activate etrl

# 1. 블루투스 GATT 서버 실행 (별도 터미널 또는 백그라운드)
python ble_server.py

# 2. 메인 AI 진단 실행
python main.py
```

### ② 수신기 실행 (원격 PC)

```bash
cd /home/team4/Documents/etrl/src
python ble_receiver.py
```

---

## 5. 실시간 출력 화면

### 🖥️ 송신기 콘솔 출력 (`main.py`)
```text
Sound 판단: 0 (Normal)
Sound 판단: 0 (Normal)
Sound 판단: 1 (Abnormal)
```

### 🖥️ 수신기 콘솔 출력 (`ble_receiver.py`)
```text
2026-09-29 11:28:50 Status: NORMAL (0) | dB: 43.1 dB
2026-09-29 11:28:51 Status: NORMAL (0) | dB: 44.3 dB
2026-09-29 11:28:52 Status: ABNORMAL (1) | dB: 52.8 dB
```