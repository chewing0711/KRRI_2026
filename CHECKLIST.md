# 📋 KRRI_2026 협업 가이드 및 함수별 기능 명세 체크리스트

본 문서는 팀원 간 원활한 협업과 쉬운 코드 이해를 위해 **누구나 알 수 있는 쉬운 용어 표준안** 및 **모듈/함수별 상세 기능 주석 정의**를 정리한 체크리스트입니다.

---

## 1. 협업을 위한 용어 및 함수명 표준안

어려운 시스템/블루투스 전문 용어(BlueZ, D-Bus, GATT, Handle 등)를 모두 배제하고 직관적인 일반 용어로 통일합니다.

### 📌 [용어 표준 매핑 테이블]

| 어려운 전문 용어 | 누구나 알 수 있는 쉬운 용어 | 직관적 의미 및 설명 |
| :--- | :--- | :--- |
| **`BlueZ / D-Bus`** | **무선 통신 시스템** | 리눅스 시스템 내장 기본 무선 드라이버 |
| **`Notify / Notification`** | **실시간 전송 (Stream / Push)** | 데이터를 1초마다 수신기로 자동 전달하는 방식 |
| **`GATT Server / BLE Server`** | **송신기 (Sender)** | 라즈베리파이 측 데이터 송출 모듈 |
| **`BLE Client / Bleak`** | **수신기 (Receiver)** | 노트북/원격 PC 측 데이터 수신 모듈 |
| **`Characteristic / UUID`** | **데이터 채널 (Data Channel)** | AI 진단 결과 및 DSP 수치가 전달되는 통로 |
| **`Advertising`** | **기기 탐색 신호 (Beacon)** | 수신기가 라즈베리파이를 찾을 수 있게 보내는 신호 |
| **`AutoAgent / Pairing`** | **무인 자동 연결 (Auto Connect)** | 비밀번호 확인 팝업 없이 0초 만에 무인 자동 접속 |

---

### 📌 [함수명 표준안]

#### ① 송신측 (`main_ble.py`, `ble_server.py`)
- [ ] **`ble_send()`** $\longrightarrow$ **`ble_send()` (기존 이름 100% 유지)** : 1초 오디오 DSP 연산 및 데이터 전송
- [ ] `run_ble_server()` $\longrightarrow$ **`start_server()`** : 무선 송신 서버 백그라운드 구동
- [ ] `check_shared_memory()` $\longrightarrow$ **`sync_memory()`** : 공유 메모리 변경 감지 및 즉시 실시간 전송
- [ ] `Characteristic.StartNotify()` $\longrightarrow$ **`start_streaming()`** : 실시간 스트리밍 시작
- [ ] `Characteristic.StopNotify()` $\longrightarrow$ **`stop_streaming()`** : 실시간 스트리밍 중단

#### ② 수신측 (`ble_receiver.py`)
- [ ] `gatt_notification_handler()` $\longrightarrow$ **`on_receive()`** : 데이터 수신 콜백 및 시간/상태/dB 출력
- [ ] `setup_bluez_agent()` $\longrightarrow$ **`enable_auto_connect()`** : 무인 자동 접속 모드 켜기
- [ ] `clean_bluetooth_environment()` $\longrightarrow$ **`reset_connection()`** : 연결 상태 초기화

---

## 2. 각 함수별 기능 상세 주석 정의서

각 파일에 작성될 핵심 함수들의 상세 기능 및 역할 주석입니다.

### ① `main_ble.py` (메인 실행 파이프라인)

* **`main()`**
  * **기능**: 
    1. 무선 통신 서비스 초기화
    2. 164바이트 공유 메모리(`/ble_shm`) 할당
    3. `ble_server.py`의 `load_all_wav_files()`를 통해 WAV 음원 메모리(RAM) 일괄 적재
    4. 송신 서버(`ble_server.py`) 자동 실행
    5. 실제 WAV 음원을 1초 단위로 슬라이싱하여 사전 학습된 AI 모델(XGBoost) 추론 수행
    6. AI 판정 결과 및 1초 음향 프레임을 `ble_send()`를 통해 공유 메모리로 전송

---

### ② `ble_server.py` (DSP 연산 및 무선 송신 데몬)

* **`load_all_wav_files(wav_dir)`**
  * **기능**: 디렉토리 내 모든 WAV 음원(`record_*.wav`)을 읽어 48,000Hz 1채널 정규화 넘파이 배열로 메모리에 일괄 적재하여 파일 전환 지연을 0초로 유지
  * **입력**: 음원 디렉토리 경로 (`wav_dir`)
  * **출력**: `[(파일명, 오디오배열, 샘플레이트, 재생시간), ...]` 리스트

* **`ble_send(predict_result, sound_1s, shm, sample_rate=48000)`** *(함수명 유지)*
  * **기능**: 
    1. 1초 오디오 신호를 20개 서브윈도우(50ms 단위)로 분할
    2. 각 윈도우별 실효값(RMSE) 및 주파수 대역폭(Bandwidth) 고속 연산
    3. 헤더(`0xAA`) + 고장 상태(`0/1`) + 20개 윈도우 수치(총 164B)로 패킹
    4. 공유 메모리(`/ble_shm`)에 기록

* **`Characteristic.ReadValue(options)`**
  * **기능**: 수신기가 데이터를 직접 읽어갈 때 공유 메모리의 최신 164B 반환

* **`sync_memory()`**
  * **기능**: 50ms 주기로 공유 메모리를 감시하여 새 데이터가 쓰이면 수신기에 실시간 자동 전송

* **`start_server()`**
  * **기능**: 
    1. 와이파이 간섭 차단 및 무선 통신 우선권 설정
    2. 시스템 드라이버에 송신 서버 등록
    3. 기기 탐색 신호 활성화 및 무선 송신 루프 실행

---

### ③ `ble_receiver.py` (수신기 클라이언트)

* **`AutoAgent` / `enable_auto_connect()`**
  * **기능**: 비밀번호 확인 팝업을 무시하고 전원만 켜지면 0초 만에 무인으로 즉시 자동 접속 승인

* **`on_receive(sender, data)`**
  * **기능**: 
    1. 164바이트 패킷 유효성 검증 (헤더 `0xAA` 확인)
    2. AI 판정 상태(`NORMAL` / `ABNORMAL`) 파싱
    3. 20개 윈도우 RMSE 평균값을 계산하여 데시벨(dB)로 환산
    4. `[YYYY-MM-DD HH:MM:SS] Status: <상태> | dB: <수치> dB` 형식으로 화면 출력

* **`reset_connection()`**
  * **기능**: 이전 연결의 잔여 상태를 초기화하여 재접속 시 연결 잠김 방지

* **`main()`**
  * **기능**: 라즈베리파이를 탐색하여 자동 연결 후 실시간 스트리밍 루프 유지 (연결 단절 시 0.2초 내 즉시 자동 복구)

---

## 3. 164바이트 패킷 구조 명세서

| 오프셋 (Byte) | 필드명 | 데이터 타입 | 설명 |
| :---: | :---: | :---: | :--- |
| **`0`** | **Header** | `uint8` (`0xAA`) | 패킷 시작 식별자 매직 넘버 |
| **`1`** | **Status** | `uint8` | `0`: 정상 (NORMAL), `1`: 고장/이상 (ABNORMAL) |
| **`2 ~ 9`** | **Subwindow #01** | `float32[2]` (`<ff`) | 0~50ms 구간 `[RMSE, Bandwidth(Hz)]` |
| **`10 ~ 17`** | **Subwindow #02** | `float32[2]` (`<ff`) | 50~100ms 구간 `[RMSE, Bandwidth(Hz)]` |
| **`...`** | **Subwindow #03 ~ #19** | `float32[2]` (`<ff`) | 50ms 단위 연속 서브윈도우 수치 |
| **`154 ~ 161`** | **Subwindow #20** | `float32[2]` (`<ff`) | 950~1000ms 구간 `[RMSE, Bandwidth(Hz)]` |
| **`162 ~ 163`** | **Padding** | `uint8[2]` (`0x00`) | 164바이트 4바이트 정렬 고정 패딩 |
