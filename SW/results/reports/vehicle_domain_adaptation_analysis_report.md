# 🚗 YF 소나타(ICE) vs 포터 EV(전기 트럭) 차종 차이 및 도메인 적응(Domain Adaptation) 분석 보고서

> **분석 목적**: 현재 오프라인 실험 데이터셋(YF 소나타 파워트레인 CAN/오디오)과 향후 실차 테스트 대상 차량(포터 EV 저상형 개조 전기 트럭) 간의 물리적 하드웨어, CAN 프로토콜 및 수집 신호 차이를 정밀 대조하고, 도메인 변경(Domain Transfer) 시 발생할 수 있는 이상 진단 오탐 방지 대책을 수립합니다.

---

## 1. 차종 간 5대 핵심 물리적 / 시스템 하드웨어 차이점 대조표

| 비교 항목 | 현재 오프라인 데이터 (YF 소나타 승용) | 실차 검증 타겟 (포터 EV 전기 트럭) | 진단 모델 및 피처 추출에 미치는 영향 |
| :--- | :--- | :--- | :--- |
| **1. 차량 하중 & 세그먼트** | 공차중량 ~1,400 kg (승용 세단) | **공차중량 ~1,900 kg + 화물 적재 ~1,000 kg (총 2.9톤)** | 하중증가로 인한 타이어 접지 슬립비 변동 및 진동 포락선 진폭 상승 |
| **2. 동력전달계 (Powertrain)** | 2.0L 가솔린/LPI 엔진 + 6단 변속기 | **135 kW 영구자석 모터 + 감속기 (Single Reducer)** | 엔진 폭발 소음 부재 $\to$ 오디오 SNR(신호 대 잡음비) 대폭 향상 |
| **3. 회전수 (RPM) 대역** | 엔진 최대 `6,000 RPM` | **구동 모터 최대 `12,000 ~ 15,000 RPM`** | 기어 맞물림 주파수(GMF)가 고주파 대역으로 2배 이상 상향 이동 |
| **4. CAN ID 및 메세지 스키마** | C-CAN 500kbps (`Engine_RPM`, `Clutch_Tq`) | **EV C-CAN 500kbps (`MCU_Motor_RPM`, `MCU_Motor_Tq`)** | CAN DBC 메세지 ID 및 비트 해상도(Scaling Factor) 불일치 |
| **5. 휠베이스 & 타이어** | 16/17인치 승용 타이어, 휠베이스 2,795mm | **소형 트럭 전륜/후륜 복륜 타이어, 휠베이스 2,640mm** | 차속 대비 휠속 회전 펄스 수 및 4휠 속도 편차 기준 재정의 필요 |

---

## 2. 도메인 변경(Domain Transfer) 시 발생 가능한 리스크

1. **주파수 대역 이탈 (Frequency Shift Risk)**:  
   전기 모터의 고속 회전($12,000\,\text{RPM}$)으로 인해 기어 씹힘 고장 주파수가 기존 $2.8\,\text{kHz}$에서 **$5\,\text{kHz} \sim 8\,\text{kHz}$ 대역으로 고주파 이동**하여 고정 주파수 필터링 적용 시 진단 누락 발생 위험.

2. **적재 하중에 의한 슬립비 오탐 (Load-Induced False Positive Risk)**:  
   트럭에 1톤 화물을 적재하고 급경사로를 오를 때 발생하는 정상적인 휠 슬립을 고장 슬립으로 오인할 위험.

3. **CAN ID 불일치에 의한 피처 추출 오류 (CAN Protocol Mismatch Risk)**:  
   엔진 ECU 메시지(`0x162`)와 전기차 VCU/MCU 메세지(`0x100`대)의 ID 번호가 달라 수신 에러가 발생할 위험.

---

## 3. 도메인 적응(Domain Adaptation) 및 실차 호환 해결 방안

### 🛠️ [솔루션 1] 차속 연동 Order Tracking 기법 기반 피처 무차원화 (Dimensionless Feature)
* **방안**: 고정 주파수($\text{Hz}$) 수치 대신, **모터 1회전 당 진동/소음 주파수를 정규화하는 `Order Tracking` 기법**을 적용.
* **효과**: 모터 RPM이 $6,000\,\text{RPM}$에서 $12,000\,\text{RPM}$으로 올라가더라도, 차속 비례 차수(Order) 피처 수치는 변하지 않아 차종 변경 시에도 동일 모델 재사용 가능.

### 🛠️ [솔루션 2] CAN 프로토콜 추상화 레이어 (CAN Abstraction Wrapper) 구축
* **방안**: 알고리즘 입력단에 표준화 인터페이스(`Standard_CAN_Input`)를 두고, 차량별 CAN DBC 파서가 소나타(`0x162`)와 포터 EV(`MCU_0x100`) 메세지를 동일한 표준 파라미터(`Motor_RPM`, `Driving_Torque`, `Wheel_Speed`)로 자동 변환하여 입력.

### 🛠️ [솔루션 3] 토크(Torque) 수치 기반 동적 임계값 (Dynamic Scaling Threshold)
* **방안**: 적재량에 따른 동적 하중 변화를 보정하기 위해, CAN으로 입력되는 **모터 토크(`Driving_Torque`) 수치에 비례하여 이상 판정 임계값을 실시간 스케일링**하는 동적 임계값 수식 적용.
  $$\text{Dynamic\_Threshold} = \text{Base\_Threshold} \times \left( 1.0 + \alpha \cdot \frac{\text{Current\_Torque}}{\text{Max\_Torque}} \right)$$
