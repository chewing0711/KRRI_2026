# 파이프라인 아키텍처 및 43개 컬럼 특성 정의서 (Pipeline Architecture & Feature Dictionary)

본 문서는 **기어박스 고장 진단 데이터 처리 파이프라인의 3단계 구축 과정**과 [data/unified_can_context_dataset.csv](../data/unified_can_context_dataset.csv) 내 **43개 전체 컬럼에 대한 1:1 상세 정의 및 물리적 의미**를 기술합니다.

---

## 1. 데이터 처리 파이프라인 3단계 아키텍처 및 주요 파일 바로가기 링크

```mermaid
flowchart TD
    A["Raw CAN 로그 (16개 세션)"] -->|CAN ID 0x386/0x220/0x2B0/0x316/0x153 파싱| B["1단계: 디코딩 CSV (data/*.csv, 16개)"]
    B -->|슬라이딩 윈도우 분할 (500샘플 / 250샘플)| C["2단계: 30차원 CAN 특성 캐시 (data/multimodal_cache_*.csv)"]
    C -->|주행 맥락 + 무차원 슬립 비율 결합| D["3단계: 최종 통합 데이터셋 (data/unified_can_context_dataset_*.csv, 43열)"]
```

### 📂 단계별 핵심 처리 스크립트 및 실측 데이터 파일 바로가기 (마크다운 프리뷰 클릭용)

1. **1단계: Raw CAN 로그 디코딩 (data/raw_decoded/*.csv 16개 생성)**
   - **생성 코드**: [decode_asc_to_official_csv.py](../src/decode_asc_to_official_csv.py)
   - **결과 데이터 폴더 (16개 디코딩 CSV)**: [data/raw_decoded](../data/raw_decoded)
   - **설명**: 16개 세션 폴더의 4대 원본 신호(휠속도 4채널 `0x386`, 횡가속도/요레이트 `0x220`, 조향각/속도 `0x2B0`, 가속/제동페달 `0x316`)를 인위적 선형 보간 없이 Zero-Order Hold(최신 실측 수신값 유지) 방식으로 병합하여 16개 공식 디코딩 CSV로 복원.

2. **2단계: 슬라이딩 윈도우 특성 추출 (30차원 피처 생성)**
   - **생성 코드**: [extract_raw_can_extra_features.py](../src/extract_raw_can_extra_features.py)
   - **변환된 캐시 데이터 CSV**: [multimodal_cache_full.csv](../data/multimodal_cache_full.csv) *(및 `multimodal_cache_w500.csv`, `w250.csv`)*
   - **설명**: 디코딩된 시계열 신호를 슬라이딩 윈도우(기본 500개 및 250개 샘플, CLI 인자로 가변 조절 가능)로 분할하여 휠속 통계(10개), 4륜 상대 속도차(8개), 차체 거동(4개), 조향 시스템(4개), 운전자 페달(4개)의 총 30차원 물리 피처 공간으로 추출하여 캐시 데이터셋으로 수집.

3. **3단계: 주행 맥락 + 무차원 슬립 비율 결합 (최종 통합 CSV 완성)**
   - **생성 코드**: [build_unified_can_context_csv.py](../src/build_unified_can_context_csv.py)
   - **최종 데이터셋 CSV**: [unified_can_context_dataset.csv](../data/unified_can_context_dataset.csv) *(및 `unified_can_context_dataset_w500.csv`, `w250.csv`)*
   - **설명**: 캐시 데이터(30차원 물리 피처)에 차속(`speed_kph`), 경사도(`grade_pct`), 무차원 바퀴 슬립 비율(`slip_ratio_fl~rr` 등 7차원)을 결합하여 총 43개 컬럼의 단일 통합 데이터셋 완성.

---

## 2. 43개 전체 컬럼 명세 및 물리적 의미 사전 (Feature Dictionary)

| 번호 | 컬럼명 (Column Name) | 데이터 타입 | 분류 (Category) | 물리적 의미 및 계산 수식 |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `state` | String | 라벨 | 차량 작동 상태 (`normal`: 정상, `abnormal`: 고장) |
| **2** | `target` | Integer | 라벨 | 이진 분류 타겟 변수 (`0`: 정상, `1`: 기어박스 감속기 고장) |
| **3** | `scenario` | String | 주행 맥락 | 주행 코스 식별자 (`high_speed_20kph`, `hills_12deg`, `steering_pad_40kph` 등) |
| **4** | `speed_kph` | Float | 주행 맥락 | 설정 주행 차속 (단위: km/h, `20.0`, `40.0`, `60.0`, `80.0`) |
| **5** | `grade_pct` | Float | 주행 맥락 | 도로 등판 경사도 각도 (단위: deg, `0.0`, `6.0`, `12.0`, `18.0`, `30.0`) |
| **6** | `window_idx` | Integer | 메타데이터 | 세션 내 윈도우 순차 인덱스 (`0, 1, 2, ...`) |
| **7** | `slip_ratio_fl` | Float | 물리 슬립 | 전좌측(FL) 바퀴 무차원 슬립 비율 ($\text{WHL\_SPD\_FL} / \text{speed\_kph}$) |
| **8** | `slip_ratio_fr` | Float | 물리 슬립 | 전우측(FR) 바퀴 무차원 슬립 비율 ($\text{WHL\_SPD\_FR} / \text{speed\_kph}$) |
| **9** | `slip_ratio_rl` | Float | 물리 슬립 | 후좌측(RL) 바퀴 무차원 슬립 비율 ($\text{WHL\_SPD\_RL} / \text{speed\_kph}$) |
| **10** | `slip_ratio_rr` | Float | 물리 슬립 | 후우측(RR) 바퀴 무차원 슬립 비율 ($\text{WHL\_SPD\_RR} / \text{speed\_kph}$) |
| **11** | `slip_diff_front_rear` | Float | 물리 슬립 | 전륜 대비 후륜 속도차 ($\text{Avg}_{\text{Front}} - \text{Avg}_{\text{Rear}}$) |
| **12** | `norm_yaw_rate` | Float | 관성 정규화 | 속도 정규화 요율 각속도 ($\text{Yaw\_Rate} / \text{speed\_kph}$) |
| **13** | `norm_lat_accel` | Float | 관성 정규화 | 경사도 정규화 횡가속도 ($\text{Lateral\_Accel} / (\text{grade\_pct} + 1.0)$) |
| **14** | `mean` | Float | 휠속 통계 | 윈도우 내 4륜 통합 평균 바퀴 속도 ($\mu$) |
| **15** | `std` | Float | 휠속 통계 | 윈도우 내 바퀴 속도 표준편차 ($\sigma$) |
| **16** | `p2p` | Float | 휠속 통계 | Peak-to-Peak 속도 진폭 ($\max(x) - \min(x)$) |
| **17** | `rms` | Float | 휠속 통계 | 제곱평균제곱근 진폭 ($\sqrt{\frac{1}{N}\sum x^2}$) |
| **18** | `variance` | Float | 휠속 통계 | 윈도우 내 바퀴 속도 분산 ($\sigma^2$) |
| **19** | `peak` | Float | 휠속 통계 | 윈도우 내 최대 절대 속도 피크 수치 ($\max \|x\|$) |
| **20** | `crest_factor` | Float | 파형 인자 | 크레스트 팩터 충격 지수 ($\text{Peak} / \text{RMS}$) |
| **21** | `shape_factor` | Float | 파형 인자 | 셰이프 팩터 파형 형상 지수 ($\text{RMS} / \text{Mean}$) |
| **22** | `skew` | Float | 통계 모멘트 | 신호 분포의 비대칭도 스큐니스 ($S$) |
| **23** | `kurt` | Float | 통계 모멘트 | 신호 첨두 충격도 커토시스 ($K$) |
| **24** | `diff_fl_rl_mean` | Float | 속도차 통계 | 전좌(FL)-후좌(RL) 휠속도 상대 속도차 평균 |
| **25** | `diff_fl_rl_std` | Float | 속도차 통계 | 전좌(FL)-후좌(RL) 상대 속도차 표준편차 |
| **26** | `diff_fl_rl_p2p` | Float | 속도차 통계 | 전좌(FL)-후좌(RL) 상대 속도차 P2P 진폭 |
| **27** | `diff_fr_rr_mean` | Float | 속도차 통계 | 전우(FR)-후우(RR) 휠속도 상대 속도차 평균 |
| **28** | `diff_fr_rr_std` | Float | 속도차 통계 | 전우(FR)-후우(RR) 상대 속도차 표준편차 |
| **29** | `diff_fr_rr_p2p` | Float | 속도차 통계 | 전우(FR)-후우(RR) 상대 속도차 P2P 진폭 |
| **30** | `diff_fl_fr_std` | Float | 속도차 통계 | 전좌(FL)-전우(FR) 좌우 속도차 표준편차 |
| **31** | `diff_rl_rr_std` | Float | 속도차 통계 | 후좌(RL)-후우(RR) 좌우 속도차 표준편차 |
| **32** | `lat_accel_mean` | Float | 차체 거동 | 윈도우 내 횡가속도(Lateral Acceleration) 평균 |
| **33** | `lat_accel_std` | Float | 차체 거동 | 윈도우 내 횡가속도 표준편차 |
| **34** | `yaw_rate_mean` | Float | 차체 거동 | 윈도우 내 회전 요레이트(Yaw Rate) 평균 |
| **35** | `yaw_rate_std` | Float | 차체 거동 | 윈도우 내 회전 요레이트 표준편차 |
| **36** | `sas_angle_mean` | Float | 조향 시스템 | 윈도우 내 조향각(Steering Angle) 평균 |
| **37** | `sas_angle_std` | Float | 조향 시스템 | 윈도우 내 조향각 표준편차 |
| **38** | `sas_angle_p2p` | Float | 조향 시스템 | 윈도우 내 조향각 P2P 진폭 |
| **39** | `sas_speed_mean` | Float | 조향 시스템 | 윈도우 내 핸들 조향속도(Steering Speed) 평균 |
| **40** | `accel_pedal_mean` | Float | 운전자 조작 | 윈도우 내 가속 페달 개도율 평균 (%) |
| **41** | `accel_pedal_max` | Float | 운전자 조작 | 윈도우 내 가속 페달 최대 개도율 (%) |
| **42** | `brake_pedal_mean` | Float | 운전자 조작 | 윈도우 내 제동 브레이크 페달 작동량 평균 |
| **43** | `brake_pedal_max` | Float | 운전자 조작 | 윈도우 내 제동 브레이크 페달 최대 작동량 |
| **44** | `dimless_std` | Float | 무차원 변동률 | 속도 불변 휠속도 표준편차 비율 ($\text{std} / \text{mean}$) |
| **45** | `dimless_p2p` | Float | 무차원 변동률 | 속도 불변 휠속도 P2P 진폭 비율 ($\text{p2p} / \text{mean}$) |
| **46** | `dimless_diff_fr_rr_std` | Float | 무차원 변동률 | 속도 불변 전우-후우 휠속차 변동 비율 ($\text{diff\_fr\_rr\_std} / \text{mean}$) |
| **47** | `pedal_slip_response` | Float | 토크 응답비 | 가속 페달 개도량 대비 구동륜 슬립 변동비 ($\text{diff\_fr\_rr\_std} / \text{accel\_pedal\_mean}$) |
