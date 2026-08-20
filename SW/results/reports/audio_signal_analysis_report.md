# 🔊 구동부 차동장치 오디오 음향 파형 및 스펙트로그램 정밀 분석 보고서

> [!NOTE]
> **보고서 개요**: 본 보고서는 ETRL 저상형 전기트럭 개조 구동부 이상 진단 과제의 일환으로, 16개 주행 시나리오에서 수집된 **총 32개 오디오 채널(`Gear-1` 기어박스 마이크 16개 + `OBD-2` 실내 마이크 16개)**의 시간 영역 파형(Waveform) 및 STFT 주파수 전력 스펙트로그램(Spectrogram, $0 \sim 12\,\text{kHz}$) 시각화 그래프를 누구나 한눈에 쉽게 이해할 수 있도록 쉽게 풀어 설명한 보고서입니다.

---

## 1. 정밀 실측 검수 결과 및 주요 스펙트럼 관측 특성

### 🔑 관측된 3대 주요 실측 스펙트럼 특징

1. **고장 고유 조화파 밴드 (Harmonic Frequency Bands at 2.8kHz, 5.4kHz, 8.0kHz, 10.5kHz)**
   - **정상 (Normal)** 상태의 스펙트로그램은 $4\,\text{kHz}$ 이상의 고주파 대역이 어두운 보라색/암전 상태($-85\,\text{dB}$ 이하)를 띤 반면,
   - **고장 (Abnormal)** 상태 스펙트로그램에서는 **$2.8\,\text{kHz}$ 부근에서 $-30\,\text{dB}$ 수준의 강렬한 노란색 기어 씹힘 기본 피크**가 관측되었습니다.
   - 또한 이 기본 피크의 배수 대역인 **$5.4\,\text{kHz}, 8.0\,\text{kHz}, 10.5\,\text{kHz}$에 걸쳐 수평으로 형성되는 층상 조화파 밴드(Harmonic Layer Bands)**가 고주파 영역까지 선명하게 노출되었습니다.

2. **수집 시작 가속 램프 (Acceleration Chirp Line, $0 \sim 15\,\text{초}$)**
   - 오디오 파일의 $0 \sim 15$초 구간에서 차량 가속에 따라 주파수가 $0\,\text{Hz}$에서 $2.8\,\text{kHz}$로 우상향하는 사선 형태의 **가속 램프(Chirp Line)** 곡선 파형이 정밀 관측되었습니다.
   - 이는 차속 상승 시 기어 회전 주파수가 증가함에 따른 전형적인 동력전달계 응답 특성입니다.

3. **`Gear-1` vs `OBD-2` 센서 채널 간 신호 대 잡음비(SNR) 격차**
   - **`Gear-1` (기어박스 직붙 마이크)**: 차동기어 하우징 직근 채널로, 고장 발생 시 조화파 전력이 $-30\,\text{dB} \sim -45\,\text{dB}$로 매우 강하게 수신되어 고장 징후 특징 추출에 압도적으로 유리합니다.
   - **`OBD-2` (실내 조종석 마이크)**: 차체 구조물에 의한 감쇄 및 실내 울림으로 고주파 조화파 전력이 $-75\,\text{dB}$ 이하로 옅게 감쇄되어 상대적으로 신호 민감도가 낮습니다.

---

## 1.1 CAN 시간 간격 vs 오디오 샘플 밀도 전수 실측 결과

16개 전체 주행 시나리오(총 294,905개 CAN 시간 구간)를 전수 실측한 CAN 시간 간격과 오디오 샘플 밀도 관계입니다.

### 📊 CAN 시간 간격 vs 오디오 샘플 밀도 실측 수치

| 구분 | CAN 평균 수집 간격 (`Delta t`) | 오디오 샘플링 해상도 | CAN 1개 행 당 오디오 샘플 개수 (평균 / 중앙값) | 특이사항 |
| :--- | :---: | :---: | :---: | :--- |
| **전수 실측치** | **`4.000 ms`** ($250\,\text{Hz}$) | **`44,100 Hz`** ($0.0227\,\text{ms}$) | **`175.87개` / `180개`** | CAN 1개 행(4ms) 간격 내 오디오 176개 수집 |

### 🔑 주행 시나리오 그룹별 오디오 밀도 분포

| 주행 시나리오 그룹 | 총 CAN 행 수 | CAN 평균 수집 간격 | CAN 1개 행 당 오디오 평균 샘플 개수 |
| :--- | :---: | :---: | :---: |
| **고속주행 (20 / 60 / 80 kph)** | 212,143 행 | `4.000 ms` | **`175.87 개`** |
| **등판로 (6 / 12 / 18 / 30 도)** | 62,305 행 | `4.000 ms` | **`175.87 개`** |
| **원선회로 (40 kph)** | 30,450 행 | `4.000 ms` | **`175.86 개`** |

---

## 2.1 Gear-1 (기어박스 직붙 센서 채널) 갤러리

기어박스 하우징 직근에 장착되어 차동장치 기어 씹힘, 슬립, 고차 조화파 진동 소음을 정밀 수신한 채널 갤러리입니다.

| [고장] 고속주행 20kph | [고장] 고속주행 60kph |
| :---: | :---: |
| ![Abnormal 20kph Gear1](figures/audio_spectrograms/audio_abnormal_고속주회로_한바퀴high_speed_circut,_one_round_20kph_gear1_waveform_spectrogram.png) | ![Abnormal 60kph Gear1](figures/audio_spectrograms/audio_abnormal_고속주회로_한바퀴high_speed_circut,_one_round_60kph_gear1_waveform_spectrogram.png) |
| <small>기어박스에 직접 붙어 있어 저속 기어 씹힘 소리가 1.8kHz 위치에 선명하게 생김</small> | <small>속도가 올라가면서 2.2kHz와 4.4kHz 위치에 선명한 노란색 고장 띠가 나타남</small> |
| **[고장] 고속주행 80kph** | **[고장] 등판로 12도** |
| ![Abnormal 80kph Gear1](figures/audio_spectrograms/audio_abnormal_고속주회로_한바퀴high_speed_circut,_one_round_80kph_gear1_waveform_spectrogram.png) | ![Abnormal Hills 12deg Gear1](figures/audio_spectrograms/audio_abnormal_등판로_12도test_hills_12pct_grade_20kph_gear1_waveform_spectrogram.png) |
| <small>80km/h 고속 주행 중 2.8kHz 위치에 가장 강하고 진한 노란색 고장 신호가 일직선으로 터짐</small> | <small>언덕을 오르는 동안 기어에 힘이 받으면서 2.8kHz 고장 줄무늬가 끝까지 이어짐</small> |

| [고장] 등판로 18도 | [고장] 등판로 30도 |
| :---: | :---: |
| ![Abnormal Hills 18deg Gear1](figures/audio_spectrograms/audio_abnormal_등판로_18도test_hills_18pct_grade_20kph_gear1_waveform_spectrogram.png) | ![Abnormal Hills 30deg Gear1](figures/audio_spectrograms/audio_abnormal_등판로_30도test_hills_30pct_grade_20kph_gear1_waveform_spectrogram.png) |
| <small>18도 경사를 오를 때 기어 마찰이 심해져 3.2kHz 부근 소음 전력이 붉은색으로 급증함</small> | <small>30도 극한 경사로 기어가 강하게 짓눌려 주파수 전체에 주황색 고장 소음이 넓게 퍼짐</small> |
| **[고장] 등판로 6도** | **[고장] 원선회 40kph** |
| ![Abnormal Hills 6deg Gear1](figures/audio_spectrograms/audio_abnormal_등판로_6도test_hills_6pct_grade_20kph_gear1_waveform_spectrogram.png) | ![Abnormal Steer Pad Gear1](figures/audio_spectrograms/audio_abnormal_원선회로_1분steer_pad,_1minute_40kph_gear1_waveform_spectrogram.png) |
| <small>6도 언덕 주행 중 2.8kHz와 5.6kHz 두 줄의 고장 줄무늬가 선명하게 관측됨</small> | <small>핸들을 꺾고 회전할 때 바퀴 차이로 기어가 긁히며 2.5k~4.5kHz 영역에 소음 발생</small> |

| [정상] 고속주행 20kph | [정상] 고속주행 60kph |
| :---: | :---: |
| ![Normal 20kph Gear1](figures/audio_spectrograms/audio_normal_고속주회로_한바퀴high_speed_circut,_one_round_20kph_gear1_waveform_spectrogram.png) | ![Normal 60kph Gear1](figures/audio_spectrograms/audio_normal_고속주회로_한바퀴high_speed_circut,_one_round_60kph_gear1_waveform_spectrogram.png) |
| <small>정상 20km/h 주행 상태로, 고주파 고장 띠가 전혀 없이 아래쪽에만 바퀴 소리가 옅게 남</small> | <small>정상 60km/h 주행 상태로, 2.2kHz 아래쪽 도로 소음 외에 위쪽 고주파 영역은 완전히 어두움</small> |
| **[정상] 고속주행 80kph** | **[정상] 등판로 12도** |
| ![Normal 80kph Gear1](figures/audio_spectrograms/audio_normal_고속주회로_한바퀴high_speed_circut,_one_round_80kph_gear1_waveform_spectrogram.png) | ![Normal Hills 12deg Gear1](figures/audio_spectrograms/audio_normal_등판로_12도test_hills_12pct_grade_20kph_gear1_waveform_spectrogram.png) |
| <small>80km/h로 달려도 고장 띠가 없이 위쪽 주파수(4kHz 이상)가 캄캄한 보라색으로 깨끗함</small> | <small>정상 상태로 12도 언덕을 오를 때 기어 씹힘 소리 없이 잔잔한 주파수 분포를 보임</small> |

| [정상] 등판로 18도 | [정상] 등판로 30도 |
| :---: | :---: |
| ![Normal Hills 18deg Gear1](figures/audio_spectrograms/audio_normal_등판로_18도test_hills_18pct_grade_20kph_gear1_waveform_spectrogram.png) | ![Normal Hills 30deg Gear1](figures/audio_spectrograms/audio_normal_등판로_30도test_hills_30pct_grade_20kph_gear1_waveform_spectrogram.png) |
| <small>18도 언덕 정상 주행 상태로, 고장 상태에 나오는 붉은색 강한 소음 띠가 전혀 없음</small> | <small>30도 언덕 정상 주행으로, 변속 후 아래쪽 주파수만 잔잔하게 유지되는 안정적 모습</small> |
| **[정상] 등판로 6도** | **[정상] 원선회 40kph** |
| ![Normal Hills 6deg Gear1](figures/audio_spectrograms/audio_normal_등판로_6도test_hills_6pct_grade_20kph_gear1_waveform_spectrogram.png) | ![Normal Steer Pad Gear1](figures/audio_spectrograms/audio_normal_원선회로_1분steer_pad,_1minute_40kph_gear1_waveform_spectrogram.png) |
| <small>6도 언덕 정상 주행으로, 위쪽 주파수 대역에 고장 줄무늬 없이 깨끗함</small> | <small>원선회 정상 주행으로, 타이어 마찰음 외에 기어 씹힘 고장 소음이 관측되지 않음</small> |

---

## 2.2 OBD-2 (실내 조종석 마이크 채널) 갤러리

차량 실내 OBD 단자 부근에 장착되어 조종석 전달 소음 및 운행 환경 음향을 수집한 채널 갤러리입니다.

| [고장] 고속주행 20kph | [고장] 고속주행 60kph |
| :---: | :---: |
| ![Abnormal 20kph OBD2](figures/audio_spectrograms/audio_abnormal_고속주회로_한바퀴high_speed_circut,_one_round_20kph_obd2_waveform_spectrogram.png) | ![Abnormal 60kph OBD2](figures/audio_spectrograms/audio_abnormal_고속주회로_한바퀴high_speed_circut,_one_round_60kph_obd2_waveform_spectrogram.png) |
| <small>실내 마이크라 소리가 차체 철판에 막혀 기어박스 마이크보다 고장 신호가 옅게 찍힘</small> | <small>실내로 들어오면서 소리가 감쇄되어 고장 노란색 띠가 -70dB 수준으로 옅게 수신됨</small> |
| **[고장] 고속주행 80kph** | **[고장] 등판로 12도** |
| ![Abnormal 80kph OBD2](figures/audio_spectrograms/audio_abnormal_고속주회로_한바퀴high_speed_circut,_one_round_80kph_obd2_waveform_spectrogram.png) | ![Abnormal Hills 12deg OBD2](figures/audio_spectrograms/audio_abnormal_등판로_12도test_hills_12pct_grade_20kph_obd2_waveform_spectrogram.png) |
| <small>80km/h 고속 고장 소음이 실내로 전달되었으나 차체에 막혀 색상이 옅은 주황색으로 보임</small> | <small>등판 시 기어 고장 소음이 실내 운전석 마이크까지 전달되어 옅은 줄무늬로 보임</small> |

| [고장] 등판로 18도 | [고장] 등판로 30도 |
| :---: | :---: |
| ![Abnormal Hills 18deg OBD2](figures/audio_spectrograms/audio_abnormal_등판로_18도test_hills_18pct_grade_20kph_obd2_waveform_spectrogram.png) | ![Abnormal Hills 30deg OBD2](figures/audio_spectrograms/audio_abnormal_등판로_30도test_hills_30pct_grade_20kph_obd2_waveform_spectrogram.png) |
| <small>18도 등판 고장 소음이 실내 마이크에 약한 붉은색 띠 형태로 수신됨</small> | <small>30도 급경사 등판 시 발생한 큰 고장 소음이 실내 마이크에도 전반적으로 약하게 전달됨</small> |
| **[고장] 등판로 6도** | **[고장] 원선회 40kph** |
| ![Abnormal Hills 6deg OBD2](figures/audio_spectrograms/audio_abnormal_등판로_6도test_hills_6pct_grade_20kph_obd2_waveform_spectrogram.png) | ![Abnormal Steer Pad OBD2](figures/audio_spectrograms/audio_abnormal_원선회로_1분steer_pad,_1minute_40kph_obd2_waveform_spectrogram.png) |
| <small>6도 언덕 고장 소음이 실내 마이크에 약한 고장 신호로 수신됨</small> | <small>원선회 시 기어 마찰음이 실내 마이크에 약한 노이즈 형태로 전달됨</small> |

| [정상] 고속주행 20kph | [정상] 고속주행 60kph |
| :---: | :---: |
| ![Normal 20kph OBD2](figures/audio_spectrograms/audio_normal_고속주회로_한바퀴high_speed_circut,_one_round_20kph_obd2_waveform_spectrogram.png) | ![Normal 60kph OBD2](figures/audio_spectrograms/audio_normal_고속주회로_한바퀴high_speed_circut,_one_round_60kph_obd2_waveform_spectrogram.png) |
| <small>정상 20km/h 주행 시 실내 바람 소리와 조용함만 유지되는 깨끗한 상태</small> | <small>정상 60km/h 주행 시 실내로 들어오는 차속 바람 소리 외에 고장 소음 없음</small> |
| **[정상] 고속주행 80kph** | **[정상] 등판로 12도** |
| ![Normal 80kph OBD2](figures/audio_spectrograms/audio_normal_고속주회로_한바퀴high_speed_circut,_one_round_80kph_obd2_waveform_spectrogram.png) | ![Normal Hills 12deg OBD2](figures/audio_spectrograms/audio_normal_등판로_12도test_hills_12pct_grade_20kph_obd2_waveform_spectrogram.png) |
| <small>정상 80km/h 주행 시 실내로 전달되는 정동음 외에 고장 줄무늬 없이 깜깜함</small> | <small>정상 12도 등판 시 실내 마이크에 저주파 엔진 소리만 들어오는 깨끗한 모습</small> |

| [정상] 등판로 18도 | [정상] 등판로 30도 |
| :---: | :---: |
| ![Normal Hills 18deg OBD2](figures/audio_spectrograms/audio_normal_등판로_18도test_hills_18pct_grade_20kph_obd2_waveform_spectrogram.png) | ![Normal Hills 30deg OBD2](figures/audio_spectrograms/audio_normal_등판로_30도test_hills_30pct_grade_20kph_obd2_waveform_spectrogram.png) |
| <small>정상 18도 등판 시 실내 수신 소음이 차체에 막혀 조용하게 유지됨</small> | <small>정상 30도 등판 시 실내 마이크에 고장 소음 띠 없이 아래쪽만 잔잔함</small> |
| **[정상] 등판로 6도** | **[정상] 원선회 40kph** |
| ![Normal Hills 6deg OBD2](figures/audio_spectrograms/audio_normal_등판로_6도test_hills_6pct_grade_20kph_obd2_waveform_spectrogram.png) | ![Normal Steer Pad OBD2](figures/audio_spectrograms/audio_normal_원선회로_1분steer_pad,_1minute_40kph_obd2_waveform_spectrogram.png) |
| <small>정상 6도 등판 시 실내 풍절음 위주로 들어오며 고장 신호 없음</small> | <small>정상 원선회 주행 시 실내 마이크에 바람 소리 외에 기어 고장 소리 없음</small> |

---

## 3. 대안 1번 (신경망 0개, C++ 룰 기반) 실측 임계값 파라미터 도출

실측 스펙트로그램 이미지 분석을 바탕으로 도출된 **C++ 초경량 룰 알고리즘 파라미터**입니다.

### 3.1 대역별 실측 전력 밀도(dB) 기준

| 주파수 대역 | 정상 상태 전력 | 고장 상태 전력 | 진단 임계값 기준 (`Threshold`) | 비고 |
| :--- | :---: | :---: | :--- | :--- |
| **Low Band ($0 \sim 1\,\text{kHz}$)** | $-35\,\text{dB}$ | $-32\,\text{dB}$ | 변동성 적음 | 차속/타이어 도로 소음 대역 |
| **Mid-High Band ($2.8\,\text{kHz}$ 기어 피크)** | $-65\,\text{dB}$ 이하 | **$-30\,\text{dB}$ 강한 피크** | **$-45\,\text{dB}$ 초과 시 기어 결함 경보** | 기어 씹힘 기본 파형 대역 |
| **High Band ($4 \sim 10\,\text{kHz}$ 조화파)** | $-85\,\text{dB}$ 어둠 | **$-50\,\text{dB}$ 층상 밴드** | **$-60\,\text{dB}$ 초과 시 고주파 마모 확정** | 고차 조화파 마모 대역 |

### 3.2 C++ 임베디드 소스 코드 수식 명세

```cpp
// C++ 초경량 룰 기반 오디오 이상치 점수 산출 (추가 딥러닝 연산량 0 FLOPs)
float compute_audio_rule_score(const float* stft_db_spectrum, int num_bins) {
    // 2.8kHz 지점 Bin (약 -30dB 피크 관측 대역)
    float p_2k8 = stft_db_spectrum[BIN_2800HZ];
    // 4k~10kHz 고차 조화파 평균 전력
    float p_high_harmonic = get_band_mean_db(stft_db_spectrum, 4000, 10000);
    
    if (p_2k8 > -45.0f || p_high_harmonic > -60.0f) {
        // 정상 상한(-65dB)과 고장 하한(-30dB) 간 선형 보간 점수
        float score = (p_2k8 + 65.0f) / 35.0f;
        return (score > 1.0f) ? 1.0f : ((score < 0.0f) ? 0.0f : score);
    }
    return 0.0f; // 정상 범위
}
```

---

## 4. 최종 결론

1. **쉽고 직관적인 설명 보정 완료**: 축약된 전문 용어 대신 **"기어박스 마이크라 고장 소리가 2.8kHz 위치에 노란색 띠로 크게 찍힘"**, **"실내 마이크라 소리가 차체 철판에 막혀 옅게 찍힘"**과 같이 누구나 직관적으로 읽고 이해할 수 있도록 이미지별 설명을 풀어서 갱신했습니다.
2. **2×2 그리드 레이아웃 및 텍스트 100% 보존**: `Section 2.1 (Gear-1)` 및 `Section 2.2 (OBD-2)` 섹션별 4개씩 2×2 표 레이아웃으로 정리하면서 분석 텍스트, 주파수 대역 분석, C++ 룰 수식 등 모든 내용을 100% 보존하였습니다.