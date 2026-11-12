# GSC_EWS 파이프라인 가이드 v2

> ⚠️ **이 문서는 반드시 미리보기로 열어서 읽는다.**
> VSCode에서 이 파일을 연 상태로 **`Ctrl+Shift+V`** (macOS: `Cmd+Shift+V`)를 누른다.
> 원문 그대로 보면 표와 그림의 정렬이 맞지 않는다.

GSC_EWS의 구조, 실행, 설정, 입출력, 알람 판정, 온라인 추론, 해석, 코드 구성을 설명한다.
모든 명령과 상대 경로는 프로젝트 루트(`projects/gsc_ews/`)를 기준으로 한다.

# 목차

1. [전체 구조와 처리 흐름](#1-전체-구조와-처리-흐름)
2. [실행 모드](#2-실행-모드)
3. [설치, 설정, 실행 방법](#3-설치-설정-실행-방법)
4. [입력 데이터와 전처리](#4-입력-데이터와-전처리)
5. [점수, 임계값, 알람](#5-점수-임계값-알람)
6. [산출물 정의](#6-산출물-정의)
7. [온라인 추론](#7-온라인-추론)
8. [모델 해석: Attention, TFT, SHAP](#8-모델-해석-attention-tft-shap)
9. [코드 구조와 모듈 역할](#9-코드-구조와-모듈-역할)

<br>

# 1. 전체 구조와 처리 흐름

## 1.1 개념

정상 운전 데이터로 모델을 학습한다. 모델은 최근 윈도우를 보고 현재 시점의 값을 복원한다.
복원값과 실측값의 차이가 정상 범위를 벗어나 일정 시간 이어지면 알람을 낸다.

## 1.2 4개 실행 모드

| 모드 | 역할 | 시작점 | 종료 |
|---|---|---|---|
| `train` | 학습만 | 학습 CSV | 체크포인트 저장 후 종료 |
| `full` | 학습부터 평가까지 전체 | 학습 CSV + 테스트 CSV | 결과 저장 후 종료 |
| `evaluate` | 저장 모델로 일괄 추론 | 체크포인트 + 보정 파일 + 입력 CSV | 결과 저장 후 종료 |
| `online` | 저장 모델로 지속 추론 | 체크포인트 + 보정 파일 + 입력 CSV | `Ctrl+C`까지 반복 |

```text
■ train
│
├─ 전처리
├─ Fold 학습
└─ 전체 재학습
     └▶ 📦 checkpoint

■ full
│
├─ 전처리
├─ Fold 학습 + OOF 점수 수집
├─ 전체 재학습
│    └▶ 📦 checkpoint
├─ 보정
│    └▶ 📄 calibration.yaml
├─ 테스트 추론 → 점수 → 알람 → 평가
└─ 저장
     └▶ 📁 results/

■ evaluate
│
├─ 📦 checkpoint 로드
├─ 📄 calibration.yaml 로드
├─ 입력 CSV 전체 추론 → 점수 → 알람 → 평가
└─ 저장
     └▶ 📁 results/

■ online
│
├─ 📦 checkpoint 로드
├─ 📄 calibration.yaml 로드
├─ 반복
│    ├─ 새 행 확인
│    ├─ 새 시점만 추론 → 점수 → 알람
│    ├─ 저장 ▶ 💾 online.sqlite3
│    └─ 대기
└─ Ctrl+C로 종료
```

> `full`이 만든 📦 checkpoint와 📄 calibration.yaml을 `evaluate`와 `online`이 재사용한다.

## 1.3 데이터 처리 단계

모든 추론 모드(`full`, `evaluate`, `online`)는 같은 단계를 거친다.

```text
📄 입력 CSV
│
├─ ① 전처리
│     시각 정렬, 수치 변환, 공정 정지 제거, 학습 결측 전략
│     └▶ Test·Evaluate·Online 결측 행은 계기 이상 알람
│
├─ ② 윈도우 생성, 스케일링
│
├─ ③ 모델 추론
│     └▶ raw_score (변수별 복원 오차)
│
├─ ④ 점수 후처리
│     └▶ score (이동 평균, 정규화)
│
├─ ⑤ 알람 판정
│     score > threshold → 연속 감지 → 레벨 → 쿨다운
│     └▶ 원본값은 CRV 상·하한과 별도 비교
│
└─ ⑥ 저장
      └▶ predictions.csv, alarms.csv, 보고서, 플롯
```

## 1.4 지원 모델

| 모델 | 구조 | 기본 이상 점수 | 점수 후처리 |
|---|---|---|---|
| `tranad` | Transformer 2단계(Phase 1·2) 복원 | `mae` | 없음 |
| `tranad_v2` (기본) | TranAD 개선판. d_model·head 수 설정, GELU, Pre-LN | `mae` | 분위수 정규화 |
| `tft` | Temporal Fusion Transformer. 분위수 출력, 변수 선택 네트워크 | `quantile_loss` | 이동 평균 5, median/MAD 정규화 |
| `lstm_ae` | LSTM AutoEncoder | `mae` | 없음 |

모델은 `config.yaml`의 `active_model` 또는 `--model`로 선택한다.

<br>

# 2. 실행 모드

## 2.1 `train`

학습과 검증만 수행한다. 보정, 테스트 추론, 알람은 실행하지 않는다.

1. 학습 CSV를 시간순 Fold로 나눠 Fold별로 학습한다.
2. Fold별 최적 epoch의 중앙값만큼 전체 정상 데이터로 다시 학습한다.
3. 체크포인트와 학습 로그를 저장한다.

```text
📁 checkpoints/<equipment_id>/<model>/<version>/<run_id>/
   ├── 📁 fold_01/ ~ fold_05/
   │      ├── 📦 epoch=...-val_loss=....ckpt  # Fold별 최적 모델
   │      └── 📦 last.ckpt
   └── 📁 final/
          ├── 📦 final-epoch=....ckpt         # ★ 운영 모델
          └── 📦 last.ckpt

📁 logs/<equipment_id>/<model>/<version>/<run_id>/
   └── 📁 fold_01/ ~ final/
          ├── 📄 metrics.csv                  # epoch별 loss
          └── 📄 hparams.yaml

📁 results/
   ├── 📄 experiments.csv                     # 전체 실행 목록
   └── 📁 <equipment_id>/<model>/<version>/<run_id>/
          ├── 📄 config.yaml                  # 실제 적용 설정
          └── 📄 summary.yaml                 # 핵심 성능 요약
```

`--fast-dev-run`을 사용하면 학습과 검증을 한 배치씩만 실행해 연결 상태를 점검한다.

## 2.2 `full`

`train`의 학습 과정에 보정, 테스트 추론, 알람, 평가, 저장을 이어서 수행한다.
**`calibration.yaml`을 만드는 유일한 모드**다.

1. `train`과 같은 방식으로 학습한다. Fold별 검증 구간의 점수(OOF 점수)를 함께 모은다.
2. OOF 점수로 점수 정규화 통계와 변수별 임계값을 계산해 `calibration.yaml`에 저장한다.
3. `final` 체크포인트로 테스트 CSV를 추론하고 점수, 알람, 평가를 계산한다.
4. 설정에 따라 해석(Attention, SHAP)을 실행한다.

```text
📁 checkpoints/ · logs/                              # train과 동일

📁 results/<equipment_id>/<model>/<version>/<run_id>/
   ├── 📄 config.yaml
   ├── 📄 summary.yaml
   ├── 📄 calibration.yaml                           # ★ evaluate · online에서 재사용
   │
   ├── 📄 predictions.csv                            # 시점별 결과
   ├── 📄 alarms.csv                                 # 알람 이벤트
   ├── 📄 thresholds.csv                             # 변수별 임계값
   │
   ├── 📄 metrics.csv / metrics.json / *.txt         # evaluation.enabled
   ├── 📄 report.xlsx / reconstruction_quality.xlsx  # output.excel
   ├── 📁 plots/                                     # output.plots
   │      └── 📁 interpretability/                   # TFT 해석
   └── 📁 analysis/                                  # analysis.enabled
```

## 2.3 `evaluate`

재학습 없이 저장된 체크포인트와 `calibration.yaml`로 입력 CSV 전체를 한 번에 추론한다.
과거 구간 진단, 다른 기간 데이터 검증에 사용한다.

- 모델 종류, 구조, 전처리, 스케일러, 특성 순서는 **체크포인트**에서 가져온다.
- 점수 정규화 통계와 임계값은 **`calibration.yaml`**에서 가져온다.
- 알람 규칙은 현재 **`alarm.yaml`**을 따른다.
- 학습 데이터가 없으므로 SHAP은 사용할 수 없다.

```text
📁 results/<equipment_id>/<model>/<version>/<run_id>/
   ├── 📄 predictions.csv
   ├── 📄 alarms.csv
   ├── 📄 thresholds.csv
   ├── 📄 metrics.* · *.txt · *.xlsx  # 설정에 따라
   ├── 📁 plots/                      # 설정에 따라
   └── 📁 analysis/                   # 설정에 따라
```

> 체크포인트, 로그, `calibration.yaml`은 새로 만들지 않는다.

## 2.4 `online`

입력 CSV에 새로 추가되는 행만 계속 추론한다. 상세 내용은 [7절](#7-온라인-추론)에 있다.

```text
📁 results/<equipment_id>/<model>/<version>/<run_id>/online/
   ├── 💾 online.sqlite3      # ★ 결과와 상태 원본
   ├── 📄 predictions.csv     # SQLite에서 재생성
   ├── 📄 alarms.csv
   ├── 📄 thresholds.csv
   ├── 📄 *.xlsx              # 설정에 따라
   └── 📁 plots/              # 설정에 따라
```

## 2.5 체크포인트와 보정 파일은 한 쌍이다

`evaluate`와 `online`에는 같은 `full` 실행에서 나온 두 파일이 모두 필요하다.

| 파일 | 담긴 내용 |
|---|---|
| `checkpoints/.../final/*.ckpt` | 모델 가중치, 모델 설정, 스케일러, 특성 순서, 전처리 규칙 |
| `results/.../calibration.yaml` | 체크포인트 SHA-256, 점수 정규화 통계, 변수별 임계값 |

보정 파일에 기록된 해시가 체크포인트와 다르면 실행을 거부한다. `full` 종료 시 출력되는
`checkpoint`, `calibration` 경로를 그대로 사용한다.

<br>

# 3. 설치, 설정, 실행 방법

## 3.1 설치

```bash
cd projects/gsc_ews
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# SHAP을 사용할 때만
python -m pip install -r requirements-analysis.txt
```

주요 의존성은 PyTorch 2.8, Lightning 2.5, pandas, NumPy, SciPy, scikit-learn, matplotlib, openpyxl이다.
서버의 CUDA 드라이버에 맞는 PyTorch를 사용한다.

## 3.2 설정 파일

사용자가 관리하는 설정은 세 개다. 각 항목의 설명과 예시는 파일 안의 주석에 있다.

| 파일 | 역할 | 읽는 모드 |
|---|---|---|
| `configs/config.yaml` | 실행 대상, 모드, 경로, 데이터, 전처리, 분할, 학습, 모델 구조, 점수 후처리, 출력, 해석, 온라인 | 전체 |
| `configs/threshold.yaml` | 임계값 계산 방식과 변수별 직접 지정값 | `full` (`evaluate`·`online`은 형식 검증만) |
| `configs/alarm.yaml` | 알람 레벨, 연속 횟수, 쿨다운, inhibit, CRV | `full`, `evaluate`, `online` |

`calibration.yaml`은 `full`이 생성하는 결과물이다. 직접 편집하지 않는다.

## 3.3 설정 적용 규칙

- CLI에서 지정한 값이 YAML 값보다 우선한다.
- YAML 안의 상대 경로는 **YAML 파일 위치** 기준, CLI의 상대 경로는 **명령을 실행한 폴더** 기준이다.
- `config.yaml`의 `model_overrides.<모델명>`이 공통 설정 위에 덮어쓰인다.
  대상은 `postprocessing`, `evaluation`, `output`, `analysis`, `online`이다.
- `alarm.yaml`과 `threshold.yaml`은 모든 모델에 공통으로 적용된다.

## 3.4 설정 변경과 재학습·재보정

| 바꾼 설정 | 반영 방법 |
|---|---|
| 데이터 경로, 특성, 전처리, 정규화, 분할, 모델 구조, 학습 | `full` 재실행 (재학습) |
| `postprocessing`, `threshold.yaml` | `full` 재실행 (보정값이 `calibration.yaml`에 고정되므로) |
| `alarm.yaml` (레벨, 연속 횟수, 쿨다운, inhibit, CRV) | 다음 `evaluate` 실행부터 바로 반영 |
| `evaluation`, `output`, `analysis` | 다음 실행부터 바로 반영 |

`online`은 실행 중 보정 파일, 알람 설정, 입력 경로가 바뀌면 중단한다. 새 출력 폴더로 다시 시작한다.

## 3.5 CLI 옵션

| 옵션 | 역할 | 사용 모드 |
|---|---|---|
| `--config` | 통합 YAML 경로. 기본 `configs/config.yaml` | 전체 |
| `--mode` | `full`, `train`, `evaluate`, `online`, `analyze` | 전체 |
| `--model` | 학습할 모델 | `full`, `train` |
| `--max-epochs` | epoch 수 덮어쓰기 | `full`, `train` |
| `--fast-dev-run` | 한 배치만 학습해 연결 점검 | `train` |
| `--checkpoint` | 불러올 체크포인트 | `evaluate`, `online`, `analyze` |
| `--calibration` | 체크포인트와 짝인 보정 파일 | `evaluate`, `online` |
| `--input` | 입력 CSV. `full`·`evaluate`·`analyze`는 데이터 경로 대체, `online`은 감시 파일 | `full`, `evaluate`, `online`, `analyze` |
| `--background` | SHAP 정상 배경 CSV | `analyze`에서 SHAP 사용 시 |
| `--output` | 결과 폴더 직접 지정 | 전체 |
| `--alarm-config`, `--threshold-config` | 알람·임계값 YAML 경로 | 전체 |
| `--device` | `auto`, `cpu`, `gpu`, `cuda`, `mps` | 전체 |
| `--print-config` | 최종 적용 설정만 출력하고 종료 | 전체 |

## 3.6 첫 실행 순서

```bash
# 1. 설정 확인: 경로, 시각 컬럼, 특성, active_model
python main.py --print-config

# 2. 연결 점검
python main.py --mode train --fast-dev-run

# 3. 전체 실행
python main.py --mode full

# 4. 저장 모델로 다른 기간 진단 (경로는 3번 종료 시 출력값 사용)
python main.py --mode evaluate \
  --checkpoint checkpoints/<equipment_id>/<model>/<version>/<run_id>/final/final-epoch=XXX.ckpt \
  --calibration results/<equipment_id>/<model>/<version>/<run_id>/calibration.yaml \
  --input data/new_period.csv

# 5. 온라인 운영
python main.py --mode online \
  --checkpoint <같은 체크포인트> \
  --calibration <같은 보정 파일> \
  --input data/online.csv

# 6. 저장 모델 해석
python main.py --mode analyze \
  --checkpoint <체크포인트> \
  --input data/analysis_target.csv \
  --background data/normal_train.csv
```

<br>

# 4. 입력 데이터와 전처리

## 4.1 CSV 규격

```csv
DateTime,5HTI596A,5HTI597A,732VI62210
2026-09-01 00:00:00,41.2,40.8,1.53
2026-09-01 00:01:00,41.3,40.9,1.55
```

| 항목 | 규칙 | 설정 |
|---|---|---|
| 형식 | CSV만 지원 | – |
| 시각 컬럼 | `online`은 필수. `full`·`evaluate`는 `null`이면 행 순서와 수집 간격으로 상대 시각 생성. 파싱 불가 시각은 제거 | `data.timestamp_column` |
| 특성 컬럼 | `null`이면 시각 외 전체. 문자열 컬럼이 있으면 목록으로 지정 | `data.feature_columns` |
| 특성 순서 | 학습, 테스트, 온라인 CSV에서 이름이 같아야 함 | – |
| 수집 간격 | 이 간격 × 1.5를 넘으면 시간 단절로 보고 윈도우를 나눔 | `data.expected_interval_seconds`, `max_gap_multiplier` |

## 4.2 학습 데이터

- 학습 CSV는 **정상 운전 구간**이어야 한다. 이상 구간이 섞이면 그 패턴도 정상으로 학습한다.
- 학습 라벨(`data.train_labels_path`)을 지정하면 어느 변수든 라벨이 0보다 큰 행을 학습에서 제외한다.
- 검증 분할은 시간순으로만 한다.
  - `time_series_split_enabled: true`: 확장형 Fold 교차검증 (기본 5개)
  - `false`: 마지막 `validation_ratio`(기본 20%)를 검증 구간으로 한 번 분할
- 스케일러는 각 Fold의 학습 구간에만 맞춘다.

## 4.3 평가 라벨

`data.labels_path`를 지정하면 탐지 지표를 계산한다. 없으면 `null`로 둔다.

- 시각 컬럼이 있으면 시각으로, 없으면 행 순서로 테스트 데이터와 맞춘다.
- 특성명과 같은 라벨 컬럼을 우선 사용한다. 라벨 컬럼이 하나면 모든 특성에 복제한다.
- 값이 0보다 크면 이상이다.

## 4.4 전처리 순서

`config.yaml`의 `preprocessing.steps` 순서대로 실행한다.

| 순서 | 단계 | 역할 | 기본 |
|---|---|---|---|
| 1 | `prepare_time_index` | 시각 파싱, 정렬, 중복 시각 처리 | 켜짐 |
| 2 | `coerce_numeric` | 특성을 숫자로 변환. 변환 불가 값과 ±inf는 결측으로 | 켜짐 |
| 3 | `drop_stopped` | 공정 정지 행 제거 | **꺼짐** |
| 4 | `missing.strategy` | Train/Validation 결측 처리 방식 선택(`drop`, `forward_fill`, `interpolate`) | `drop` |

제거된 행 앞뒤의 윈도우는 서로 다른 구간으로 나뉜다. 윈도우는 구간 경계를 넘지 않는다.

## 4.5 공정 정지 판단

값이 0이라는 이유만으로 정지로 보지 않는다. **지정한 운전 tag 조건**으로만 판단한다.

```yaml
# config.yaml (학습·추론 데이터에서 정지 행 제거)
- name: drop_stopped
  enabled: true
  params:
    conditions:
      - {variable: Motor_Power.PV, operator: '<', value: 5.0}
      - {variable: Pump_RPM.PV, operator: '<=', value: 100.0}
    logic: any          # any = OR, all = AND
```

- `variable`: 특성에 포함된 tag명, `value`: 원본 공정 단위
- `operator`: `<`, `<=`, `>`, `>=`
- 알람 억제용 정지 조건은 `alarm.yaml`의 `inhibit_conditions`에 따로 입력한다([5.5절](#55-inhibit)).

학습과 Validation에서는 정지 행을 모델 입력에서 제외한다. `full`의 Test와 `evaluate`에서도
정지 행을 추론하지 않는다. `online`에서는 해당 입력 시각을 처리 완료로 기록하지만
`predictions.csv`에는 예측 행을 만들지 않고 Window를 초기화한다. 따라서 정지 전 데이터와
재가동 후 데이터가 하나의 Window로 연결되지 않는다.

`drop_stopped`를 켰는데 조건이 비어 있거나 tag·연산자가 잘못되면 오류로 중단한다.
`drop_stopped`는 데이터·추론 대상 자체를 제거하고, Inhibit은 점수 계산을 유지하면서 알람만 억제한다.
Online에서 0·NA 계기 이상 알람까지 정지로 억제하려면 같은 운전 조건을
`alarm.yaml.inhibit_conditions`에도 설정해야 한다.

## 4.6 결측 처리

Train/Validation은 `preprocessing.missing.strategy`에서 결측 처리 방식을 선택한다.

- `drop`: 결측이 하나라도 있는 행을 제거한다.
- `forward_fill`: 같은 연속 구간의 최근 정상값으로 제한 개수만큼 채운다.
- `interpolate`: 같은 연속 구간의 앞뒤 정상값으로 제한 개수만큼 보간한다.

Full의 Test, Evaluate, Online은 학습 전략과 관계없이 원본 결측 행을 추론하지 않고
계기 이상(`INSTRUMENT_FAULT`)으로 기록한다. 판정 기준은 모드마다 다르다.

| 모드 | 계기 이상으로 보는 값 |
|---|---|
| `full`, `evaluate` | NA, 비수치 |
| `online` | NA, 비수치, **0** |

공정 정지(inhibit) 조건에 해당하는 시점은 계기 이상으로 기록하지 않는다.
Online은 `drop`으로 학습한 체크포인트만 허용한다.

## 4.7 윈도우

- 각 시점은 자신을 포함한 직전 `window_size`개 시점으로 추론한다.
- `pad_start: true`: 구간 시작부의 부족한 과거를 첫 값으로 채워 처음부터 결과를 낸다.
- `pad_start: false`: 관측이 `window_size`개 쌓인 뒤부터 결과를 낸다.

<br>

# 5. 점수, 임계값, 알람

## 5.1 전체 흐름

```text
■ 모델 알람
│
├─ ① raw_score: 모델이 계산한 변수별 복원 오차
├─ ② score: 이동 평균 → 정규화  [postprocessing.py]
├─ ③ 임계값 비교: score > threshold ?  [threshold.py]
└─ ④ 알람 판정: inhibit → 연속 N회 → 레벨 → 쿨다운  [alarm.py]
      └▶ MODEL_ANOMALY 이벤트

■ CRV 알람
│
└─ 원본 공정값 vs 사용자 지정 상·하한
      └▶ CRV_LOW / CRV_HIGH 이벤트

■ 계기 이상
│
└─ 입력값이 결측(NA)
      └▶ INSTRUMENT_FAULT 이벤트
```

| 구분 | 비교 대상 | 기준 | 설정 |
|---|---|---|---|
| 모델 알람 | 정규화된 score | 변수별 threshold (자동 계산 또는 직접 지정) | `threshold.yaml`, `alarm.yaml` |
| CRV 알람 | 원본 공정값 | 사용자가 지정한 상·하한 | `alarm.yaml`의 `crv` |
| 계기 이상 | 원본 입력 | 결측 여부 | 자동 |

## 5.2 이상 점수

### raw_score

모델이 변수별로 계산한 복원 오차다. 스케일링된 값 기준이다.

| 모델 | 선택 가능한 점수 (`models.<모델>.anomaly_score`) |
|---|---|
| `tranad`, `tranad_v2` | `mae`(Phase 2 절대 오차), `phase2_mse`, `hybrid` |
| `tft` | `quantile_loss`, `median_mse`, `band_mse`, `band_violation` |
| `lstm_ae` | `mae`, `mse`, `hybrid` |

### score

raw_score를 변수 간에 비교할 수 있도록 바꾼 값이다. `config.yaml`의 `postprocessing`에서 설정한다.

| 단계 | 처리 | 설정 |
|---|---|---|
| 1 | 이동 평균: 현재와 과거만 사용. 시간 단절에서 초기화 | `smooth_window` |
| 2 | 정규화: 아래 방식 중 하나 | `normalization_method: none \| robust \| quantile` |

| 정규화 방식 | 계산 | 해석 | 사용 모델 |
|---|---|---|---|
| 없음 | `raw_score` 그대로 | 복원 오차 | `tranad`, `lstm_ae` |
| robust | `max((raw − 정상 중앙값) / (1.4826 × MAD), 0)` | 정상 대비 표준편차 배수 | `tft` |
| quantile | `(1 − w) × raw / 정상 raw의 q99 + w × 변수 평균 raw / 정상 변수 평균 raw의 q99` | 정상 상위 1% 오차 대비 배수 | `tranad_v2` |

quantile 방식에서 `w`는 `mae_weight`(기본 0.5), q99는 `quantile_level`(기본 0.99)이다.
뒤의 항은 모든 변수에 같은 값으로 더해지므로, 한 변수만 튀는 경우보다 여러 변수가 함께 커질 때 점수가 더 오른다.
정규화 방식은 하나의 `normalization_method`로 선택하므로 두 방식이 동시에 적용되지 않는다.

## 5.3 보정 (calibration)

`full`에서만 수행한다.

1. Fold별로 **학습에 쓰지 않은 검증 구간**의 raw_score(OOF 점수)를 모은다. 정상 데이터 점수지만
   모델이 처음 보는 데이터이므로 실제 운영 오차 분포에 가깝다.
2. OOF 점수로 정규화 통계(중앙값, MAD, 분위수)를 계산한다.
3. 정규화한 OOF score로 변수별 임계값을 계산한다.
4. 결과를 체크포인트 해시와 함께 `calibration.yaml`에 저장한다.

임계값은 `thresholds.<피처명>: <값>` 매핑으로 저장하므로 피처 순서를 세어 볼 필요가 없다.
`manual` 방식에서 `per_variable`에 없는 피처는 `threshold.yaml`의 `default` 값을 사용하고,
적용한 방식과 기본값은 `threshold_settings`에 함께 기록한다.

보정에 쓰는 OOF 점수는 Fold 모델들이 낸 점수다. 운영에는 전체 데이터로 재학습한 `final` 모델을 쓰므로,
`final` 모델의 정상 점수 분포는 보정 분포와 조금 다를 수 있다.

`evaluate`와 `online`은 이 파일을 그대로 적용하며 다시 계산하지 않는다.
`time_series_split_enabled: false`이면 단일 검증 구간의 점수로 보정한다.

## 5.4 임계값

`threshold.yaml`에서 계산 방식을 고르고, 실제 값은 정상 OOF score에서 자동 계산된다.

| `method` | 계산 | 사용자가 정하는 값 |
|---|---|---|
| `manual` | 모든 변수에 같은 값 | `default` |
| `percentile` (기본) | 변수별 정상 score의 분위수 | `percentile` (기본 99.5) |
| `sigma` | 변수별 평균 + k × 표준편차 | `sigma_k` (기본 3.0) |
| `pot` | 상위 꼬리에 일반화 파레토 분포를 맞춘 초과 확률 지점. 꼬리 표본이 부족하면 percentile로 대체 | `pot_risk`, `pot_init_quantile`, `pot_multiplier` |

`per_variable`에 지정한 변수는 계산값 대신 지정값을 쓴다. 특성명 또는 0부터 시작하는 인덱스로 지정한다.

```yaml
method: percentile
percentile: 99.5
per_variable:
  5HTI596A: 1.2
```

임계값은 원본 공정값이 아니라 **score와 같은 단위**다. 직접 지정할 때는 [5.2절](#52-이상-점수)의
정규화 방식을 확인한다. 공정값 기준의 상·하한은 [CRV](#56-crv)로 설정한다.

## 5.5 모델 알람

`alarm.yaml`에서 설정한다. 각 시점에서 아래 순서로 판정한다.

| 순서 | 판정 | 설정 | 기본 |
|---|---|---|---|
| 1 | **inhibit**: 정지 조건이면 판정을 생략하고 연속 횟수를 0으로 초기화 | `inhibit_enabled`, `inhibit_conditions`, `inhibit_logic` | 조건 없음 |
| 2 | **연속 감지**: `score > threshold`가 연속 N회면 변수 알람. 한 번 끊기면 다시 셈 | `consecutive_count` | 3 |
| 3 | **레벨**: `score / threshold` 비율로 결정 | `levels` | CAUTION 1.0, WARNING 1.5 |
| 4 | **전체 알람**: 알람 중인 변수 수가 기준 이상이면 OVERALL | `min_variables` | 1 |
| 5 | **쿨다운**: 같은 변수·레벨의 이벤트를 N초 안에 다시 기록하지 않음. 알람 상태는 유지 | `cooldown_seconds` | 60 |

시간 단절이 생겨도 연속 횟수를 초기화한다.

threshold가 0.8인 변수의 예:

| 시각 | score | 비율 | 연속 | 상태 | 레벨 | 이벤트 |
|---|---|---|---|---|---|---|
| 10:00 | 0.9 | 1.13 | 1 | – | NORMAL | – |
| 10:01 | 1.0 | 1.25 | 2 | – | NORMAL | – |
| 10:02 | 1.1 | 1.38 | 3 | 알람 | CAUTION | 기록 |
| 10:03 | 1.3 | 1.63 | 4 | 알람 | WARNING | 기록 (레벨 변경) |
| 10:04 | 1.3 | 1.63 | 5 | 알람 | WARNING | 생략 (쿨다운) |
| 10:05 | 0.7 | 0.88 | 0 | – | NORMAL | – |

`score_overall`은 변수별 score의 평균이며 표시용이다. 전체 알람 판정에는 쓰지 않는다.

### inhibit

공정 정지 중 알람을 막는다. score와 추론 결과는 계속 계산하고 저장한다.

```yaml
inhibit_enabled: true
inhibit_conditions:
  - {variable: Motor_Power.PV, operator: '<', value: 5.0}
inhibit_logic: any      # any = OR, all = AND
```

- `variable`은 모델 특성에 포함된 tag여야 한다.
- `operator`는 `<`, `<=`, `>`, `>=`, `==`, `!=`를 지원한다.
- `drop_stopped`(4.5절)는 정지 행을 데이터에서 제거하고, inhibit은 알람만 막는다. 두 설정은 따로 입력한다.

## 5.6 CRV

모델과 무관하게 **원본 공정값**을 tag별 상·하한과 비교한다.

```yaml
crv:
  enabled: true
  per_variable:
    Shaft_Position.PV:
      low: -0.15
      high: 0.15
      level: WARNING
    5HTI596A:
      low: null         # 한쪽만 쓸 때는 null
      high: 85.0
      level: CAUTION
```

- 기본값은 `enabled: false`다.
- tag는 모델 특성에 포함되어야 한다. 특성이 아닌 tag에는 CRV를 걸 수 없다.
- tag마다 `low`, `high` 중 하나 이상을 지정한다. `level`은 `levels`에 있는 이름이어야 한다.
- 이벤트는 `CRV_LOW`, `CRV_HIGH`로 기록된다.
- 연속 횟수, 쿨다운, inhibit은 모델 알람 설정을 공유한다.
- 추론 대상 행에서만 판정한다. 결측이나 정지로 제거된 행은 판정하지 않는다.

## 5.7 계기 이상

결측 입력([4.6절](#46-결측-처리))을 `INSTRUMENT_FAULT` 이벤트로 기록한다.
레벨은 `levels`에 `WARNING`이 있으면 `WARNING`, 없으면 가장 낮은 레벨이다.
해당 시점은 추론하지 않으며, 앞뒤 윈도우와 알람 연속 횟수가 끊어진다.

## 5.8 평가 지표

`evaluation.enabled: true`일 때 계산한다.

| 지표 | 필요 조건 | 단위 |
|---|---|---|
| 복원 품질: MAE, RMSE, R², MAPE | 없음 | 원본 공정 단위 |
| 탐지 지표: Precision, Recall, F1, TP/FP/FN/TN, ROC-AUC | 평가 라벨 | – |
| 순위 지표: Hit, NDCG | 평가 라벨 | – |

- 탐지 지표는 변수별 `score > threshold`를 시점×변수 단위로 집계한다. 연속 감지, 쿨다운, inhibit을 적용한 알람 기준 성능이 아니다.
- 라벨이 없으면 탐지 지표는 `N/A`, `metrics.json`의 `labels_provided`는 `false`다.
- `evaluation.enabled: false`이면 지표 파일을 만들지 않는다. 추론, 점수, 알람은 저장한다.

<br>

# 6. 산출물 정의

## 6.1 디렉토리 구조

```text
📁 projects/gsc_ews/
│
├── 📁 checkpoints/<equipment_id>/<model>/<version>/<run_id>/
│      ├── 📁 fold_01/ ~ fold_NN/                # Fold별 체크포인트
│      └── 📁 final/                             # ★ 운영 모델 체크포인트
│
├── 📁 logs/<equipment_id>/<model>/<version>/<run_id>/
│      └── 📁 fold_01/ ~ final/                  # metrics.csv, hparams.yaml
│
└── 📁 results/
       ├── 📄 experiments.csv                    # 모든 학습 실행 요약
       │
       └── 📁 <equipment_id>/<model>/<version>/<run_id>/
              │
              │   ── 실행 정보 ──
              ├── 📄 config.yaml                 # 실제 적용 설정
              ├── 📄 summary.yaml                # 실행 요약
              ├── 📄 calibration.yaml            # ★ 보정값
              │
              │   ── 핵심 결과 ──
              ├── 📄 predictions.csv             # 시점별 결과
              ├── 📄 alarms.csv                  # 알람 이벤트
              ├── 📄 thresholds.csv              # 변수별 임계값
              │
              │   ── 평가 보고서 ──
              ├── 📄 metrics.csv / metrics.json  # 평가 지표
              ├── 📄 overall_metrics.txt         # 전체 지표 요약
              ├── 📄 reconstruction_quality.*    # 복원 품질 (txt, xlsx)
              ├── 📄 report.xlsx                 # 알람·지표 시트
              │
              │   ── 그림 · 해석 ──
              ├── 📁 plots/                      # 공통 플롯
              │      └── 📁 interpretability/    # TFT 해석 플롯
              ├── 📁 analysis/                   # Attention, SHAP 원시 결과
              ├── 📄 tft_interpretability.xlsx   # TFT 해석 요약
              │
              └── 📁 online/                     # online 모드 결과
```

**예시**: `equipment_id: 512K102A`, `active_model: tranad_v2`, `model_version: v0_1_0`, `run_id: base`

```text
📁 results/512K102A/tranad_v2/v0_1_0/base/
   ├── 📄 calibration.yaml
   ├── 📄 predictions.csv
   └── ...
📁 checkpoints/512K102A/tranad_v2/v0_1_0/base/final/final-epoch=012.ckpt
```

- 경로의 `<equipment_id>`, `<model>`, `<version>`, `<run_id>`는 `config.yaml`의 `equipment_id`, `active_model`, `model_version`, `run.run_id` 값이다. `run_id`를 지정하지 않으면 `<날짜시각>_<UUID 8자리>`로 자동 생성된다.
- `--output`을 지정하면 `results/` 아래 경로 대신 그 폴더를 그대로 쓴다. `online`도 `online/` 하위 폴더를 붙이지 않는다.
  체크포인트와 로그 위치는 `--output`과 관계없이 `checkpoints/`, `logs/` 아래에 저장된다.
- 같은 폴더를 다시 쓰면 같은 이름의 파일만 덮어쓴다. 실행마다 폴더를 구분한다.

## 6.2 파일별 생성 조건

| 파일 | `train` | `full` | `evaluate` | `online` | `analyze` | 조건 |
|---|---|---|---|---|---|---|
| `config.yaml`, `summary.yaml`, `experiments.csv` | ✓ | ✓ |  |  |  |  |
| `calibration.yaml` |  | ✓ |  |  |  |  |
| `predictions.csv`, `alarms.csv`, `thresholds.csv` |  | ✓ | ✓ | ✓ |  | 추론 모드에서 항상 |
| `metrics.*`, `overall_metrics.txt`, `reconstruction_quality.txt` |  | ✓ | ✓ | ✓ |  | `evaluation.enabled` |
| `reconstruction_quality.xlsx` |  | ✓ | ✓ | ✓ |  | `evaluation.enabled` + `output.excel` |
| `report.xlsx` |  | ✓ | ✓ | ✓ |  | `output.excel` |
| `plots/` |  | ✓ | ✓ | ✓ |  | `output.plots` |
| `analysis/` |  | ✓ | ✓ |  | ✓ | full/evaluate는 `analysis.enabled`, analyze는 항상 |
| `plots/interpretability/` |  | ✓ | ✓ |  | ✓ | Attention 또는 TFT 해석 사용 시 |
| `tft_interpretability.xlsx` |  | ✓ | ✓ |  | ✓ | TFT + `output.excel` |
| `online.sqlite3` |  |  |  | ✓ |  |  |

## 6.3 predictions.csv

시점마다 한 행이다. `<변수>`는 실제 특성명이다.

| 열 | 의미 |
|---|---|
| `timestamp`, `index` | 시각, 전처리 후 행 번호 |
| `score_overall` | 변수별 score 평균 (표시용) |
| `alarm_overall` | 전체 모델 알람 (0/1) |
| `crv_alarm_overall` | CRV 알람 변수 존재 여부 (0/1) |
| `inhibited` | inhibit 적용 여부 |
| `<변수>__actual`, `<변수>__prediction` | 원본 단위 실측값, 복원값 |
| `<변수>__mae`, `__mse`, `__rmse` | 시점별 오차 (원본 단위) |
| `<변수>__raw_score`, `<변수>__score` | 모델 원시 점수, 후처리 점수 |
| `<변수>__threshold` | 임계값 |
| `<변수>__alarm`, `<변수>__level` | 변수 알람 (0/1), 레벨 |
| `<변수>__crv_alarm`, `<변수>__crv_direction` | CRV 알람 (0/1), 방향 (`LOW`/`HIGH`/`NORMAL`) |
| `<변수>__label` | 평가 라벨. 없으면 비어 있음 |
| 입력 부가 컬럼 | 입력에 있던 기타 컬럼. 이름이 겹치면 `input__` 접두어 |

계기 이상 시점은 `timestamp`, `__actual`, `alarm_overall`, `inhibited`만 채워진 행으로 들어간다.

## 6.4 alarms.csv

이벤트마다 한 행이다. 한 시점에 여러 행이 있을 수 있다.

| 열 | 의미 |
|---|---|
| `timestamp`, `variable`, `level` | 시각, 변수명 (`OVERALL` 포함), 레벨 |
| `event_type` | `MODEL_ANOMALY`, `CRV_LOW`, `CRV_HIGH`, `INSTRUMENT_FAULT` |
| `score`, `threshold`, `ratio` | 모델 알람의 score, 임계값, 비율 |
| `raw_value`, `low_limit`, `high_limit` | CRV 알람의 원본값과 기준값 |
| `consecutive_count` | 연속 감지 횟수 |
| `anomalous_count` | `OVERALL` 행의 알람 변수 수 |
| `score_overall` | 해당 시점 score 평균 |
| `message` | 요약 메시지 |

쿨다운 때문에 `predictions.csv`의 알람 시점 수와 `alarms.csv`의 행 수는 다를 수 있다.

## 6.5 지표 파일

| 파일 | 내용 |
|---|---|
| `metrics.csv` | 변수별 MAE, RMSE, R², MAPE, threshold. 라벨이 있으면 탐지 지표 추가 |
| `metrics.json` | 전체 복원 품질, `labels_provided`, 탐지·순위 지표 |
| `overall_metrics.txt` | 전체 지표 요약 |
| `reconstruction_quality.txt`, `.xlsx` | 전체·변수별 복원 품질, 우수·취약 변수, 지표 설명 |
| `report.xlsx` | `alarms` 시트, 평가 시 `metrics`와 품질 시트 |

## 6.6 플롯

| 파일 | 개수 | 내용 | 조건 |
|---|---|---|---|
| `variable_<번호>_<변수>.png` | 변수 수 | 실측·복원, score·임계값, 알람 구간 | 항상 |
| `all_scoring_metrics.png` | 1 | 점수 방식별 비교 | 항상 |
| `recon_quality_<번호>_<변수>.png` | 변수 수 | 변수별 복원 품질 | 평가 |
| `reconstruction_quality_*.png` | 3 | 품질 요약·표 | 평가 |
| `performance_table.png`, `performance_per_variable.png` | 2 | 탐지 성능 표 | 평가 |
| `training_curves.png` | 1 | 학습·검증 loss | `full` |

변수 플롯의 `Input (Ground Truth)`는 복원 대상 실측값이며 이상 라벨이 아니다.
시간 단절 구간은 선을 끊어서 표시한다.

<br>

# 7. 온라인 추론

## 7.1 동작 방식

모델, 스케일러, 보정값을 한 번 로드한다. 입력 CSV를 `online.poll_interval_seconds`(기본 60초)마다
확인하고 새로 추가된 행만 처리한다. 새 행이 없으면 모델을 실행하지 않는다.
첫 실행에서는 CSV에 이미 있는 행을 순서대로 처리해 윈도우와 알람 상태를 준비한다.

## 7.2 입력 CSV 작성 규칙

- 헤더와 기존 행은 바꾸지 않고 **끝에 행만 추가**한다.
- 시각은 중복 없이 오름차순이어야 한다.
- 외부 작성기는 완성된 파일로 원자적으로 교체한다. 열 개수가 맞지 않는 미완성 행이 있으면 중단한다.
- 체크포인트의 특성 컬럼이 모두 있어야 한다.

## 7.3 한 행의 처리 순서

1. 시각 유효성과 순서를 확인한다.
2. inhibit 조건이 아니면서 특성에 0·NA·비수치가 있으면 계기 이상으로 기록하고 추론하지 않는다.
3. 전처리하고 최근 윈도우를 갱신한다. 간격이 단절 기준을 넘으면 윈도우를 새로 시작한다.
4. 새 시점의 윈도우 하나만 추론한다.
5. 이동 평균, 연속 감지, 쿨다운 상태를 이어서 score와 알람을 계산한다.
6. 결과와 상태를 SQLite에 한 트랜잭션으로 저장한다.
7. 저장된 결과로 CSV, Excel, 플롯 보고서를 갱신한다.

## 7.4 상태 저장과 재시작

- `online.sqlite3`가 결과와 상태의 원본이다. CSV와 보고서는 여기서 재생성한다.
- 같은 출력 폴더, 체크포인트, 보정 파일, 알람 설정, 입력 경로로 재시작하면 이미 처리한 행은 건너뛴다.
- 처리 중 오류가 나면 해당 갱신의 결과와 상태가 함께 롤백된다.
- 다음 경우에는 중단한다. 새 출력 폴더로 다시 시작한다.
  - 이미 처리한 입력 행이 변경되거나 삭제됨
  - 입력 컬럼이 바뀜
  - 체크포인트, 보정 파일, 알람 설정, 입력 경로가 바뀜
- 같은 출력 폴더에 두 실행이 동시에 쓰지 못하도록 파일 잠금을 건다.
- 종료는 `Ctrl+C`다.

## 7.5 운영 설정과 한계

- 정답 라벨과 해석(Attention, SHAP)은 처리하지 않는다.
- 보고서 갱신 비용은 누적 행 수에 비례한다. 1분 주기 운영에서는 다음 설정을 권장한다.

```yaml
evaluation: {enabled: false}
output: {excel: false, plots: false}
```

- 입력은 CSV만 지원한다. RTDB와 API 연동은 구현되어 있지 않다.

<br>

# 8. 모델 해석: Attention, TFT, SHAP

## 8.1 실행 조건

| 설정 | 역할 | 기본 |
|---|---|---|
| `analysis.enabled` | 해석 실행 여부. `full`, `evaluate`에서만 동작 | `true` |
| `analysis.max_samples` | 해석할 윈도우 표본 수 | 100 |
| `analysis.top_k` | 중요도 그림에 표시할 상위 변수 수 | 20 |
| `analysis.attention` | Attention 저장 | `true` |
| `analysis.shap` | SHAP 계산. `requirements-analysis.txt` 필요 | `false` |

선택한 표본은 `analysis/samples.csv`에 표본 번호, 행 번호, 시각으로 남는다.

## 8.2 TranAD Attention

| 파일 | 내용 |
|---|---|
| `analysis/attention.npz` | Attention 가중치 원본 |
| `analysis/<모듈>.png` | 모듈별 평균 Attention |
| `analysis/<모듈>_per_head.png` | head별 Attention |
| `analysis/<모듈>_temporal_importance.png` | 윈도우 내 시점별 중요도 |

어느 과거 시점을 많이 참고해 복원했는지 보여준다.

## 8.3 TFT 변수 중요도

| 파일 | 내용 |
|---|---|
| `analysis/variable_importance.csv` | 변수 선택 가중치와 score 기여도 |
| `analysis/tft_interpretation.npz` | 표본별 가중치, 기여도, 분위수 출력 |
| `tft_interpretability.xlsx` | `Variable Importance`, `Lag Importance`, `Alarm Root Causes` 시트 |
| `plots/interpretability/interp_var_importance_*.png` | 변수 중요도와 히트맵 |
| `plots/interpretability/interp_attention_*.png` | Attention 히트맵, 시간 패턴 |
| `plots/interpretability/interp_qforecast_<번호>_<변수>.png` | 변수별 분위수 예측 구간 |
| `plots/interpretability/interp_alarm_root_cause.png` | 알람 시점의 변수 기여도 (알람이 있을 때) |

알람 시점이 있으면 최대 10개를 표본에 포함한다.

## 8.4 SHAP

| 설정 | 역할 |
|---|---|
| `shap_method` | `kernel`, `gradient`, `permutation` |
| `shap_target` | `null`이면 전체 변수 raw_score 평균, 특성명이면 해당 변수의 raw_score |
| `background_samples` | 학습 구간에서 뽑는 배경 윈도우 수 |
| `shap_nsamples` | 근사 계산 샘플 수 |

생성 파일은 `analysis/shap.npz`, `shap_importance.csv`, `shap_importance.png`다.
학습 데이터가 배경으로 필요하므로 `full`에서만 사용할 수 있다.
계산량은 윈도우 길이, 특성 수, 표본 수에 따라 크게 늘어난다.

## 8.5 해석의 한계

- SHAP은 모델 raw_score를 설명한다. 정규화 후 score나 연속 감지를 거친 알람 판정을 설명하지 않는다.
- Attention, 변수 선택 가중치, SHAP은 모델이 무엇을 참고했는지를 보여준다. 설비 이상 원인을 확정하지 않는다.
- 모든 그림을 끄려면 `output.plots: false`와 `analysis.enabled: false`를 함께 설정한다.

<br>

# 9. 코드 구조와 모듈 역할

## 9.1 디렉토리

```text
📁 projects/gsc_ews/
   ├── 📄 main.py                                       # CLI 진입점
   ├── 📁 configs/                                      # config · alarm · threshold YAML
   ├── 📁 models/                                       # 모델 네트워크, Lightning 어댑터
   │      ├── 📄 base.py
   │      ├── 📄 tranad.py · tranad_v2.py
   │      ├── 📄 tft.py
   │      └── 📄 lstm_ae.py
   ├── 📁 src/                                          # 공통 파이프라인
   │      ├── 📄 workflow.py                            # 모드별 단계 연결
   │      ├── 📄 preprocessing.py · dataloader.py       # 데이터
   │      ├── 📄 model.py · trainer.py · inference.py   # 모델·학습·추론
   │      ├── 📄 scoring.py · alarm.py · evaluation.py  # 점수·알람·평가
   │      ├── 📄 save.py · explainer.py                 # 저장·해석
   │      ├── 📄 online_runner.py                       # 온라인
   │      └── 📁 udf/                                   # 계산 도구
   │             ├── 📄 configuration.py
   │             ├── 📄 postprocessing.py · threshold.py
   │             ├── 📄 quality.py · metrics.py
   │             └── 📄 visualization.py
   ├── 📁 tests/                                        # unittest
   ├── 📁 data/                                         # 입력 CSV
   ├── 📁 checkpoints/ · logs/ · results/               # 실행 산출물
   └── 📁 docs/                                         # 문서
```

## 9.2 파일별 역할

### 진입점과 설정

| 파일 | 역할 |
|---|---|
| `main.py` | CLI 옵션을 읽어 YAML과 합치고 `run_pipeline()` 호출 |
| `src/workflow.py` | 모드별 단계 연결. `run_prediction()`이 추론 → 점수 → 알람 → 계기 이상 → 평가를 묶음 |
| `src/udf/configuration.py` | YAML 로딩, 검증, 병합, 경로 해석, 모델별 덮어쓰기 |

### 데이터

| 파일 | 역할 |
|---|---|
| `src/preprocessing.py` | 전처리 단계 함수(`prepare_time_index`, `coerce_numeric`, `drop_stopped`, 결측 전략 등), 온라인용 `OnlinePreprocessor` |
| `src/dataloader.py` | CSV 로드, 라벨 정렬, Fold·holdout 분할, 스케일러(`FeatureScaler`), 윈도우 Dataset, DataLoader |

### 모델과 학습

| 파일 | 역할 |
|---|---|
| `models/base.py` | 공통 `AnomalyLightningModule`: 학습·검증·예측 step, optimizer, 체크포인트 상태 |
| `models/tranad.py`, `tranad_v2.py`, `tft.py`, `lstm_ae.py` | 모델별 네트워크, 손실, 이상 점수 |
| `src/model.py` | `MODEL_REGISTRY`(이름 → 클래스), 모델 생성, 체크포인트 복원 |
| `src/trainer.py` | Trainer 구성, Fold 학습과 OOF 점수 수집, 전체 재학습 |
| `src/inference.py` | `Trainer.predict` 실행, 결과를 시점 순서와 원본 단위로 복원 |

### 점수, 임계값, 알람, 평가

| 파일 | 역할 |
|---|---|
| `src/scoring.py` | 보정 관리: `calibrate`(통계·임계값 계산), `validate_calibration`(짝 검증), `score_predictions`(적용) |
| `src/udf/postprocessing.py` | `ScoreProcessor`: 이동 평균과 정규화 |
| `src/udf/threshold.py` | 임계값 계산: manual, percentile, sigma, POT, `per_variable` |
| `src/alarm.py` | `AlarmGenerator`: 연속 감지, 레벨, 쿨다운, inhibit, CRV, 계기 이상 이벤트 |
| `src/evaluation.py` | 복원 품질과 탐지 지표 계산 호출 |
| `src/udf/quality.py` | MAE, RMSE, R², MAPE |
| `src/udf/metrics.py` | Precision, Recall, F1, ROC-AUC, Hit, NDCG, Point-Adjust |

### 저장, 해석, 온라인

| 파일 | 역할 |
|---|---|
| `src/save.py` | CSV, JSON, TXT, Excel, `calibration.yaml`, `experiments.csv` 저장, 플롯 호출 |
| `src/udf/visualization.py` | 그림 그리기 |
| `src/explainer.py` | Attention, TFT 변수 중요도, SHAP |
| `src/online_runner.py` | CSV 감시, 새 행 처리, SQLite 상태 저장, 재시작, 파일 잠금 |

## 9.3 모드별 실행 모듈

| 단계 | `train` | `full` | `evaluate` | `online` | `analyze` |
|---|---|---|---|---|---|
| 설정 `configuration` | ✓ | ✓ | ✓ | ✓ | ✓ |
| 전처리 `preprocessing`, `dataloader` | 학습 데이터 | 학습 + 테스트 | 입력 CSV | 새 행 단위 | 분석 CSV, 선택적 배경 CSV |
| 모델 `model`, `models/*` | 생성 | 생성 | 복원 | 복원 | 복원 |
| 학습 `trainer` | ✓ | ✓ |  |  |  |
| 보정 `scoring.calibrate` |  | 생성 | 파일 로드 | 파일 로드 |  |
| 추론 `inference` |  | ✓ | ✓ | 새 시점만 | 분석용 모델 출력만 직접 계산 |
| 점수 `postprocessing` |  | ✓ | ✓ | 상태 유지 |  |
| 알람 `alarm` |  | ✓ | ✓ | 상태 유지 |  |
| 평가 `evaluation` |  | 선택 | 선택 | 선택 |  |
| 해석 `explainer` |  | 선택 | 선택 (SHAP 제외) |  | 항상, 선택적 SHAP 포함 |
| 저장 `save`, `online_runner` | 요약 | 전체 | 전체 | SQLite → 보고서 | 해석 산출물만 |

## 9.4 수정 위치

| 바꾸려는 내용 | 수정할 곳 |
|---|---|
| 데이터 경로, 배치, 윈도우, 학습률 | `configs/config.yaml` |
| 임계값 방식, 변수별 임계값 | `configs/threshold.yaml` |
| 알람 레벨, 연속 횟수, inhibit, CRV | `configs/alarm.yaml` |
| 전처리 단계 추가 | `src/preprocessing.py`의 함수와 `PREPROCESS_FUNCTIONS` |
| 모델 구조, 손실, 이상 점수 | `models/<모델>.py` |
| 점수 정규화 방식 | `src/udf/postprocessing.py` |
| 임계값 계산 방식 | `src/udf/threshold.py` |
| 알람 판정 로직 | `src/alarm.py` |
| 모드별 실행 단계 | `src/workflow.py` |
| 출력 열, 파일 | `src/save.py` |
| 그림 | `src/udf/visualization.py` |

## 9.5 새 모델 추가

1. `models/<새모델>.py`에 네트워크, 손실, 이상 점수와 `AnomalyLightningModule`을 상속한 어댑터를 작성한다.
2. `src/model.py`의 `MODEL_REGISTRY`에 등록한다.
3. `src/udf/configuration.py`의 `MODEL_NAMES`와 모델 설정 검증(`validate_model_config`)에 반영한다.
4. `configs/config.yaml`의 `models.<새모델>`에 설정을 추가한다. 후처리와 알람은 공통 설정을 사용한다.
5. 해석이나 그림이 모델별로 달라야 하면 `src/explainer.py`, `src/save.py`를 확인한다.

모델은 공통 배치를 받아 공통 형식으로 반환해야 한다.

| 항목 | 형태 | 의미 |
|---|---|---|
| 입력 `context` | `[B, W, F]` | 배치 × 윈도우 × 특성 |
| 입력 `target` | `[B, F]` | 윈도우 마지막 시점 실측값 |
| 입력 `label` | `[B, F]` | 평가 라벨. 없으면 0 |
| 입력 `index`, `timestamp` | `[B]` | 행 번호, 유닉스 나노초 시각 |
| 반환 `loss`, `validation_loss` | scalar | 학습·검증 손실 |
| 반환 `prediction` | `[B, F]` | 복원값 (스케일링 단위) |
| 반환 `anomaly_score` | `[B, F]` | 변수별 raw_score |
| 반환 `score_<방식>` | `[B, F]` | 점수 비교 그림용 후보 점수 |
