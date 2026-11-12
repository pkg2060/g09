# 코드 공통화 검토 및 이관 결과

## 검토 결과

루트 전처리·DataLoader와 Lightning 어댑터가 있었지만 공통화는 완료되지 않은 상태였습니다.
어댑터는 두 모델의 `src/model.py`를 가져왔고, 각 `main.py`는 별도 수동 Trainer와
전처리·윈도우 생성 코드로 실행됐습니다. EWS·분석 코드도 Python config의 전역 변수와
구형 체크포인트 형식에 의존했습니다.

## 이관표

| 기존 파일 또는 기능 | 현재 구현 | 처리 내용 |
| --- | --- | --- |
| 양쪽 `src/model.py` | `models/tft.py`, `models/tranad.py` | 모델별 파일에 네트워크와 Lightning 어댑터를 통합하고 수식·차원 설명 보존. alias, 주석 처리된 구형 구현, 최적화·체크포인트 헬퍼 제거 |
| 양쪽 `src/trainer.py` | `models/base.py`, 모델 어댑터, `src/trainer.py` | 수동 학습·검증 루프 제거. Lightning 최적화·TimeSeriesSplit·최적 가중치 복원으로 통일 |
| 양쪽 `src/dataloader.py` | `src/preprocessing.py`, `src/dataloader.py` | 정규화·역변환·CSV·윈도우 생성 통합. 지연 윈도우 및 저장 스케일러 복원 사용 |
| 양쪽 `src/scoring.py` | 모델 어댑터, `src/udf/metrics.py` | 모델 점수 정의 단일화. 분류·Hit/NDCG·별도 Point-Adjust 평가 공통화 |
| `TFT/src/tft_postprocess.py` | `models/tft.py`, `src/udf/postprocessing.py` | 분위수·구간 위반 점수는 어댑터로, 스무딩·MAD 보정은 공통 모듈로 이관 |
| 양쪽 `src/threshold.py`, `src/spot.py` | `src/udf/threshold.py` | Manual·POT·Percentile·Sigma 통합. 테스트값을 받지 않는 보정 API. 사용하지 않는 biSPOT/dSPOT/bidSPOT 제거 |
| 양쪽 `src/alarm.py` | `src/alarm.py` | 연속 감지·쿨다운·최소 변수·원본 단위 억제 조건 통합. 상태/이벤트 판정의 중복 제거 |
| 양쪽 `src/online_runner.py` | `src/online_runner.py`, `main.py` | CSV 감시를 표준 추론 경로로 연결. 입력 파일 삭제·trim 및 가짜 시각의 모의 루프 제거 |
| 양쪽 `src/visualizer.py` | `src/save.py` | 실측·복원·점수·임계값·알람 그래프 통합. 반복되는 표·차트별 구현은 CSV/Excel 보고서로 대체 |
| 양쪽 `src/reconstruction_quality.py` | `src/udf/quality.py`, `src/save.py` | 기존 MAE/RMSE/R²/MAPE 수식과 설명 주석 보존. 출력 코드 공통화 |
| `TranAD/src/output_saver.py` | `src/save.py` | 실제 시각·원본 단위 복원·오차·알람을 같은 행에 저장. 소스 CSV와 위치로 재결합하는 중복 경로 제거 |
| `TFT/src/interpretability.py` | `src/explainer.py` | VSN 중요도·시점별 가중치·분위수·점수 기여도·Attention 저장과 시각화 통합 |
| `TranAD/src/attention_analyzer.py` | `src/explainer.py` | 공통 모델의 Attention hook 사용. 헤드별 원본 배열 및 평균 히트맵 제공 |
| `TranAD/src/shap_explainer.py` | `src/explainer.py` | 표준 어댑터 점수로 Kernel/Gradient/Permutation SHAP. 학습 배경과 dtype을 공통 처리 |
| 양쪽 `main.py` | `main.py`, `src/workflow.py` | 단일 진입점의 full/train/evaluate/online/analyze 모드로 통합하고 개별 CLI 삭제 |
| 모델별 `config.py`, `explainer_config.py`, `config_compat.py`, `config_bindings.py` | `configs/`의 YAML 3개와 `src/udf/configuration.py` | Python 설정 래퍼·대문자 별칭·전역 설정 제거 |
| 모델별 YAML, requirements, 패키지 초기화 파일 | 공통 설정·의존성·패키지 | 중복 파일 삭제 |
| 양쪽 `preprocess_benchmark.py`, 합성 데이터 생성기 | 제거 | 현재 CSV 파이프라인에서 사용하지 않는 벤치마크 변환·다운로드·예제 경로 삭제 |
| `legacy_code/`의 소스 전체 | 제거 | 중복 사본 삭제. 실행 가능한 레거시 경로를 보존하지 않음 |

## 파이프라인 계층

- `src/`는 전처리·로더·모델 선택·학습·추론·점수 처리·알람·평가·해석·저장·온라인 감시 단계와 `workflow.py`를 둡니다.
- `configuration.py`, `metrics.py`, `quality.py`, `postprocessing.py`, `threshold.py`는 공통 보조 모듈로 분류해 `src/udf/`로 이동했습니다.
- 기존 함수·클래스는 기능 모듈에 함께 유지합니다. 각 상위 모듈에 대응하는 보조 파일을 별도로 만들지 않았습니다.
- `pipeline.py`의 추론과 점수 보정을 `inference.py`, `scoring.py`로 이관했습니다. 알람 판정과 평가 호출은 `workflow.py`가 연결합니다.
- 저장 함수의 지표 계산은 `evaluation.py`로 옮겼습니다. `save.py`는 계산된 지표와 결과를 기록합니다.
- `models/__init__.py`의 중복 레지스트리·생성 함수를 제거하고 `src/model.py`로 통합했습니다. 학습은 `active_model`, 복원은 저장된 모델명을 사용합니다.
- 전처리·DataLoader·학습·해석·저장·온라인 모듈명은 각각 `preprocessing`, `dataloader`, `trainer`, `explainer`, `save`, `online_runner`로 정리했습니다. 기존 파일·호환 래퍼는 제거했습니다.
- 온라인 감시는 `workflow.py`에서 전달받은 처리 함수를 재사용해 순환 import 없이 배치 실행과 같은 결과를 만듭니다.
- 보정·평가·독립 추론·CSV 감시는 `Trainer.predict()`를 사용합니다. 수동 배치 반복문과
  `predict_step()` 직접 호출을 제거하고 장치 이동·정밀도·CPU 결과 수집을 Lightning에 맡깁니다.
- `full`은 학습 Trainer에 복원된 최적 모델을 명시적으로 전달합니다. 독립 평가·추론은
  저장된 모델 설정으로 Trainer를 한 번 구성하며 CSV 감시 중에도 재사용합니다.
- 학습 전용 로거·체크포인트·조기 종료는 학습 Trainer에만 구성합니다. 후처리·알람·최종
  지표와 파일 저장은 공통 파이프라인에서 처리하고, 해석은 추론 루프 밖에서 실행합니다.

## 설정 구조

- `configs/config.yaml`에 실행·데이터·전처리·DataLoader·모델·학습과 후처리·출력·분석·감시를 통합했습니다.
- 알람 규칙은 `configs/alarm.yaml`, 임계값은 `configs/threshold.yaml`에서만 관리합니다.
- `config.yaml`의 `run.alarm_config_path`와 `run.threshold_config_path`로 두 파일을 연결합니다.
- 후처리·출력·분석·감시 설정은 모델별 덮어쓰기 없이 모든 모델에 공통으로 적용합니다.
- 알람과 임계값은 모델을 구분하지 않고 모든 모델에 공통으로 적용합니다.
- 임계값은 `threshold.yaml`의 `method`를 명시적으로 적용하고, 평가·추론은 저장된 보정값을 따릅니다.
- 상대 경로는 YAML 위치를 기준으로 해석하며 기본 데이터·산출물은 프로젝트 루트 위치를 유지합니다.

## 예외 검증 정리

동일한 조건을 하위 구현에서 반복하는 검사만 제거했습니다. `ValueError` 발생 지점은
93개에서 81개로 줄었으며, 예외 종류를 바꾸어 개수만 줄이지 않았습니다.

| 정리한 중복 검사 | 검증을 담당하는 위치 |
| --- | --- |
| 모델 공통부의 window_size·dtype | `configuration.validate_model_config` — YAML·직접 어댑터 생성·체크포인트 복원에서 사용 |
| optimizer·scheduler 재검사 | `configuration.validate_training_config`와 어댑터의 `load_training_config` |
| TFT 내부 head 수·분위수 범위/순서·손실 가중치 및 Attention assert | `configuration.validate_model_config` — 내부 네트워크 생성 전에 검사 |
| Trainer의 장치·validation_metric | 설정의 `validate_run_config`·`validate_training_config`; 실제 장치 지원은 Lightning이 확인 |
| DataLoader의 분할 비율·최소 행 수 재검사 | `preprocessing.preprocess_training_splits` — 전처리 전후 분할을 확인 |
| CSV 존재 여부 사전 검사 | pandas의 `FileNotFoundError` — 파일 읽기 시 경로를 포함해 보고 |
| load_config의 deterministic 불리언 재검사 | 같은 함수에서 먼저 호출하는 `validate_run_config` |

다음 검증은 서로 다른 입력·처리 단계의 오류를 막으므로 유지했습니다.

- `preprocessing`·`dataloader`: 누락 컬럼·결측/무한대·빈 데이터·라벨 정렬·시간 순서·윈도우 생성 가능 여부·스케일러 복원 상태.
- 모델 어댑터: 직접 생성 시 특성 수, 실제 입력 텐서 shape, 모델별 손실·점수·예측 길이, 특성 수에 따라 정해지는 TranAD 구조 제약.
- `inference`·`scoring`: 중복 추론 행·비정상 점수, 체크포인트/보정 해시·특성 순서·배열 크기·스케일의 일치.
- `postprocessing`·`threshold`·`alarm`: 독립 함수로 전달되는 배열·구간·특성 수, 계산 결과의 유효성, POT 실패 시 학습 백분위수 대체.
- `online_runner`: 기존 입력 행의 수정·삭제와 모델/보정 변경 감지. `explainer`는 학습 배경과 지원 SHAP 방식을 확인.
- `configuration`·`model`·`workflow`·`main`: 잘못된 설정·모델명·필수 파일·실행 모드와 체크포인트 생성 실패를 명확히 보고.

## 동작 변경

- 학습/검증 분할 전 보간을 제거해 검증 값이 학습 경계로 유입되지 않게 했습니다.
- TFT의 중앙 이동 평균을 과거 방향 이동 평균으로 바꾸고 단절에서 초기화합니다.
- MAD 보정의 음수 편차는 정상 영역인 0으로 제한합니다.
- POT는 SciPy의 GPD 추정으로 단순화했습니다. 구형 SPOT 추정기와 숫자가 같다고 가정하지
  않으며 새 모델을 반드시 다시 보정합니다. 표본 부족 시 학습 백분위수로 대체하고 알립니다.
- 알람 레벨의 높은 값이 낮은 레벨로 덮이는 문제를 제거했습니다.
- 이상 점수와 알람 판정은 파일 출력·해석에서도 같은 구현을 사용합니다.
- 구형 체크포인트와 별칭·Python config를 읽는 호환 코드는 제거했습니다.
  새 통합 모델을 학습해 체크포인트를 생성해야 합니다.
- CSV 감시는 전체 입력 이력을 재평가합니다. 결과 재현성을 우선하며 누적 행 수에 따른
  처리 비용이 있습니다. 입력에는 시간순 append만 허용합니다.

유용한 네트워크 설명과 복원 품질 수식의 주석을 가능한 그대로 이관했습니다.
실행하지 않는 코드 블록과 삭제된 API만 설명하는 주석은 함께 제거했습니다.

## 검증 결과

- 원격 환경에서 회귀 테스트 통과. 두 모델의 학습·보정·평가·독립 추론,
  세 SHAP 방식, 단일 main의 모드 분기와 YAML/CLI 우선순위, 세 설정 파일의 모델별 병합·경로·오타 검증을 확인했습니다.
- 모델명의 공백·대소문자 정리, 잘못된 이름·설정의 조기 오류, float32/float64 일치와 저장된 모델·dtype 복원을 확인했습니다.
- Trainer 추론으로 전환한 뒤에도 회귀 테스트 29개가 통과했습니다. 모드별 Trainer 재사용,
  저장 dtype 우선 적용, 온라인 진입점, 독립 추론의 학습 콜백·로거 비활성화를 함께 검증합니다.
- 두 모델의 기존 체크포인트로 CPU·GPU 각각 보정 및 라벨 유무별 추론을 실행했습니다.
  보정값·예측·알람·임계값·지표 파일 44개가 수동 추론 경로와 바이트 단위로 일치했습니다.
- 예외 검증 정리 후에도 동일한 44개 파일의 일치를 확인했습니다. 직접 어댑터 생성 시
  잘못된 구조·dtype·optimizer·scheduler와 전처리의 잘못된 분할·누락 CSV가 계속 거부됩니다.
- 전환한 GPU 전체 모드에서 실제 sample1의 학습 64행·테스트 32행·7개 특성을 사용해
  두 모델의 1 epoch 학습, 보정·평가, Attention·Gradient SHAP, 독립 추론을 확인했습니다.
  학습 로그와 최적·마지막 체크포인트를 생성하고 독립 추론은 학습 산출물을 추가하지 않았습니다.
- 계층 정리 전후에 기존 두 모델의 체크포인트·보정값·실제 sample1 입력으로 실행했습니다. 라벨 유무별 예측·알람·임계값·지표 파일 20개가 바이트 단위로 일치했습니다.
- 기존 YAML과 새 세 파일을 대조해 두 모델의 설정값이 모두 보존된 것을 확인했습니다.
- 전체 모드에서 학습·DataLoader 구성이 한 번만 실행되는지 확인했고, 평가·추론 모드는
  최적화를 호출하지 않도록 검증했습니다. 저장 보정값을 쓰면 학습 CSV 없이 평가할 수 있고,
  추론은 학습·테스트 CSV를 삭제한 상태에서도 별도 입력만으로 실행됐습니다.
- 실제 sample1의 학습 64행·테스트 32행·7개 특성으로 두 모델의 A100 GPU 전체 모드를
  실행했습니다. main CLI가 학습 1 epoch부터 보정·평가·결과 저장까지 완료했습니다.
- 이 검증은 실행·수치 계약 점검이며 전체 데이터 학습이나 탐지 성능 평가가 아닙니다.


## 기존 플롯 복원

- 기존 시각화 함수의 레이아웃·색상·범례·변수명 포함 파일명을 `src/udf/visualization.py`로 이관했습니다.
- `save.py`가 변수별 분석·절대 오차, 점수 비교, 성능·복원 품질 표와 요약 그림을 연결합니다.
- 20개 변수 제한을 제거했습니다. 학습 곡선은 Lightning CSVLogger의 epoch별 loss와 실제 학습률을 사용합니다.
- 모델 어댑터가 이미 계산한 모든 점수를 추론 결과로 전달하므로 그리기 코드에서 점수 공식을 재구현하지 않습니다.
- TFT 해석 그림은 기존 `plots/interpretability/interp_*.png` 이름으로 저장합니다.
- 원본 행·시각 정렬과 원본 단위 복원을 유지합니다. 라벨 없는 데이터의 탐지 지표는 N/A로 표시합니다.


## 모드 구체화와 온라인 처리

- `full/train/evaluate/online/analyze`로 실행 수명주기를 구분하고 저장 모델 해석을 독립 실행할 수 있다.
- `train`은 테스트·라벨 CSV를 읽지 않는다. 일회성 추론은 `evaluate`와 `evaluation.enabled: false`를 사용한다.
- `online`은 새 시점만 추론하고 제한된 윈도우·점수·알람 상태를 SQLite 결과와 함께 저장한다.
- Train/Validation 결측은 `drop`, `forward_fill`, `interpolate` 중 선택한다. Test/Evaluate/Online은 원본 결측을 계기 이상으로 기록하고 추론하지 않는다.
- 온라인 CSV 무결성 확인과 누적 보고서는 전체 이력을 읽는다. 입력 커서·증분 보고서는 아직 구현하지 않았다.
- 사용자 보안 지침은 `docs/secure_coding_guide.md`에 보관하고 `AGENTS.md`에서 개발 규칙으로 연결했다.
