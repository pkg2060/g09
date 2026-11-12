# GSC_EWS

TFT·TranAD가 전처리·DataLoader·Lightning 학습·추론·점수·알람·저장을 공유합니다.
전체 설명과 기존 산출물 대응표는 [통합 파이프라인 가이드](docs/pipeline_guide.md)를 참고하세요.
개발 시 [AGENTS.md](AGENTS.md)와 [시큐어 코딩 지침](docs/secure_coding_guide.md)을 따릅니다.

## 설치

Python 3.10 이상과 서버에 맞는 PyTorch 환경을 사용합니다.

```bash
python -m pip install -r requirements.txt
# SHAP을 사용할 때 추가 설치합니다.
python -m pip install -r requirements-analysis.txt
```

## 네 가지 실행 모드

| 모드 | 역할 |
| --- | --- |
| `full` | 시계열 교차 검증·전체 정상 데이터 재학습 → OOF 보정 → 일괄 추론·선택적 평가·해석·저장 |
| `train` | 학습·검증·체크포인트·로그 저장. 테스트 CSV 불필요 |
| `evaluate` | 저장 체크포인트·보정값으로 유한한 CSV를 일괄 추론·평가 |
| `online` | 모델·보정값을 한 번 로드하고 새 관측값만 지속 추론 |

이전 `all`, `predict`, `--watch`, `run.watch`는 제거했습니다.
`full`은 온라인 운영을 자동으로 시작하지 않습니다.

```bash
cd /workspace/GSC_EWS
python main.py --mode full --model tranad_v2 --output results/tranad_run01
python main.py --mode full --model tft --output results/tft_run01
python main.py --mode train --model tranad_v2 --fast-dev-run

# 실제_최적모델.ckpt는 학습 종료 시 출력된 경로로 바꿉니다.
python main.py --mode evaluate \
  --checkpoint checkpoints/tranad/실제_최적모델.ckpt \
  --calibration results/tranad_run01/calibration.yaml \
  --input data/new_data.csv --output results/tranad_batch01

python main.py --mode online \
  --checkpoint checkpoints/tranad/실제_최적모델.ckpt \
  --calibration results/tranad_run01/calibration.yaml \
  --input data/online.csv --output results/tranad_online01
```

평가·온라인의 모델 종류와 정밀도는 체크포인트를 따릅니다. `--model`은 full/train 전용입니다.
`evaluate`는 `--input`이 없으면 `data.test_path`를 사용합니다.
`evaluate`에는 체크포인트와 보정 파일이 모두 필요합니다. 체크포인트와 보정 파일은 해시가 일치해야 합니다.
`TimeSeriesSplit` 학습은 fold마다 새 모델을 만들기 때문에 단일 체크포인트 재개를 지원하지 않습니다.
각 fold의 최저 검증 손실 epoch를 구한 뒤 그 중앙값만큼 전체 정상 데이터를 다시 학습해 운영 모델을 만듭니다.
보정에는 모든 fold의 OOF(Out-of-Fold) 검증 점수를 사용합니다.

## 설정

- `configs/config.yaml`: 실행·데이터·전처리·DataLoader·모델·학습·후처리·평가·출력·분석·온라인
- `configs/alarm.yaml`: 알람 레벨·연속 횟수·최소 변수 수·쿨다운·억제 조건
- `configs/threshold.yaml`: 수동·백분위수·Sigma·POT 임계값

YAML 상대 경로는 YAML 위치, CLI 상대 경로는 실행 폴더 기준입니다.
모델별 EWS 설정은 각 파일의 `model_overrides`가 공통값보다 우선합니다.
현재 TranAD v2는 검증 분위수 기반 점수 정규화와 백분위수 임계값을 사용하고, TFT는 POT를 사용합니다.
`data.train_labels_path`를 지정하면 정답 라벨이 1인 행을 제외하고 정상 행만 학습·교차 검증에 사용합니다.

```yaml
# evaluate에서 저장 보정값으로 추론·점수·알람만 저장할 때 사용합니다.
evaluation:
  enabled: false
analysis:
  enabled: false
```

평가를 켜면 복원 품질을 계산하고 정답 라벨이 있으면 탐지 성능도 계산합니다.
라벨이 없으면 탐지 지표는 N/A입니다. 평가를 끄면 품질·지표 보고서 자체를 생성하지 않습니다.
해석은 full/evaluate에서 별도로 제어하며 SHAP에는 학습 배경이 필요합니다.
`output.plots`·`output.excel`은 공통 그림·Excel을 제어합니다.

## 온라인 처리

현재 입력은 시간순으로 행이 추가되는 CSV입니다. 기본 수집·확인 간격은 60초입니다.
새 행이 없으면 추론하지 않으며, 여러 행이 들어오면 각각의 새 시점을 순서대로 처리합니다.
최근 윈도우·과거값 채움·이동 평균·연속 알람·쿨다운 상태를 유지합니다.
결과와 상태는 SQLite의 같은 트랜잭션에 저장하여 재시작·오류 복구 시 중복 추론을 막습니다.

기본 전처리는 `forward_fill`입니다. 미래 관측값을 쓰는 `interpolate` 체크포인트는
온라인에서 거부하므로 인과적 전처리로 다시 학습·보정해야 합니다.
CSV 작성기는 완성된 파일을 원자적으로 교체하고 기존 행·컬럼을 변경하지 않아야 합니다.

추론은 새 시점에만 수행하지만 입력 앞부분 확인·누적 보고서 갱신은 전체 이력 크기에 비례합니다.
장시간 운영 시 불필요한 평가·플롯·Excel을 끄고, 더 큰 규모의 입력 커서·보고서 분리는 별도 확장 대상입니다.
RTDB/API 어댑터는 아직 구현되어 있지 않습니다.

## 결과와 코드 구조

기본 출력은 `results/<설비>/<모델>/<버전>/<날짜시각_UUID 8자리>/`이며 online은 그 아래 `online/`입니다.
학습 실행 폴더에는 적용된 `config.yaml`과 `summary.yaml`을 저장하고, 전체 학습 결과는
`results/experiments.csv`에서 설비·모델·버전과 관계없이 한 번에 조회할 수 있습니다.
`--output`으로 지정한 `<실험 폴더>` 바로 아래 CSV·TXT·Excel과 `plots/`, `analysis/`를 저장합니다.

- `predictions.csv`, `alarms.csv`, `thresholds.csv`: 시점별 결과·알람·임계값
- `config.yaml`, `summary.yaml`: 해당 학습 실행의 적용 설정과 핵심 성능 요약
- `results/experiments.csv`: 모든 설비·모델·버전의 학습 실행 통합 목록
- 평가 활성화: `metrics.csv/json`, `overall_metrics.txt`, `reconstruction_quality.txt/xlsx`
- Excel 활성화: `report.xlsx`; 평가를 끄면 알람 시트만 기록
- 공통 플롯: 평가 활성화 시 변수 N개 기준 `2N+6`, 실제 학습 이력이 있으면 1개 추가
- 평가를 끈 공통 플롯: 변수 분석 N개·점수 비교 1개, 학습 이력이 있으면 1개 추가
- 해석 활성화: Attention·TFT 해석·선택적 SHAP; online에서는 해석하지 않음
- 온라인의 `online.sqlite3`: 결과와 처리 상태의 원본. CSV·보고서는 이 결과에서 갱신

학습은 `Trainer.fit()`, 보정·일괄·온라인 추론은 `Trainer.predict()`를 사용합니다.
학습 로그와 체크포인트는 모델 버전·실행 ID·fold별 폴더에 저장합니다.
`--output`은 학습 로그·체크포인트 위치를 바꾸지 않습니다.

- `main.py`: CLI와 모드 선택
- `src/workflow.py`: 실행 단계 연결
- `src/`: 전처리·데이터·학습·추론·평가·저장·온라인 단계
- `src/udf/`: 설정 검증·지표·점수·임계값·그리기 보조 함수
- `models/<모델명>.py`: 모델 구조·손실·점수·Lightning 어댑터
- `src/model.py`: 모델 클래스 연결을 관리하는 단일 레지스트리

## 검증

```bash
python -m unittest discover -s tests -v
```
