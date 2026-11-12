# 기존 출력물 복원 결과

## 검토 기준

`/workspace/GSC_EWS/outputs/prior_runs/tranad_ews_v2/`의 실제 파일·CSV 열·Excel 시트와
공통화 전 TFT·TranAD 저장 코드를 대조했다. TFT 보관 산출물은 없어 기존 소스를 기준으로 확인했다.

## 복원한 항목

| 항목 | 현재 저장 결과 |
| --- | --- |
| 복원 품질 텍스트 | `reconstruction_quality.txt`: 전체·변수별 지표, Best/Worst, 지표 설명을 기존 형식으로 유지 |
| 복원 품질 Excel | `reconstruction_quality.xlsx`: `Overall Summary`, `Per Variable`, `Metric Guide`와 변수별 평균 행 복원. `report.xlsx`에도 같은 3개 시트를 기록 |
| 종합 텍스트 | `overall_metrics.txt`: 탐지·순위 지표와 복원 품질 요약. 라벨이 없거나 정의할 수 없는 지표는 N/A |
| 예측 CSV의 전체 점수 | `predictions.csv`의 `score_overall`: 해당 시점의 변수별 최종 점수 평균 |
| 예측 CSV의 시점별 RMSE | `<변수명>__rmse`: 기존처럼 각 시점·변수의 `sqrt(error²)` 값이며 같은 단위의 MAE와 같음 |
| 입력 부가 컬럼 | 전처리·윈도우 선택 후 예측과 같은 행 순서로 보존. 생성 컬럼과 이름이 충돌하면 `input__` 접두어로 구분 |
| 알람 CSV의 전체 점수·메시지 | `alarms.csv`의 `score_overall`, `message`: 실제 이벤트 시각에 해당하는 평균 점수와 변수별·전체 알람 설명 |
| 변수별 CSV의 임계값 | `metrics.csv`에 `threshold` 포함. `thresholds.csv`도 유지 |
| TFT 해석 Excel | `tft_interpretability.xlsx`: `Variable Importance`, `Lag Importance`, `Alarm Root Causes` 3개 시트. 실제 알람이 없으면 원인 시트는 빈 표 |
| TranAD 헤드별 그림 | `analysis/<모듈명>_per_head.png`: 원본 헤드 가중치의 표본 평균을 헤드마다 표시 |
| TranAD 시간별 중요도 | `analysis/<모듈명>_temporal_importance.png`: 평균 Attention의 키 방향 중요도를 표시 |
| 대상 변수별 SHAP | `analysis.shap_target`으로 선택. 해당 모델 점수의 SHAP을 계산하고 자기 변수를 붉게 표시. null이면 전체 평균 점수 |

공통 PNG는 변수 7개·학습 이력 포함 실행 기준 기존 21개를 유지한다. TFT 기존 `interp_*.png`
그림도 유지한다. 새 TranAD 그림은 예전 화면 표시 기능을 저장 파일로 연결한 것이다.

## 유지하는 현재 데이터 계약

- 출력 디렉터리에 보고서·CSV를 함께 저장한다. 예전의 `results/` 하위 폴더 구조를 추가로 복제하지 않는다.
- `predictions.csv`, `alarms.csv`, `metrics.csv`는 현재 파일명을 사용한다.
  이전 이름의 `System_Output_database.csv`, `Alarm_database.csv`, `per_variable_metrics.csv`를 중복 생성하지 않는다.
- `thresholds.csv`의 열 이름은 현재의 `feature,threshold`를 유지한다.
- 복원 품질과 CSV 오차는 현재 원본 공정 단위를 따른다. 정규화 단위였던 과거 실험의 숫자를 재현하는 변경이 아니다.
- 라벨 없는 데이터에 가짜 탐지 성능을 채우지 않는다. 예측·점수·임계값·실제 알람 판정은 저장 복원과 분리한다.
- 알람은 공통 AlarmGenerator의 쿨다운·변수별 이벤트 판정을 그대로 저장한다. 이전 배치 저장기의 필터로 이벤트를 다시 제거하지 않는다.
- Excel은 `output.excel`, 공통 그림은 `output.plots`, 해석은 `analysis.enabled` 및 해당 Attention/SHAP 설정에 따른다.
- TFT 해석 Excel은 `analysis.max_samples` 내 표본을 집계한다. 지연 축은 현재 윈도우 마지막 입력인 `lag_0` 기준이며,
  알람 원인 행에는 표본 순번 대신 실제 입력 인덱스를 저장한다. 표본 목록은 `analysis/samples.csv`에서 확인한다.
- 체크포인트는 `checkpoints/<설비>/<모델>/<버전>/<실행ID>/<fold>/` 아래에 Lightning의
  `epoch=...ckpt`, `last.ckpt`를 저장한다.

보고서·시트·열·해석 그림의 누락을 복원했다. 파일명 변경·공통 데이터 계약과 과거 실험 조건까지 동일해진다는 뜻은 아니다.


## 실행 옵션에 따른 생성 범위

현재 실행 모드는 `full/train/evaluate/online/analyze`이다. 위 복원 산출물 중 지표·품질 보고서와
품질·성능 그림은 `evaluation.enabled: true`일 때 생성한다. false에서는 시점별 예측·점수·알람·임계값,
선택적 변수 그림·점수 비교·알람 Excel을 저장한다. full/evaluate의 후속 해석은 `analysis.enabled`를
따르고, `analyze`는 저장된 Checkpoint와 분석 CSV를 사용해 해석 산출물만 생성한다. SHAP을 사용할 때는
정상 배경 CSV가 추가로 필요하며 Calibration·점수 후처리·알람·평가는 실행하지 않는다.
온라인은 `online.sqlite3`에 결과·상태를 함께 확정하고 누적 보고서를 갱신한다.
