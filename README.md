# 회전기계 EWS

1분 간격의 센서 데이터에서 정상 운전 패턴을 학습하고, 현재 값과 복원값의 차이로 이상 징후를 감지하는 TranAD 기반 Early Warning System입니다. 진동 Spectrum 대신 센서 시계열을 사용합니다.

이 저장소는 최종 A 전달본의 실행 소스, 선정 설정과 최종 통합 HTML을 제공합니다. HTML에는 그래프를 확인할 수 있도록 실제 센서 시계열과 평가 결과가 내장되어 있습니다. 별도 원본 CSV, 학습 checkpoint, scaler, 평가 DB, Codex 세션 로그는 포함하지 않습니다. 모델을 다시 실행할 때 필요한 데이터와 기존 모델은 별도로 준비합니다.

## 최종 결과 HTML

[최종 통합 HTML 다운로드](https://github.com/pkg2060/g09/raw/refs/heads/main/reports/EWS_final_offline.html) · [파일 위치](reports/EWS_final_offline.html)

다운로드한 `EWS_final_offline.html`을 Chrome 또는 Edge에서 열면 SageMaker에 접속하지 않고 최종 정리 화면과 선정 모델의 대시보드를 볼 수 있습니다. Tag 그래프의 확대·축소, 실험 기록, 랄프 증거와 저장된 SHAP 분석을 한 파일에 포함했습니다. SHAP 탭의 저장된 분석 구간은 2026년 6월 3일 14:00부터 14:10까지이며, 새로운 구간의 SHAP 계산에는 Python 실행 환경이 필요합니다.

HTML은 원본 그대로 약 229 MiB이며 Git LFS로 관리합니다. GitHub 화면에서 직접 실행되는 페이지가 아니므로 위 링크로 파일을 다운로드해서 여세요. 저장소를 clone해서 받을 경우 Git LFS를 설치한 뒤 `git lfs pull`을 실행하세요.

## Vercel 웹 화면

[EWS 최종 결과 사이트](https://g09project.vercel.app)

연결된 Vercel 프로젝트 `g09_project`는 `main` 변경을 배포합니다. `vercel.json`의 빌드 명령은 원본 HTML을 검증한 뒤 `site/`에 정적 웹 화면을 생성합니다. 첫 화면에는 최종 정리를 표시하고, 대시보드의 시계열은 Tag를 선택할 때 불러옵니다. 전체 데이터 포인트와 저장된 SHAP 결과를 유지합니다.

웹 배포는 완료된 결과를 조회하는 용도입니다. Python 모델 학습·추론·새 SHAP 계산은 위 웹 서버에서 실행하지 않으며, 아래 실행 환경에서 별도로 수행합니다.

Node.js 22에서 로컬 웹 산출물을 생성하려면 `npm run build`를 실행하세요. Git LFS 원본이 내려받아지지 않은 빌드 환경에서는 공개 GitHub 파일을 받은 뒤 크기와 고정 SHA-256을 검증합니다. 원본이 달라지면 빌드는 실패하도록 구성했습니다. 생성된 `site/`만 웹에 배포하며, Python 실행 소스나 로컬 파일은 웹 경로로 노출하지 않습니다.

## 주요 기능

- 시간순 Train 앞 80%에서만 정규화 기준을 학습하고, 동일 기준으로 Validation과 Test를 변환합니다.
- 각 입력 시점의 복원값과 정규화 절대오차를 계산합니다. 초기 Window 행도 보존하며 시간 불연속 구간을 구분합니다.
- Tag별 오차가 Threshold를 엄격하게 초과(`>`)하면 Warning을 생성합니다.
- 공정 정지는 원단위 센서 조건으로 판단해 Alarm을 억제합니다. `Inhibit`과 `Alarm_*`는 채점 전용이며 모델 입력에 사용하지 않습니다.
- 전체 시점의 CSV·SQLite·NPZ, 정적 그래프, 확대·축소·이동·구간 CSV가 가능한 대시보드를 생성합니다.
- Python 서버에서 선택 구간의 SHAP를 계산합니다. 감사 기록은 실행 명령, 종료코드, 파일 해시와 검증 주장을 연결합니다.

## 구조

```text
mode.py                    Jupyter / CLI 진입점
src/
  config.py                기본 설정
  profiles/delivered.json  최종 선정 모델의 재실행 설정
  data.py                  CSV 계약, 시간 분할, 정규화, Window
  model.py, trainer.py     TranAD 모델, 학습, 복원
  detection.py             Threshold, 정지 억제, 채점
  pipeline.py              full / train / eval 실행
  dashboard.py, reports.py 결과 화면과 보고서 생성
  explain.py, serve.py     SHAP와 로컬 분석 서버
  audit.py, verify.py      감사와 산출물 검증
  test_contracts.py        데이터·추론·감사 계약시험
  test_bounded_output.py   출력층 계약시험
  vendor/                  외부 코드와 라이선스 고지
data/README.md             입력 데이터 안내
docs/THIRD_PARTY.md        출처와 파일별 라이선스
reports/EWS_final_offline.html  최종 정리 + 오프라인 대시보드 (Git LFS)
```

## 실행 환경

기존 실행 환경은 SageMaker의 Python 3.12, PyTorch 2.8 CUDA, NVIDIA Tesla T4입니다. 실제 학습·평가·SHAP에는 CUDA를 사용할 수 있는 PyTorch 환경이 필요합니다. 아래 계약시험은 CPU에서 실행할 수 있습니다.

저장소를 받은 후, `mode.py`가 있는 프로젝트 루트를 Jupyter의 작업 폴더로 사용하세요.

```python
%pip install -r src/requirements.txt
```

PyTorch는 현재 장비와 드라이버에 맞는 CUDA 설치본을 준비합니다. 코드는 CUDA 사용 가능 여부를 확인하며, 실제 학습을 CPU로 조용히 바꾸지 않습니다.

## 데이터 준비

정상 운전 Train과 평가용 Test를 다음 위치에 둡니다. 저장소는 이 파일들을 Git에서 제외합니다.

```text
data/train.csv
data/test.csv
```

CSV는 `DateTime` 시각 열과 `.PV`로 끝나는 센서 열을 사용합니다. `Alarm_<Tag>`는 Tag별 이상 정답, `Inhibit`는 공정 정지 정답입니다. 자세한 입력 규칙은 [데이터 안내](data/README.md)를 확인하세요.

## Jupyter 파이프라인

최종 선정 설정은 **Window 20, FF 64, 목표 Epoch 16, scaled_sigmoid margin 0.25, L1 목적함수**입니다. 아래 예시는 반드시 `--config src/profiles/delivered.json`을 사용합니다. 설정 인수를 생략하면 초기 기본값인 Window 10, Epoch 10을 사용합니다.

아래 명령은 목적에 맞게 선택합니다. `-full`과 `-train`을 둘 다 실행하면 각각 새 학습을 수행합니다. 실행마다 새로운 `--run-id`를 사용하세요.

**새 학습부터 평가와 출력까지 한 번에 실행**

```python
%run mode -full --config src/profiles/delivered.json --run-id full_001
```

**학습만 실행**

```python
%run mode -train --config src/profiles/delivered.json --run-id train_001
```

**저장 모델을 다시 평가**

```python
%run mode -eval --config src/profiles/delivered.json --checkpoint results/train_001/last.pt --run-id eval_001
```

`-eval`도 Train/Test CSV와 checkpoint 옆의 동일한 `scaler.pkl`이 필요합니다. 기존 실험의 결과를 재현하려면 당시의 checkpoint·scaler·CSV를 함께 사용해야 합니다. 선정 설정으로 새로 학습하는 것은 당시 가중치를 그대로 사용하는 것과 다릅니다. 조기 종료 설정에 따라 실제 완료 Epoch가 목표값보다 작을 수 있습니다.

터미널에서는 `%run mode` 대신 `python mode.py`를 사용합니다.

## 대시보드와 SHAP

평가 완료 후 대시보드와 `dashboard_data/`를 함께 보관합니다. 이 자산에는 실제 센서값과 정답이 들어가므로 공개 저장소에서는 제외합니다.

```python
%run mode -serve --run-id eval_001 --checkpoint results/train_001/last.pt
```

서버가 안내하는 주소로 접속합니다. SHAP에는 해당 평가의 NPZ, 동일한 checkpoint와 scaler, Train 데이터가 필요합니다. GitHub 저장소나 정적 HTML만으로 Python 모델을 실행하지는 않습니다.

## 설정과 검증 범위

`src/profiles/delivered.json`은 기존 7개 Tag와 그 Tag별 수동 Threshold를 대상으로 합니다. 다른 설비를 적용할 때는 `n_heads`의 호환성, Tag별 Threshold, 공정 정지 조건을 검토해 별도 설정 파일을 만드세요. `n_heads: null`은 센서 수를 기준으로 head 수를 정합니다.

모델 성능 기준은 학습 기준으로 정규화한 전체 Test MAE `< 0.02`와 모든 Tag의 동일 시점 Label·최종 Alarm 일치율 `≥ 95%`입니다. 일치율과 Recall은 다른 지표이므로 Precision, Recall, 오탐·미탐도 함께 확인합니다. 기존 선정은 동일 Test를 반복 활용한 개발평가이며, 새 데이터의 성능을 보장하지 않습니다.

`src/optimize.py`와 `src/experiment_configs/`는 기존 실험 캠페인 및 회귀시험을 위해 보존했습니다. 과거 마감시간과 Linux 실행 감시를 전제로 하는 `-optimize`는 위의 일반 재실행 절차에 포함하지 않습니다. 이번 캠페인을 자동으로 재개하지 마세요.

## CPU 계약시험

```bash
python -m unittest src.test_contracts src.test_bounded_output -v
python mode.py --help
```

계약시험은 합성·모의 데이터로 정규화, 시점 정렬, 정답 분리, 정지 억제, 감사, 모델 출력층을 검증합니다. 실제 데이터의 GPU 학습 결과나 성능 평가를 대신하지 않습니다.

## 외부 코드

TranAD 구조를 참조한 모델과 SPOT 코드의 출처는 [파일별 라이선스 안내](docs/THIRD_PARTY.md)에 정리했습니다. `src/vendor/LICENSE.TranAD`와 `src/vendor/LICENSE.GPL-3.0`, 외부 파일의 원래 고지를 함께 보존합니다.
