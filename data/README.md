# 입력 데이터

이 폴더에 `train.csv`와 `test.csv`를 직접 둡니다. 실제 데이터는 Git에 포함하지 않습니다.

- `DateTime`: 시각 열. 기본 설정은 1분 간격입니다.
- 센서 열: `.PV` 접미사를 사용합니다. Train과 Test의 센서 구성이 일치해야 합니다.
- `Alarm_<Tag>`: 해당 센서의 이상 여부를 채점하는 정답 열입니다. 예를 들어 `Example.PV`의 정답 이름은 `Alarm_Example.PV`입니다.
- `Inhibit`: 공정 정지 여부의 정답 열입니다. 이상 정답으로 사용하지 않습니다.

Train은 정상 운전 구간입니다. 전처리는 원본 행을 임의로 삭제하지 않으며, 잘못된 값은 기본 설정에서 명시적으로 실패합니다. 정답 열이 없는 Test도 추론할 수 있지만 해당 채점은 미검증으로 표시합니다.

공정 정지 억제에는 정답 `Inhibit`가 아니라 `src/config.py` 또는 선택한 JSON 설정의 원단위 센서 조건을 사용합니다. 대상 설비에 해당 센서가 없으면 정지 억제를 미적용했다고 표시하고 Score 기반 Alarm 계산은 계속합니다.

다른 센서 구성으로 실행할 경우, `src/profiles/delivered.json`의 7Tag 전용 `n_heads`, `threshold.per_tag`, `alarm.conditions`를 대상 데이터에 맞춰 검토하세요.
