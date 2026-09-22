# 외부 소스와 파일별 라이선스

참조한 논문, 저장소의 일반 라이선스, 개별 파일에 적힌 라이선스는 구분한다.
전체 프로젝트를 하나의 외부 라이선스로 일괄 표시하지 않는다.

| 대상 | 출처 | 확인한 고지 | 함께 보존할 위치 |
|---|---|---|---|
| TranAD 모델의 구조·attention/positional encoding을 참고·수정한 부분 | [imperial-qore/TranAD](https://github.com/imperial-qore/TranAD), revision `7ffb98d0c18189cc3d9ab732b4cb0278200a0af0` | 저장소 일반 LICENSE: BSD-3-Clause, Copyright (c) 2022 Shreshth Tuli | `src/model.py` 출처·고지, `src/vendor/LICENSE.TranAD` |
| `src/vendor/spot.py` | [공식 TranAD 저장소의 SPOT 파일](https://github.com/imperial-qore/TranAD/blob/7ffb98d0c18189cc3d9ab732b4cb0278200a0af0/src/spot.py) | 파일 자체 헤더: author Alban Siffer, company Amossys, **GNU GPLv3** | 원본 파일 헤더, `src/vendor/LICENSE.GPL-3.0` |
| TranAD 방법 설명 | [TranAD 논문](https://arxiv.org/abs/2201.07284) | Tuli, Casale, Jennings, PVLDB 15(6), 1201–1214, 2022 | README·요구사항 문서의 논문 링크 및 모델 출처 |

**SPOT의 파일별 GPLv3 고지는 TranAD 저장소의 일반 BSD-3-Clause 고지와 다르다.**
`spot.py`를 가져왔다는 이유로 그 파일의 헤더를 BSD로 바꾸거나, BSD LICENSE 하나만
남기고 GPL 고지를 삭제하지 않는다. GPLv3 원문은
[GNU 공식 원문](https://www.gnu.org/licenses/gpl-3.0.txt)과 함께 확인할 수 있다.

`src/detection.py`의 `method='spot'`은 이 별도 파일을 불러와 초기 고정 임계값을
계산한다. SciPy GPD 최대우도 방식인 `method='pot'`과 같은 구현이라고 표현하지 않는다.
기본 `method='manual'`은 SPOT 파일을 실행하지 않지만, 소스 배포 폴더에 해당 파일을
포함했다면 파일의 출처·고지를 유지해야 한다.

TranAD 공개 코드와 논문의 학습 절차는 동일하지 않다. 현재 모델은 공개 코드의
두 단계 self-conditioning 구조를 참조하며, 현재 시점 정렬, float32/GPU batch,
MAE Scoring, 실험별 목적함수 및 선택 가능한 출력층 등의 변경을 기록한다.
논문에 설명된 MAML이나 별도 adversarial 최적화가 구현·실행됐다고 주장하지 않는다.

Python 패키지 의존성은 `src/requirements.txt`에 있으며 각 패키지의 배포 라이선스가
별도로 존재한다. 이 문서는 패키지 파일을 단일 라이선스로 재지정하지 않는다.
실제 설치 버전은 실행별 환경 기록과 함께 보존한다.
