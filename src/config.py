"""회전기계 EWS 통합 설정. 경로는 이 파일이 있는 프로젝트를 기준으로 한다.

센서 값은 원래 단위에서 정지 판단을 하고, 복원 점수/Threshold는 Train의
앞 80%에만 fit한 MinMax 공간에서 계산한다. Inhibit/Alarm_*는 검증 정답이다.
변경 실험은 --run-id로 결과를 분리한다. 기존 data 파일은 읽기만 한다.
"""
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    'source': 'rtdb_csv',  # benchmark도 동일 DateTime/센서/Label CSV 계약을 사용한다.
    'paths': {'train': str(ROOT/'data/train.csv'), 'test': str(ROOT/'data/test.csv'),
              'results': str(ROOT/'results'), 'benchmark_train': str(ROOT/'data/benchmark_train.csv'),
              'benchmark_test': str(ROOT/'data/benchmark_test.csv')},
    'data': {'timestamp': 'DateTime', 'validation_fraction': .2,
             'sample_minutes': 1, 'missing_policy': 'error',
             'sensor_suffix': '.PV', 'label_prefix': 'Alarm_', 'stop_label': 'Inhibit'},
    # window는 현재 시점을 포함한다. 분할/불연속 시각 경계에서 첫 값을 반복해
    # 좌측 padding하여 각 원본 시점마다 정확히 하나의 복원값을 출력한다.
    # window: 복원할 현재 분과 직전 기록 수. 늘리면 더 긴 패턴을 보지만
    # attention 계산/메모리도 늘어난다. dropout: 학습 중 확률적 제거 비율;
    # 크게 하면 과적합을 줄일 수 있으나 세밀한 복원이 어려울 수 있다.
    # ff_dim: attention 뒤 은닉층 폭. encoder/decoder_layers: 반복 블록 수;
    # 용량을 늘릴수록 시간/메모리가 증가하고 정상 패턴 과적합 가능성이 있다.
    # n_heads=None은 공식 구현처럼 센서 개수. 2*센서개수를 나누는 수만 유효.
    # dtype=float32는 T4 실행 효율을 위한 선택이며 공식 double과 차이를 기록.
    'model': {'window': 10, 'dropout': .1, 'ff_dim': 16,
              'encoder_layers': 1, 'decoder_layers': 1, 'n_heads': None, 'dtype': 'float32',
              'output_activation': 'sigmoid', 'output_margin': 0.0},
    # 독립 bounded-output 실험만 scaled_sigmoid와 고정 margin .1/.25 사용.
    # yhat=(1+2*margin)*sigmoid(logits)-margin, 점수/입력/scaler 재척도화 없음.
    # sigmoid/linear에서는 margin=0만 허용하며 과거 checkpoint의 누락값도 0.
    # epochs: 이번 모델의 누적 목표(추가 횟수 아님); 이어학습까지 최대30.
    # lr: AdamW 갱신 크기. 크면 빠르지만 불안정; 작으면 느리고 안정적이다.
    # weight_decay: 가중치 규제. scheduler_step마다 lr*=scheduler_gamma.
    # patience: Validation MAE가 min_delta 이상 개선되지 않는 연속 Epoch 수.
    # batch_size: 한번에 CUDA로 보낼 윈도우 수. 메모리/학습동역학에 영향.
    # grad_clip: gradient norm 상한(0이면 끔). seed는 초기화/샘플순서 재현용.
    # num_workers: CPU 데이터 준비 프로세스 수; GPU 대신 학습하지 않는다.
    'train': {'epochs': 10, 'max_epochs': 30, 'lr': .001, 'weight_decay': 1e-5,
              'scheduler_step': 5, 'scheduler_gamma': .9, 'patience': 3, 'min_delta': 1e-5,
              'batch_size': 128, 'seed': 42, 'grad_clip': 1., 'num_workers': 2,
              'shuffle': True, 'resume_from': None, 'early_stopping': True,
              'progress_every_batches': 500, 'progress_every_seconds': 30,
              'min_epochs': 10, 'objective': 'mse', 'huber_delta': .05, 'l1_weight': .5,
              'save_epoch_checkpoints': True, 'inference_batch_size': 2048},
    # 현재 Manual 설정은 저장된 threshold보다 우선한다. score > threshold만 Alarm.
    # 낮은 threshold는 민감도와 오경보를 함께 늘릴 수 있다.
    # percentile: Validation 점수의 백분위수, sigma: 평균+계수*표준편차.
    # pot: SciPy GPD 최대우도 추정; spot: 공식 Grimshaw 초기화 극단분위수.
    # level: 상위 꼬리 시작 분위수(0~1), q: 목표 초과 위험확률(작으면보수적).
    # SPOT의 본 배치 파이프라인은 초기 고정 Threshold이며 스트리밍 갱신은 아님.
    # 최초 실험은 per_tag={}로 모든 Tag에 0.2를 적용한다.
    # 변경된 수동값은 별도 실험 Config와 변경 전후 채점 결과에 기록한다.
    'threshold': {'method': 'manual', 'manual': .2, 'per_tag': {},
                  'percentile': 99.5, 'sigma': 3., 'level': .98, 'q': 1e-4,
                  'reference_split': 'validation', 'spot_source_path': str(ROOT/'src/vendor/spot.py'),
                  'reuse_saved_on_eval': True},
    # 아래 값은 정규화하지 않은 공정 센서 단위다. Inhibit을 조건에 넣지 않는다.
    # 경계 진단 후 채택한 보수적 재기동 확인: 센서 정지는 즉시 억제하고,
    # 해제 뒤 연속1분을 추가 억제한다. 실제 시각 gap에서는 유지하지 않는다.
    # 이 설정 검토는 제공 Test를 사용한 개발평가이며 독립 holdout 증명이 아니다.
    'alarm': {'enabled': True, 'mode': 'or', 'release_hold_minutes': 1, 'conditions': [
        {'tag': 'PDC4303_7.PV', 'op': '<=', 'value': .2},
        {'tag': 'PPVI43032A.PV', 'op': '<=', 'value': 2.5}]},
    # 합격 기준은 최신 사용자 요청 그대로 고정. 실험에서 완화하지 않는다.
    'success': {'normalized_test_mae_lt': .02, 'every_tag_accuracy_ge': .95},
    # 일반 -full/-train은 다른 CSV로 나중에 재실행할 수 있도록 시간 제한 없음.
    # 이번 야간 -optimize만 아래 search.deadline_utc를 각 실험에 명시 주입한다.
    'deadline_utc': None,
    'report': {'max_chart_points': 2500, 'dpi': 120, 'export_database': True,
               'dashboard': True, 'static_plots': True},
    'explain': {'server_port': 8767, 'exact_max_features': 8, 'permutations': 16,
                'max_duration_minutes': 1440, 'inference_batch_size': 2048, 'cache_size': 8},
    'search': {'seed_epochs': 10, 'maximum_epochs': 30, 'maximum_consecutive_errors': 3,
               'continue_until_deadline': True,
               'deadline_utc': '2026-09-21T22:00:00+00:00', # 2026-09-22 07:00 KST
               # 학습 시작 전 고정한 수동 Threshold 비교 후보. Label 값으로
               # cutoff를 fitting하지 않으며 각 후보는 Test 개발평가로 기록한다.
               'manual_threshold_candidates': [.2, .15, .1, .05, .02]},
}

def get_config():
    return deepcopy(CONFIG)
