"""마감시각까지 수행하는 증거 기반 GPU 실험 탐색.

Jupyter의 ``%run mode -optimize``는 Phase1 검증 후 분리된 Linux 작업을
시작하고 반환한다. Test 정답을 본 반복 선택은 개발평가이며 독립 검증 성능이
아니다. 모델·전처리에는 Alarm_* 및 Inhibit 정답을 전달하지 않는다.
실행 소스·정적 자산은 보호하지만 실행 노트북(.ipynb)과 Jupyter의
.ipynb_checkpoints, Python bytecode/cache는 자동 저장을 위해 보호에서 제외한다.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid

from .audit import AuditJournal, AuditIntegrityError, file_sha256
from .config import ROOT, get_config

RANK_DESCRIPTION = [
    'MAE<0.02와 모든 Tag 일치율>=0.95를 동시에 통과한 후보 우선',
    '동시 합격 후보끼리는 전체 Test MAE 오름차순 최우선; MAE 동률이면 Tag 최저 일치율 내림차순',
    '미합격 후보끼리는 95% 미달 Tag 개수 오름차순, Tag 최저 일치율 내림차순',
    '미합격 후보의 다음 순위는 MAE 0.02 초과분 오름차순, 전체 Test MAE 오름차순',
    '합격 후에도 마감까지 탐색하며 모든 Tag 일치율95% 이상을 유지하는 범위에서 MAE 최소화',
]


def _now():
    return dt.datetime.now(dt.timezone.utc)


def _deadline(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('탐색 마감시각에는 UTC offset이 필요합니다.')
    return parsed.astimezone(dt.timezone.utc)


def _read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).is_file() else default


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def _trace(path, event, **values):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time_utc': _now().isoformat(), 'event': event, **values},
                                ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _diff(before, after, prefix=''):
    changes = {}
    for key in sorted(set(before) | set(after)):
        name = prefix + key
        old, new = before.get(key), after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            changes.update(_diff(old, new, name + '.'))
        elif old != new:
            changes[name] = {'before': old, 'after': new}
    return changes


def _merge(base, changes):
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f'알 수 없는 설정: {key}')
        if isinstance(value, dict) and isinstance(base[key], dict) and key != 'per_tag':
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _process(pid):
    """Linux /proc의 실제 PID·시작 tick·실행 명령을 조회한다."""
    try:
        process = Path('/proc') / str(int(pid))
        stat = (process/'stat').read_text()
        rest = stat.rsplit(')', 1)[1].split()
        if rest[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': int(rest[19]),
                'ppid': int(rest[1]), 'cwd': str((process/'cwd').resolve()),
                'argv': [part.decode('utf-8', errors='replace')
                         for part in (process/'cmdline').read_bytes().split(b'\0') if part]}
    except (OSError, ValueError, IndexError):
        return None


def _worker_live(state, root):
    if not isinstance(state, dict) or not state.get('pid'):
        return False
    actual = _process(state['pid'])
    return bool(actual and actual['start_ticks'] == state.get('start_ticks')
                and actual['cwd'] == str(Path(root).resolve())
                and 'src.optimize' in actual['argv'] and '--worker' in actual['argv']
                and state.get('project_root') == str(Path(root).resolve()))


@contextlib.contextmanager
def _lock(path, blocking=False):
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('분리된 최적화 작업은 SageMaker Linux에서 실행해야 합니다.')
    import fcntl
    with Path(path).open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _journal(root):
    from .test_contracts import GATE_REGISTRY
    from .verify import protected_source_paths
    root = Path(root).resolve()
    return AuditJournal(root, GATE_REGISTRY, protected_paths=protected_source_paths(root))


def _source_files(root):
    from .verify import protected_source_paths
    root = Path(root).resolve()
    paths = protected_source_paths(root)
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(paths)}


def _verify_source_manifest(root, path, expected_sha256):
    """이번 Phase1 실행에 결속된 원본 목록을 검증한다. 과거 보호를 주장하지 않는다."""
    path = Path(path)
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise AuditIntegrityError('Phase1 원본 목록 자체의 hash가 변경되었습니다.')
    payload = _read(path)
    if payload.get('files') != _source_files(root):
        raise AuditIntegrityError('Phase1 검증 이후 실행 소스가 변경되었습니다. 새 검증이 필요합니다.')
    return payload


def _validate_search_config(config):
    required = {'seed_epochs': 10, 'maximum_epochs': 30,
                'maximum_consecutive_errors': 3, 'continue_until_deadline': True}
    for key, value in required.items():
        if config.get('search', {}).get(key) != value:
            raise ValueError(f'search.{key}={value!r}는 이번 사용자 요구에 따른 고정 조건입니다.')


def _check_initial(root, results, run_id):
    """완료 표식뿐 아니라 데이터 hash·분할·epoch·체크포인트 출처를 확인한다."""
    run = results/run_id
    integrity = _read(run/'integrity.json', {})
    summary = _read(run/'training_summary.json', {})
    config = _read(run/'config.json', {})
    profile = _read(run/'data_profile.json', {})
    if not integrity.get('complete') or not integrity.get('original_data_unchanged'):
        raise RuntimeError(f'초기 학습이 아직 완료되지 않았습니다: {run}/integrity.json')
    if not all(integrity['original_data_unchanged'].values()):
        raise RuntimeError('초기 학습 도중 원본 데이터 변경이 감지되었습니다.')
    if (config.get('model', {}).get('window') != 10 or
            int(config.get('train', {}).get('epochs', 0)) != 10 or
            int(summary.get('completed_epoch_count', 0)) < 10 or
            int(summary.get('epochs_consumed', 0)) != 10):
        raise RuntimeError('최초 실험은 Window10, 실제 완료 Epoch10이어야 합니다.')
    if summary.get('status') not in {'target_epochs_reached', 'early_stopping'}:
        raise RuntimeError('초기 학습의 정상 종료 증거가 없습니다.')
    for name in ('best.pt', 'last.pt', 'scaler.pkl'):
        if not (run/name).is_file():
            raise RuntimeError(f'초기 산출물이 없습니다: {name}')
    train = profile.get('train', {})
    source = Path(config.get('paths', {}).get('train', ''))
    if not source.is_file() or file_sha256(source) != train.get('sha256'):
        raise RuntimeError('현재 Train과 초기 학습의 원본 hash가 다릅니다.')
    return run, config, summary


def candidate_status(metrics):
    """고정 합격 조건을 재계산하고 합격 후보에서는 MAE를 먼저 최소화한다.

    rank 첫 값0은 동시합격, 1은 미합격이다. 이후 값의 의미는 조건부이며
    합격에서는 MAE→최저일치율, 미합격에서는 실패Tag수→최저일치율→MAE다.
    """
    gate = metrics.get('gate', {})
    mae = float(gate.get('mae', math.nan))
    accuracies = gate.get('per_tag_accuracy', {})
    expected = int(metrics.get('denominators', {}).get('test_features', 0))
    if not math.isfinite(mae) or expected < 1 or len(accuracies) != expected:
        raise ValueError('전체 Test MAE 또는 Tag별 채점 분모가 없습니다.')
    values = []
    for tag, value in accuracies.items():
        if value is None or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f'Tag 정확도가 유효하지 않습니다: {tag}')
        values.append(float(value))
    failing = sum(value < .95 for value in values)
    passed = mae < .02 and failing == 0
    if gate.get('phase2_numeric_pass') is not passed:
        raise ValueError('저장된 합격 주장과 고정 기준 재계산이 다릅니다.')
    rank = ([0, mae, -min(values), 0., 0.] if passed else
            [1, failing, -min(values), max(0., mae-.02), mae])
    return {'mae': mae, 'per_tag_accuracy': accuracies,
            'phase2_numeric_pass': passed, 'failing_tag_count': failing,
            'minimum_tag_accuracy': min(values),
            'rank': rank,
            'ranking_branch': '합격: MAE 최소화' if passed else '미합격: 실패 Tag·최저 정확도 개선 우선',
            'truth_stop_alarm_count': metrics.get('stop_safety', {}).get('truth_stop_alarm_count')}


def _variants(linear_evidence):
    grid = [
        ('w20_l1', {'model': {'window': 20, 'dropout': .0, 'ff_dim': 32},
                     'train': {'lr': .001, 'objective': 'l1'}},
         '20분 문맥과 L1 손실로 MAE 목적에 대한 민감도 비교'),
        ('w30_huber', {'model': {'window': 30, 'dropout': .05, 'ff_dim': 32},
                        'train': {'lr': .0003, 'objective': 'huber'}},
         '30분 문맥·작은 학습률·Huber 손실로 점진 변화 복원 비교'),
        ('w60_mix', {'model': {'window': 60, 'dropout': .0, 'ff_dim': 64},
                      'train': {'lr': .001, 'objective': 'mse_l1'}},
         '60분 문맥과 넓은 은닉층으로 장기 패턴 표현 비교'),
        ('w10_l1', {'model': {'window': 10, 'dropout': .0, 'ff_dim': 64},
                     'train': {'lr': .0003, 'objective': 'l1'}},
         '초기 창 길이를 유지하면서 dropout·용량·손실·학습률 비교'),
        ('w20_mse', {'model': {'window': 20, 'dropout': .05, 'ff_dim': 64},
                      'train': {'lr': .0003, 'objective': 'mse'}},
         '공식 MSE 손실에서 창·용량·학습률 영향 비교'),
        ('w30_mix', {'model': {'window': 30, 'dropout': .0, 'ff_dim': 64},
                      'train': {'lr': .001, 'objective': 'mse_l1'}},
         '중간 문맥 길이와 MAE 혼합 손실 비교'),
    ]
    if linear_evidence and linear_evidence['outside_fraction'] > 0:
        grid.insert(0, ('w10_linear_l1',
                       {'model': {'window': 10, 'output_activation': 'linear', 'dropout': .0,
                                  'ff_dim': 32}, 'train': {'lr': .001, 'objective': 'l1'}},
                       'Train 정규화 Test 범위가 [0,1] 밖이라는 실측 근거로 sigmoid 출력 제약 완화'))
        grid.append(('w30_linear_huber',
                     {'model': {'window': 30, 'output_activation': 'linear', 'dropout': .05,
                                'ff_dim': 64}, 'train': {'lr': .0003, 'objective': 'huber'}},
                     '범위 이탈 실측에 따라 linear head와 긴 문맥 조합 비교'))
    return grid


class SearchWorker:
    def __init__(self, root, results, initial_run, search_id, gate_evidence):
        self.root, self.results = Path(root).resolve(), Path(results).resolve()
        self.initial_run, self.search_id = initial_run, search_id
        self.directory = self.results/'search'/search_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trace = self.directory/'trials.jsonl'
        self.state_path = self.results/'search_state.json'
        self.journal = _journal(self.root)
        self.base = get_config()
        _validate_search_config(self.base)
        self.end = _deadline(self.base['search']['deadline_utc'])
        self.best = None
        self.candidates = []
        self.last_config = copy.deepcopy(self.base)
        self.errors = 0
        self.source_manifest_path = self.directory/'source_manifest.json'
        events = self.journal.events()
        commands = [event for event in events if event['kind'] == 'command_start'
                    and event['data']['evidence_id'] == gate_evidence]
        if len(commands) != 1:
            raise AuditIntegrityError('원본 목록과 결속된 Phase1 실행 기록이 없습니다.')
        gate_loops = [event for event in events if event['kind'] == 'loop_start'
                      and event['loop_id'] == commands[0]['loop_id']]
        source = gate_loops[0]['data'].get('config', {}).get('search_source_manifest', {}) if len(gate_loops) == 1 else {}
        if source.get('path') != str(self.source_manifest_path):
            raise AuditIntegrityError('Phase1 증거가 이번 탐색 원본 목록에 속하지 않습니다.')
        self.source_manifest_sha256 = source.get('sha256')
        self.assert_source()
        self.state = {'schema_version': 1, 'search_id': search_id,
                      'project_root': str(self.root), 'results_root': str(self.results),
                      'pid': os.getpid(), 'start_ticks': _process(os.getpid())['start_ticks'],
                      'started_utc': _now().isoformat(), 'deadline_utc': self.end.isoformat(),
                      'log': str(self.directory/'worker.log'), 'trace': str(self.trace),
                      'phase1_gate_evidence': gate_evidence, 'status': 'starting',
                      'source_manifest_sha256': self.source_manifest_sha256,
                      'ranking': RANK_DESCRIPTION, 'initial_run': initial_run,
                      'development_evaluation': True}
        self.save_state()

    def assert_source(self):
        return _verify_source_manifest(self.root, self.source_manifest_path,
                                       self.source_manifest_sha256)

    def save_audit_summary(self):
        summary = self.journal.summary()
        _write(self.results/'ralph_summary.json', summary)
        return summary

    def save_state(self, **values):
        self.state.update(values, updated_utc=_now().isoformat())
        _write(self.state_path, self.state)

    def command(self, purpose, argv, config):
        """실행은 감사 장부에 맡기고 현재 프로세스 핸들을 주기적으로 관찰한다."""
        loop_id = self.journal.start_loop(purpose, config)
        if not self.journal.check_protected(loop_id):
            self.save_audit_summary()
            raise AuditIntegrityError('보호 파일 변경으로 탐색 중단')
        try:
            self.assert_source()
        except AuditIntegrityError:
            self.journal.abort_loop(loop_id, 'Phase1 검증 후 원본 목록 또는 실행 소스 변경',
                                    category='evidence_integrity')
            self.save_audit_summary()
            raise
        self.save_audit_summary()
        self.save_state(status='running', current_loop=loop_id, current_command=argv,
                        command_started_utc=_now().isoformat(), command_process=None)
        holder = {}
        def execute():
            try:
                holder['result'] = self.journal.run_command(loop_id, argv, cwd=self.root,
                                                           require_nonce=True)
            except BaseException as error:
                holder['error'] = error
        thread = threading.Thread(target=execute, daemon=False)
        thread.start()
        while thread.is_alive():
            thread.join(timeout=15)
            children = []
            for process_dir in Path('/proc').iterdir():
                if process_dir.name.isdigit():
                    proc = _process(process_dir.name)
                    if proc and proc['ppid'] == os.getpid() and proc['cwd'] == str(self.root):
                        children.append(proc)
            self.save_state(command_process=children, live_command_thread=thread.is_alive())
        if 'error' in holder:
            raise holder['error']
        evidence = holder['result']
        try:
            self.assert_source()
        except AuditIntegrityError:
            self.journal.abort_loop(loop_id, '명령 실행 도중 원본 목록 또는 실행 소스 변경',
                                    evidence_id=evidence['evidence_id'], category='evidence_integrity')
            self.save_audit_summary()
            raise
        if not self.journal.record_claim(loop_id, evidence['evidence_id'], evidence['success']):
            self.save_audit_summary()
            raise AuditIntegrityError('실제 실행 결과와 검증 증거가 일치하지 않습니다.')
        self.journal.end_loop(loop_id, 'completed' if evidence['success'] else 'failed',
                              {'evidence_id': evidence['evidence_id'], 'purpose': purpose})
        self.save_audit_summary()
        _trace(self.trace, 'command_completed', purpose=purpose, argv=argv, loop_id=loop_id,
               evidence=evidence)
        self.save_state(command_process=None, live_command_thread=False,
                        last_evidence=evidence, current_command=None)
        if not evidence['success']:
            raise RuntimeError(f'실행 실패: {purpose}; evidence={evidence["evidence_id"]}')
        return evidence

    def config_file(self, config, run_id, reason):
        if config['success'] != {'normalized_test_mae_lt': .02, 'every_tag_accuracy_ge': .95}:
            raise ValueError('합격 기준 변경 금지')
        path = self.directory/(run_id+'.json')
        _write(path, config)
        _trace(self.trace, 'configuration', run_id=run_id, config=str(path),
               reason=reason, changes_from_previous=_diff(self.last_config, config),
               full_config_sha256=file_sha256(path))
        self.last_config = copy.deepcopy(config)
        return path

    def record_candidate(self, eval_dir, checkpoint, config, evidence, provenance):
        from .trainer import load_trusted_checkpoint, canonical_scaler_state
        from .data import load_scaler
        metrics = _read(eval_dir/'metrics.json', {})
        integrity = _read(eval_dir/'integrity.json', {})
        if (not integrity.get('complete') or not integrity.get('original_data_unchanged')
                or not all(integrity['original_data_unchanged'].values())):
            raise RuntimeError('완료·원본 무결성을 확인하지 못한 평가 결과')
        evaluation = _read(eval_dir/'eval_provenance.json', {})
        if evaluation.get('checkpoint_sha256') != file_sha256(checkpoint):
            raise RuntimeError('평가한 체크포인트와 후보 체크포인트 hash 불일치')
        checkpoint_state = load_trusted_checkpoint(checkpoint, map_location='cpu')
        scaler_state = load_scaler(checkpoint.parent/'scaler.pkl')
        if canonical_scaler_state(checkpoint_state['scaler_state']) != canonical_scaler_state(scaler_state):
            raise RuntimeError('후보 체크포인트·scaler 상태 불일치')
        if checkpoint_state['feature_names'] != scaler_state['features']:
            raise RuntimeError('후보 feature 순서 불일치')
        if not 1 <= int(checkpoint_state['epochs_consumed']) <= 30:
            raise RuntimeError('후보 누적 Epoch 예산 위반')
        for name, profile in _read(eval_dir/'data_profile.json', {}).items():
            if profile and file_sha256(profile['path']) != profile['sha256']:
                raise RuntimeError(f'후보 {name} 원본 데이터 hash 불일치')
        provenance.update(model_id=checkpoint_state['model_id'],
                          model_weights_epoch=checkpoint_state.get('model_weights_epoch', checkpoint_state['completed_epochs']),
                          epochs_consumed=checkpoint_state['epochs_consumed'],
                          source_info=checkpoint_state.get('source_info'))
        status = candidate_status(metrics)
        effective_config=_read(eval_dir/'config.json',{})
        if not effective_config or effective_config.get('model')!=checkpoint_state['model_cfg']:
            raise RuntimeError('평가 실행 Config와 실제 Checkpoint 모델 설정이 다릅니다.')
        if effective_config.get('success')!={'normalized_test_mae_lt':.02,'every_tag_accuracy_ge':.95}:
            raise RuntimeError('평가 실행 Config의 고정 합격 기준이 다릅니다.')
        candidate = {'evaluation_dir': str(eval_dir), 'checkpoint': str(checkpoint.resolve()),
                     'checkpoint_sha256': file_sha256(checkpoint),
                     'metrics_sha256': file_sha256(eval_dir/'metrics.json'),
                     'config': copy.deepcopy(effective_config), 'config_sha256':file_sha256(eval_dir/'config.json'),
                     'evidence_id': evidence['evidence_id'],
                     'provenance': provenance, 'completed_utc': _now().isoformat(), **status}
        self.candidates.append(candidate)
        improved = self.best is None or tuple(candidate['rank']) < tuple(self.best['rank'])
        if improved:
            self.best = candidate
            _write(self.directory/'best_so_far.json', candidate)
        _trace(self.trace, 'candidate_evaluated', improved=improved, candidate=candidate)
        self.save_state(candidate_count=len(self.candidates), best=self.best,
                        latest_candidate=candidate)
        print(f'후보 {len(self.candidates)}: MAE={status["mae"]:.7f}, '
              f'Tag 최저={status["minimum_tag_accuracy"]:.4%}, '
              f'동시 통과={status["phase2_numeric_pass"]}', flush=True)
        return candidate

    def evaluate(self, checkpoint, config, run_id, reason, final=False, threshold_sweep=True):
        if (self.results/run_id).exists():
            raise RuntimeError(f'평가 폴더 재사용 금지: {run_id}')
        path = self.config_file(config, run_id, reason)
        argv = [sys.executable, '-u', str(self.root/'mode.py'), '-eval',
                '--checkpoint', str(checkpoint), '--run-id', run_id, '--config', str(path)]
        if not final:
            argv.append('--metrics-only')
        started = time.time()
        evidence = self.command(reason, argv, config)
        if (self.results/run_id/'metrics.json').stat().st_mtime < started-2:
            raise RuntimeError('평가 metrics.json이 이번 실행보다 오래되었습니다.')
        candidate = self.record_candidate(self.results/run_id, checkpoint, config, evidence,
                                          {'mode': 'eval', 'reason': reason, 'final_export': final})
        if not final:
            self.write_trial_report(candidate)
            self.record_mandatory_case(candidate)
        self.errors=0
        self.save_state(consecutive_runtime_errors=0)
        # 기본값은 빈 목록이다. 운영자가 명시한 고정 수동 후보만 비교하며
        # Label 값에서 cutoff를 fitting하거나 TP/FP 최적 분할값을 추출하지 않는다.
        if not final and threshold_sweep:
            thresholds = self.base.get('search', {}).get('manual_threshold_candidates', [])
            for index, value in enumerate(thresholds):
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError('수동 Threshold 후보는 유한한 0 이상의 값이어야 합니다.')
                if (config['threshold']['method'] == 'manual'
                        and not config['threshold']['per_tag']
                        and value == float(config['threshold']['manual'])):
                    continue
                if _now() >= self.end:
                    break
                altered = copy.deepcopy(config)
                altered['threshold'].update(method='manual', manual=value, per_tag={})
                alternative = self.evaluate(checkpoint, altered, run_id+f'_threshold_{index:02d}',
                                            f'사전 지정 수동 Threshold {value} 비교; Test 기반 개발평가',
                                            threshold_sweep=False)
                if tuple(alternative['rank']) < tuple(candidate['rank']):
                    candidate = alternative
        return candidate

    def write_trial_report(self, candidate):
        """보호된 코드로 보고서를 만들고 수치·설정·원본·HTML의 결속을 남긴다."""
        from .trial_reports import create_trial_report
        run=Path(candidate['evaluation_dir'])
        self.assert_source()
        try:
            report=create_trial_report(run,candidate)
            self.assert_source()
            manifest=_read(run/'trial_report_manifest.json',{})
            digest=file_sha256(report)
            if manifest.get('dashboard_sha256')!=digest:
                raise AuditIntegrityError('실험 HTML과 완료 manifest hash 불일치')
        except Exception as error:
            candidate['html_complete']=False
            _trace(self.trace,'trial_report_failed',run_id=run.name,error=repr(error),
                   numerical_candidate_recorded=True,html_complete=False)
            self.save_state(latest_trial_report_error={'run_id':run.name,'error':repr(error)})
            raise
        candidate['html_complete']=True
        candidate['trial_report']={'dashboard':str(report),'dashboard_sha256':digest,
                                  'manifest':str(run/'trial_report_manifest.json'),
                                  'manifest_sha256':file_sha256(run/'trial_report_manifest.json')}
        _trace(self.trace,'trial_report_created',run_id=run.name,**candidate['trial_report'],
               metrics_sha256=candidate['metrics_sha256'],config_sha256=candidate['config_sha256'],
               checkpoint_sha256=candidate['checkpoint_sha256'],
               full_resolution_preserved=True,storage='shared_gzip_assets')
        if self.best is candidate:
            _write(self.directory/'best_so_far.json',candidate)
        self.save_state(latest_candidate=candidate,best=self.best,
                        trial_reports_complete=sum(bool(c.get('html_complete')) for c in self.candidates))
        return report

    def save_mandatory_case(self, status=None, **values):
        payload=_read(self.directory/'mandatory_cases.json',None)
        if payload is None:
            payload={'search_id':self.search_id,'window10_epoch20':{
                'required':True,'status':'pending','requested_window':10,'requested_epochs':20,
                'reason':'최초 Window10 모델을 Epoch11부터20까지 매회 평가; 오류 시 독립 Window10 Epoch20 대체 시도'}}
        case=payload['window10_epoch20']
        if status is not None:
            case['status']=status
        case.update(values,updated_utc=_now().isoformat())
        _write(self.directory/'mandatory_cases.json',payload)
        _write(self.results/'mandatory_cases.json',payload)
        self.save_state(mandatory_cases=payload,mandatory_cases_complete=case['status']=='completed')
        return payload

    def record_mandatory_case(self,candidate):
        """설정값이 아니라 실제 가중치 Epoch20·누적20·HTML 증거를 확인한다."""
        provenance=candidate.get('provenance',{})
        if (candidate.get('config',{}).get('model',{}).get('window')!=10 or
                provenance.get('model_weights_epoch')!=20 or provenance.get('epochs_consumed')!=20):
            return False
        self.assert_source()
        run=Path(candidate['evaluation_dir']);report=run/'dashboard.html'
        manifest=_read(run/'trial_report_manifest.json',{})
        if (not candidate.get('html_complete') or not report.is_file() or
                manifest.get('dashboard_sha256')!=file_sha256(report) or
                manifest.get('source_identities',{}).get('metrics.json')!=candidate['metrics_sha256'] or
                manifest.get('checkpoint_sha256')!=candidate['checkpoint_sha256'] or
                file_sha256(Path(candidate['checkpoint']))!=candidate['checkpoint_sha256']):
            raise AuditIntegrityError('필수 Window10 Epoch20 평가·HTML 증거가 완전하지 않습니다.')
        current=self.save_mandatory_case()
        if current['window10_epoch20']['status']=='completed':
            return True
        payload=self.save_mandatory_case('completed',window=10,model_weights_epoch=20,epochs_consumed=20,
            checkpoint=candidate['checkpoint'],checkpoint_sha256=candidate['checkpoint_sha256'],
            evaluation_dir=str(run),dashboard=str(report),dashboard_sha256=file_sha256(report),
            report_manifest=str(run/'trial_report_manifest.json'),report_manifest_sha256=file_sha256(run/'trial_report_manifest.json'),
            metrics_sha256=candidate['metrics_sha256'],config_sha256=candidate['config_sha256'],
            evidence_id=candidate['evidence_id'],completed_utc=_now().isoformat())
        _trace(self.trace,'mandatory_case_completed',case=payload['window10_epoch20'])
        return True

    def ensure_mandatory_case(self,initial_config,prefix):
        """최초 이어학습 실패 시 선택적 탐색 전에 독립 Epoch20을 재시도한다."""
        if self.save_mandatory_case()['window10_epoch20']['status']=='completed':
            return True
        for attempt in range(1,4):
            if _now()>=self.end:
                break
            config=copy.deepcopy(initial_config)
            config['train'].update(epochs=20,resume_from=None,early_stopping=False)
            config['model']['window']=10
            run_id=f'{prefix}_mandatory_w10_e20_fallback_{attempt:02d}'
            _trace(self.trace,'mandatory_case_fallback',run_id=run_id,attempt=attempt,
                   reason='최초 계보의 실제 Epoch20 평가·HTML 미완료; 다른 설정 탐색 전 독립 학습')
            try:
                trained=self.train(config,run_id,'필수 Window10 Epoch20 대체 독립 학습')
                if trained is None:
                    break
                run,summary=trained
                if int(summary.get('epochs_consumed',0))!=20 or int(summary.get('completed_epochs',0))!=20:
                    if _now()>=self.end or summary.get('status')=='deadline_reached':
                        break
                    raise RuntimeError('필수 대체 학습이 실제 Epoch20까지 완료되지 않았습니다.')
                candidate=self.evaluate(run/'last.pt',config,run_id+'_eval',
                                        '필수 Window10 실제 Epoch20 last.pt 평가 및 HTML',threshold_sweep=False)
                if not self.record_mandatory_case(candidate):
                    raise RuntimeError('필수 대체 평가의 실제 Epoch/Window가 목표와 다릅니다.')
                return True
            except Exception as error:
                self.handle_error(error,run_id)
        self.save_mandatory_case('unfulfilled',reason='마감 또는 실행 오류로 실제 Epoch20 평가·HTML 완료 증거가 없음')
        return False

    def train(self, config, run_id, reason):
        if _now() >= self.end:
            return None
        if (self.results/run_id).exists():
            raise RuntimeError(f'학습 폴더 재사용 금지: {run_id}')
        path = self.config_file(config, run_id, reason)
        self.command(reason, [sys.executable, '-u', str(self.root/'mode.py'), '-train',
                              '--run-id', run_id, '--config', str(path), '--metrics-only'], config)
        run = self.results/run_id
        summary, integrity = _read(run/'training_summary.json', {}), _read(run/'integrity.json', {})
        if (not integrity.get('complete') or not integrity.get('original_data_unchanged')
                or not all(integrity['original_data_unchanged'].values())):
            raise RuntimeError('학습 완료·원본 무결성 확인 실패')
        if not 1 <= int(summary.get('epochs_consumed', 0)) <= 30:
            raise RuntimeError('모델당 누적 Epoch 상한 위반')
        if not (run/'last.pt').is_file():
            raise RuntimeError('완료 epoch checkpoint 없음')
        if int(summary.get('completed_epoch_count', 0)) < 10:
            _trace(self.trace, 'incomplete_minimum_epochs', run_id=run_id, summary=summary,
                   reason='마감 전 실제 완료 Epoch이10 미만이므로 최종 선택 후보에서 제외')
            return None
        return run, summary

    def lower_bound_evidence(self, candidate):
        import numpy as np
        with np.load(Path(candidate['evaluation_dir'])/'predictions.npz', allow_pickle=False) as data:
            actual = data['actual_scaled'].astype(np.float64)
            bound = np.maximum(-actual, 0) + np.maximum(actual-1., 0)
            evidence = {'source': str(Path(candidate['evaluation_dir'])/'predictions.npz'),
                        'source_sha256': file_sha256(Path(candidate['evaluation_dir'])/'predictions.npz'),
                        'outside_fraction': float(np.mean((actual < 0) | (actual > 1))),
                        'sigmoid_irreducible_mae_lower_bound': float(bound.mean()),
                        'minimum_normalized': float(actual.min()), 'maximum_normalized': float(actual.max()),
                        'uses_labels': False}
        _write(self.directory/'sigmoid_range_evidence.json', evidence)
        return evidence

    def handle_error(self, error, context):
        self.errors += 1
        _trace(self.trace, 'runtime_error', context=context, consecutive=self.errors,
               error=repr(error), traceback=traceback.format_exc())
        self.save_state(consecutive_runtime_errors=self.errors, last_error=repr(error))
        print(f'실행 오류 {self.errors}: {context}: {error}', flush=True)
        if isinstance(error, AuditIntegrityError):
            # 타입을 보존해야 worker_main이 최종 선택/보고를 시도하지 않는다.
            raise error
        if self.errors >= 3:
            raise RuntimeError('연속 세 번 실행 오류로 탐색 중단') from error

    def search(self):
        initial, initial_config, initial_summary = _check_initial(self.root, self.results, self.initial_run)
        self.base['paths'] = copy.deepcopy(initial_config['paths'])
        self.base['paths']['results'] = str(self.results)
        self.base['train']['early_stopping'] = False
        self.base['deadline_utc'] = self.end.isoformat()
        # 먼저 사용자 지정 최초 조건의 best checkpoint를 평가하고 전체 epoch10도
        # 평가한다. 두 snapshot은 실제 가중치 epoch를 각각 provenance에 기록한다.
        initial_eval_cfg = _merge(copy.deepcopy(self.base),initial_config)
        initial_eval_cfg['alarm']=copy.deepcopy(self.base['alarm'])
        initial_eval_cfg['search']=copy.deepcopy(self.base['search'])
        initial_eval_cfg['paths']['results']=str(self.results)
        initial_eval_cfg['deadline_utc'] = self.end.isoformat()
        _trace(self.trace,'initial_alarm_policy',initial_run=self.initial_run,
               before=initial_config.get('alarm'),after=initial_eval_cfg['alarm'],
               original_initial_config_unchanged=True,
               reason='Inhibit 1분 지연은 개발평가에서 확인; 현재 센서 즉시 정지 + 해제 1분 확인; 정답 입력 사용 없음')
        prefix = self.search_id.replace('-', '_')
        self.save_mandatory_case()
        attempt = 0
        while True:
            initial_id = prefix+'_initial_eval'+(f'_retry{attempt}' if attempt else '')
            try:
                candidate = self.evaluate(initial/'best.pt', initial_eval_cfg, initial_id,
                                          '최초 Window10·Epoch10 학습의 validation 최상 모델 평가')
                break
            except Exception as error:
                self.handle_error(error, initial_id)
                attempt += 1
        linear_evidence = self.lower_bound_evidence(candidate)
        attempt = 0
        while True:
            initial_id = prefix+'_initial_epoch10_eval'+(f'_retry{attempt}' if attempt else '')
            try:
                self.evaluate(initial/'last.pt', initial_eval_cfg, initial_id,
                              '최초 실험 실제 Epoch10 가중치의 별도 채점')
                break
            except Exception as error:
                self.handle_error(error, initial_id)
                attempt += 1
        previous_run = initial
        config = copy.deepcopy(initial_eval_cfg)
        config['train']['early_stopping'] = False
        config['deadline_utc'] = self.end.isoformat()
        variants = [('initial', {}, '초기 모델 누적 Epoch을 한 회씩 증가')] + _variants(linear_evidence)
        variant_index, cycle = 0, 0
        while _now() < self.end:
            variant_name, overrides, reason = variants[variant_index]
            local_best = tuple(candidate['rank']) if variant_index == 0 and cycle == 0 else None
            no_improvement = 0
            if variant_index > 0 or cycle > 0:
                config = _merge(copy.deepcopy(self.base), overrides)
                config['train'].update(epochs=10, resume_from=None, early_stopping=False,
                                       seed=int(self.base['train']['seed'])+cycle)
                config['paths']['results'] = str(self.results)
                run_id = f'{prefix}_c{cycle:02d}_v{variant_index:02d}_{variant_name}_e10'
                try:
                    trained = self.train(config, run_id, reason)
                    if trained is None:
                        break
                    previous_run, summary = trained
                    candidate = self.evaluate(previous_run/'last.pt', config, run_id+'_eval',
                                              reason+' / 실제 최종 완료 Epoch 평가')
                    local_best = tuple(candidate['rank'])
                except Exception as error:
                    self.handle_error(error, run_id)
                    variant_index = (variant_index+1) % len(variants)
                    cycle += int(variant_index == 0)
                    continue
            # 최초 Window10은11→20을 매회 직접 평가한다. 이후에만 미개선 중단을
            # 허용하며 다른 설정은11·12까지 평가한다. 모델별 누적30 상한은 유지한다.
            mandatory_initial=variant_index==0 and cycle==0
            summary = _read(previous_run/'training_summary.json', initial_summary)
            consumed = int(summary['epochs_consumed'])
            for epoch in range(consumed+1, 31):
                if _now() >= self.end:
                    break
                config = copy.deepcopy(config)
                config['train'].update(epochs=epoch, resume_from=str(previous_run/'last.pt'),
                                       early_stopping=False)
                run_id = f'{prefix}_c{cycle:02d}_v{variant_index:02d}_{variant_name}_e{epoch:02d}'
                try:
                    trained = self.train(config, run_id,
                                         f'{variant_name}: 누적 Epoch {epoch-1}→{epoch}, 동일 모델·scaler 이어학습')
                    if trained is None:
                        break
                    previous_run, summary = trained
                    actual_epoch = int(summary['completed_epochs'])
                    candidate = self.evaluate(previous_run/'last.pt', config, run_id+'_eval',
                                              f'{variant_name}: 목표 Epoch{epoch}, 실제 완료 Epoch{actual_epoch} checkpoint 채점')
                    rank = tuple(candidate['rank'])
                    improved = local_best is None or rank < local_best
                    if improved:
                        local_best, no_improvement = rank, 0
                    else:
                        no_improvement += 1
                    _trace(self.trace, 'epoch_decision', variant=variant_name, epoch=epoch,
                           improved=improved, no_improvement=no_improvement,
                           mandatory_epoch20=mandatory_initial,
                           reason=('필수 최초 Window10 Epoch20까지 매회 평가 후 3회 연속 미개선 허용' if mandatory_initial else
                                   '최소 Epoch11·12 평가 후 3회 연속 미개선 또는 누적30에서 다음 설정'))
                    minimum_epoch=20 if mandatory_initial else 12
                    if summary.get('status') == 'deadline_reached' or (epoch >= minimum_epoch and no_improvement >= 3):
                        break
                except Exception as error:
                    self.handle_error(error, run_id)
                    # 오류 checkpoint를 반복 이어학습하지 않고 다음 독립 설정으로.
                    break
            if mandatory_initial and not self.ensure_mandatory_case(initial_eval_cfg,prefix):
                # 필수 케이스를 남겨 둔 채 선택적 변형 탐색을 계속하지 않는다.
                break
            variant_index = (variant_index+1) % len(variants)
            cycle += int(variant_index == 0)
        self.finalize('deadline_reached')

    def finalize(self, reason):
        self.assert_source()
        self.save_audit_summary()
        mandatory=self.save_mandatory_case()
        if mandatory['window10_epoch20']['status']!='completed':
            mandatory=self.save_mandatory_case('unfulfilled',reason='탐색 종료 시 실제 Epoch20 평가·HTML 완료 증거 없음: '+reason)
        # 실행 수치가 보존된 보고서 생성 실패만 CPU로 재시도한다. HTML 없는
        # 실험을 모두 완료했다고 표시하지 않으며 무결성 실패는 즉시 중단한다.
        for prior in self.candidates:
            if prior.get('provenance',{}).get('final_export'):
                continue
            if not prior.get('html_complete'):
                self.write_trial_report(prior)
                self.record_mandatory_case(prior)
            elif (not Path(prior['trial_report']['dashboard']).is_file() or
                    file_sha256(prior['trial_report']['dashboard'])!=prior['trial_report']['dashboard_sha256']):
                raise AuditIntegrityError('이미 완료된 실험 HTML이 변경되거나 없어졌습니다.')
        mandatory=self.save_mandatory_case()
        if self.best is None:
            self.save_state(status='failed', stop_reason=reason, ended_utc=_now().isoformat())
            return
        selected = copy.deepcopy(self.best)
        selected.update(selected_utc=_now().isoformat(), selection_reason=reason,
                        rank_description=RANK_DESCRIPTION, candidate_count=len(self.candidates),
                        search_id=self.search_id, deadline_utc=self.end.isoformat(),
                        mandatory_cases=mandatory,mandatory_cases_complete=mandatory['window10_epoch20']['status']=='completed',
                        criteria={'normalized_test_mae_lt': .02, 'every_tag_accuracy_ge': .95},
                        independent_holdout_evaluation=False)
        _write(self.results/'selected_checkpoint.json', selected)
        final_config = copy.deepcopy(selected['config'])
        final_config['train']['resume_from'] = None
        final_config['report'].update(export_database=True, dashboard=True, static_plots=True)
        _write(self.results/'final_config.json', final_config)
        self.save_state(status='finalizing', stop_reason=reason, selected=selected)
        final_base = 'final_'+self.search_id
        attempt = 0
        final_id = final_base
        while (self.results/final_id).exists():
            attempt += 1
            final_id = final_base+f'_retry{attempt:02d}'
        _trace(self.trace, 'finalization_attempt', attempt=attempt, run_id=final_id,
               reason=reason, preserves_previous_partial_outputs=True)
        final = self.evaluate(Path(selected['checkpoint']), final_config, final_id,
                              '마감 후 최상 checkpoint로 전체 DB·그림·대시보드 생성', final=True)
        selected['final_run'] = final['evaluation_dir']
        selected['final_metrics_sha256'] = final['metrics_sha256']
        selected['final_evidence_id'] = final['evidence_id']
        _write(self.results/'selected_checkpoint.json', selected)
        _write(self.results/'final'/'selection.json', selected)
        _write(self.results/'final'/'config.json', final_config)
        # 실제 평가 루프가 닫힌 뒤 발표용 A/B/C/D를 다시 반영한다.
        self.save_audit_summary()
        from .reports import refresh_reports_from_saved
        from .package import build_package
        refresh_reports_from_saved(Path(final['evaluation_dir']))
        package = build_package(self.root, Path(final['evaluation_dir']))
        _trace(self.trace, 'github_package_created', **package)
        from .serve import launch_dashboard
        server = launch_dashboard(Path(final['evaluation_dir']),Path(selected['checkpoint']))
        self.save_state(status='completed', final_run=final['evaluation_dir'], selected=selected,
                        ended_utc=_now().isoformat(), phase2_numeric_pass=final['phase2_numeric_pass'],
                        mandatory_cases_complete=mandatory['window10_epoch20']['status']=='completed',
                        github_package=package,dashboard_server=server)
        self.save_audit_summary()


def run_search(initial_run='initial_w10_e10'):
    """Phase1 검증 후 단일 detached worker를 시작하고 상태를 반환한다."""
    root = Path(ROOT).resolve()
    config = get_config()
    _validate_search_config(config)
    results = Path(config['paths']['results']).resolve()
    results.mkdir(parents=True, exist_ok=True)
    state_path = results/'search_state.json'
    existing = _read(state_path, {})
    if _worker_live(existing, root):
        print(f'이미 실행 중인 탐색 PID={existing["pid"]}; 상태={state_path}', flush=True)
        return existing
    with _lock(results/'.search.lock'):
        # 잠금을 얻는 사이 다른 호출이 작업을 시작했을 수 있으므로 다시 확인.
        existing = _read(state_path, {})
        if _worker_live(existing, root):
            return existing
        _check_initial(root, results, initial_run)
        search_id = _now().strftime('s%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
        directory = results/'search'/search_id
        directory.mkdir(parents=True, exist_ok=False)
        source_path = directory/'source_manifest.json'
        _write(source_path, {'created_utc': _now().isoformat(), 'files': _source_files(root),
                             'protection_scope': '이번 탐색의 Phase1 실행 직전부터 적용'})
        source_sha256 = file_sha256(source_path)
        journal = _journal(root)
        loop_id = journal.start_loop('최적화 시작 전 Phase1 외부 검증',
                                     {'search_source_manifest': {'path': str(source_path),
                                                                 'sha256': source_sha256}})
        evidence = journal.run_command(loop_id,
                                      [sys.executable, '-u', '-m', 'src.test_contracts', '-v'],
                                      cwd=root, require_nonce=True)
        try:
            _verify_source_manifest(root, source_path, source_sha256)
        except AuditIntegrityError:
            journal.abort_loop(loop_id, 'Phase1 검증 도중 소스 변경',
                               evidence_id=evidence['evidence_id'], category='evidence_integrity')
            _write(results/'ralph_summary.json', journal.summary())
            raise
        if not journal.record_claim(loop_id, evidence['evidence_id'], evidence['success']):
            _write(results/'ralph_summary.json', journal.summary())
            raise AuditIntegrityError('Phase1 실행 증거를 확인하지 못했습니다.')
        journal.end_loop(loop_id, 'completed' if evidence['success'] else 'failed')
        _write(results/'ralph_summary.json', journal.summary())
        if not evidence['success']:
            raise RuntimeError('Phase1 실제 검증이 실패하여 GPU 탐색을 시작하지 않았습니다.')
        argv = [sys.executable, '-u', '-m', 'src.optimize', '--worker',
                '--initial-run', initial_run, '--search-id', search_id,
                '--gate-evidence', evidence['evidence_id']]
        with (directory/'worker.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        actual = _process(process.pid)
        if actual is None:
            raise RuntimeError(f'탐색 작업이 시작 직후 종료되었습니다: {directory}/worker.log')
        state = {'status': 'starting', 'pid': process.pid, 'start_ticks': actual['start_ticks'],
                 'project_root': str(root), 'results_root': str(results), 'search_id': search_id,
                 'log': str(directory/'worker.log'), 'trace': str(directory/'trials.jsonl'),
                 'started_utc': _now().isoformat(), 'phase1_gate_evidence': evidence['evidence_id'],
                 'deadline_utc': config['search']['deadline_utc']}
        _write(state_path, state)
    print(f'GPU 최적화 시작 PID={process.pid}\n상태: {state_path}\n로그: {state["log"]}', flush=True)
    return state


def worker_main(initial_run, search_id, gate_evidence):
    root = Path(ROOT).resolve()
    results = Path(get_config()['paths']['results']).resolve()
    with _lock(results/'.search.lock', blocking=True):
        worker = SearchWorker(root, results, initial_run, search_id, gate_evidence)
        try:
            # 분리된 진입점을 직접 호출해 검증 게이트를 우회할 수 없도록 검증한다.
            matches = [event for event in worker.journal.events()
                       if event['kind'] == 'command_result'
                       and event['data']['evidence_id'] == gate_evidence]
            if len(matches) != 1 or not matches[0]['data']['success']:
                raise AuditIntegrityError('유효한 Phase1 외부 실행 증거가 없습니다.')
            starts = [event for event in worker.journal.events()
                      if event['kind'] == 'command_start'
                      and event['data']['evidence_id'] == gate_evidence]
            if (len(starts) != 1 or starts[0]['data']['argv'] !=
                    [sys.executable, '-u', '-m', 'src.test_contracts', '-v']
                    or starts[0]['data']['cwd'] != str(root)):
                raise AuditIntegrityError('Phase1 전체 검증 명령의 증거가 아닙니다.')
            worker.journal.verify(verify_evidence=True)
            worker.search()
        except BaseException as error:
            _trace(worker.trace, 'search_stopped', error=repr(error), traceback=traceback.format_exc())
            worker.save_state(status='failed', last_error=repr(error), ended_utc=_now().isoformat())
            mandatory=worker.save_mandatory_case()
            if mandatory['window10_epoch20']['status']!='completed':
                worker.save_mandatory_case('unfulfilled',reason='작업 중단으로 필수 케이스 증거 미완료: '+repr(error))
            try:
                worker.save_audit_summary()
            except AuditIntegrityError:
                # 장부 자체가 손상되면 검증되지 않은 새 요약을 만들지 않는다.
                pass
            if worker.best is not None and not isinstance(error, AuditIntegrityError):
                try:
                    worker.finalize('runtime_error_limit_or_unexpected_error')
                except BaseException as final_error:
                    worker.save_state(status='failed', finalization_error=repr(final_error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--initial-run', default='initial_w10_e10')
    parser.add_argument('--search-id')
    parser.add_argument('--gate-evidence')
    options = parser.parse_args()
    if options.worker:
        if not options.search_id or not options.gate_evidence:
            parser.error('--worker requires --search-id and --gate-evidence')
        worker_main(options.initial_run, options.search_id, options.gate_evidence)
    else:
        run_search(options.initial_run)
