"""독립 Phase1 계약시험 진입점. 통과는 실제 종료 코드로 판단한다."""
import json
import os
from pathlib import Path
from .audit import AuditJournal
from .test_contracts import GATE_REGISTRY, gate_checks


def protected_source_paths(root):
    """실행 소스·화면·vendor를 보호하며 노트북 자동저장/캐시는 제외한다."""
    root=Path(root).resolve()
    ignored_directories={'__pycache__', '.ipynb_checkpoints'}
    ignored_suffixes={'.ipynb', '.pyc', '.pyo'}
    sources=[path for path in (root/'src').rglob('*')
             if path.is_file() and path.suffix.lower() not in ignored_suffixes
             and not ignored_directories.intersection(path.relative_to(root/'src').parts)]
    return [root/'mode.py', *sorted(sources)]


def verify():
    root=Path(__file__).resolve().parents[1]
    # 최초 실행부터 진입점·검증기·모델·화면 자산·vendor·의존성을 고정한다.
    # 기존에 빈 baseline으로 생성한 장부는 소급 수정하지 않고 fail-closed한다.
    protected=protected_source_paths(root)
    journal=AuditJournal(root, gate_registry=GATE_REGISTRY, protected_paths=protected)
    loop=journal.start_loop('Phase1 고정 계약 검증', {})
    result=journal.run_gates(loop,gate_checks())
    print(json.dumps(result,ensure_ascii=False,indent=2))
    failed=[key for key,row in result['results'].items() if not row['passed']]
    if not result['all_passed']:
        journal.abort_loop(loop,'검증 실패: '+','.join(failed))
        raise SystemExit(1)
    journal.end_loop(loop)
    summary=journal.summary()
    (root/'results').mkdir(exist_ok=True)
    (root/'results'/'ralph_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if os.environ.get('EWS_AUDIT_NONCE'):
        print('EWS_AUDIT_NONCE='+os.environ['EWS_AUDIT_NONCE'],flush=True)
    return result
