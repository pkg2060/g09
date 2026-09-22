"""실제 배열과 채점 결과에서 한국어 보고서·그래프를 생성한다."""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import numpy as np
import pandas as pd


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.item() if hasattr(v, 'item') else str(v)), encoding='utf-8')


def _read_audit_summary(results, project_root=None):
    """최신 장부에서 집계한다. 과거 verify 스냅샷을 최신 실행 횟수로 오인하지 않는다."""
    root=Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    registry=root/'.ralph'/'gate_registry.json'
    events_path=root/'.ralph'/'events.jsonl'
    if registry.is_file() and events_path.is_file():
        from .audit import AuditJournal
        gates=json.loads(registry.read_text(encoding='utf-8'))['gates']
        journal=AuditJournal(root,gates)
        summary=journal.summary()
        events=journal.events()
        loop_sources={e['loop_id']:bool(e['data']['injected']) for e in events if e['kind']=='loop_start'}
        evidence_kinds={'evidence_rejected','gate_run','bug_registered','bug_resolved','loop_abort'}
        summary['evidence']=[{'event':e['kind'],'loop_id':e['loop_id'],'timestamp':e['utc'],
            'injected':loop_sources.get(e['loop_id'],False), 'details':e['data']}
            for e in events if e['kind'] in evidence_kinds]
        summary['snapshot']={'source':str(events_path),'captured_at_utc':datetime.now(timezone.utc).isoformat(),
            'kind':'live_journal_snapshot','note':'보고서 생성 시점의 집계이며 현재 평가 루프가 아직 열려 있을 수 있습니다.'}
        return summary
    cached=Path(results)/'ralph_summary.json'
    if cached.is_file():
        summary=json.loads(cached.read_text(encoding='utf-8'))
        summary['snapshot']={'source':str(cached),'kind':'saved_summary_only',
            'note':'원장에 접근하지 못해 저장된 요약을 표시합니다. 이후 실행을 포함하는 최신 집계인지는 미확인입니다.'}
        return summary
    return {}


def _read_experiments(results):
    """실제 trials.jsonl의 설정·실행·평가 기록을 연결한다. Epoch history와 별도다."""
    records=[]
    sources=[]
    for trace in sorted((Path(results)/'search').glob('*/trials.jsonl')):
        by_id={}
        source={'path':str(trace),'search_id':trace.parent.name,'parsed_events':0,'incomplete_tail':False}
        lines=trace.read_text(encoding='utf-8').splitlines()
        for line_number,line in enumerate(lines,1):
            if not line.strip():
                continue
            try:
                event=json.loads(line)
            except json.JSONDecodeError:
                if line_number==len(lines):
                    source['incomplete_tail']=True
                    break
                raise ValueError(f'실험 로그 {trace}:{line_number} JSON 형식 오류')
            source['parsed_events']+=1
            kind=event.get('event')
            run_id=event.get('run_id')
            candidate=event.get('candidate',{}) if kind=='candidate_evaluated' else {}
            if candidate:
                run_id=Path(candidate['evaluation_dir']).name
            argv=event.get('argv',[])
            if kind=='command_completed' and '--run-id' in argv:
                run_id=argv[argv.index('--run-id')+1]
            if not run_id and kind=='runtime_error':
                run_id=str(event.get('context') or f'error_line_{line_number}')
            if not run_id:
                continue
            row=by_id.setdefault(run_id,{'trial_id':run_id,'run_id':run_id,'search_id':trace.parent.name,
                'timestamp':event.get('time_utc'),'status':'기록됨','source_log':str(trace),'source_lines':[]})
            row['source_lines'].append(line_number)
            if kind=='configuration':
                row.update(changes=event.get('changes_from_previous',{}),reason=event.get('reason'),
                    config_path=event.get('config'),config_sha256=event.get('full_config_sha256'),status='설정 저장')
                path=Path(event.get('config',''))
                if path.is_file() and path.resolve().is_relative_to(trace.parent.resolve()):
                    row['config']=json.loads(path.read_text(encoding='utf-8'))
            elif kind=='command_completed':
                row.update(status='실행 완료' if event.get('evidence',{}).get('success') else '실행 실패',
                    evidence_id=event.get('evidence',{}).get('evidence_id'),completed_utc=event.get('time_utc'))
            elif kind=='runtime_error':
                row.update(status='실행 오류',error=event.get('error'),error_context=event.get('context'))
            elif kind=='incomplete_minimum_epochs':
                row.update(status='최소 Epoch 미달·후보 제외',training_summary=event.get('summary'))
            elif kind=='candidate_evaluated':
                provenance=candidate.get('provenance',{})
                row.update(config=candidate.get('config',{}),checkpoint=candidate.get('checkpoint'),
                    checkpoint_sha256=candidate.get('checkpoint_sha256'),evidence_id=candidate.get('evidence_id'),
                    mae=candidate.get('mae'),minimum_tag_accuracy=candidate.get('minimum_tag_accuracy'),
                    per_tag_accuracy=candidate.get('per_tag_accuracy'),phase2_numeric_pass=candidate.get('phase2_numeric_pass'),
                    failing_tag_count=candidate.get('failing_tag_count'),improved=event.get('improved'),
                    completed_utc=candidate.get('completed_utc'),provenance=provenance,
                    model_weights_epoch=provenance.get('model_weights_epoch'),
                    epochs_consumed=provenance.get('epochs_consumed'),
                    status='두 조건 통과' if candidate.get('phase2_numeric_pass') else '수치 목표 미달')
            config=row.get('config',{})
            row['epochs']=config.get('train',{}).get('epochs')
            row['window']=config.get('model',{}).get('window')
        records.extend(by_id.values())
        sources.append(source)
    return records,sources


def _audit_markdown(summary):
    actual=summary.get('actual',{})
    injected=summary.get('injected_tests',{})
    def number(key,gate=False,scope=actual):
        value=scope.get(key)
        return ('미실행' if gate and key in scope else '미집계') if value is None else str(value)
    a=number('loops_started'); b=number('evidence_rejected_loops')
    c1=number('gate_initial_passed',True); c2=number('gate_final_passed',True)
    total=number('p0_discovered'); fixed=number('p0_resolved')
    lines=['## 랄프 루프 발표용 최종 집계', '',
        f'총 **{a} 회** 실제 랄프 루프를 시작했습니다. (A) 그중 **{b} 회**, 검증 증거 불일치 등 '
        '정의된 부정직 신호를 탐지하여 증거를 거부했습니다. (B) 이 값은 의도적인 기만이 확인된 횟수가 아닙니다.', '',
        f'게이트 통과 항목은 **{c1} 개에서 {c2} 개로** 변화했습니다. (C1 → C2) '
        f'P0 등급 버그, 즉 크래시 없이 잘못된 값을 내보내는 버그는 **{total} 건 중 {fixed} 건**의 해결이 기록됐습니다. (D 전체 / 해결)', '']
    if actual.get('evidence_rejected_loops')==0:
        lines += ['실제 검증 증거 거부 횟수는 **0 회**입니다. 부정직 신호를 발견했다는 발표 사례를 만들어 채우지 않았습니다.','']
    elif isinstance(actual.get('evidence_rejected_loops'),int):
        lines += [f'이 **{b} 회**의 증거 불일치를 잡지 못했다면 해당 결과를 그대로 신뢰하기 어려웠을 것입니다.','']
    lines += ['`null`은 해당 검증 미실행 또는 집계 근거 미제공을 뜻합니다. 기록된 `0`은 0회/0건이며 미집계와 구별합니다. '
        'A는 시작한 실제 루프 수로, 학습 Epoch 수가 아닙니다. C1/C2는 같은 고정 게이트 집합의 최초/최종 실행입니다.', '',
        f'의도적 오류 주입 시험은 별도입니다: 루프 **{number("loops_started",scope=injected)} 회**, '
        f'증거 거부 **{number("evidence_rejected_loops",scope=injected)} 회**, '
        f'P0 **{number("p0_discovered",scope=injected)} 건 중 {number("p0_resolved",scope=injected)} 건** 해결. '
        '이 수치를 실제 A/B/D에 더하지 않습니다.', '',
        '정규화 이중 적용 사례는 실제 재현·수정 검증 증거가 확인된 경우에만 실제 해결 사례로 발표합니다.', '',
        '집계 스냅샷: '+str(summary.get('snapshot',{}).get('note','감사 원장 스냅샷 정보 미제공'))]
    return lines


def create_reports(run_dir, test, reconstruction, scores, thresholds, alarms, stopped,
                   metrics, history, config, *, regenerate_static_plots=True):
    run_dir=Path(run_dir)
    features=test['features']
    gate=metrics['gate']
    rows=[]
    for j,tag in enumerate(features):
        regression=metrics['normalized_regression']['per_tag'][tag]
        classification=metrics['classification']['per_tag'][tag]
        rows.append({'Tag':tag, 'MAE':regression['mae'], 'RMSE':regression['rmse'],
            'R2':regression['r2'], 'MAPE_percent':regression['mape'],
            'Threshold':float(thresholds[j]), 'Accuracy':classification['accuracy'],
            'Precision':classification['precision'], 'Recall':classification['recall'],
            'F1':classification['f1'], 'TP':classification['tp'], 'FP':classification['fp'],
            'FN':classification['fn'], 'TN':classification['tn'],
            'Alarm_count':int(alarms[:,j].sum()), 'Label_count':int(np.nansum(test['labels'][:,j])),
            'Evaluation_count':classification['n'], 'Missing_labels':classification['missing_label_count'],
            'Delay_seconds':classification['mean_detection_delay_seconds']})
    pd.DataFrame(rows).to_csv(run_dir/'tag_metrics.csv', index=False, encoding='utf-8-sig')
    stop_metric=metrics['stop_validation']['per_tag']['ProcessStopped']
    stop_status=metrics.get('stop_rule_status',{})
    stop_applicable=stop_status.get('applicable',True)
    stop_known=metrics['stop_safety'].get('truth_known_minutes',int(np.isfinite(test['inhibit']).sum()))
    truth_stop_minutes=metrics['stop_safety'].get('truth_stop_minutes',int(np.sum(test['inhibit']==1)))
    missing_stop_tags=stop_status.get('missing_sensor_tags',[])
    stop_notice=('센서 기반 정지 규칙 적용' if stop_applicable else
        '정지 억제 미적용: '+('조건 센서 없음 ('+', '.join(missing_stop_tags)+')' if missing_stop_tags else '설정 비활성 또는 조건 없음')+
        '. 공정을 운전 중이라고 확정한 것이 아니며, score 기반 Alarm 탐지는 계속합니다.')
    release_hold=int(stop_status.get('release_hold_minutes',0))
    hold_notice=(f'센서 정지는 즉시 억제하고, 해제 후 {release_hold}분 동안 억제를 유지합니다. '
                 '연속 1분 시각에서 현재·과거 센서만 사용하며 시간 공백을 넘겨 유지하지 않습니다.'
                 if release_hold else '센서의 현재 정지 조건만 적용하며 해제 지연은 없습니다.')
    if stop_applicable:
        stop_notice+=' · '+hold_notice
    stop_accuracy=(str(stop_metric['accuracy']) if stop_applicable and stop_known else
        '검증 불가 ('+('Inhibit 정답 없음' if not stop_known else '센서 정지 규칙 미적용')+')')
    truth_alarm_text=('검증 불가 (Inhibit 정답 없음)' if not stop_known else
        '해당 없음 (제공된 Inhibit 정답에 정지 시점 없음)' if not truth_stop_minutes else
        str(metrics['stop_safety']['truth_stop_alarm_count'])+f' (정답 정지 {truth_stop_minutes}분 기준)')
    label_known_count=sum(row['Evaluation_count'] for row in rows)
    _json(run_dir/'stop_validation.json', {'metrics':stop_metric,'safety':metrics['stop_safety'],
        'stop_rule_status':stop_status,'validation_available':bool(stop_applicable and stop_known),
        'prediction_source':'raw_sensor_conditions_only' if stop_applicable else 'not_applied',
        'truth_source':'input CSV Inhibit' if stop_known else 'unavailable',
        'interpretation':stop_notice,'truth_stop_alarm_interpretation':truth_alarm_text})
    audit_summary=_read_audit_summary(config['paths']['results'])
    experiments,experiment_sources=_read_experiments(config['paths']['results'])
    _json(run_dir/'ralph_summary.json',audit_summary)
    _json(run_dir/'experiment_history.json',{'experiments':experiments,'sources':experiment_sources})
    lines=['# 회전기계 EWS 실행 결과', '', f'- 실행 폴더: `{run_dir.name}`',
        '- 평가 구분: 제공된 Test를 반복 확인하는 개발평가. 독립적인 일반화 성능 검증 아님.',
        f'- 모델 입력: 센서 {len(features)}개. Alarm_* 및 Inhibit은 채점에만 사용.',
        '- 정규화: 정상 train 시간순 앞 80%로 fit한 MinMaxScaler. Test 재학습·clip 없음.',
        '- 전체 MAE: 모든 Test 시점×Tag의 정규화 절대 복원오차 평균. 초기 window·정지·이상 구간 포함.',
        f'- 전체 Test MAE: **{gate["mae"]:.8f}** / 기준 <0.02 / 통과: {gate["mae_pass"]}',
        f'- 최종 Alarm 발생 시점×Tag 수: **{int(alarms.sum())}**. 정답 열이 없어도 점수 기반 탐지를 수행합니다.',
        f'- Label 비교 가능 시점×Tag: {label_known_count} / {len(test["values"])*len(features)}',
        f'- 모든 Tag 일치율 ≥95%: **{gate["classification_pass"] if label_known_count else "검증 불가 (Alarm_* 정답 없음)"}**',
        f'- 두 수치 조건 동시 통과: **{gate["phase2_numeric_pass"] if label_known_count else "미판정 (정답 없는 탐지 실행)"}**',
        f'- 정지 규칙 상태: {stop_notice}',
        f'- 공정 정지 판정 일치율: {stop_accuracy}',
        f'- 정답 정지(Inhibit=1) 중 발생한 Tag Alarm 수: {truth_alarm_text}',
        '', '## Tag별 같은 시점의 Label·최종 Alarm 비교', '',
        '|Tag|일치율|Precision|Recall|F1|오경보|놓친 이상|', '|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v: '계산 불가' if v is None else f'{100*v:.3f}%'
    for row in rows:
        lines.append('|'+ '|'.join([row['Tag'],fmt(row['Accuracy']),fmt(row['Precision']),
            fmt(row['Recall']),fmt(row['F1']),str(row['FP']),str(row['FN'])])+'|')
    lines += ['', '## 랄프 루프 증거', '',
        '아래 통계는 감사 기록에서만 채운다. Epoch 수를 외부 루프 수로 세지 않는다. '
        '일반 검증 실패와 의도적인 오류 주입 시험을 실제 부정직 신호로 세지 않는다.',
        '```json', json.dumps(audit_summary,ensure_ascii=False,indent=2), '```',
        'A/B/C1/C2/D 수치는 아래 마지막 발표용 집계에 원장 그대로 표기한다. '
        '이중 정규화 등 예시 버그가 실제로 발견되었다고 가정하지 않는다.',
        '', '## 재현', '', '```python', '%run mode -full', '%run mode -train', '%run mode -eval', '```',
        '', '## 검증 범위와 한계', '',
        'Excel 첨부는 DRM 때문에 원본 셀을 읽지 못했다. 접근 가능한 요구문서 및 사용자 최신 요청을 검증 대상으로 삼았다.',
        '수치 목표가 미달인 경우 결과를 숨기거나 기준을 낮추지 않는다. 정상 비중이 크면 일치율만 높아질 수 있으므로 Recall·F1도 함께 해석한다.',
        '', '## 하이퍼파라미터 실험 기록', '',
        f'실제 trials.jsonl {len(experiment_sources)}개에서 실험·실행 기록 {len(experiments)}개를 읽었습니다. '
        '`experiment_history.json` 및 HTML 실험 탭에서 설정 변경 전후 값, 실제 가중치 Epoch, 체크포인트, MAE, Tag별 정확도를 확인할 수 있습니다. '
        '학습 Epoch별 history는 선택 모델 학습곡선에 별도로 유지합니다.', '']
    lines += _audit_markdown(audit_summary)
    (run_dir/'결과보고.md').write_text('\n'.join(lines),encoding='utf-8')
    if config['report']['static_plots'] and regenerate_static_plots:
        make_plots(run_dir,test,reconstruction,scores,thresholds,alarms,stopped,history,config)
    dashboard_path=None
    if config['report']['dashboard']:
        from .dashboard import render_dashboard
        provenance_path=run_dir/'eval_provenance.json'
        provenance=json.loads(provenance_path.read_text(encoding='utf-8')) if provenance_path.is_file() else {}
        stop_conditions=' '+str(config['alarm'].get('mode','or')).upper()+' '
        stop_conditions=stop_conditions.join(f"{c['tag']} {c['op']} {c['value']}" for c in config['alarm'].get('conditions',[]))
        payload={'metadata':{'title':'회전기계 조기경보 시스템', 'run_id':run_dir.name,
            'config':config, 'evaluation_kind':'개발평가', 'value_unit':'원래 센서 단위',
            'selected_checkpoint':provenance.get('checkpoint'),
            'provenance':provenance,
            'generated_at':datetime.now(timezone.utc).isoformat(),
            'normalization':'정상 train 시간순 앞 80%에만 fit한 MinMaxScaler. Test는 재학습·clip 없이 동일 scaler로 변환합니다.',
            'stop_rule':stop_conditions+' · '+hold_notice if stop_applicable else stop_notice,
            'alarm_rule':'학습 기준 정규화 공간의 절대 복원오차가 Tag 임계값을 초과하면 Alarm입니다. '+
                ('센서로 정지를 판정한 시점은 최종 Alarm을 억제합니다. ' if stop_applicable else '정지 억제는 미적용이며 score 기반 탐지는 계속합니다. ')+
                'Alarm_* 및 Inhibit 정답은 탐지 입력에 사용하지 않습니다.',
            'description':'정상 운전에서 학습한 TranAD의 복원값과 현재 센서값을 비교합니다. 정답 열이 없는 다른 CSV에서도 Alarm을 생성하며, Label·Inhibit 정답이 제공된 경우에만 비교 채점합니다.'},
            'metrics':{**metrics,'normalized_test_mae':gate['mae']},'tags':features,
            'timestamps':test['timestamps'].astype('datetime64[ms]').astype(str).tolist(),
            'actual':test['values'],'reconstructed':reconstruction,'scores':scores,
            'thresholds':thresholds,'final_alarm':alarms,'labels':test['labels'],
            'predicted_inhibit':stopped,'inhibit_labels':test['inhibit'],
            'history':history,'experiments':experiments,'experiment_sources':experiment_sources,'audit_summary':audit_summary}
        dashboard_path=render_dashboard(run_dir,payload)
        _json(run_dir/'report_manifest.json',{'dashboard':str(dashboard_path),
            'source_rows':len(test['values']),'full_resolution_preserved':True,
            'metrics':'tag_metrics.csv','stop_validation':'stop_validation.json'})
    return dashboard_path


def refresh_reports_from_saved(run_dir):
    """평가 루프 종료 후 저장된 CPU 배열로만 보고서를 갱신한다.

    모델 로드·GPU 추론·학습을 수행하지 않는다. 현재 감사 원장 집계와 완료된
    trials 기록을 다시 읽는다. config.json 및 HTML의 설정 표시는 보존하며,
    이미 갖춰진 정적 그림은 다시 생성하지 않는다. 반환값은 dashboard.html
    Path이며 저장된 설정에서 대시보드가 비활성화된 경우 None이다.
    """
    run_dir=Path(run_dir)
    metrics=json.loads((run_dir/'metrics.json').read_text(encoding='utf-8'))
    config=json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    history_path=run_dir/'history.csv'
    if not history_path.is_file():
        provenance_path=run_dir/'eval_provenance.json'
        provenance=json.loads(provenance_path.read_text(encoding='utf-8')) if provenance_path.is_file() else {}
        checkpoint=provenance.get('checkpoint')
        history_path=Path(checkpoint).parent/'history.csv' if checkpoint else history_path
    history=pd.read_csv(history_path).to_dict('records') if history_path.is_file() else []
    with np.load(run_dir/'predictions.npz',allow_pickle=False) as saved:
        features=saved['features'].astype(str).tolist()
        test={'features':features,'timestamps':saved['timestamps'].astype('datetime64[ns]'),
              'values':saved['actual_raw'],'labels':saved['labels'],'inhibit':saved['inhibit']}
        expected_plots=[run_dir/'plot'/('variable_'+re.sub(r'[^\w.-]','_',tag)+'.png') for tag in features]
        if history:
            expected_plots.append(run_dir/'plot'/'training_curve.png')
        plots_complete=bool(expected_plots) and all(path.is_file() for path in expected_plots)
        return create_reports(run_dir,test,saved['reconstruction_raw'],saved['scores'],
            saved['thresholds'],saved['alarms'],saved['stop'],metrics,history,config,
            regenerate_static_plots=not plots_complete)


def make_plots(run_dir,test,reconstruction,scores,thresholds,alarms,stopped,history,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    choices=[p for p in Path('/usr/share/fonts').rglob('*') if p.suffix in {'.ttf','.otf','.ttc'}
             and any(v in p.name.lower() for v in ['noto','nanum','malgun'])]
    if choices:
        font_manager.fontManager.addfont(str(choices[0]))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(choices[0])).get_name()
    plt.rcParams['axes.unicode_minus']=False
    plot=Path(run_dir)/'plot'; plot.mkdir(exist_ok=True)
    times=pd.to_datetime(test['timestamps'])
    # PNG is a static overview; interactive HTML retains every source point.
    for j,tag in enumerate(test['features']):
        fig,ax=plt.subplots(2,1,figsize=(14,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        ax[0].plot(times,test['values'][:,j],lw=.5,label='Actual')
        ax[0].plot(times,reconstruction[:,j],lw=.5,label='TranAD reconstruction')
        hits=np.flatnonzero(alarms[:,j]); ax[0].scatter(times[hits],test['values'][hits,j],s=4,c='#d14545',label='Final alarm',zorder=5)
        ax[1].plot(times,scores[:,j],lw=.6,color='#2963b5',label='MAE score')
        ax[1].axhline(thresholds[j],color='#d98b1d',ls='--',label='Threshold')
        for a in ax:
            a.fill_between(times,0,1,where=stopped,transform=a.get_xaxis_transform(),color='gray',alpha=.15)
            a.fill_between(times,0,1,where=alarms[:,j],transform=a.get_xaxis_transform(),color='red',alpha=.08)
            a.grid(alpha=.15); a.legend(loc='upper right',fontsize=8)
        ax[0].set_title(tag); ax[0].set_ylabel('Sensor unit'); ax[1].set_ylabel('Normalized score')
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(plot/('variable_'+re.sub(r'[^\w.-]','_',tag)+'.png'),dpi=config['report']['dpi']); plt.close(fig)
    if history:
        table=pd.DataFrame(history); fig,ax=plt.subplots(figsize=(9,4))
        ax.plot(table.epoch,table.train_mae,label='Train MAE'); ax.plot(table.epoch,table.val_mae,label='Validation MAE')
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(plot/'training_curve.png',dpi=config['report']['dpi']); plt.close(fig)
