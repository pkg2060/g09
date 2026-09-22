"""Full/train/eval entry point; all outputs are scoped to a new run directory."""
from __future__ import annotations
import argparse
from contextlib import closing
import json
import os
import platform
import random
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import get_config
from .data import read_sensor_csv, fit_training_scaler, save_scaler, load_scaler, make_loader, file_sha256
from .model import build_model
from .trainer import Trainer, reconstruct, load_trusted_checkpoint, canonical_scaler_state
from .detection import point_scores, process_stop, apply_release_hold, generate_alarms, calculate_thresholds, classification_metrics, regression_metrics


def stage(number, title):
    print('\n' + '='*64 + f'\n단계 {number}: {title}\n' + '='*64, flush=True)


def print_regression(title, report):
    print(title + ' (MAPE는 실제값 0 제외; 나머지 지표는 전체 점)', flush=True)
    for tag, row in [('전체', report['overall']), *report['per_tag'].items()]:
        display = {key: row[key] for key in ('n', 'mae', 'rmse', 'r2', 'mape')}
        print(tag + ': ' + json.dumps(display, ensure_ascii=False), flush=True)


def predict_process_stop(sensor_frame, alarm_config, timestamps=None):
    """설비가 바뀌면 정지 조건 센서의 존재부터 확인한다. 정답 열은 받지 않는다."""
    lookup={str(column).casefold():column for column in sensor_frame.columns}
    resolved=dict(alarm_config)
    resolved['conditions']=[dict(condition) for condition in alarm_config.get('conditions',[])]
    missing=[]
    for condition in resolved['conditions']:
        key=str(condition['tag']).casefold()
        if key not in lookup: missing.append(condition['tag'])
        else: condition['tag']=lookup[key]
    applicable=bool(alarm_config.get('enabled',True) and resolved['conditions'] and not missing)
    if not applicable:
        print('정지 억제 미적용: '+('조건 센서 없음 '+', '.join(missing) if missing else '설정 비활성/조건 없음')+
              '. Score 기반 Alarm 탐지는 계속합니다. 새 설비 정지 조건은 src/config.py에서 지정하세요.',flush=True)
        return np.zeros(len(sensor_frame),dtype=bool), {'applicable':False,'missing_sensor_tags':missing,
                'truth_used':False,'reason':'disabled_or_unavailable_sensor_conditions',
                'release_hold_minutes':alarm_config.get('release_hold_minutes',0),'release_hold_applied':False}
    hold=resolved.get('release_hold_minutes',0)
    raw_stop=process_stop(sensor_frame,{**resolved,'release_hold_minutes':0})
    stopped=apply_release_hold(raw_stop,timestamps,hold)
    hold=int(hold)
    gaps=(int(np.sum(np.diff(np.asarray(timestamps,dtype='datetime64[ns]')) != np.timedelta64(1,'m')))
          if timestamps is not None and len(timestamps)>1 else 0)
    return stopped, {'applicable':True,'missing_sensor_tags':[],
                'truth_used':False,'resolved_conditions':resolved['conditions'],'mode':resolved['mode'],
                'release_hold_minutes':int(hold),'release_hold_applied':bool(hold),
                'raw_sensor_stop_minutes':int(raw_stop.sum()),
                'additional_release_hold_minutes':int(np.sum(stopped & ~raw_stop)),
                'timestamp_discontinuities':gaps,'hold_propagates_across_gap':False,'uses_future_values':False,
                'release_hold_reason':f'센서 정지는 즉시 억제, 해제 후 연속{hold}분 재기동 안정 확인; 제공 Test 경계 진단을 참고한 개발 설정'}

def write_json(path, value):
    def default(v):
        if isinstance(v, (np.integer,np.floating,np.bool_)): return v.item()
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,Path): return str(v)
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=default,allow_nan=False),encoding='utf-8')

def merge_dict(base, override):
    for k,v in override.items():
        if k not in base: raise ValueError(f'Unknown configuration key: {k}')
        if isinstance(v,dict) and isinstance(base[k],dict):
            if k=='per_tag': base[k]=v
            else: merge_dict(base[k],v)
        else: base[k]=v
    return base

def export_databases(run_dir, test, actual, reconstruction, scores, thresholds, alarms, overall, stopped):
    """Wide OutputDB has exactly one row per input minute; no downsampling."""
    frame = pd.DataFrame({'DateTime':test['timestamps'], 'ProcessStopped':stopped.astype('int8'),
                          'Inhibit_Truth':test['inhibit'], 'Overall_MAE':scores.mean(axis=1),
                          'Overall_MSE':np.square(scores).mean(axis=1),
                          'Overall_RMSE':np.sqrt(np.square(scores).mean(axis=1)),
                          'Overall_Alarm':overall.astype('int8')})
    for j,tag in enumerate(test['features']):
        for suffix,value in {'Actual':actual[:,j],'Reconstruction':reconstruction[:,j],
                             'Score_MAE':scores[:,j],'MSE':scores[:,j]**2,
                             'RMSE':scores[:,j], 'Threshold':thresholds[j],
                             'Alarm':alarms[:,j].astype('int8'), 'Label':test['labels'][:,j]}.items():
            frame[f'{tag}__{suffix}']=value
    frame.to_csv(run_dir/'OutputDB.csv',index=False)
    entries=[]
    for j,tag in enumerate(test['features']):
        ix=np.flatnonzero(alarms[:,j])
        entries.append(pd.DataFrame({'DateTime':test['timestamps'][ix], 'Level':'Warning',
                                     'Scope':'Tag','Tag':tag,'Score':scores[ix,j],
                                     'Threshold':thresholds[j]}))
    ix=np.flatnonzero(overall)
    # Overall OR has no single MAE threshold; record the largest score/threshold
    # ratio (dimensionless, threshold 1) in this separate scope.
    ratios=np.divide(scores,thresholds[None,:],out=np.full_like(scores,np.inf),where=thresholds[None,:]>0)
    ratios[(thresholds[None,:]==0)&(scores==0)]=0
    entries.append(pd.DataFrame({'DateTime':test['timestamps'][ix], 'Level':'Warning',
                                 'Scope':'OverallMaxRatio','Tag':'ANY','Score':ratios[ix].max(axis=1),'Threshold':1.}))
    alarm_frame=pd.concat(entries,ignore_index=True).sort_values(['DateTime','Scope','Tag'])
    alarm_frame.to_csv(run_dir/'AlarmDB.csv',index=False)
    with closing(sqlite3.connect(run_dir/'ews.sqlite')) as connection, connection:
        frame.to_sql('OutputDB',connection,index=False,if_exists='fail',chunksize=2000)
        alarm_frame.to_sql('AlarmDB',connection,index=False,if_exists='fail',chunksize=2000)
        connection.execute('CREATE INDEX output_time ON OutputDB(DateTime)')
        connection.execute('CREATE INDEX alarm_time ON AlarmDB(DateTime)')
    return {'output_rows':len(frame),'alarm_rows':len(alarm_frame),'tag_alarm_points':int(alarms.sum()),
            'overall_alarm_minutes':int(overall.sum())}

def evaluate(model, scaler_payload, train, test, config, run_dir, device, history=None, saved_threshold_path=None):
    features=test['features']; scaler=scaler_payload['scaler']
    test_scaled=scaler.transform(test['values']).astype(np.float32)
    loader=make_loader(test_scaled,test['timestamps'],config)
    print(f'EVAL GPU {device}: {len(test_scaled):,} minutes x {len(features)} Tags',flush=True)
    started=time.monotonic()
    reconstruction=reconstruct(model,loader,device)
    if reconstruction.shape != test_scaled.shape or not np.isfinite(reconstruction).all():
        raise ValueError('Inference must reconstruct every original timestamp and Tag')
    scores=point_scores(test_scaled,reconstruction)
    normalized_report=regression_metrics(test_scaled,reconstruction,features)
    print_regression('Test 정규화 복원 성능', normalized_report)
    train_quality=None
    if config['report']['export_database']:
        stage('4-1', '전체 Train 복원 성능 확인 (재학습 없음)')
        train_scaled=scaler.transform(train['values']).astype(np.float32)
        train_recon=reconstruct(model,make_loader(train_scaled,train['timestamps'],config),device)
        train_quality={'normalized':regression_metrics(train_scaled,train_recon,features),
                       'raw':regression_metrics(train['values'],scaler.inverse_transform(train_recon),features)}
        write_json(run_dir/'train_reconstruction_metrics.json',train_quality)
        print_regression('Train 정규화 복원 성능',train_quality['normalized'])
        del train_scaled,train_recon
    stage(5, '현재 설정에 따른 Threshold 산출·적용')
    reference=None
    thresholds=None
    threshold_source='current_config' if config['threshold']['method'].lower()=='manual' else 'validation_calibration'
    if config['threshold']['method'].lower()!='manual' and config['threshold']['reuse_saved_on_eval'] and saved_threshold_path and Path(saved_threshold_path).exists():
        saved=json.loads(Path(saved_threshold_path).read_text(encoding='utf-8'))
        if saved['method']==config['threshold']['method']:
            if set(saved['values'])!=set(features) or saved['space']!='train80_minmax_absolute_error':
                raise ValueError('Saved thresholds use incompatible features or score space')
            thresholds=np.array([saved['values'][tag] for tag in features],dtype=float)
            if not np.isfinite(thresholds).all() or (thresholds<0).any():
                raise ValueError('Invalid saved thresholds')
            threshold_source='saved_threshold.json'
    if thresholds is None and config['threshold']['method'].lower()!='manual':
        cut=scaler_payload['training_rows']
        if config['threshold']['reference_split']!='validation':
            raise ValueError('This pipeline calibrates automatic thresholds on validation only')
        val_scaled=scaler.transform(train['values'][cut:]).astype(np.float32)
        val_recon=reconstruct(model,make_loader(val_scaled,train['timestamps'][cut:],config),device)
        reference=point_scores(val_scaled,val_recon)
    if thresholds is None:
        thresholds=calculate_thresholds(reference,config['threshold'],features)
    threshold_record={'space':'train80_minmax_absolute_error','method':config['threshold']['method'],
                      'values':dict(zip(features,map(float,thresholds))), 'config':config['threshold'],
                      'manual_precedence':'current_config_over_saved_thresholds','source':threshold_source}
    write_json(run_dir/'threshold.json',threshold_record)
    print('THRESHOLDS '+json.dumps(threshold_record['values']),flush=True)
    stage(6, 'Tag별 절대오차 Score·센서 정지 조건·최종 Alarm 산출')
    # 채점용 열은 이 호출에 물리적으로 전달하지 않는다.
    stopped,stop_rule_status=predict_process_stop(test['frame'][features],config['alarm'],test['timestamps'])
    alarms,overall=generate_alarms(scores,thresholds,stopped)
    raw_reconstruction=scaler.inverse_transform(reconstruction)
    metrics={'evaluation_kind':'development_evaluation_on_provided_test',
             'normalized_regression':normalized_report,
             'raw_regression':regression_metrics(test['values'],raw_reconstruction,features),
             'classification':classification_metrics(test['labels'],alarms,test['timestamps'],features),
             'operating':classification_metrics(test['labels'],alarms,test['timestamps'],features,valid_mask=~stopped),
             'stopped':classification_metrics(test['labels'],alarms,test['timestamps'],features,valid_mask=stopped),
             'normal_points':regression_metrics(test_scaled,reconstruction,features,mask=test['labels']==0),
             'anomaly_points':regression_metrics(test_scaled,reconstruction,features,mask=test['labels']==1),
             'stop_validation':classification_metrics(test['inhibit'][:,None],stopped[:,None],test['timestamps'],['ProcessStopped'],
                                valid_mask=np.full(len(stopped),stop_rule_status['applicable'],dtype=bool)),
             'stop_rule_status':stop_rule_status,
             'denominators':{'test_rows':len(test_scaled),'test_features':len(features),
                              'initial_window_rows_included':True,'stopped_rows_included':True,
                              'rows_deleted':0,'stop_truth_used_as_model_or_alarm_input':False},
             'inference_seconds':time.monotonic()-started}
    metrics['stop_safety'] = {
        'predicted_stop_alarm_count': int(alarms[stopped].sum()),
        'truth_stop_alarm_count': int(alarms[test['inhibit']==1].sum()),
        'truth_stop_missed_minutes': int(np.sum((test['inhibit']==1)&~stopped)),
        'predicted_stop_false_minutes': int(np.sum((test['inhibit']==0)&stopped)),
        'truth_used_for_prediction': False,
        'truth_known_minutes': int(np.isfinite(test['inhibit']).sum()),
        'truth_stop_minutes': int(np.sum(test['inhibit']==1)),
        'stop_rule_applicable': stop_rule_status['applicable'],
    }
    print('ALARM '+json.dumps({'score_shape':list(scores.shape),'score_method':'정규화 절대오차',
                'threshold_comparison':'strict >','predicted_stop_minutes':int(stopped.sum()),
                'tag_alarm_points':dict(zip(features,map(int,alarms.sum(axis=0)))),
                'overall_alarm_minutes':int(overall.sum()),'stop_safety':metrics['stop_safety']},ensure_ascii=False),flush=True)
    for tag,row in metrics['classification']['per_tag'].items():
        print('정답 비교 '+tag+': '+json.dumps({key:row.get(key) for key in
              ('n','accuracy','precision','recall','f1','tp','fp','tn','fn','missing_label_count')},ensure_ascii=False),flush=True)
    print('Inhibit 정답 비교 '+json.dumps(metrics['stop_validation']['overall'],ensure_ascii=False),flush=True)
    metrics['gate']={'mae_lt':config['success']['normalized_test_mae_lt'],
                     'every_tag_accuracy_ge':config['success']['every_tag_accuracy_ge'],
                     'mae':float(scores.mean()),
                     'mae_pass':bool(scores.mean()<config['success']['normalized_test_mae_lt'])}
    metrics['gate']['per_tag_accuracy']={tag:float(np.mean(alarms[np.isfinite(test['labels'][:,j]),j]==test['labels'][np.isfinite(test['labels'][:,j]),j]))
                                        if np.isfinite(test['labels'][:,j]).any() else None for j,tag in enumerate(features)}
    metrics['gate']['classification_pass']=all(v is not None and v>=config['success']['every_tag_accuracy_ge'] for v in metrics['gate']['per_tag_accuracy'].values())
    metrics['gate']['phase2_numeric_pass']=metrics['gate']['mae_pass'] and metrics['gate']['classification_pass']
    write_json(run_dir/'metrics.json',metrics)
    np.savez_compressed(run_dir/'predictions.npz',timestamps=test['timestamps'].astype(str),features=np.array(features),
                        actual_raw=test['values'],reconstruction_raw=raw_reconstruction,actual_scaled=test_scaled,
                        reconstruction_scaled=reconstruction,scores=scores,thresholds=thresholds,
                        labels=test['labels'],alarms=alarms,stop=stopped,inhibit=test['inhibit'])
    print('GATE '+json.dumps(metrics['gate']),flush=True)
    if config['report']['export_database']:
        stage(7, '분 단위 OutputDB·AlarmDB·SQLite 저장')
        export=export_databases(run_dir,test,test['values'],raw_reconstruction,scores,thresholds,alarms,overall,stopped)
        write_json(run_dir/'export_manifest.json',export)
        print('DATABASE '+json.dumps(export),flush=True)
    from .reports import create_reports
    stage(8, '한국어 설명·그래프·채점 결과·검증 증거 생성')
    create_reports(run_dir, test, raw_reconstruction, scores, thresholds, alarms, stopped,
                   metrics, history or [], config)
    return metrics

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['full','train','eval'],default='full')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('-full', action='store_true')
    modes.add_argument('-train', action='store_true')
    modes.add_argument('-eval', action='store_true')
    modes.add_argument('-optimize', action='store_true', help='마감시각까지 랄프 최적화 루프')
    modes.add_argument('-verify', action='store_true', help='Phase1 독립 검증')
    modes.add_argument('-serve', action='store_true', help='대시보드·요청형 SHAP 로컬 서버 시작')
    parser.add_argument('--port', type=int, help='대시보드 서버 포트')
    parser.add_argument('--run-id',default=None,help='New unique output subfolder')
    parser.add_argument('--checkpoint',type=Path,help='Own best.pt for eval')
    parser.add_argument('--config',type=Path,help='Optional JSON experiment overrides of Config.py defaults')
    parser.add_argument('--metrics-only', action='store_true', help='탐색 중 대용량 DB·그림 생성을 마지막까지 미룸')
    args=parser.parse_args(argv)
    if args.run_id is not None and Path(args.run_id).name!=args.run_id:
        raise ValueError('run-id must be a simple folder name')
    if args.serve:
        from .serve import launch_dashboard
        run_dir=Path(get_config()['paths']['results'])/args.run_id if args.run_id else None
        return launch_dashboard(run_dir,args.checkpoint,args.port)
    if args.optimize:
        from .optimize import run_search
        return run_search()
    if args.verify:
        from .verify import verify
        return verify()
    args.mode = 'train' if args.train else 'eval' if args.eval else 'full' if args.full else args.mode
    args.run_id = args.run_id or (args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f'))
    if Path(args.run_id).name!=args.run_id: raise ValueError('run-id must be a simple folder name')
    config=get_config()
    if args.config: merge_dict(config,json.loads(args.config.read_text(encoding='utf-8-sig')))
    if config['success'] != {'normalized_test_mae_lt': .02, 'every_tag_accuracy_ge': .95}:
        raise ValueError('합격 기준 완화 금지: MAE<0.02, 모든 Tag 정확도>=0.95')
    if args.metrics_only:
        config['report'].update(export_database=False, dashboard=False, static_plots=False)
    if args.mode=='eval' and args.checkpoint is None:
        selection=Path(config['paths']['results'])/'selected_checkpoint.json'
        if selection.exists():
            args.checkpoint=Path(json.loads(selection.read_text(encoding='utf-8'))['checkpoint'])
        else:
            candidates=sorted(Path(config['paths']['results']).glob('*/best.pt'),key=lambda p:p.stat().st_mtime)
            if not candidates: raise ValueError('저장된 모델이 없습니다. 먼저 -train/-full 실행 또는 --checkpoint 지정')
            args.checkpoint=candidates[-1]
    if config['source'] not in {'rtdb_csv','benchmark_csv'}:
        raise ValueError('source must be rtdb_csv or benchmark_csv')
    if not torch.cuda.is_available(): raise RuntimeError('TranAD requires GPU/CUDA. CPU fallback forbidden.')
    device=torch.device('cuda:0')
    stage(0, '실행 설정 확인')
    print(json.dumps(config,ensure_ascii=False,indent=2),flush=True)
    torch.manual_seed(config['train']['seed']); np.random.seed(config['train']['seed']); random.seed(config['train']['seed'])
    run_dir=Path(config['paths']['results'])/args.run_id
    run_dir.mkdir(parents=True,exist_ok=False)
    write_json(run_dir/'config.json',config)
    write_json(run_dir/'environment.json',{'time_utc':datetime.now(timezone.utc).isoformat(),
                'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
                'cuda_runtime':torch.version.cuda,'cuda_available':True,'gpu_name':torch.cuda.get_device_name(device),
                'command':sys.argv,'pid':os.getpid()})
    source_keys=('train','test') if config['source']=='rtdb_csv' else ('benchmark_train','benchmark_test')
    stage(1, '데이터 로딩·무결성·정답 분리')
    train=read_sensor_csv(config['paths'][source_keys[0]],config['data'])
    test=read_sensor_csv(config['paths'][source_keys[1]],config['data'],train['features']) if args.mode!='train' else None
    profile={'train':train['profile'],'test':test['profile'] if test else None}
    write_json(run_dir/'data_profile.json',profile)
    print(json.dumps({'run':args.run_id,'mode':args.mode,'train_rows':len(train['values']),
                      'test_rows':len(test['values']) if test else None,'features':train['features'],
                      'gpu':torch.cuda.get_device_name(device)},ensure_ascii=False),flush=True)
    history=None
    stage(2, '학습 구간 MinMax 정규화·TranAD 구성')
    eval_immutable=None
    if args.mode=='eval':
        if args.checkpoint is None: raise ValueError('eval requires --checkpoint from this project')
        eval_immutable={str(path.resolve()):file_sha256(path) for path in
                        (args.checkpoint,args.checkpoint.parent/'scaler.pkl')}
        checkpoint=load_trusted_checkpoint(args.checkpoint,map_location=device)
        scaler_payload=load_scaler(args.checkpoint.parent/'scaler.pkl')
        if scaler_payload['features']!=train['features']: raise ValueError('Saved scaler feature mismatch')
        if checkpoint['feature_names']!=train['features']: raise ValueError('Checkpoint feature order mismatch')
        if canonical_scaler_state(checkpoint['scaler_state'])!=canonical_scaler_state(scaler_payload):
            raise ValueError('Checkpoint and scaler state differ; refuse silently incorrect normalization')
        config['model']=checkpoint['model_cfg']
        model=build_model(len(train['features']),config['model']).to(device)
        model.deadline_utc=None # 마감 후에도 최상 체크포인트 평가·보고 가능
        model.load_state_dict(checkpoint['model_state_dict'])
        shutil.copy2(args.checkpoint.parent/'scaler.pkl',run_dir/'scaler.pkl')
        history=checkpoint.get('history')
        write_json(run_dir/'eval_provenance.json',{'checkpoint':str(args.checkpoint.resolve()),
                    'checkpoint_sha256':file_sha256(args.checkpoint),'scaler_sha256':file_sha256(run_dir/'scaler.pkl'),
                    'model_weights_epoch':checkpoint.get('model_weights_epoch',checkpoint.get('completed_epochs')),
                    'epochs_consumed':checkpoint.get('epochs_consumed'),
                    'training_performed':False,'scaler_fit_performed':False})
    else:
        if config['train'].get('resume_from'):
            previous=Path(config['train']['resume_from']).resolve().parent
            scaler_payload=load_scaler(previous/'scaler.pkl')
            scaler,cut=scaler_payload['scaler'],scaler_payload['training_rows']
            previous_profile=json.loads((previous/'data_profile.json').read_text(encoding='utf-8'))
            if previous_profile['train']['sha256']!=train['profile']['sha256']:
                raise ValueError('Resume training source changed; use the preserved original dataset')
            if scaler_payload['features']!=train['features']:
                raise ValueError('Resume scaler feature order mismatch')
            shutil.copy2(previous/'scaler.pkl',run_dir/'scaler.pkl')
            write_json(run_dir/'resume_provenance.json',{'checkpoint':config['train']['resume_from'],
                       'scaler_fit_performed':False,'scaler_sha256':file_sha256(run_dir/'scaler.pkl')})
        else:
            scaler,cut=fit_training_scaler(train['values'],config['data']['validation_fraction'])
            scaler_payload=save_scaler(scaler,train['features'],cut,run_dir/'scaler.pkl')
        scaled=scaler.transform(train['values']).astype(np.float32)
        train_loader=make_loader(scaled[:cut],train['timestamps'][:cut],config,training=True)
        val_loader=make_loader(scaled[cut:],train['timestamps'][cut:],config)
        model=build_model(len(train['features']),config['model'])
        print(f'Model TranAD / GPU {device} / parameters={sum(p.numel() for p in model.parameters()):,}',flush=True)
        stage(3, 'GPU 학습·Epoch별 Loss·시간·체크포인트 기록')
        engine=Trainer(model,config['train'],device,run_dir,config['deadline_utc'])
        portable_scaler={'features':train['features'],'training_rows':cut,
                         **{key:getattr(scaler,key).tolist() if isinstance(getattr(scaler,key),np.ndarray)
                            else int(getattr(scaler,key)) for key in
                            ['data_min_','data_max_','scale_','min_','n_samples_seen_']}}
        summary=engine.fit(train_loader,val_loader,scaler_state=portable_scaler,feature_names=train['features'])
        history=pd.read_csv(run_dir/'history.csv').to_dict('records') if (run_dir/'history.csv').exists() else None
        write_json(run_dir/'training_summary.json',summary)
    write_json(run_dir/'config.json',config)
    if args.mode in ['full','eval']:
        stage(4, '저장 모델 복원값·정규화 전체 Test MAE 평가')
        evaluate(model,scaler_payload,train,test,config,run_dir,device,history,
                 saved_threshold_path=args.checkpoint.parent/'threshold.json' if args.mode=='eval' else None)
    if eval_immutable is not None:
        preserved={path:file_sha256(path)==expected for path,expected in eval_immutable.items()}
        write_json(run_dir/'eval_immutability.json',{'files_unchanged':preserved,'expected_sha256':eval_immutable})
        if not all(preserved.values()): raise RuntimeError('Eval 중 원본 Checkpoint/Scaler 변경 감지')
    unchanged={name:file_sha256(p['path'])==p['sha256'] for name,p in profile.items() if p}
    write_json(run_dir/'integrity.json',{'original_data_unchanged':unchanged,'complete':True,
                                      'ended_utc':datetime.now(timezone.utc).isoformat()})
    if not all(unchanged.values()): raise RuntimeError('Source data changed while running')
    print(f'COMPLETE {run_dir}',flush=True)
    if os.environ.get('EWS_AUDIT_NONCE'):
        print('EWS_AUDIT_NONCE='+os.environ['EWS_AUDIT_NONCE'],flush=True)
    return str(run_dir)

if __name__=='__main__':
    main()
