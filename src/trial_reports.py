"""각 평가 실험의 전체 해상도 HTML과 내용 해시 기반 공유 압축 시계열.

정상 원본/정답, 모델별 복원/점수, 임계값 후보별 Alarm을 나눠 저장한다.
동일 데이터는 results/_trial_shared 아래 한 번 저장하고 각 평가의
dashboard_data에는 hardlink를 만든다. gzip 해제는 브라우저 내장 API로
수행하므로 CDN/서버 계산 없이 로컬 HTML에서도 원본 전체를 탐색한다.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from .dashboard import _json, _native
from .audit import AuditIntegrityError, file_sha256


_LOADER = r"""
window.EWS_TRIAL_BLOBS=Object.create(null);
window.EWS_TRIAL=(function(){
  const decoded=new Map(),inflight=new Map();
  async function blob(key){
    if(decoded.has(key)){const v=decoded.get(key);decoded.delete(key);decoded.set(key,v);return v;}
    if(inflight.has(key))return inflight.get(key);
    const promise=(async()=>{
      if(typeof DecompressionStream!=='function')throw new Error('전체 원본 그래프는 gzip 해제를 지원하는 최신 Edge/Chrome에서 열어주세요.');
      await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src='dashboard_data/blob_'+key+'.js';script.onload=()=>{script.remove();resolve();};script.onerror=()=>{script.remove();reject(new Error('공유 시계열 파일을 읽지 못했습니다: '+key));};document.head.appendChild(script);});
      const encoded=window.EWS_TRIAL_BLOBS[key];delete window.EWS_TRIAL_BLOBS[key];
      if(typeof encoded!=='string')throw new Error('압축 시계열 응답이 없습니다.');
      const binary=atob(encoded),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
      const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
      const value=JSON.parse(await new Response(stream).text());decoded.set(key,value);
      while(decoded.size>9)decoded.delete(decoded.keys().next().value);
      return value;
    })();inflight.set(key,promise);
    try{return await promise;}finally{inflight.delete(key);}
  }
  return{blob:blob,async tag(index,keys,tag,threshold){
    try{const parts=await Promise.all(keys.map(blob));window.EWS_TAG_READY(index,Object.assign({tag:tag,threshold:threshold},...parts));}
    catch(error){if(window.EWS_TRIAL_DATA_ERROR)window.EWS_TRIAL_DATA_ERROR(index,error.message);else throw error;}
  }};
})();
"""


def _read(path, default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def _save(path, value):
    _atomic_text(path,json.dumps(_native(value),ensure_ascii=False,indent=2,allow_nan=False))


def _atomic_text(path,text):
    path=Path(path)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    # Windows에서도 해시 계산에 사용한 LF byte를 그대로 보존한다.
    temporary.write_bytes(text.encode('utf-8'))
    os.replace(temporary,path)


def _blob(data,shared,destination):
    compressed=gzip.compress(_json(data).encode('utf-8'),compresslevel=4,mtime=0)
    key=hashlib.sha256(compressed).hexdigest()
    source=shared/('blob_'+key+'.js')
    encoded=base64.b64encode(compressed).decode('ascii')
    content='window.EWS_TRIAL_BLOBS['+json.dumps(key)+']='+json.dumps(encoded)+';\n'
    expected=hashlib.sha256(content.encode('utf-8')).hexdigest()
    if not source.is_file():
        _atomic_text(source,content)
    elif file_sha256(source)!=expected:
        raise AuditIntegrityError('공유 시계열 파일이 내용 해시와 다릅니다: '+str(source))
    target=destination/source.name
    linked=True
    if not target.exists():
        try: os.link(source,target)
        except OSError:
            shutil.copyfile(source,target)
            linked=False
    else:
        linked=os.path.samefile(source,target)
        if file_sha256(target)!=expected:
            raise AuditIntegrityError('평가 시계열 파일이 내용 해시와 다릅니다: '+str(target))
    return key,source.stat().st_size,linked


def _verify_counts(row,labels,alarms,name,applicable=True):
    selected=np.isfinite(labels)&bool(applicable)
    truth=labels[selected].astype(bool);predicted=alarms[selected].astype(bool)
    counts={'tp':int(np.sum(truth&predicted)),'tn':int(np.sum(~truth&~predicted)),
            'fp':int(np.sum(~truth&predicted)),'fn':int(np.sum(truth&~predicted)),
            'n':int(selected.sum())}
    if any(row.get(key)!=value for key,value in counts.items()):
        raise AuditIntegrityError(f'{name} 저장된 혼동행렬/분모가 원본 배열과 다릅니다.')
    def ratio(a,b): return a/b if b else None
    rates={'accuracy':ratio(counts['tp']+counts['tn'],counts['n']),
        'precision':ratio(counts['tp'],counts['tp']+counts['fp']),
        'recall':ratio(counts['tp'],counts['tp']+counts['fn']),
        'f1':ratio(2*counts['tp'],2*counts['tp']+counts['fp']+counts['fn'])}
    for key,value in rates.items():
        stored=row.get(key)
        if (value is None and stored is not None) or (value is not None and
                (not isinstance(stored,(int,float)) or not math.isclose(stored,value,rel_tol=1e-9,abs_tol=1e-12))):
            raise AuditIntegrityError(f'{name} {key}가 원본 배열과 다릅니다.')


def _comparison(row,n):
    row=row or {}
    return {**{key:row.get(key) for key in ('accuracy','precision','recall','f1')},
        **{key:int(row.get(key,0)) for key in ('tp','tn','fp','fn')},
        'evaluated':int(row.get('n',0)),'omitted':n-int(row.get('n',0)),
        'positive_rate':(row.get('tp',0)+row.get('fn',0))/row['n'] if row.get('n') else None}


def create_trial_report(run_dir,candidate=None):
    """실제 평가 run마다 dashboard.html을 생성한다. GPU 실행은 없다."""
    run_dir=Path(run_dir).resolve()
    metrics=_read(run_dir/'metrics.json')
    config=_read(run_dir/'config.json')
    if metrics is None or config is None:
        raise FileNotFoundError('실험 HTML에는 해당 run의 metrics.json/config.json이 필요합니다.')
    results=Path(config['paths']['results']).resolve()
    shared=results/'_trial_shared';shared.mkdir(parents=True,exist_ok=True)
    data_dir=run_dir/'dashboard_data';data_dir.mkdir(exist_ok=True)
    provenance=_read(run_dir/'eval_provenance.json',{})
    candidate=candidate or {}
    identities={name:file_sha256(run_dir/name) for name in ('metrics.json','config.json','predictions.npz')}
    if candidate.get('metrics_sha256') and candidate['metrics_sha256']!=identities['metrics.json']:
        raise AuditIntegrityError('평가 후보와 HTML의 metrics.json이 다릅니다.')
    if candidate.get('config_sha256') and candidate['config_sha256']!=identities['config.json']:
        raise AuditIntegrityError('평가 후보와 HTML의 config.json이 다릅니다.')
    checkpoint=candidate.get('checkpoint') or provenance.get('checkpoint')
    if checkpoint is None and (run_dir/'best.pt').is_file():
        checkpoint=str((run_dir/'best.pt').resolve())
    checkpoint_sha=candidate.get('checkpoint_sha256') or provenance.get('checkpoint_sha256')
    model_evidence={**provenance,**candidate.get('provenance',{})}
    if checkpoint:
        actual_checkpoint_sha=file_sha256(Path(checkpoint))
        if checkpoint_sha and checkpoint_sha!=actual_checkpoint_sha:
            raise AuditIntegrityError('HTML의 평가 Checkpoint hash가 저장 증거와 다릅니다.')
        checkpoint_sha=actual_checkpoint_sha
        if model_evidence.get('model_weights_epoch') is None or model_evidence.get('epochs_consumed') is None:
            from .trainer import load_trusted_checkpoint
            state=load_trusted_checkpoint(checkpoint,map_location='cpu')
            model_evidence.update(model_weights_epoch=state.get('model_weights_epoch',state['completed_epochs']),
                                  epochs_consumed=state['epochs_consumed'],model_id=state['model_id'])
    history_path=Path(checkpoint).parent/'history.csv' if checkpoint else run_dir/'history.csv'
    history=pd.read_csv(history_path).to_dict('records') if history_path.is_file() else []
    from .reports import _read_audit_summary,_read_experiments
    audit=_read_audit_summary(results)
    experiments,sources=_read_experiments(results)
    blobs=[]
    def save_blob(data):
        key,size,linked=_blob(data,shared,data_dir)
        blobs.append({'sha256':key,'bytes':size,'hardlinked':linked})
        return key
    with np.load(run_dir/'predictions.npz',allow_pickle=False) as saved:
        features=saved['features'].astype(str).tolist()
        timestamps=saved['timestamps'].astype(str).tolist()
        n=len(timestamps)
        actual=saved['actual_raw'];recon=saved['reconstruction_raw'];scores=saved['scores']
        labels=saved['labels'];alarms=saved['alarms'];thresholds=saved['thresholds']
        stopped=saved['stop'];truth_stop=saved['inhibit']
        expected=(n,len(features))
        if not n or any(array.shape!=expected for array in (actual,recon,scores,labels,alarms)):
            raise ValueError('실험 원본 시계열 shape 또는 시점 정렬이 일치하지 않습니다.')
        if len(set(features))!=len(features) or stopped.shape!=(n,) or truth_stop.shape!=(n,) or thresholds.shape!=(len(features),):
            raise ValueError('Tag 중복 또는 정지/임계값 배열 shape이 올바르지 않습니다.')
        for array,name,missing in [(alarms,'Alarm',False),(stopped,'정지 예측',False),(labels,'Label',True),(truth_stop,'Inhibit 정답',True)]:
            known=np.isfinite(array)
            if np.isinf(array).any() or (not missing and not known.all()) or not np.isin(array[known],[0,1]).all():
                raise ValueError(name+' 값은 0/1이며 정답 누락만 NaN일 수 있습니다.')
        if any(not np.isfinite(array).all() for array in (actual,recon,scores,thresholds)) or (scores<0).any() or (thresholds<0).any():
            raise ValueError('실제값/복원값/점수/임계값의 유효성 확인에 실패했습니다.')
        if not np.array_equal(alarms.astype(bool),(scores>thresholds)&~stopped.astype(bool)[:,None]):
            raise AuditIntegrityError('최종 Alarm이 실제 점수·임계값·정지 억제 규칙과 다릅니다.')
        threshold_config=config.get('threshold',{})
        if threshold_config.get('method')=='manual':
            expected_thresholds=[threshold_config.get('per_tag',{}).get(tag,threshold_config['manual']) for tag in features]
            if not np.allclose(thresholds,expected_thresholds,rtol=1e-7,atol=1e-12):
                raise AuditIntegrityError('수동 Threshold 설정과 실제 평가 임계값이 다릅니다.')
        times=np.asarray(timestamps,dtype='datetime64[ns]')
        if np.isnat(times).any() or (len(times)>1 and (np.diff(times)<=np.timedelta64(0,'ns')).any()):
            raise ValueError('원본 시각이 중복·역순이거나 유효하지 않습니다.')
        if set(metrics['classification']['per_tag'])!=set(features):
            raise AuditIntegrityError('저장된 Tag별 지표와 배열의 Tag 집합이 다릅니다.')
        denom=metrics.get('denominators',{})
        if denom.get('test_rows')!=n or denom.get('test_features')!=len(features):
            raise AuditIntegrityError('저장된 Test 평가 분모와 원본 배열이 다릅니다.')
        for j,tag in enumerate(features):
            _verify_counts(metrics['classification']['per_tag'][tag],labels[:,j],alarms[:,j],tag)
            row=metrics['classification']['per_tag'][tag]
            if row.get('missing_label_count',int(np.isnan(labels[:,j]).sum()))!=int(np.isnan(labels[:,j]).sum()):
                raise AuditIntegrityError(tag+' Label 누락 개수가 원본 배열과 다릅니다.')
        stop_applicable=metrics.get('stop_rule_status',{}).get('applicable',True)
        _verify_counts(metrics['stop_validation']['per_tag']['ProcessStopped'],truth_stop,stopped,'정지 검증',stop_applicable)
        actual_mae=float(scores.mean())
        if not math.isclose(actual_mae,float(metrics['gate']['mae']),rel_tol=1e-7,abs_tol=1e-9):
            raise AuditIntegrityError('전체 MAE와 저장된 원본 이상점수 평균이 다릅니다.')
        tag_pass=all(row.get('accuracy') is not None and row['accuracy']>=.95 for row in metrics['classification']['per_tag'].values())
        if metrics['gate'].get('per_tag_accuracy')!={tag:metrics['classification']['per_tag'][tag]['accuracy'] for tag in features}:
            raise AuditIntegrityError('합격 기준의 Tag별 정확도가 실제 혼동행렬과 다릅니다.')
        if metrics['gate'].get('mae_lt',.02)!=.02 or metrics['gate'].get('every_tag_accuracy_ge',.95)!=.95:
            raise AuditIntegrityError('고정 합격 기준값이 변경되었습니다.')
        safety=metrics.get('stop_safety',{})
        safety_counts={'predicted_stop_alarm_count':int(alarms[stopped.astype(bool)].sum()),
            'truth_stop_alarm_count':int(alarms[truth_stop==1].sum()),
            'truth_stop_missed_minutes':int(np.sum((truth_stop==1)&~stopped.astype(bool))),
            'predicted_stop_false_minutes':int(np.sum((truth_stop==0)&stopped.astype(bool)))}
        if any(safety.get(key)!=value for key,value in safety_counts.items()):
            raise AuditIntegrityError('정지 시 Alarm/누락 계수가 원본 배열과 다릅니다.')
        if metrics['gate']['mae_pass']!=(actual_mae<.02) or metrics['gate']['classification_pass']!=tag_pass or metrics['gate']['phase2_numeric_pass']!=((actual_mae<.02) and tag_pass):
            raise AuditIntegrityError('고정 합격 기준과 저장된 합격 주장이 다릅니다.')
        common=save_blob({'timestamps':timestamps,'predicted_inhibit':stopped.astype(np.int8),
                          'inhibit_labels':truth_stop})
        tag_metrics=[]
        for j,tag in enumerate(features):
            source_key=save_blob({'actual':actual[:,j],'labels':labels[:,j]})
            model_key=save_blob({'reconstructed':recon[:,j],'scores':scores[:,j]})
            alarm_key=save_blob({'final_alarm':alarms[:,j].astype(np.int8)})
            _atomic_text(data_dir/f'tag_{j:04d}.js',
                f'window.EWS_TRIAL.tag({j},{_json([source_key,model_key,alarm_key])},{_json(tag)},{_json(float(thresholds[j]))});\n')
            row=_comparison(metrics['classification']['per_tag'][tag],n)
            row.update(tag=tag,threshold=float(thresholds[j]),alarm_count=int(alarms[:,j].sum()))
            tag_metrics.append(row)
        gate=metrics['gate'];stop_status=metrics.get('stop_rule_status',{})
        stop_applicable=stop_status.get('applicable',True)
        stop_metric=_comparison(metrics.get('stop_validation',{}).get('per_tag',{}).get('ProcessStopped'),n)
        missing=stop_status.get('missing_sensor_tags',[])
        hold=stop_status.get('release_hold_minutes',config.get('alarm',{}).get('release_hold_minutes',0))
        stop_rule=(f'현재 센서 조건 즉시 정지 · 해제 {hold}분 확인(현재·과거만 사용)' if stop_applicable else
            '정지 억제 미적용'+(' · 조건 센서 없음: '+', '.join(missing) if missing else ' · 설정 비활성/조건 없음')+'; score 기반 Alarm 탐지는 계속합니다.')
        summary={'metadata':{'title':'회전기계 EWS · 실험 '+run_dir.name,'run_id':run_dir.name,
            'config':config,'selected_checkpoint':checkpoint,
            'model_weights_epoch':model_evidence.get('model_weights_epoch'),
            'epochs_consumed':model_evidence.get('epochs_consumed'),
            'value_unit':'원래 센서 단위','generated_at':candidate.get('completed_utc'),
            'normalization':'정상 train 시간순 앞 80%에 fit한 MinMaxScaler. Test 재학습·clip 없음.',
            'alarm_rule':'Tag별 정규화 절대 복원오차 > 해당 임계값. 적용 가능한 센서 정지 규칙으로만 억제합니다.',
            'stop_rule':stop_rule,
            'description':'이 평가 run의 실제 저장 배열과 채점 결과입니다. 원본 모든 시점을 압축 보존하며, 확대하면 해당 구간을 1분 단위로 확인할 수 있습니다.',
            'evaluation_note':'Test를 반복 확인하는 개발평가입니다. Alarm_*와 Inhibit은 채점용이며 탐지 입력에 사용하지 않습니다.'},
            'metrics':{**metrics,'normalized_test_mae':gate['mae']},'tags':features,'n':n,
            'tag_metrics':tag_metrics,'normalized_mae':gate['mae'],
            'all_labels_complete':all(row['evaluated']==n for row in tag_metrics),
            'mae_pass':gate['mae_pass'],'tag_pass':gate['classification_pass'],
            'phase2_pass':gate['phase2_numeric_pass'],'stop_metrics':stop_metric,
            'stop_rule_applicable':stop_applicable,'stop_truth_known_minutes':int(np.isfinite(truth_stop).sum()),
            'stop_truth_stop_minutes':int(np.sum(truth_stop==1)),
            'history':history,'experiments':experiments,'experiment_sources':sources,'audit_summary':audit}
    assets=Path(__file__).with_name('dashboard_assets')
    js=(assets/'dashboard.js').read_text(encoding='utf-8')
    old="script.onload=()=>{script.remove();if(pending.has(index)){pending.delete(index);reject(new Error('태그 파일에 데이터가 없습니다.'));}};"
    if old not in js:
        raise RuntimeError('대시보드 Tag 로더 계약이 변경되어 압축 실험 보고서 통합을 확인해야 합니다.')
    js=js.replace(old,'script.onload=()=>{script.remove();};')
    js=js.replace('window.EWS_TAG_READY=(index,data)=>{',
        "window.EWS_TRIAL_DATA_ERROR=(index,message)=>{if(pending.has(index)){pending.get(index).reject(new Error(message));pending.delete(index);}};\n  window.EWS_TAG_READY=(index,data)=>{")
    template=(assets/'dashboard.html').read_text(encoding='utf-8')
    old_scripts='<script>window.EWS_SUMMARY=__SUMMARY__;</script><script src="dashboard_data/common.js"></script><script>__SCRIPT__</script><script>__SHAP_SCRIPT__</script>'
    if old_scripts not in template:
        raise RuntimeError('대시보드 script 통합 계약이 변경되었습니다.')
    scripts='<script>window.EWS_SUMMARY=__SUMMARY__;\n'+_LOADER+'\nwindow.EWS_TRIAL.blob('+_json(common)+').then(common=>{window.EWS_COMMON=common;\n__SCRIPT__\n__SHAP_SCRIPT__\n}).catch(error=>{document.getElementById("chart-message").textContent=error.message;document.getElementById("overall-status").textContent="시계열 읽기 실패: "+error.message;});</script>'
    page=template.replace(old_scripts,scripts).replace('__TITLE__',html.escape(summary['metadata']['title']))
    page=page.replace('__STYLE__',(assets/'dashboard.css').read_text(encoding='utf-8')).replace('__SUMMARY__',_json(summary)).replace('__SCRIPT__',js).replace('__SHAP_SCRIPT__',(assets/'shap.js').read_text(encoding='utf-8'))
    path=run_dir/'dashboard.html'
    if any(file_sha256(run_dir/name)!=digest for name,digest in identities.items()):
        raise AuditIntegrityError('실험 HTML 생성 도중 평가 입력 파일이 변경되었습니다.')
    if checkpoint and file_sha256(Path(checkpoint))!=checkpoint_sha:
        raise AuditIntegrityError('실험 HTML 생성 도중 Checkpoint가 변경되었습니다.')
    _save(run_dir/'trial_report_manifest.json',{'dashboard':'dashboard.html','run_id':run_dir.name,
        'source':'predictions.npz','source_rows':n,'tags':features,'full_resolution_preserved':True,
        'compression':'gzip_json_base64','shared_directory':str(shared),'blobs':blobs,
        'physical_deduplication':all(blob['hardlinked'] for blob in blobs),
        'metrics_source':'metrics.json','source_identities':identities,'checkpoint':checkpoint,
        'checkpoint_sha256':checkpoint_sha,'model_evidence':model_evidence,
        'dashboard_sha256':hashlib.sha256(page.encode('utf-8')).hexdigest()})
    _atomic_text(path,page)
    return path
