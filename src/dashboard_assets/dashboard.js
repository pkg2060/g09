/* 한국어 결과 탐색기. 입력 배열·Label은 수정하지 않는다. 외부 라이브러리 불필요. */
(() => {
  'use strict';
  const S = window.EWS_SUMMARY;
  const C = window.EWS_COMMON;
  const $ = id => document.getElementById(id);
  const esc = value => String(value == null ? '미정' : value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const pct = value => finite(value) ? (100 * value).toFixed(2) + '%' : '미정';
  const num = (value, digits=5) => finite(value) ? Number(value.toPrecision(digits)).toLocaleString('ko-KR', {maximumFractionDigits:12}) : '미정';
  const count = value => finite(value) ? value.toLocaleString('ko-KR') : '미집계';
  const text = value => typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '기록 없음');
  const pick = (obj, keys) => {for (const key of keys) if (obj[key] != null) return obj[key]; return null;};
  const meta = S.metadata || {};
  const stopStatus=S.metrics?.stop_rule_status||{};
  const stopApplicable=S.stop_rule_applicable!==false&&stopStatus.applicable!==false;
  const knownLabels=S.tag_metrics.some(item=>item.evaluated>0);
  let chartsReady=false;
  function details(target, items) {
    $(target).innerHTML = items.map(([key,value]) => `<dt>${esc(key)}</dt><dd>${esc(text(value))}</dd>`).join('');
  }
  function kpi(label, value, note, state='') {
    return `<div class="kpi"><p>${esc(label)}</p><strong class="${state}">${esc(value)}</strong><small>${esc(note)}</small></div>`;
  }
  function showTab(id) {
    document.querySelectorAll('.tab-panel').forEach(el => el.classList.toggle('active', el.id === id));
    document.querySelectorAll('.tabs button').forEach(el => el.setAttribute('aria-selected', String(el.dataset.tab === id)));
    if (id === 'trend' && chartsReady) requestAnimationFrame(render);
  }
  document.querySelectorAll('.tabs button').forEach(el => el.addEventListener('click', () => showTab(el.dataset.tab)));
  $('overall-status').textContent = knownLabels?(S.phase2_pass ? 'Phase 2 · 두 수치 조건 통과' : 'Phase 2 · 조건 미충족 / 미확인')+(S.all_labels_complete?'':' · 비교 누락 경고'):'Alarm 탐지 결과 · 정답 비교 불가';
  $('overall-status').classList.add(S.phase2_pass ? 'pass' : 'fail');
  const accuracies = S.tag_metrics.map(m => m.accuracy).filter(finite);
  const minAccuracy = accuracies.length ? Math.min(...accuracies) : null;
  $('kpis').innerHTML = kpi('정규화 전체 Test MAE', num(S.normalized_mae, 7), '기준: 0.02 미만', S.mae_pass?'good':'bad') + kpi('가장 낮은 Tag 일치율', pct(minAccuracy), `기준: 모든 Tag 95% 이상 · ${S.all_labels_complete?'정답 비교 완료':'누락 정답/예측 있음'}`, S.tag_pass?'good':'bad') + kpi('분석 Tag', count(S.tags.length), '태그마다 동일 시점에서 개별 평가') + kpi('원본 시점', count(S.n), '다운샘플링 없이 모든 결과 배열 보존');
  $('system-description').textContent = meta.description || '고정 High/Low Alarm만으로 놓치기 쉬운 순간 Peak, 헌팅, 운전 흔들림, 지속적인 증가·감소를 살펴보기 위한 조기 경보 시스템입니다. 이번 결과에서 실제로 감지했는지는 정답과 최종 Alarm을 비교해 확인합니다.';
  const runConfig=meta.config||meta.hyperparameters||{};
  const modelConfig=runConfig.model||{},trainConfig=runConfig.train||{},thresholdConfig=runConfig.threshold||{};
  const recorded=value=>value!==null&&value!==undefined;
  const modelWindow=modelConfig.window??runConfig.window;
  const sampleMinutes=runConfig.data?.sample_minutes;
  const windowText=recorded(modelWindow)?`${modelWindow}시점${finite(modelWindow)&&finite(sampleMinutes)?' ('+numberMinutes(modelWindow*sampleMinutes)+'분)':''}`:'기록 없음';
  function numberMinutes(value){return value.toLocaleString('ko-KR');}
  const epochTarget=trainConfig.epochs??runConfig.epochs;
  const actualEpoch=meta.model_weights_epoch??meta.checkpoint_weights_epoch??meta.checkpoint_epoch??meta.provenance?.model_weights_epoch;
  const lossNames={mse:'MSE',l1:'L1 (MAE)',huber:'Huber',mse_l1:'MSE + L1 혼합'};
  const thresholdNames={manual:'수동',percentile:'Validation 백분위수',sigma:'평균 + 표준편차',pot:'POT',spot:'SPOT'};
  const thresholdMethod=thresholdConfig.method;
  let thresholdText=thresholdNames[thresholdMethod]||thresholdMethod||'기록 없음';
  if(thresholdMethod==='manual'&&recorded(thresholdConfig.manual))thresholdText+=` · 기본 ${num(thresholdConfig.manual,7)}`;
  const thresholdOverrides=Object.keys(thresholdConfig.per_tag||{}).length;
  if(thresholdOverrides)thresholdText+=` · Tag별 지정 ${thresholdOverrides}개`;
  let deadlineText='기록 없음';
  if(runConfig.deadline_utc){const deadline=new Date(runConfig.deadline_utc);deadlineText=Number.isFinite(deadline.getTime())?new Intl.DateTimeFormat('ko-KR',{timeZone:'Asia/Seoul',dateStyle:'medium',timeStyle:'short'}).format(deadline)+' (한국시간)':runConfig.deadline_utc;}
  const configRows=[['모델 / 입력 창',`TranAD / ${windowText}`],['목표 학습 Epoch',recorded(epochTarget)?`${epochTarget}회 누적${recorded(trainConfig.max_epochs)?' · 모델당 상한 '+trainConfig.max_epochs+'회':''}`:'기록 없음']];
  if(recorded(actualEpoch))configRows.push(['선택 가중치 Epoch',`${actualEpoch}회 실제 가중치`]);
  configRows.push(['학습률 / 배치 크기',`${num(trainConfig.lr??runConfig.lr,7)} / ${recorded(trainConfig.batch_size??runConfig.batch_size)?(trainConfig.batch_size??runConfig.batch_size):'기록 없음'}`],['Dropout / 손실함수',`${finite(modelConfig.dropout)?numberMinutes(modelConfig.dropout*100)+'%':'기록 없음'} / ${lossNames[trainConfig.objective]||trainConfig.objective||'기록 없음'}`],['이상점수 임계값',thresholdText],['최적화 마감',deadlineText],['정규화 방식',meta.normalization],['값의 표시 단위',meta.value_unit||'전달된 실제값·복원값의 원 단위'],['Alarm 결정 규칙',meta.alarm_rule],['센서 기반 정지 규칙',meta.stop_rule],['실행 시각',meta.generated_at||meta.created_at],['선택 체크포인트',meta.selected_checkpoint||meta.checkpoint]);
  details('run-details',configRows);
  $('run-config-json').textContent=JSON.stringify(runConfig,null,2);
  $('run-config-detail').hidden=Object.keys(runConfig).length===0;
  $('evaluation-note').textContent = meta.evaluation_note || 'test 결과를 반복 확인하며 설정을 선택한 경우, 이 값은 해당 test에 대한 개발·최적화 성능입니다. 별도 미사용 데이터에 대한 일반화 성능을 입증하는 결과와 구분해 해석합니다.';
  $('tag-table').querySelector('tbody').innerHTML = S.tag_metrics.map((m,i) => {
    const complete = m.evaluated === S.n;
    const pass = m.evaluated > 0 && m.accuracy !== null && m.accuracy >= .95;
    return `<tr data-tag="${i}" tabindex="0"><td>${esc(m.tag)}</td><td class="${pass?'good':'bad'}"><strong>${pct(m.accuracy)}</strong></td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td>${pct(m.f1)}</td><td>${pct(m.positive_rate)}</td><td>${count(m.tp)}</td><td>${count(m.fp)}</td><td>${count(m.fn)}</td><td>${count(m.tn)}</td><td>${count(m.evaluated)} / ${count(S.n)} (${pct(m.evaluated/S.n)})</td><td><span class="pill ${pass?'':'fail'}">${pass?'수치 통과':'미달 / 미정'}</span>${complete?'':' <span class="pill missing">누락 '+count(m.omitted)+'</span>'}</td></tr>`;
  }).join('');
  $('tag-table').querySelectorAll('tbody tr').forEach(row => {
    const select = () => {showTab('trend'); if(chartsReady)selectTag(Number(row.dataset.tag));};
    row.addEventListener('click',select); row.addEventListener('keydown',event => {if(event.key==='Enter')select();});
  });
  const sm = S.stop_metrics;
  const stopKnown=S.stop_truth_known_minutes??sm.evaluated;
  const stopValidation=stopApplicable&&sm.evaluated>0;
  const stopUnavailable=!stopKnown?'검증 불가 · Inhibit 정답 없음':!stopApplicable?'검증 불가 · 센서 정지 규칙 미적용':'검증 불가 · 비교할 예측/정답 없음';
  const stopNotice=stopApplicable?'센서 조건 적용':`미적용 · ${stopStatus.missing_sensor_tags?.length?'조건 센서 없음: '+stopStatus.missing_sensor_tags.join(', '):'설정 비활성 또는 조건 없음'}. 운전 중이라고 확정한 것이 아니며 score 기반 Alarm 탐지는 계속합니다.`;
  const truthStopAlarm=!stopKnown?'검증 불가 · Inhibit 정답 없음':S.stop_truth_stop_minutes===0?'해당 없음 · 정답에 정지 시점 없음':count(S.metrics?.stop_safety?.truth_stop_alarm_count);
  details('stop-details', [['정지 규칙 상태',stopNotice],['제공된 Inhibit 정답',`${count(stopKnown)} / ${count(S.n)} 시점`],['정지 판정 일치율',stopValidation?pct(sm.accuracy):stopUnavailable],['정지 재현율 / 정밀도',stopValidation?`${pct(sm.recall)} / ${pct(sm.precision)}`:'비교하지 않음'],['TP / FP / FN / TN',stopValidation?`${sm.tp} / ${sm.fp} / ${sm.fn} / ${sm.tn}`:'비교하지 않음'],['비교한 시점 / 전체',`${count(sm.evaluated)} / ${count(S.n)}`],['정답 정지 중 Tag Alarm',truthStopAlarm]]);

  const experiments=Array.isArray(S.experiments)?S.experiments:[];
  const experimentSources=Array.isArray(S.experiment_sources)?S.experiment_sources:[];
  if(!experiments.length)$('experiment-content').innerHTML='<p class="empty">실제 trials.jsonl 실험 기록이 아직 없습니다. 아래 Epoch별 history를 실험 횟수로 대신 세지 않습니다.</p>';
  else {
    const headers=['실험 / 실행','목표 Epoch','가중치 Epoch','Window','전체 MAE','최저 Tag 일치율','상태','설정 변경','체크포인트'];
    $('experiment-content').innerHTML='<div class="table-scroll"><table><thead><tr>'+headers.map(h=>`<th>${esc(h)}</th>`).join('')+'</tr></thead><tbody>'+experiments.map(item=>{
      const change=Object.entries(item.changes||{}).map(([key,value])=>`${key}: ${text(value.before)} → ${text(value.after)}`).join(' / ')||'이전 설정과 차이 없음 / 기록 없음';
      const row=[item.trial_id||item.run_id,item.epochs,item.model_weights_epoch,item.window,num(item.mae,7),pct(item.minimum_tag_accuracy),item.status,change,item.checkpoint];
      return '<tr>'+row.map(v=>`<td>${esc(text(v))}</td>`).join('')+'</tr>';
    }).join('')+'</tbody></table></div><h3 style="margin-top:28px">실험별 시각·Tag 정확도·증거</h3>'+experiments.map((item,index)=>`<details><summary>${index+1}. ${esc(item.trial_id||item.run_id)} · ${esc(item.status)} · ${esc(item.timestamp)}</summary><pre>${esc(JSON.stringify(item,null,2))}</pre></details>`).join('');
  }
  if(experimentSources.length)$('experiment-content').innerHTML+=`<details><summary>읽은 실험 로그 ${count(experimentSources.length)}개${experimentSources.some(item=>item.incomplete_tail)?' · 작성 중인 마지막 행은 다음 집계에서 반영':''}</summary><pre>${esc(JSON.stringify(experimentSources,null,2))}</pre></details>`;
  const history=Array.isArray(S.history)?S.history:[];
  if(!history.length)$('history-content').innerHTML='<p class="empty">선택 체크포인트의 Epoch별 학습 기록이 전달되지 않았습니다.</p>';
  else {
    const valid=history.filter(item=>finite(item.epoch));
    const values=valid.flatMap(item=>[item.train_mae,item.val_mae]).filter(finite);
    let curve='';
    if(valid.length&&values.length){
      const w=960,h=250,left=76,right=28,top=18,bottom=42,minEpoch=Math.min(...valid.map(item=>item.epoch)),maxEpoch=Math.max(...valid.map(item=>item.epoch));
      const lo=Math.min(0,...values),hi=Math.max(...values),span=hi-lo||1;
      const px=epoch=>left+(epoch-minEpoch)/(maxEpoch-minEpoch||1)*(w-left-right),py=value=>top+(hi-value)/span*(h-top-bottom);
      let contents='';
      for(let i=0;i<=4;i++){const value=lo+span*i/4,y=py(value);contents+=`<line x1="${left}" x2="${w-right}" y1="${y}" y2="${y}" stroke="#e3edf3"/><text x="${left-10}" y="${y+4}" text-anchor="end" font-size="11" fill="#667b8b">${num(value,4)}</text>`;}
      for(const [key,color,label] of [['train_mae','#126ab2','학습 MAE'],['val_mae','#e89050','Validation MAE']]){
        let path='',pen=false;
        for(const item of valid){if(!finite(item[key])){pen=false;continue;}path+=`${pen?' L':' M'}${px(item.epoch)},${py(item[key])}`;pen=true;}
        contents+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="2"/>`;
        for(const item of valid)if(finite(item[key]))contents+=`<circle cx="${px(item.epoch)}" cy="${py(item[key])}" r="3" fill="${color}"><title>Epoch ${item.epoch} · ${label}: ${item[key]}</title></circle>`;
      }
      contents+=`<text x="${left}" y="${h-12}" font-size="12" fill="#667b8b">Epoch ${minEpoch}</text><text x="${w-right}" y="${h-12}" text-anchor="end" font-size="12" fill="#667b8b">Epoch ${maxEpoch}</text>`;
      curve=`<p class="legend"><span class="actual">학습 MAE</span><span class="recon">Validation MAE</span></p><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="선택 모델의 Epoch별 학습 및 Validation MAE" style="width:100%;max-height:300px">${contents}</svg>`;
    }
    $('history-content').innerHTML=curve+'<div class="table-scroll"><table><thead><tr><th>Epoch</th><th>학습 MAE</th><th>Validation MAE</th></tr></thead><tbody>'+history.map(item=>`<tr><td>${esc(item.epoch)}</td><td>${num(item.train_mae,7)}</td><td>${num(item.val_mae,7)}</td></tr>`).join('')+'</tbody></table></div>';
  }
  const audit=S.audit_summary||{},actualAudit=audit.actual||audit,injectedAudit=audit.injected_tests||{};
  const A=pick(actualAudit,['loops_started','loops_total','total_loops','loop_count','A']);
  const B=pick(actualAudit,['evidence_rejected_loops','dishonest_signals_detected','dishonesty_stops','integrity_stops','B']);
  const C1=pick(actualAudit,['gate_initial_passed','gates_initial_passed','initial_gates_passed','C1']);
  const C2=pick(actualAudit,['gate_final_passed','gates_final_passed','final_gates_passed','C2']);
  const D1=pick(actualAudit,['p0_discovered','p0_total','p0_found','D1']);
  const D2=pick(actualAudit,['p0_resolved','p0_fixed','D2']);
  const gateCount=(value,key,scope=actualAudit)=>finite(value)?count(value):(Object.hasOwn(scope,key)&&scope[key]===null?'미실행':'미집계');
  const c1=gateCount(C1,'gate_initial_passed'),c2=gateCount(C2,'gate_final_passed');
  $('audit-kpis').innerHTML=kpi('A · 실제 랄프 루프 시작',count(A),'학습 Epoch와 구별해 집계')+kpi('B · 증거 거부 루프',count(B),'실제 증거 불일치 탐지 횟수')+kpi('C1 → C2 · 게이트 통과',`${c1} → ${c2}`,'동일한 고정 검증 항목 기준')+kpi('D 해결 / 전체 · P0',`${count(D2)} / ${count(D1)}`,'재현 및 수정 검증 증거 필요');
  $('audit-narrative').innerHTML=`총 <strong>${count(A)} 회</strong> 실제 랄프 루프를 시작했습니다. 이 중 검증 증거 불일치 등 정의된 부정직 신호를 탐지해 증거를 거부한 횟수는 <strong>${count(B)} 회</strong>입니다. 게이트 통과 항목은 <strong>${c1} → ${c2}</strong>입니다. 크래시 없이 잘못된 값을 내보내는 P0 버그는 <strong>${count(D1)} 건 중 ${count(D2)} 건</strong>의 해결이 기록됐습니다.`+(B===0?' 실제 증거 거부는 0회이며 부정직 신호 탐지 사례를 만들어 채우지 않았습니다.':finite(B)&&B>0?' 이 증거 불일치를 잡지 못했다면 해당 결과를 그대로 신뢰하기 어려웠을 것입니다.':'')+([A,B,C1,C2,D1,D2].some(v=>!finite(v))?' 미실행·미집계를 숫자 0으로 바꾸지 않았습니다.':'');
  $('audit-snapshot').textContent=[audit.snapshot?.captured_at_utc,audit.snapshot?.note,`완료 ${count(actualAudit.loops_completed)} · 실패 ${count(actualAudit.loops_failed)} · 중단 ${count(actualAudit.loops_aborted)} · 진행 중 ${count(actualAudit.loops_open)}`].filter(Boolean).join(' / ');
  details('injected-details',[['주입 시험 루프',count(injectedAudit.loops_started)],['주입 증거 거부',count(injectedAudit.evidence_rejected_loops)],['주입 게이트 최초 → 최종',`${gateCount(injectedAudit.gate_initial_passed,'gate_initial_passed',injectedAudit)} → ${gateCount(injectedAudit.gate_final_passed,'gate_final_passed',injectedAudit)}`],['주입 P0 해결 / 발견',`${count(injectedAudit.p0_resolved)} / ${count(injectedAudit.p0_discovered)}`]]);
  const evidence=pick(audit,['evidence','events','findings','issues']);
  $('audit-evidence').innerHTML=(Array.isArray(evidence)&&evidence.length?evidence.map((item,i)=>`<div class="evidence-row"><h4>${item.injected?'주입 시험':'실제 실행'} · ${esc(item.title||item.id||item.event||`증거 ${i+1}`)}</h4><pre>${esc(JSON.stringify(item,null,2))}</pre></div>`).join(''):'<p class="empty">상세 증거 목록이 전달되지 않았습니다. 예시 버그를 실제 해결 사례로 기재하지 않습니다.</p>')+`<details><summary>집계 원본 확인</summary><pre>${esc(JSON.stringify(audit,null,2))}</pre></details>`;

  if(!C || !Array.isArray(C.timestamps) || C.timestamps.length!==S.n){
    $('chart-message').textContent='공통 시계열 파일을 읽지 못했습니다. dashboard.html과 dashboard_data 폴더를 함께 보관해 주세요.';
    return;
  }
  const n=S.n, timestamps=C.timestamps;
  const parsed=timestamps.map(t=>typeof t==='number'?(Math.abs(t)<1e11?t*1000:t):Date.parse(t));
  const temporal=parsed.every((v,i)=>Number.isFinite(v)&&(i===0||v>parsed[i-1]));
  const xvalues=temporal?parsed:timestamps.map((_,i)=>i);
  const state={tag:0,data:null,a:0,b:n-1,pan:false,geometry:null,drag:null,hover:null};
  const canvas=$('trend-canvas'),ctx=canvas.getContext('2d');
  const base=document.createElement('canvas'),bctx=base.getContext('2d');
  const cache=new Map(),pending=new Map();
  let selectionSerial=0,renderPending=false;
  window.EWS_TAG_READY=(index,data)=>{
    cache.set(index,data);
    while(cache.size>3){const victim=[...cache.keys()].find(key=>key!==state.tag&&key!==index);if(victim===undefined)break;cache.delete(victim);}
    if(pending.has(index)){pending.get(index).resolve(data);pending.delete(index);}
  };
  function loadTag(index){
    if(cache.has(index)){const data=cache.get(index);cache.delete(index);cache.set(index,data);return Promise.resolve(data);}
    if(pending.has(index))return pending.get(index).promise;
    let resolve,reject;const promise=new Promise((res,rej)=>{resolve=res;reject=rej;});pending.set(index,{promise,resolve,reject});
    const script=document.createElement('script');script.src=`dashboard_data/tag_${String(index).padStart(4,'0')}.js`;
    script.onerror=()=>{pending.delete(index);script.remove();reject(new Error('태그 데이터 파일을 읽지 못했습니다. dashboard_data 폴더와 HTML을 함께 보관해 주세요.'));};
    script.onload=()=>{script.remove();if(pending.has(index)){pending.delete(index);reject(new Error('태그 파일에 데이터가 없습니다.'));}};
    document.head.appendChild(script);return promise;
  }
  $('tag-selector').innerHTML=S.tags.map((tag,i)=>`<option value="${i}">${esc(tag)}</option>`).join('');
  async function selectTag(index){
    const serial=++selectionSerial;state.tag=index;state.data=null;$('tag-selector').value=String(index);$('chart-message').textContent='원본 시계열을 읽고 있습니다…';
    try {const data=await loadTag(index);if(serial!==selectionSerial)return;state.data=data;$('chart-message').textContent='';
      const m=S.tag_metrics[index];const descriptions=meta.tag_descriptions||{};
      $('tag-detail').innerHTML=`<h3>${esc(data.tag)} · 결과 해석</h3><p>${esc(descriptions[data.tag]||'실제값과 복원값의 차이, 이상점수의 임계값 초과, 최종 Alarm과 정답 구간을 같은 시점에서 확인하세요. 확대해 순간 Peak와 연속 변화가 원본 1분 데이터에서 어떻게 나타나는지 검증할 수 있습니다.')}</p><p>일치율 <strong>${pct(m.accuracy)}</strong> · 재현율 <strong>${pct(m.recall)}</strong> · 정밀도 <strong>${pct(m.precision)}</strong> · 임계값 <strong>${num(data.threshold,7)}</strong> · Alarm 시점 <strong>${count(m.alarm_count)}</strong></p><p class="note">붉은 영역은 최종 Alarm, 청록 띠는 Label 정답, 회색 영역은 센서 기반 정지 판정입니다. 짙은 회색 하단 띠는 Inhibit 정답이며 비교를 위해 표시합니다. 정답을 표시하거나 숨겨도 Alarm과 점수는 바뀌지 않습니다.</p>`;render();
    }catch(error){if(serial===selectionSerial)$('chart-message').textContent=error.message;}
  }
  $('tag-selector').addEventListener('change',event=>selectTag(Number(event.target.value)));
  const leftBound=value=>{let l=0,r=n;while(l<r){const m=(l+r)>>1;if(xvalues[m]<value)l=m+1;else r=m;}return Math.min(n-1,l);};
  const rightBound=value=>{let l=0,r=n;while(l<r){const m=(l+r)>>1;if(xvalues[m]<=value)l=m+1;else r=m;}return Math.max(0,l-1);};
  function dateInput(index){if(!temporal)return'';const d=new Date(parsed[index]);const pad=v=>String(v).padStart(2,'0');return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;}
  function setRange(a,b){a=Math.round(a);b=Math.round(b);const span=Math.max(0,b-a);if(a<0){a=0;b=span;}if(b>=n){b=n-1;a=Math.max(0,b-span);}state.a=Math.max(0,Math.min(n-1,a));state.b=Math.max(state.a,Math.min(n-1,b));state.hover=null;$('chart-tooltip').hidden=true;render();}
  function zoom(factor,center=(state.a+state.b)/2){const oldSpan=Math.max(1,state.b-state.a),span=Math.max(Math.min(2,n-1),Math.min(n-1,Math.round(oldSpan*factor))),fraction=(center-state.a)/oldSpan;const a=center-span*fraction;setRange(a,a+span);}
  $('zoom-in').onclick=()=>zoom(.5);$('zoom-out').onclick=()=>zoom(2);$('reset-range').onclick=()=>setRange(0,n-1);
  $('pan-left').onclick=()=>{const shift=Math.max(1,Math.round((state.b-state.a)*.7));setRange(state.a-shift,state.b-shift);};
  $('pan-right').onclick=()=>{const shift=Math.max(1,Math.round((state.b-state.a)*.7));setRange(state.a+shift,state.b+shift);};
  $('drag-mode').onclick=()=>{state.pan=!state.pan;$('drag-mode').setAttribute('aria-pressed',String(state.pan));$('drag-mode').textContent=state.pan?'드래그: 좌우 이동':'드래그: 구간 확대';canvas.style.cursor=state.pan?'grab':'crosshair';};
  $('show-labels').onchange=render;$('show-stop').onchange=render;
  if(!stopApplicable){$('show-stop').checked=false;$('show-stop').disabled=true;$('show-stop').title='정지 조건 센서 없음 또는 설정 비활성으로 미적용';}
  $('apply-date').onclick=()=>{if(!temporal)return;const a=Date.parse($('date-start').value),b=Date.parse($('date-end').value);if(!Number.isFinite(a)||!Number.isFinite(b)||a>b){$('chart-message').textContent='시작·끝 시간을 올바르게 입력해 주세요.';return;}const first=leftBound(a),last=rightBound(b);if(b<xvalues[0]||a>xvalues[n-1]||first>last||xvalues[first]>b||xvalues[last]<a){$('chart-message').textContent='입력한 기간에 원본 데이터가 없습니다.';return;}$('chart-message').textContent='';setRange(first,last);};
  ['range-start','range-end'].forEach(id=>{$(id).max=String(n-1);$(id).addEventListener('input',()=>{if(id==='range-start')setRange(Math.min(Number($(id).value),state.b),state.b);else setRange(state.a,Math.max(Number($(id).value),state.a));});});
  if(!temporal){$('date-start').disabled=true;$('date-end').disabled=true;$('apply-date').disabled=true;}

  // 표시 영역을 다시 샘플링하며 각 bucket의 최소·최대 및 결측 경계를 보존한다.
  // 반환값은 원본 인덱스다. 원본 배열은 영구적으로 축소하거나 수정하지 않는다.
  function envelope(values,a,b,buckets){
    const length=b-a+1;
    if(length<=buckets*2)return Array.from({length},(_,i)=>a+i);
    const selected=new Set([a,b]),step=length/buckets;
    for(let bucket=0;bucket<buckets;bucket++){
      const lo=a+Math.floor(bucket*step),hi=Math.min(b,a+Math.floor((bucket+1)*step)-1);
      let min=-1,max=-1,lastFinite=null;
      for(let i=lo;i<=hi;i++){
        const valid=finite(values[i]);
        if(valid){if(min<0||values[i]<values[min])min=i;if(max<0||values[i]>values[max])max=i;}
        if(lastFinite!==null&&valid!==lastFinite){selected.add(i-1);selected.add(i);}
        lastFinite=valid;
      }
      // A gap at a bucket boundary must also break a line.
      if(lo>a&&finite(values[lo])!==finite(values[lo-1])){selected.add(lo-1);selected.add(lo);}
      if(min>=0){selected.add(min);selected.add(max);}else selected.add(lo);
    }
    return [...selected].sort((a,b)=>a-b);
  }
  function valueRange(series,a,b,include=null){let lo=Infinity,hi=-Infinity;for(const values of series)for(let i=a;i<=b;i++)if(finite(values[i])){lo=Math.min(lo,values[i]);hi=Math.max(hi,values[i]);}if(finite(include)){lo=Math.min(lo,include);hi=Math.max(hi,include);}if(!Number.isFinite(lo))return [0,1];const pad=(hi-lo||Math.abs(hi)*.1||1)*.09;return[lo-pad,hi+pad];}
  function geometry(width,height){const l=72,r=22,top=36,gap=84,bottom=42,available=height-top-bottom-gap;const h1=available*.6,h2=available*.4;return{l,r,w:Math.max(1,width-l-r),p1:{top,h:h1},p2:{top:top+h1+gap,h:h2},width,height};}
  function draw(){
    if(!state.data||!$('trend').classList.contains('active'))return;
    const rect=canvas.getBoundingClientRect();if(rect.width<2)return;
    const dpr=window.devicePixelRatio||1,width=rect.width,height=rect.height;
    canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);base.width=canvas.width;base.height=canvas.height;
    bctx.setTransform(dpr,0,0,dpr,0,0);bctx.clearRect(0,0,width,height);bctx.fillStyle='#ffffff';bctx.fillRect(0,0,width,height);bctx.font='11px "Malgun Gothic",Arial,sans-serif';
    const g=geometry(width,height),{a,b,data}=state;const xlo=xvalues[a],xhi=xvalues[b],span=xhi-xlo||1;
    g.x=i=>g.l+(xvalues[i]-xlo)/span*g.w;g.toIndex=x=>{const value=xlo+Math.max(0,Math.min(1,(x-g.l)/g.w))*span;const right=leftBound(value),left=Math.max(a,right-1);return Math.min(b,Math.abs(xvalues[left]-value)<Math.abs(xvalues[right]-value)?left:right);};
    const vr=valueRange([data.actual,data.reconstructed],a,b),sr=valueRange([data.scores],a,b,data.threshold);
    g.y1=v=>g.p1.top+g.p1.h-(v-vr[0])/(vr[1]-vr[0])*g.p1.h;g.y2=v=>g.p2.top+g.p2.h-(v-sr[0])/(sr[1]-sr[0])*g.p2.h;state.geometry=g;
    function axis(panel,range,title){bctx.fillStyle='#304d60';bctx.textAlign='left';bctx.fillText(title,g.l,panel.top-14);for(let tick=0;tick<=4;tick++){const y=panel.top+panel.h*tick/4;bctx.strokeStyle='#e7eef2';bctx.lineWidth=1;bctx.beginPath();bctx.moveTo(g.l,y);bctx.lineTo(g.l+g.w,y);bctx.stroke();bctx.fillStyle='#7c909e';bctx.textAlign='right';bctx.fillText(num(range[1]-(range[1]-range[0])*tick/4,4),g.l-10,y+4);}}
    axis(g.p1,vr,`실제값 / 복원값${meta.value_unit?' · '+meta.value_unit:''}`);axis(g.p2,sr,'이상점수 / 임계값');
    function shade(values,color,panel,strip=false){bctx.fillStyle=color;let start=-1;for(let i=a;i<=b+1;i++){const active=i<=b&&values[i]===1;if(active&&start<0)start=i;if(!active&&start>=0){const from=start===a?g.l:(g.x(start-1)+g.x(start))/2;const end=i>b?g.l+g.w:(g.x(i-1)+g.x(i))/2;bctx.fillRect(from,strip?panel.top+panel.h+6:panel.top,Math.max(1,end-from),strip?5:panel.h);start=-1;}}}
    for(const panel of [g.p1,g.p2]){
      if($('show-stop').checked)shade(C.predicted_inhibit,'rgba(120,137,149,.17)',panel);
      shade(data.final_alarm,'rgba(220,71,83,.13)',panel);
    }
    if($('show-labels').checked){shade(data.labels,'#008f8a',g.p1,true);shade(C.inhibit_labels,'#6c7885',{top:g.p1.top+8,h:g.p1.h},true);}
    const bucketCount=Math.max(32,Math.floor(g.w*.65));
    const ai=envelope(data.actual,a,b,bucketCount),ri=envelope(data.reconstructed,a,b,bucketCount),si=envelope(data.scores,a,b,bucketCount);
    function line(values,indices,y,color,panel){bctx.save();bctx.beginPath();bctx.rect(g.l,panel.top,g.w,panel.h);bctx.clip();bctx.strokeStyle=color;bctx.lineWidth=1.35;bctx.beginPath();let pen=false;for(const i of indices){if(!finite(values[i])){pen=false;continue;}const x=g.x(i),yy=y(values[i]);if(!pen)bctx.moveTo(x,yy);else bctx.lineTo(x,yy);pen=true;}bctx.stroke();if(b-a<=120){bctx.fillStyle=color;for(const i of indices)if(finite(values[i])){bctx.beginPath();bctx.arc(g.x(i),y(values[i]),b-a<=30?2.6:1.7,0,2*Math.PI);bctx.fill();}}bctx.restore();}
    line(data.actual,ai,g.y1,'#126ab2',g.p1);line(data.reconstructed,ri,g.y1,'#e89050',g.p1);line(data.scores,si,g.y2,'#7656af',g.p2);
    // 최종 Alarm의 원본 실제값 위치를 붉은 점으로 표시한다. Label을 점 생성에 사용하지 않는다.
    let alarmPointCount=0;
    bctx.save();bctx.beginPath();bctx.rect(g.l,g.p1.top,g.w,g.p1.h);bctx.clip();bctx.fillStyle='#d43f4c';bctx.beginPath();
    const alarmRadius=b-a<=120?2.8:1.5;
    for(let i=a;i<=b;i++)if(data.final_alarm[i]===1&&finite(data.actual[i])){const x=g.x(i),y=g.y1(data.actual[i]);bctx.moveTo(x+alarmRadius,y);bctx.arc(x,y,alarmRadius,0,2*Math.PI);alarmPointCount++;if(alarmPointCount%4096===0){bctx.fill();bctx.beginPath();}}
    bctx.fill();bctx.restore();
    if(finite(data.threshold)){bctx.strokeStyle='#b97614';bctx.lineWidth=1.2;bctx.setLineDash([6,4]);bctx.beginPath();bctx.moveTo(g.l,g.y2(data.threshold));bctx.lineTo(g.l+g.w,g.y2(data.threshold));bctx.stroke();bctx.setLineDash([]);}
    const ticks=Math.max(2,Math.min(7,Math.floor(g.w/130)));bctx.textAlign='center';bctx.fillStyle='#7c909e';
    for(let tick=0;tick<ticks;tick++){const fraction=tick/(ticks-1),i=g.toIndex(g.l+g.w*fraction),x=g.x(i);let label=String(timestamps[i]);if(temporal){const d=new Date(parsed[i]);label=`${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;}if(label.length>23)label=label.slice(0,23);bctx.fillText(label,Math.max(g.l+20,Math.min(g.l+g.w-25,x)),g.p2.top+g.p2.h+25);bctx.strokeStyle='#e7eef2';for(const p of [g.p1,g.p2]){bctx.beginPath();bctx.moveTo(x,p.top);bctx.lineTo(x,p.top+p.h);bctx.stroke();}}
    $('chart-meta').textContent=`${String(timestamps[a])} ~ ${String(timestamps[b])} · 현재 구간 원본 ${count(b-a+1)} / 전체 ${count(n)} 시점 · 표시 실제 ${count(ai.length)}, 복원 ${count(ri.length)}, 점수 ${count(si.length)} 포인트 · Alarm 점 ${count(alarmPointCount)}개${b-a+1<=bucketCount*2?' · 원본 전체 표시':' · 화면용 최소·최대 요약'}${temporal?'':' · 날짜 해석 불가/중복·역순으로 원본 행 순서 표시'}`;
    $('range-start').value=String(a);$('range-end').value=String(b);$('date-start').value=dateInput(a);$('date-end').value=dateInput(b);paintOverlay();
  }
  function render(){if(renderPending)return;renderPending=true;requestAnimationFrame(()=>{renderPending=false;draw();});}
  function paintOverlay(){ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(base,0,0);const g=state.geometry;if(!g)return;const dpr=window.devicePixelRatio||1;ctx.setTransform(dpr,0,0,dpr,0,0);if(state.hover!=null){const x=g.x(state.hover);ctx.strokeStyle='#5b778c';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x,g.p1.top);ctx.lineTo(x,g.p2.top+g.p2.h);ctx.stroke();ctx.setLineDash([]);}if(state.drag&&!state.drag.pan){const lo=Math.min(state.drag.start,state.drag.last),hi=Math.max(state.drag.start,state.drag.last);ctx.fillStyle='rgba(18,106,178,.13)';ctx.fillRect(lo,g.p1.top,hi-lo,g.p2.top+g.p2.h-g.p1.top);ctx.strokeStyle='#126ab2';ctx.strokeRect(lo,g.p1.top,hi-lo,g.p2.top+g.p2.h-g.p1.top);}}
  function position(event){const r=canvas.getBoundingClientRect();return{x:event.clientX-r.left,y:event.clientY-r.top};}
  canvas.addEventListener('wheel',event=>{if(!state.geometry)return;event.preventDefault();zoom(event.deltaY>0?1.4:.72,state.geometry.toIndex(position(event).x));},{passive:false});
  canvas.addEventListener('dblclick',()=>setRange(0,n-1));
  canvas.addEventListener('pointerdown',event=>{if(!state.geometry)return;const p=position(event),g=state.geometry;if(p.x<g.l||p.x>g.l+g.w)return;canvas.setPointerCapture(event.pointerId);state.drag={start:p.x,last:p.x,a:state.a,b:state.b,pan:state.pan||event.shiftKey};$('chart-tooltip').hidden=true;});
  canvas.addEventListener('pointermove',event=>{
    const g=state.geometry;if(!g||!state.data)return;const p=position(event);
    if(state.drag){state.drag.last=Math.max(g.l,Math.min(g.l+g.w,p.x));if(state.drag.pan){const shift=Math.round((state.drag.start-p.x)/g.w*(state.drag.b-state.drag.a));setRange(state.drag.a+shift,state.drag.b+shift);}else paintOverlay();return;}
    if(p.x<g.l||p.x>g.l+g.w||p.y<g.p1.top||p.y>g.p2.top+g.p2.h){state.hover=null;$('chart-tooltip').hidden=true;paintOverlay();return;}
    const i=g.toIndex(p.x),d=state.data;state.hover=i;const yes=v=>v===1?'1 · 해당':v===0?'0 · 해당 없음':'미정';
    $('chart-tooltip').innerHTML=`<strong>${esc(String(timestamps[i]))}</strong><br>원본 행: ${count(i+1)}<br>실제값: ${num(d.actual[i],8)}<br>복원값: ${num(d.reconstructed[i],8)}<br>이상점수: ${num(d.scores[i],8)}<br>임계값: ${num(d.threshold,8)}<br>최종 Alarm: ${yes(d.final_alarm[i])}<br>Label 정답: ${yes(d.labels[i])}<br>센서 정지: ${stopApplicable?yes(C.predicted_inhibit[i]):'미적용 (조건 센서 없음/비활성)'}<br>Inhibit 정답: ${yes(C.inhibit_labels[i])}`;
    const tip=$('chart-tooltip');tip.hidden=false;tip.style.left=Math.max(0,Math.min(g.width-tip.offsetWidth-6,p.x+15))+'px';tip.style.top=Math.max(0,Math.min(g.height-tip.offsetHeight-5,p.y+12))+'px';paintOverlay();
  });
  function finishDrag(event){if(!state.drag)return;const drag=state.drag;state.drag=null;if(!drag.pan&&Math.abs(drag.last-drag.start)>5){const g=state.geometry;setRange(g.toIndex(Math.min(drag.start,drag.last)),g.toIndex(Math.max(drag.start,drag.last)));}else paintOverlay();if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);}
  canvas.addEventListener('pointerup',finishDrag);canvas.addEventListener('pointercancel',event=>{state.drag=null;paintOverlay();if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);});
  canvas.addEventListener('pointerleave',()=>{if(!state.drag){state.hover=null;$('chart-tooltip').hidden=true;paintOverlay();}});
  $('download-visible').onclick=()=>{
    if(!state.data)return;const quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"',d=state.data;
    const lines=[['timestamp','tag','actual','reconstructed','score','threshold','final_alarm','label','predicted_inhibit','inhibit_label','stop_rule_applicable'].join(',')];
    for(let i=state.a;i<=state.b;i++)lines.push([timestamps[i],d.tag,d.actual[i],d.reconstructed[i],d.scores[i],d.threshold,d.final_alarm[i],d.labels[i],C.predicted_inhibit[i],C.inhibit_labels[i],stopApplicable].map(quote).join(','));
    const blob=new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`EWS_${d.tag.replace(/[\\/:*?"<>|]/g,'_')}_${state.a+1}-${state.b+1}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  new ResizeObserver(render).observe($('chart-wrap'));
  chartsReady=true;
  selectTag(0);
})();
