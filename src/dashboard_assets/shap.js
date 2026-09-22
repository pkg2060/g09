/* 실제 SHAP 서버 결과만 표시한다. 정적 HTML에서 계산값을 생성하지 않는다. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const S=window.EWS_SUMMARY||{},C=window.EWS_COMMON||{};
  const form=$('shap-form');
  if(!form)return;
  const escape=value=>String(value??'미제공').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const finite=value=>typeof value==='number'&&Number.isFinite(value);
  const number=value=>!finite(value)?'미제공':Math.abs(value)>0&&Math.abs(value)<1e-6?value.toExponential(3):Number(value.toPrecision(7)).toLocaleString('ko-KR',{maximumFractionDigits:10});
  const instruction='Python 분석 서버에서 열어주세요. Jupyter에서 %run mode -serve를 실행하고 안내된 대시보드 주소를 여세요.';
  const controls=[...form.querySelectorAll('input,select,button')];
  let busy=false;
  function localInput(timestamp){const d=new Date(timestamp);if(!Number.isFinite(d.getTime()))return'';const p=v=>String(v).padStart(2,'0');return`${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;}
  if(C.timestamps?.length)$('shap-start').value=localInput(C.timestamps[0]);
  $('shap-target').innerHTML='<option value="">전체 Tag 평균 이상점수</option>'+(S.tags||[]).map(tag=>`<option value="${escape(tag)}">${escape(tag)} 점수</option>`).join('');
  $('shap-use-trend').addEventListener('click',()=>{
    const value=$('date-start')?.value;
    if(value){$('shap-start').value=value;$('shap-status').textContent='추세 그래프의 시작 시각을 가져왔습니다. 분석 기간과 대상을 확인하세요.';}
    else $('shap-status').textContent='추세 그래프에서 먼저 기간을 선택하거나 시작 시각을 직접 입력하세요.';
  });
  if(!/^https?:$/.test(location.protocol))$('shap-status').textContent=instruction;
  function loading(value){busy=value;controls.forEach(el=>el.disabled=value);$('shap-run').textContent=value?'SHAP 계산 중…':'SHAP 분석 실행';form.setAttribute('aria-busy',String(value));}
  function error(message){$('shap-error').hidden=false;$('shap-error').textContent=message;$('shap-status').textContent='SHAP 분석을 완료하지 못했습니다. 입력 또는 분석 서버 상태를 확인하세요.';}
  function average(rows,key){const values=rows.map(row=>row[key]).filter(finite);return values.length?values.reduce((a,b)=>a+b,0)/values.length:null;}
  function validateResult(result){
    if(!result||!Array.isArray(result.contributions)||!result.contributions.length)throw new Error('분석 서버가 변수별 SHAP 기여도를 반환하지 않았습니다.');
    if(result.contributions.some(row=>!row||typeof row.tag!=='string'||!finite(row.mean_signed)||!finite(row.mean_abs)||row.mean_abs<0))throw new Error('분석 서버의 SHAP 기여도 형식이 올바르지 않습니다. 결과를 표시하지 않습니다.');
    if(new Set(result.contributions.map(row=>row.tag)).size!==result.contributions.length)throw new Error('분석 서버 응답에 중복 Tag가 있습니다. 결과를 표시하지 않습니다.');
    return result;
  }
  function barChart(rows){
    const ordered=[...rows].sort((a,b)=>b.mean_abs-a.mean_abs);
    const w=960,left=240,right=188,top=36,rowHeight=42,bottom=43,h=top+ordered.length*rowHeight+bottom;
    const chartWidth=w-left-right,center=left+chartWidth/2;
    const maximum=Math.max(...ordered.map(row=>Math.abs(row.mean_signed)),1e-12);
    const scale=chartWidth*.46/maximum;
    let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Tag별 평균 부호 SHAP 기여도. 양수는 점수 증가, 음수는 점수 감소."><text x="${left}" y="18" fill="#126ab2" font-size="12">점수 감소 (−)</text><text x="${left+chartWidth}" y="18" fill="#c84649" font-size="12" text-anchor="end">점수 증가 (+)</text><text x="${w-8}" y="18" fill="#667b8b" font-size="11" text-anchor="end">평균 기여도 / 평균 절대 영향</text>`;
    svg+=`<line x1="${center}" x2="${center}" y1="${top-9}" y2="${h-bottom+1}" stroke="#9cafba"/>`;
    ordered.forEach((row,index)=>{
      const y=top+index*rowHeight,width=Math.abs(row.mean_signed)*scale,x=row.mean_signed>=0?center:center-width,color=row.mean_signed>0?'#c84649':row.mean_signed<0?'#126ab2':'#a4b4bf';
      const label=row.tag.length>28?row.tag.slice(0,26)+'…':row.tag;
      svg+=`<line x1="8" x2="${w-8}" y1="${y+rowHeight-8}" y2="${y+rowHeight-8}" stroke="#eef3f6"/><text x="${left-14}" y="${y+15}" text-anchor="end" fill="#304d60" font-size="12"><title>${escape(row.tag)}</title>${escape(label)}</text><rect x="${x}" y="${y}" width="${Math.max(.7,width)}" height="23" rx="2" fill="${color}"><title>${escape(row.tag)} · 평균 기여도 ${number(row.mean_signed)} · 평균 절대 영향 ${number(row.mean_abs)}</title></rect><text x="${w-8}" y="${y+15}" text-anchor="end" fill="${color}" font-size="11">${number(row.mean_signed)} / ${number(row.mean_abs)}</text>`;
    });
    svg+=`<text x="${center}" y="${h-14}" text-anchor="middle" fill="#667b8b" font-size="11">0 · 정상 학습 배경값 대비 이상점수 변화</text></svg>`;
    $('shap-bars').innerHTML=svg;
  }
  function showResult(result,request){
    validateResult(result);
    const rows=Array.isArray(result.per_minute)?result.per_minute:[];
    const prediction=finite(result.prediction)?result.prediction:average(rows,'model_output');
    const baseline=finite(result.base_value)?result.base_value:average(rows,'baseline');
    const total=finite(result.sum_contributions)?result.sum_contributions:result.contributions.reduce((sum,row)=>sum+row.mean_signed,0);
    const residual=finite(result.additivity_residual)?result.additivity_residual:finite(prediction)&&finite(baseline)?prediction-baseline-total:null;
    const maxResidual=finite(result.max_abs_additivity_residual)?result.max_abs_additivity_residual:rows.some(row=>finite(row.additivity_residual))?Math.max(...rows.map(row=>finite(row.additivity_residual)?Math.abs(row.additivity_residual):0)):null;
    const method=result.exact===true?'모든 특징 조합 계산 · 정확':result.exact===false?'고정 시드 순열 계산 · 근사':'계산 방법 확인 필요';
    const cache=result.cache_hit===true?'저장된 동일 요청 분석 재사용':result.cache_hit===false?'이번 요청에서 새로 계산':'미제공';
    const background=typeof result.background==='string'?result.background:JSON.stringify(result.background??'미제공');
    const details=[['분석 대상',request.target_tag||'전체 Tag 평균 이상점수'],['분석한 시점 수',finite(result.n_samples)?result.n_samples:rows.length],['모델 이상점수 평균',number(prediction)],['배경값 점수 평균',number(baseline)],['SHAP 기여도 합계',number(total)],['평균 가산 일치 오차',number(residual)],['분별 최대 절대 가산 오차',number(maxResidual)],['계산 방법',method+(result.method?' ('+result.method+')':'')],['계산 소요시간',finite(result.elapsed_seconds)?number(result.elapsed_seconds)+' 초':'미제공'],['캐시',cache],['학습 배경값',background]];
    $('shap-details').innerHTML=details.map(([key,value])=>`<dt>${escape(key)}</dt><dd>${escape(value)}</dd>`).join('');
    $('shap-result-range').textContent=`${request.start_time.replace('T',' ')}부터 ${request.duration_minutes}분 · ${request.target_tag||'전체 Tag 평균 점수'} · 평균 절대 영향 순서`;
    barChart(result.contributions);
    $('shap-minute-details').innerHTML=rows.length?'<table><thead><tr><th>시각</th><th>배경값 점수</th><th>모델 점수</th><th>가산 일치 오차</th></tr></thead><tbody>'+rows.map(row=>`<tr><td>${escape(row.timestamp)}</td><td>${number(row.baseline)}</td><td>${number(row.model_output)}</td><td>${number(row.additivity_residual)}</td></tr>`).join('')+'</tbody></table>':'<p class="note">분 단위 상세 결과가 반환되지 않았습니다.</p>';
    $('shap-result').hidden=false;
    $('shap-status').textContent=`SHAP 분석 완료 · ${result.contributions.length}개 Tag · ${cache}. 가산 일치 오차는 모델 점수 − 배경값 점수 − 기여도 합계입니다.`;
  }
  form.addEventListener('submit',async event=>{
    event.preventDefault();if(busy)return;
    $('shap-error').hidden=true;$('shap-result').hidden=true;
    const start=$('shap-start').value,duration=Number($('shap-duration').value),target=$('shap-target').value;
    if(!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(start)||!Number.isFinite(Date.parse(start))){error('유효한 분석 시작 시각을 입력하세요.');return;}
    if(!Number.isInteger(duration)||duration<1||duration>1440){error('분석 기간은 1~1,440 사이의 정수 분으로 입력하세요.');return;}
    if(target&&!(S.tags||[]).includes(target)){error('대시보드에 있는 Tag 또는 전체 점수를 선택하세요.');return;}
    if(!/^https?:$/.test(location.protocol)){error(instruction);return;}
    const request={start_time:start.length===16?start+':00':start,duration_minutes:duration,target_tag:target||null};
    loading(true);$('shap-result').hidden=true;$('shap-status').textContent='선택 구간을 실제 모델로 계산하고 있습니다. 분석 구간이 길면 시간이 더 걸릴 수 있습니다.';
    const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),300000);
    try{
      const response=await fetch(new URL('api/shap',location.href),{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json','X-EWS-Request':'1'},body:JSON.stringify(request),signal:controller.signal});
      let result;
      try{result=await response.json();}catch(_){throw new Error('Python 분석 API의 JSON 응답을 받지 못했습니다. '+instruction);}
      if(!response.ok){const message=result?.error||`분석 요청 오류 (HTTP ${response.status})`;throw new Error(message+([404,405,503].includes(response.status)?' '+instruction:''));}
      if(result?.error)throw new Error(String(result.error));
      showResult(result,request);
    }catch(cause){error(cause.name==='AbortError'?'300초 안에 응답을 받지 못했습니다. 분석 기간을 줄여 다시 실행하거나 서버 상태를 확인하세요. 서버 계산이 종료됐는지는 별도로 확인해야 합니다.':cause instanceof TypeError?'분석 서버에 연결하지 못했습니다. '+instruction:cause.message||String(cause));}
    finally{clearTimeout(timeout);loading(false);}
  });
})();
