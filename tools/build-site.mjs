/** Build the static site solely from the pinned, original Git LFS HTML. Node >=22. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {gzipSync,gunzipSync} from 'node:zlib';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.join(root,'site');
const expectedSHA='1963310d723b1fe01740ab36006ff81df387c43671050a10745b3dc53bfd0248';
const expectedBytes=240070294;
const download='https://media.githubusercontent.com/media/pkg2060/g09/main/reports/EWS_final_offline.html';
const chunkBytes=8*1024*1024;
const sha=buffer=>createHash('sha256').update(buffer).digest('hex');
function require(value,message){if(!value)throw Error(message);}
async function input(){
  let bytes;
  try{bytes=await fs.readFile(path.join(root,'reports','EWS_final_offline.html'));}catch(error){if(error.code!=='ENOENT')throw error;}
  if(!bytes||bytes.length<1024&&bytes.toString('utf8').startsWith('version https://git-lfs.github.com/spec/v1')){
    console.log('Fetching the SHA-pinned final HTML from its public Git LFS URL.');
    const response=await fetch(download,{signal:AbortSignal.timeout(180000)});
    require(response.ok,'Pinned Git LFS report could not be downloaded.');
    const pieces=[];let total=0;
    for await(const piece of response.body){total+=piece.length;require(total<=expectedBytes,'Downloaded report exceeds pinned size.');pieces.push(piece);}
    bytes=Buffer.concat(pieces,total);
  }
  require(bytes.length===expectedBytes&&sha(bytes)===expectedSHA,'Original final HTML size/SHA-256 mismatch.');
  return bytes;
}
const original=await input(),source=original.toString('utf8');
const scripts=[...source.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)];
const block=id=>{
  const found=scripts.filter(match=>new RegExp(`\\bid=["']${id}["']`).test(match[1]));
  require(found.length===1,`Expected one ${id} block.`);return found[0][2];
};
let summary=Buffer.from(block('offline-summary-shell'),'base64').toString('utf8');
let dashboard=Buffer.from(block('offline-dashboard-shell'),'base64').toString('utf8');
const dashboardScripts=[...dashboard.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)];
const mainScript=dashboardScripts.find(match=>match[2].includes('  function loadTag(index){'));
const shapScript=dashboardScripts.find(match=>match[2].includes('const saved=parent.EWSOffline.shap();'));
const commonScript=dashboardScripts.find(match=>match[2]==='window.EWS_COMMON=parent.EWSOffline.common();');
const summaryScript=dashboardScripts.find(match=>match[2].startsWith('window.EWS_SUMMARY='));
require(mainScript&&shapScript&&commonScript&&summaryScript,'Unexpected dashboard shell.');
const official=JSON.parse(summaryScript[2].slice('window.EWS_SUMMARY='.length).replace(/;\s*$/,''));
require(official.n===476641&&official.tags.length===7,'Unexpected selected model data dimensions.');
const manifest={schema:'ews-static-site-v1',source:{bytes:expectedBytes,sha256:expectedSHA},rows:official.n,tag_count:official.tags.length,full_numeric_arrays_preserved:true,payloads:{},files:{}};
await fs.mkdir(output,{recursive:true});
async function write(relative,value){
  const buffer=Buffer.isBuffer(value)?value:Buffer.from(value,'utf8');
  require(buffer.length<20*1024*1024,`Static asset exceeds 20 MiB: ${relative}`);
  const destination=path.join(output,relative);await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,buffer);
  manifest.files[relative]={bytes:buffer.length,sha256:sha(buffer)};
}
async function payload(key,text){
  const buffer=Buffer.from(text,'utf8'),entry={bytes:buffer.length,sha256:sha(buffer),encoding:'gzip',chunks:[]};
  for(let offset=0,index=0;offset<buffer.length;offset+=chunkBytes,index++){
    const decoded=buffer.subarray(offset,Math.min(offset+chunkBytes,buffer.length));
    const compressed=gzipSync(decoded,{level:9,mtime:0});
    require(gunzipSync(compressed).equals(decoded),'Compression changed payload bytes.');
    const relative=`data/${key}/${String(index).padStart(3,'0')}.gz.bin`;
    await write(relative,compressed);entry.chunks.push({path:relative,bytes:compressed.length,decoded_bytes:decoded.length,sha256:sha(compressed)});
  }
  manifest.payloads[key]=entry;return JSON.parse(text);
}
const common=await payload('common',block('offline-common'));
for(const key of ['timestamps','predicted_inhibit','inhibit_labels'])require(common[key]?.length===official.n,`Common ${key} row count mismatch.`);
const savedShap=await payload('shap',block('offline-shap'));
require(savedShap.start_time==='2026-06-03T14:00:00'&&savedShap.duration_minutes===10,'Saved SHAP timestamp changed.');
manifest.saved_shap={start_time:savedShap.start_time,duration_minutes:savedShap.duration_minutes};
let scoreSum=0;
manifest.tag_validation=[];
for(let index=0;index<official.tags.length;index++){
  const value=await payload('tag-'+index,block('offline-tag-'+index));
  require(value.tag===official.tags[index],'Tag order mismatch.');
  for(const key of ['actual','reconstructed','scores','final_alarm','labels'])require(value[key]?.length===official.n,`Tag ${index} ${key} row count mismatch.`);
  const counts={tp:0,fp:0,fn:0,tn:0};let omitted=0;
  for(let row=0;row<official.n;row++){
    const score=value.scores[row],alarm=value.final_alarm[row],label=value.labels[row];
    require(Number.isFinite(score)&&score>=0,'Invalid reconstruction score.');
    require(alarm===0||alarm===1,'Invalid final Alarm.');
    require(!common.predicted_inhibit[row]||alarm===0,'Alarm present during a detected stop.');
    scoreSum+=score;
    if(label===null){omitted++;continue;}
    require(label===0||label===1,'Invalid grading label.');counts[label?(alarm?'tp':'fn'):(alarm?'fp':'tn')]++;
  }
  const recorded=official.tag_metrics[index];
  for(const key of Object.keys(counts))require(counts[key]===recorded[key],`Tag ${index} confusion count mismatch.`);
  manifest.tag_validation.push({tag:value.tag,rows:official.n,omitted,...counts});
}
const calculatedMAE=scoreSum/(official.n*official.tags.length);
require(Math.abs(calculatedMAE-official.normalized_mae)<1e-12,'Full MAE verification failed.');
manifest.mae={recorded:official.normalized_mae,calculated:calculatedMAE,tolerance:1e-12};
const catalogBlock=[...summary.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)].find(match=>match[1].includes('id="offline-catalog-data"'));
const catalogScript=[...summary.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)].find(match=>match[2].includes("const data=JSON.parse(document.getElementById('offline-catalog-data').textContent);"));
require(catalogBlock&&catalogScript,'Original experiment catalog missing.');
const catalog=await payload('catalog',catalogBlock[2]);
manifest.catalog={evaluations:catalog.evaluations.length,preserved_code_paths:catalog.codes.length};
require(manifest.catalog.evaluations===464&&manifest.catalog.preserved_code_paths===162,'Catalog coverage mismatch.');
summary=summary.replace(catalogBlock[0],'').replace(catalogScript[0],'');
for(const [before,after] of [
  ['회전기계 EWS · 오프라인 최종 요약','회전기계 EWS · 웹 최종 요약'],
  ['요약 수치는 실제 완료·검증 메타데이터에 근거하며 외부 서버에 요청하지 않습니다.','요약 수치는 실제 완료·검증 메타데이터에 근거하며 같은 사이트의 검증된 정적 자료를 표시합니다.'],
  ['이 요약 탭은 검증 메타데이터를 설명하며 외부 CDN·네트워크 요청을 사용하지 않습니다.','이 요약 탭은 검증 메타데이터를 설명하며 외부 CDN 없이 같은 사이트에서 필요한 자료만 불러옵니다.'],
]){
  require(summary.split(before).length===2,'Unexpected web summary wording.');
  summary=summary.replace(before,after);
}
const catalogJS=catalogScript[2].replace("const data=JSON.parse(document.getElementById('offline-catalog-data').textContent);",'const data=window.EWSWebCatalog;');
const originalLoader="queueMicrotask(()=>{try{window.EWS_TAG_READY(index,parent.EWSOffline.tag(index));}catch(error){pending.delete(index);reject(error);}});";
require(mainScript[2].includes(originalLoader),'Unrecognized offline Tag loader.');
const mainJS=mainScript[2].replace(originalLoader,"window.EWSWeb.tag(index).then(data=>window.EWS_TAG_READY(index,data)).catch(error=>{pending.delete(index);reject(error);window.EWSWeb.status(error.message,true);});");
const shapJS=shapScript[2].replace('parent.EWSOffline.shap()','window.EWSWeb.shap()');
dashboard=dashboard.replace(commonScript[0],'').replace(mainScript[0],'').replace(shapScript[0],'');
const csp="default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; frame-src 'self'; media-src data: blob:; object-src 'none'; base-uri 'none'; form-action 'none'";
function adapt(page,boot){
  const matches=page.match(/<meta\s+http-equiv="Content-Security-Policy"[^>]*>/gi)||[];
  require(matches.length===(boot==='summary-boot.js'?2:1),'Unexpected inner CSP count.');
  page=page.replace(/<meta\s+http-equiv="Content-Security-Policy"[^>]*>/gi,'');
  page=page.replace(/<head\b[^>]*>/i,match=>match+`<meta http-equiv="Content-Security-Policy" content="${csp}">`);
  return page.replace('</head>',`<script src="runtime.js" defer></script><script src="${boot}" defer></script></head>`);
}
await write('summary.html',adapt(summary,'summary-boot.js'));
await write('dashboard.html',adapt(dashboard,'dashboard-boot.js'));
await write('dashboard-main.js',mainJS);await write('dashboard-shap.js',shapJS);await write('summary-catalog.js',catalogJS);
for(const [from,to] of [['site-runtime.js','runtime.js'],['site-dashboard-boot.js','dashboard-boot.js'],['site-summary-boot.js','summary-boot.js']])await write(to,await fs.readFile(path.join(root,'tools',from)));
const index=`<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="${csp}"><title>회전기계 EWS · 최종 결과</title><style>html,body{margin:0;height:100%;font:14px system-ui,"Malgun Gothic",sans-serif;background:#f4f7fb;color:#172c42}*{box-sizing:border-box}header{padding:14px 20px;background:#112b43;color:white;display:flex;align-items:center;gap:16px;flex-wrap:wrap}header strong{font-size:18px}nav{display:flex;gap:8px}button,a{font:inherit}button{border:1px solid #7892ac;background:#fff;color:#163b5a;border-radius:7px;padding:9px 14px;cursor:pointer}button[aria-selected=true]{background:#16838b;color:white;border-color:#16838b}header a{color:#d3e3f2;margin-left:auto}.note{margin:0;padding:8px 20px;color:#425c6d}main{height:calc(100vh - 108px);min-height:400px}iframe{width:100%;height:100%;border:0;background:white}[hidden]{display:none!important}@media(max-width:650px){header{gap:9px;padding:12px}header a{margin-left:0}.note{font-size:12px}main{height:calc(100vh - 156px)}}</style></head><body><header><strong>회전기계 EWS · 최종 결과</strong><nav aria-label="결과 화면"><button data-tab="summary" aria-selected="true">최종 정리</button><button data-tab="dashboard" aria-selected="false">선정 모델 대시보드</button></nav><a href="https://github.com/pkg2060/g09" target="_blank" rel="noopener noreferrer">코드와 오프라인 HTML</a></header><p class="note">전체 476,641시점 × 7Tag를 보존했습니다. 선택한 화면·Tag만 불러옵니다. SHAP는 2026-06-03 14:00의 실제 저장 결과입니다.</p><main><iframe id="summary-frame" title="최종 정리" sandbox="allow-scripts allow-same-origin allow-downloads"></iframe><iframe id="dashboard-frame" title="선정 모델 대시보드" sandbox="allow-scripts allow-same-origin allow-downloads" hidden></iframe></main><script src="index.js"></script></body></html>`;
await write('index.html',index);
await write('index.js',`(()=>{'use strict';const frames={summary:document.getElementById('summary-frame'),dashboard:document.getElementById('dashboard-frame')};function activate(name){if(!(name in frames))return;for(const[key,frame]of Object.entries(frames))frame.hidden=key!==name;document.querySelectorAll('[data-tab]').forEach(button=>button.setAttribute('aria-selected',String(button.dataset.tab===name)));const frame=frames[name];if(!frame.hasAttribute('src'))frame.src=name+'.html';requestAnimationFrame(()=>{try{frame.contentWindow.dispatchEvent(new Event('resize'));}catch{}});history.replaceState(null,'','#'+name);}document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>activate(button.dataset.tab)));window.addEventListener('message',event=>{const value=event.data;if(event.origin!==location.origin||!value||value.type!=='ews:activate-tab')return;if(event.source===frames.summary.contentWindow&&value.source==='offline-summary'&&value.tab==='dashboard')activate('dashboard');if(event.source===frames.dashboard.contentWindow&&value.source==='offline-dashboard'&&value.tab==='summary')activate('summary');});activate(location.hash==='#dashboard'?'dashboard':'summary');})();`);
require(sha(original)===expectedSHA,'Source bytes changed unexpectedly.');
await fs.writeFile(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n','utf8');
console.log(JSON.stringify({status:'ok',source_sha256:expectedSHA,rows:manifest.rows,tags:manifest.tag_count,payloads:Object.keys(manifest.payloads).length,static_files:Object.keys(manifest.files).length+1,initial_index_bytes:manifest.files['index.html'].bytes,summary_shell_bytes:manifest.files['summary.html'].bytes,catalog:manifest.catalog,total_asset_bytes:Object.values(manifest.files).reduce((sum,item)=>sum+item.bytes,0)},null,2));
