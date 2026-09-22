/** Independent byte, syntax, and browser-reader contract checks. No server required. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash,webcrypto} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {spawnSync} from 'node:child_process';
import vm from 'node:vm';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),site=path.join(root,'site');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const require=(test,message)=>{if(!test)throw Error(message);};
const manifest=JSON.parse(await fs.readFile(path.join(site,'manifest.json'),'utf8'));
require(manifest.source.sha256==='1963310d723b1fe01740ab36006ff81df387c43671050a10745b3dc53bfd0248','Unpinned source.');
let syntaxChecks=0;
for(const [relative,entry] of Object.entries(manifest.files)){
  const bytes=await fs.readFile(path.join(site,relative));
  require(bytes.length===entry.bytes&&sha(bytes)===entry.sha256,'Static file identity mismatch.');
  require(bytes.length<20*1024*1024,'Oversized static asset.');
  if(relative.endsWith('.js')){const result=spawnSync(process.execPath,['--check',path.join(site,relative)],{encoding:'utf8'});require(result.status===0,`JavaScript syntax error: ${relative}`);syntaxChecks++;}
}
let checkedPayloads=0;
for(const entry of Object.values(manifest.payloads)){
  const chunks=[];
  for(const chunk of entry.chunks){const compressed=await fs.readFile(path.join(site,chunk.path));require(sha(compressed)===chunk.sha256,'Compressed SHA mismatch.');const decoded=gunzipSync(compressed);require(decoded.length===chunk.decoded_bytes,'Decoded size mismatch.');chunks.push(decoded);}
  const decoded=Buffer.concat(chunks);require(decoded.length===entry.bytes&&sha(decoded)===entry.sha256,'JSON payload byte mismatch.');JSON.parse(decoded.toString('utf8'));checkedPayloads++;
}
const requests=[];
const statusNode={textContent:'',style:{},hidden:false};
const sandbox={window:{},document:{getElementById:()=>statusNode},location:{href:'https://ews.example/dashboard.html'},URL,Blob,Response,DecompressionStream,TextDecoder,Uint8Array,crypto:webcrypto,console,
  fetch:async url=>{const parsed=new URL(url);require(parsed.origin==='https://ews.example','Unexpected external request.');const relative=parsed.pathname.slice(1);require(!relative.includes('..'),'Path traversal.');requests.push(relative);return new Response(await fs.readFile(path.join(site,relative)),{status:200});}};
vm.createContext(sandbox);vm.runInContext(await fs.readFile(path.join(site,'runtime.js'),'utf8'),sandbox);
const runtime=sandbox.window.EWSWeb;
const saved=await runtime.read('shap');require(saved.start_time==='2026-06-03T14:00:00','Browser-reader SHAP mismatch.');
require(!requests.some(name=>name.includes('tag-')),'Tag data eagerly loaded.');
const first=await runtime.tag(0);require(first.actual.length===manifest.rows&&first.scores.length===manifest.rows,'Browser-reader full Tag mismatch.');
require(!requests.some(name=>/tag-[1-6]/.test(name)),'Unselected Tag data eagerly loaded.');
const outer=await fs.readFile(path.join(site,'index.html'),'utf8'),summary=await fs.readFile(path.join(site,'summary.html'),'utf8'),dashboard=await fs.readFile(path.join(site,'dashboard.html'),'utf8');
require(!outer.includes('data/tag-')&&!summary.includes('offline-catalog-data'),'Landing embeds heavy data.');
require(!dashboard.includes('parent.EWSOffline.'),'Offline synchronous API remains in dashboard.');
require(manifest.rows===476641&&manifest.tag_count===7&&manifest.catalog.evaluations===464&&manifest.catalog.preserved_code_paths===162,'Coverage changed.');
console.log(JSON.stringify({status:'ok',syntax_checks:syntaxChecks,payloads_verified:checkedPayloads,full_rows_per_tag:manifest.rows,tags:manifest.tag_count,catalog:manifest.catalog,browser_reader_contract:'gzip/SHA/JSON decode and lazy selected Tag passed',browser_visual_interaction:'not tested by this script'},null,2));
