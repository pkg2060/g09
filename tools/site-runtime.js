/* Same-origin, lazy reader for the verified full-resolution final report. */
(() => {
  'use strict';
  const status = (message, failed=false) => {
    let node=document.getElementById('web-load-status');
    if(!node){node=document.createElement('div');node.id='web-load-status';node.setAttribute('role','status');node.style.cssText='position:sticky;top:0;z-index:10000;padding:10px 18px;background:#e8f3f6;color:#143d50;font:14px system-ui';document.body.prepend(node);}
    node.textContent=message;node.style.background=failed?'#ffe9e5':'#e8f3f6';node.hidden=!message;
  };
  const checked = async (bytes, expected) => {
    const digest=await crypto.subtle.digest('SHA-256',bytes);
    const actual=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');
    if(actual!==expected)throw Error('파일 무결성 확인에 실패했습니다. 페이지를 새로 고쳐주세요.');
  };
  const request=async path=>{const response=await fetch(new URL(path,location.href),{credentials:'same-origin'});if(!response.ok)throw Error(`결과 파일을 불러오지 못했습니다 (${response.status}).`);return response;};
  const manifest=request('manifest.json').then(response=>response.json());
  const read=async (key,label='저장된 결과')=>{
    const metadata=await manifest,entry=metadata.payloads[key];
    if(!entry)throw Error('알 수 없는 결과 항목입니다.');
    if(typeof DecompressionStream!=='function')throw Error('최신 Chrome 또는 Edge에서 열어주세요. 압축 데이터 읽기를 지원하는 브라우저가 필요합니다.');
    const output=new Uint8Array(entry.bytes);let offset=0;
    for(let i=0;i<entry.chunks.length;i++){
      status(`${label} 불러오는 중 · ${i+1}/${entry.chunks.length}`);
      const chunk=entry.chunks[i],compressed=new Uint8Array(await (await request(chunk.path)).arrayBuffer());
      if(compressed.length!==chunk.bytes)throw Error('결과 파일 크기가 일치하지 않습니다.');
      await checked(compressed,chunk.sha256);
      const bytes=new Uint8Array(await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      if(bytes.length!==chunk.decoded_bytes)throw Error('압축 해제 결과 크기가 일치하지 않습니다.');
      output.set(bytes,offset);offset+=bytes.length;
    }
    if(offset!==entry.bytes)throw Error('결과 행렬이 완전히 다운로드되지 않았습니다.');
    await checked(output,entry.sha256);
    const value=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(output));
    status('');return value;
  };
  let savedShap=null;
  window.EWSWeb=Object.freeze({read,status,manifest,tag:index=>read('tag-'+index,'선택한 Tag의 전체 원본 시점'),shap:()=>savedShap,setShap:value=>{savedShap=value;}});
})();
