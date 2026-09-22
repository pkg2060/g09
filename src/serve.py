"""동일 출처 대시보드와 요청 시 SHAP를 제공하는 로컬 분석 서버.

127.0.0.1에만 바인딩하며 SageMaker에서는 기존 Jupyter 인증 프록시로 연다.
모델 경로는 서버 시작 때 고정된다. 브라우저 요청으로 파일/명령을 실행하지 않는다.
"""
from __future__ import annotations
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import URLError, HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

from .config import ROOT, get_config


class DashboardServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self, address, run_dir, checkpoint, engine_factory=None):
        self.run_dir=Path(run_dir).resolve()
        self.checkpoint=Path(checkpoint).resolve()
        self.engine_factory=engine_factory
        self.engine=None
        self.engine_lock=threading.Lock()
        super().__init__(address, partial(DashboardHandler,directory=str(self.run_dir)))

    def explain(self, data):
        if not isinstance(data,dict) or set(data)-{'start_time','duration_minutes','target_tag'}:
            raise ValueError('SHAP 요청 형식이 올바르지 않습니다.')
        if 'start_time' not in data or 'duration_minutes' not in data:
            raise ValueError('시작 시점과 분석 기간을 입력하세요.')
        with self.engine_lock:
            if self.engine is None:
                factory=self.engine_factory
                if factory is None:
                    from .explain import ExplanationEngine
                    factory=ExplanationEngine
                self.engine=factory(self.run_dir,self.checkpoint)
            return self.engine.explain(data['start_time'],data['duration_minutes'],data.get('target_tag'))


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version='EWS/1'
    def send_json(self, status, payload):
        raw=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Length',str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path=='/api/health':
            self.send_json(200,{'status':'ok','application':'ews-shap-dashboard',
                 'run_dir':str(self.server.run_dir),'checkpoint':str(self.server.checkpoint),
                 'pid':os.getpid(),'engine_loaded':self.server.engine is not None})
            return
        if path=='/':
            path='/dashboard.html'
        if '\\' in path or '..' in path.split('/'):
            self.send_error(404,'Not found'); return
        relative=Path(path.lstrip('/'))
        candidate=(self.server.run_dir/relative).resolve()
        if not candidate.is_relative_to(self.server.run_dir):
            self.send_error(404,'Not found'); return
        normalized=candidate.relative_to(self.server.run_dir)
        allowed=(normalized.as_posix()=='dashboard.html' or
                 (normalized.parts and normalized.parts[0] in {'dashboard_data','plot','plots'}))
        if (not allowed or not candidate.is_file() or candidate.suffix.lower() not in {'.html','.js','.css','.png','.svg'}):
            self.send_error(404,'Not found')
            return
        # 검증 뒤 부모 클래스가 경로를 다시 해석하거나 디렉터리를 열지 않게 한다.
        with candidate.open('rb') as stream:
            self.send_response(200)
            self.send_header('Content-Type',self.guess_type(str(candidate)))
            self.send_header('Content-Length',str(candidate.stat().st_size))
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.copyfile(stream,self.wfile)

    def do_HEAD(self):
        # SimpleHTTPRequestHandler의 일반 파일탐색 경로를 노출하지 않는다.
        self.send_error(405,'Method not allowed')

    def do_POST(self):
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=16384: raise ValueError('요청 크기가 올바르지 않습니다.')
            raw=self.rfile.read(length)
        except ValueError as error:
            self.send_json(400,{'error':str(error)}); return
        if urlsplit(self.path).path!='/api/shap':
            self.send_json(404,{'error':'지원하지 않는 API입니다.'}); return
        if self.headers.get('X-EWS-Request')!='1' or self.headers.get_content_type()!='application/json':
            self.send_json(403,{'error':'동일 대시보드에서 분석을 요청하세요.'}); return
        origin=self.headers.get('Origin')
        if origin:
            host=self.headers.get('X-Forwarded-Host') or self.headers.get('Host','')
            if urlsplit(origin).netloc!=host.split(',')[0].strip():
                self.send_json(403,{'error':'다른 출처의 SHAP 요청은 허용하지 않습니다.'}); return
        try:
            data=json.loads(raw)
            result=self.server.explain(data)
            self.send_json(200,result)
        except (ValueError,TypeError,KeyError) as error:
            self.send_json(400,{'error':str(error)})
        except Exception as error:
            self.log_error('SHAP 실행 실패: %s',error)
            self.send_json(500,{'error':'SHAP 실행 실패: '+str(error)})


def _resolve_selection(run_dir=None,checkpoint=None):
    results=Path(get_config()['paths']['results'])
    selection_path=results/'selected_checkpoint.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.is_file() else {}
    if run_dir is None:
        selected_run=selection.get('final_run')
        if selected_run is None:
            raise ValueError('최종 결과가 없으면 --run-id와 --checkpoint를 지정하세요.')
        run_dir=selected_run
    run_dir=Path(run_dir).resolve()
    if checkpoint is None:
        provenance=run_dir/'eval_provenance.json'
        if provenance.is_file():
            checkpoint=json.loads(provenance.read_text(encoding='utf-8'))['checkpoint']
        elif (run_dir/'best.pt').is_file(): checkpoint=run_dir/'best.pt'
        else: checkpoint=selection.get('checkpoint')
    if not checkpoint: raise ValueError('SHAP용 저장 Checkpoint가 없습니다.')
    checkpoint=Path(checkpoint).resolve()
    for path in [run_dir/'dashboard.html',run_dir/'predictions.npz',run_dir/'config.json',
                 checkpoint,checkpoint.parent/'scaler.pkl']:
        if not path.is_file(): raise FileNotFoundError(f'분석 서버에 필요한 파일이 없습니다: {path}')
    return run_dir,checkpoint


def _health(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/api/health',timeout=1) as response:
            return json.load(response)
    except (URLError,HTTPError,OSError,ValueError): return None


def launch_dashboard(run_dir=None,checkpoint=None,port=None):
    run_dir,checkpoint=_resolve_selection(run_dir,checkpoint)
    port=int(port or get_config().get('explain',{}).get('server_port',8767))
    if not 1024<=port<=65535: raise ValueError('서버 포트는 1024~65535로 지정하세요.')
    existing=_health(port)
    if existing:
        if (existing.get('application')!='ews-shap-dashboard' or
            existing.get('run_dir')!=str(run_dir) or existing.get('checkpoint')!=str(checkpoint)):
            raise RuntimeError('해당 포트는 다른 결과를 제공 중입니다. --port로 다른 포트를 지정하세요.')
        existing.update(port=port,local_url=f'http://127.0.0.1:{port}/',
                        jupyter_proxy_path=f'/jupyterlab/default/proxy/{port}/')
        print('기존 SHAP 서버 사용: '+json.dumps(existing,ensure_ascii=False),flush=True)
        return existing
    log=run_dir/'dashboard_server.log'
    argv=[sys.executable,'-u','-m','src.serve','--worker','--directory',str(run_dir),
          '--checkpoint',str(checkpoint),'--port',str(port)]
    kwargs={'start_new_session':True} if os.name=='posix' else {'creationflags':subprocess.CREATE_NO_WINDOW}
    with log.open('ab',buffering=0) as stream:
        process=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,**kwargs)
    health=None
    for _ in range(40):
        health=_health(port)
        if health or process.poll() is not None: break
        time.sleep(.25)
    if not health: raise RuntimeError(f'대시보드 서버 시작 실패. 로그: {log}')
    health.update(port=port,local_url=f'http://127.0.0.1:{port}/',
                  jupyter_proxy_path=f'/jupyterlab/default/proxy/{port}/',log=str(log))
    (run_dir/'dashboard_server.json').write_text(json.dumps(health,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SHAP 대시보드 준비 완료: '+json.dumps(health,ensure_ascii=False),flush=True)
    print('SageMaker: 현재 Jupyter 주소의 /lab... 부분을 /proxy/'+str(port)+'/ 로 바꾸어 여세요.',flush=True)
    return health


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--port',type=int,default=8767)
    args=parser.parse_args()
    directory,checkpoint=_resolve_selection(args.directory,args.checkpoint)
    with DashboardServer(('127.0.0.1',args.port),directory,checkpoint) as server:
        print(f'LISTEN 127.0.0.1:{args.port} run={directory}',flush=True)
        server.serve_forever()
