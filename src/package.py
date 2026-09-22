"""완료된 평가를 GitHub 전달용 전체 소스 폴더와 ZIP으로 묶는다.

원본 CSV·모델·Scaler·감사 원장은 포함하지 않는다. 사용자가 요청한 HTML의
dashboard_data에는 실제 센서값과 정답의 모든 시점이 들어 있다. 외부 업로드는
수행하지 않으며 기존 배포 폴더를 덮어쓰지 않는다. Python 표준 라이브러리만 사용한다.
"""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import tempfile
import zipfile


_IGNORED_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", ".ralph",
                 "data", "results", "checkpoints", ".venv", "venv"}
_IGNORED_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".pkl", ".pickle", ".npz",
                     ".safetensors", ".sqlite", ".db", ".log", ".jsonl"}
_SUMMARY_FILES = ("metrics.json", "tag_metrics.csv", "stop_validation.json",
                  "ralph_summary.json", "experiment_history.json", "결과보고.md",
                  "report_manifest.json", "threshold.json", "eval_provenance.json",
                  "eval_immutability.json", "integrity.json", "train_reconstruction_metrics.json")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed_files(directory):
    """지정된 소스/문서 트리만 읽으며 symlink를 통한 외부 파일 포함은 거부한다."""
    directory = Path(directory)
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if _IGNORED_DIRS.intersection(relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"패키지에는 symlink를 포함할 수 없습니다: {path}")
        if not path.is_file() or path.suffix.lower() in _IGNORED_SUFFIXES:
            continue
        if path.name in {"train.csv", "test.csv"} or path.name.startswith(".env"):
            continue
        yield path, relative


def _copy_tree(source, destination):
    for path, relative in _allowed_files(source):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".ipynb":
            # 셀 출력에는 데이터·원격 경로·토큰이 남을 수 있다. 원본은 수정하지 않는다.
            notebook = _read_json(path)
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            _write_json(target, notebook)
        else:
            shutil.copy2(path, target)


def _portable_config(config):
    result = copy.deepcopy(config)
    # get_config()의 상대 프로젝트 ROOT 경로를 사용하도록 override에서 제거한다.
    result.pop("paths", None)
    result.setdefault("train", {})["resume_from"] = None
    result["deadline_utc"] = None
    result.setdefault("threshold", {}).pop("spot_source_path", None)
    if "search" in result:
        result["search"]["deadline_utc"] = None

    def check(value, key="config"):
        if isinstance(value, dict):
            for name, child in value.items():
                check(child, key + "." + name)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check(child, key + f"[{index}]")
        elif isinstance(value, str) and (PurePosixPath(value).is_absolute()
                                       or PureWindowsPath(value).is_absolute()):
            raise ValueError(f"portable 설정에 처리되지 않은 절대 경로가 있습니다: {key}")
    check(result)
    return result


def _dashboard_contract(run_dir):
    """저장된 JS를 실행하지 않고 JSON 부분만 읽어 모든 시점·Tag의 보존을 확인한다."""
    report = _read_json(run_dir / "report_manifest.json")
    if report.get("full_resolution_preserved") is not True:
        raise ValueError("report_manifest에 원본 시점 보존 확인이 없습니다.")
    count = report.get("source_rows")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("report_manifest.source_rows는 양의 정수여야 합니다.")
    data_dir = run_dir / "dashboard_data"
    common_text = (data_dir / "common.js").read_text(encoding="utf-8").strip()
    prefix = "window.EWS_COMMON="
    if not common_text.startswith(prefix) or not common_text.endswith(";"):
        raise ValueError("dashboard_data/common.js 형식이 올바르지 않습니다.")
    common = json.loads(common_text[len(prefix):-1])
    for key in ("timestamps", "predicted_inhibit", "inhibit_labels"):
        if not isinstance(common.get(key), list) or len(common[key]) != count:
            raise ValueError(f"dashboard common.{key}의 원본 행수가 일치하지 않습니다.")
    del common, common_text
    tag_paths = sorted(data_dir.glob("tag_*.js"))
    tags = []
    for index, path in enumerate(tag_paths):
        if path.name != f"tag_{index:04d}.js":
            raise ValueError("dashboard_data의 Tag 파일 번호가 연속적이지 않습니다.")
        content = path.read_text(encoding="utf-8").strip()
        prefix = f"window.EWS_TAG_READY({index},"
        if not content.startswith(prefix) or not content.endswith(");"):
            raise ValueError(f"{path.name} 형식이 올바르지 않습니다.")
        tag = json.loads(content[len(prefix):-2])
        for key in ("actual", "reconstructed", "scores", "final_alarm", "labels"):
            if not isinstance(tag.get(key), list) or len(tag[key]) != count:
                raise ValueError(f"{path.name}.{key}의 원본 행수가 일치하지 않습니다.")
        if not isinstance(tag.get("tag"), str) or not tag["tag"]:
            raise ValueError(f"{path.name}에 Tag 이름이 없습니다.")
        tags.append(tag["tag"])
        del content, tag
    expected = _read_json(run_dir / "metrics.json").get("classification", {}).get("per_tag", {})
    if not tags or len(set(tags)) != len(tags) or set(tags) != set(expected):
        raise ValueError("대시보드의 Tag 집합과 metrics.json의 채점 Tag 집합이 일치하지 않습니다.")
    return {"source_rows": count, "tag_count": len(tags), "tags": tags,
            "full_resolution_preserved": True}


def _copy_mandatory_case(root, staged):
    """의무 케이스 원문을 보존하고 완료 증거가 있는 Epoch20 HTML을 별도 복사한다."""
    source = root / "results/mandatory_cases.json"
    if not source.exists():
        return {"status": "not_recorded", "included": False}
    if source.is_symlink():
        raise ValueError("의무 케이스 manifest symlink는 허용하지 않습니다.")
    source_digest = _sha256(source)
    payload = _read_json(source)
    case = payload.get("window10_epoch20")
    if not isinstance(case, dict) or case.get("status") not in {"pending", "unfulfilled", "completed"}:
        raise ValueError("의무 Epoch20 케이스 상태 형식이 올바르지 않습니다.")
    target = staged / "docs/mandatory_cases.json"
    shutil.copy2(source, target)
    if _sha256(target) != source_digest:
        raise RuntimeError("의무 케이스 manifest 복사 hash 불일치")
    result = {"status": case["status"], "included": False,
              "manifest_source_sha256": source_digest, "search_id": payload.get("search_id"),
              "manifest": "docs/mandatory_cases.json"}
    if case["status"] != "completed":
        result["reason"] = case.get("reason", "실제 Epoch20 평가·HTML 완료 증거 없음")
        return result
    if (case.get("required") is not True or case.get("window") != 10
            or case.get("model_weights_epoch") != 20 or case.get("epochs_consumed") != 20):
        raise ValueError("완료된 의무 케이스에 실제 Window10/Epoch20 증거가 없습니다.")

    def evidence_path(value):
        if not isinstance(value, str) or not value:
            raise ValueError("의무 케이스 증거 경로가 없습니다.")
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if path.is_symlink():
            raise ValueError("의무 케이스 symlink는 허용하지 않습니다.")
        resolved = path.resolve()
        if not resolved.is_relative_to((root / "results").resolve()):
            raise ValueError("의무 케이스 증거는 프로젝트 results 내부여야 합니다.")
        return resolved

    run = evidence_path(case.get("evaluation_dir"))
    dashboard = evidence_path(case.get("dashboard"))
    report_path = evidence_path(case.get("report_manifest"))
    if dashboard != run / "dashboard.html" or report_path != run / "trial_report_manifest.json":
        raise ValueError("의무 케이스 HTML/manifest와 평가 경로가 일치하지 않습니다.")
    identities = {"dashboard.html": case.get("dashboard_sha256"),
                  "trial_report_manifest.json": case.get("report_manifest_sha256"),
                  "metrics.json": case.get("metrics_sha256"),
                  "config.json": case.get("config_sha256")}
    for name, digest in identities.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"의무 케이스 {name} SHA-256 증거가 없습니다.")
        if not (run / name).is_file():
            raise FileNotFoundError(f"완료된 의무 케이스 파일 누락: {run / name}")
        if _sha256(run / name) != digest:
            raise ValueError(f"의무 케이스 {name} hash 불일치")
    checkpoint = evidence_path(case.get("checkpoint"))
    if not checkpoint.is_file() or _sha256(checkpoint) != case.get("checkpoint_sha256"):
        raise ValueError("의무 케이스 Checkpoint hash 불일치")
    report = _read_json(report_path)
    evidence = report.get("model_evidence", {})
    rows = report.get("source_rows")
    tags = report.get("tags")
    if (report.get("full_resolution_preserved") is not True
            or isinstance(rows, bool) or not isinstance(rows, int) or rows < 1
            or not isinstance(tags, list) or not tags or len(set(tags)) != len(tags)
            or report.get("dashboard_sha256") != identities["dashboard.html"]
            or report.get("checkpoint_sha256") != case["checkpoint_sha256"]
            or evidence.get("model_weights_epoch") != 20 or evidence.get("epochs_consumed") != 20
            or _read_json(run / "config.json").get("model", {}).get("window") != 10):
        raise ValueError("의무 케이스 실제 모델·HTML·원본 시점 보존 증거가 일치하지 않습니다.")
    for name in ("metrics.json", "config.json"):
        if report.get("source_identities", {}).get(name) != identities[name]:
            raise ValueError(f"의무 케이스 manifest의 {name} hash 불일치")
    assets = run / "dashboard_data"
    if assets.is_symlink() or not assets.is_dir():
        raise ValueError("의무 케이스 dashboard_data 폴더가 올바르지 않습니다.")
    expected = {f"tag_{i:04d}.js" for i in range(len(tags))}
    blobs = report.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("의무 케이스 압축 원본 파일 목록이 없습니다.")
    for blob in blobs:
        key = blob.get("sha256", "")
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("의무 케이스 압축 원본 hash 형식이 올바르지 않습니다.")
        name = "blob_" + key + ".js"
        path = assets / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != blob.get("bytes"):
            raise ValueError(f"의무 케이스 압축 원본 파일 누락 또는 크기 불일치: {name}")
        body = path.read_text(encoding="utf-8").strip()
        prefix = 'window.EWS_TRIAL_BLOBS["' + key + '"]='
        if not body.startswith(prefix) or not body.endswith(";"):
            raise ValueError(f"의무 케이스 압축 원본 JS 형식 불일치: {name}")
        compressed = base64.b64decode(json.loads(body[len(prefix):-1]), validate=True)
        if hashlib.sha256(compressed).hexdigest() != key:
            raise ValueError(f"의무 케이스 압축 원본 hash 불일치: {name}")
        series = json.loads(gzip.decompress(compressed))
        if not isinstance(series, dict) or not series or any(
                not isinstance(values, list) or len(values) != rows for values in series.values()):
            raise ValueError(f"의무 케이스 압축 원본 행수 불일치: {name}")
        expected.add(name)
    actual_files = list(_allowed_files(assets))
    if {relative.as_posix() for _, relative in actual_files} != expected:
        raise ValueError("의무 케이스 dashboard_data 파일 목록이 manifest와 다릅니다.")
    for i, tag in enumerate(tags):
        wrapper = (assets / f"tag_{i:04d}.js").read_text(encoding="utf-8").strip()
        prefix = f"window.EWS_TRIAL.tag({i},"
        if not wrapper.startswith(prefix) or not wrapper.endswith(");"):
            raise ValueError("의무 케이스 Tag 로더 형식이 올바르지 않습니다.")
        keys, stored_tag, threshold = json.loads("[" + wrapper[len(prefix):-2] + "]")
        if (stored_tag != tag or len(keys) != 3
                or any("blob_" + key + ".js" not in expected for key in keys)):
            raise ValueError("의무 케이스 Tag 참조가 압축 원본과 일치하지 않습니다.")
    destination = staged / "cases/epoch20"
    destination.mkdir(parents=True)
    # 내부 hardlink도 독립적인 일반 파일로 복사하므로 _trial_shared가 없어도 열 수 있다.
    _copy_tree(assets, destination / "dashboard_data")
    source_hashes = {}
    for name in identities:
        if name == "config.json":
            continue  # 원격 절대 경로가 들어간 실행 설정은 별도 증거 hash로만 보존한다.
        shutil.copy2(run / name, destination / name)
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            relative = path.relative_to(destination)
            digest = _sha256(run / relative)
            if _sha256(path) != digest:
                raise RuntimeError(f"의무 케이스 산출물 복사 hash 불일치: {relative}")
            source_hashes[relative.as_posix()] = digest
    if _sha256(source) != source_digest:
        raise RuntimeError("패키지 생성 도중 의무 케이스 manifest가 변경되었습니다.")
    result.update(included=True, dashboard="cases/epoch20/dashboard.html", source_rows=rows,
                  tag_count=len(tags), checkpoint_sha256=case["checkpoint_sha256"],
                  source_artifact_sha256=source_hashes)
    return result


def _readme(original, run_name, metrics, dashboard, mandatory):
    gate = metrics.get("gate", {})
    return f"""# GitHub 전달용 회전기계 EWS 전체 소스와 최종 대시보드

이 폴더는 `{run_name}`의 최종 평가에서 생성했습니다. GitHub 업로드·push나 웹 게시를 수행한 결과가 아닙니다.
전체 Test MAE는 `{gate.get('mae', '미제공')}`, 두 수치 조건 동시 통과는 `{gate.get('phase2_numeric_pass', '미제공')}`입니다.
제공된 Test를 반복 확인한 개발평가이며 독립 데이터 성능을 뜻하지 않습니다.

- `mode.py`, `src/`: 실행 가능한 전체 코드·화면 자산·의존성·외부 라이선스.
- `docs/dashboard.html`, `docs/dashboard_data/`: 원래 해상도 {dashboard['source_rows']:,}시점 × {dashboard['tag_count']}Tag의 실제값·복원값·정답·경보.
- `docs/metrics.json`, `docs/결과보고.md`: 평가 수치와 한국어 보고서.
- `selected_config.json`: 선택 설정을 새 프로젝트에서 사용할 수 있게 경로와 이어학습 정보를 정리한 override.
- `package_manifest.json`: 실제 포함 파일의 SHA-256, 원본 결과 hash와 제외 정책.
- 의무 Window10 Epoch20: `{mandatory['status']}`. 별도 HTML 포함 여부: `{mandatory['included']}`.
  완료된 경우 `cases/epoch20/dashboard.html`과 같은 폴더의 `dashboard_data/`를 함께 엽니다.
  기록이 있는 경우 `docs/mandatory_cases.json`에 원본 완료/미완료 증거를 보존합니다.

**dashboard_data에는 실제 공정 센서값과 정답이 포함됩니다.** 사용자가 선택한 Private 전체소스 전달용으로 준비했습니다.
원본 `data/train.csv`, `data/test.csv`, Checkpoint, Scaler, `.ralph` 원본 감사 로그는 포함하지 않습니다.
포함된 RALPH 요약은 집계 스냅샷이며 원본 감사 증거를 대체하지 않습니다.

## 새 SageMaker 프로젝트에서 실행

이 폴더를 프로젝트 루트로 열고, 승인된 CSV를 새 `data/` 폴더에 배치합니다.
CUDA PyTorch 환경을 선택한 뒤 아래 명령을 사용합니다. 기존 모델은 미포함이므로 먼저 학습이 필요합니다.

```python
%pip install -r src/requirements.txt
%run mode -full --config selected_config.json --run-id new_full_01
```

`-train`과 `-eval`도 동일한 config 옵션을 사용합니다. Eval은 별도로 확보한 모델과 Scaler가 함께 필요합니다.
`selected_config.json`은 선택 모델 설정을 보존하지만 저장된 가중치를 포함하지 않으므로 재학습 수치의 완전 일치를 보장하지 않습니다.
일반 full/train의 `deadline_utc=None`은 시간 제한이 없다는 의미입니다.
새 `-optimize`에는 `src/config.py`의 `search.deadline_utc`를 미래 시각(UTC offset 포함)으로 설정해야 합니다.
현재 optimizer는 `--config`를 사용하지 않으므로 선택 설정으로 새 탐색하려면 `src/config.py`에 명시적으로 반영합니다.

```text
python -m src.test_contracts -v
```

`docs/dashboard.html`은 같은 위치의 `dashboard_data/`와 함께 열어야 합니다. 원본 JS 배열은 축소하지 않았습니다.
노트북은 셀 출력과 실행 번호만 제거했으며 원본 프로젝트의 실행 안내 경로는 새 폴더에 맞춰 바꿉니다.
파일별 라이선스는 `docs/THIRD_PARTY.md`를 확인합니다. SPOT의 GPLv3 고지를 저장소의 BSD 고지와 구분합니다.

---

아래는 원본 프로젝트 안내입니다. 원래 해커톤의 경로·마감·체크포인트 예시는 위의 새 프로젝트 절차에 맞춰 적용합니다.

{original}
"""


def build_package(root, final_run_dir):
    """완성 폴더/ZIP/manifest 경로를 반환한다. 누락·불완전 결과·기존 대상은 오류다."""
    root = Path(root).resolve()
    run_dir = Path(final_run_dir)
    run_dir = (root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    if not root.is_dir() or not run_dir.is_dir():
        raise FileNotFoundError("프로젝트 루트와 완료된 final_run_dir가 모두 필요합니다.")
    required = [root / "mode.py", root / "README.md", root / "src/requirements.txt",
                run_dir / "config.json", run_dir / "metrics.json", run_dir / "integrity.json",
                run_dir / "dashboard.html", run_dir / "report_manifest.json",
                run_dir / "dashboard_data/common.js", run_dir / "결과보고.md"]
    if not (root / "docs").is_dir():
        required.append(root / "docs")
    if (root / "src/vendor/spot.py").is_file():
        required.append(root / "src/vendor/LICENSE.GPL-3.0")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("최종 패키지 필수 파일 누락: " + ", ".join(missing))
    integrity = _read_json(run_dir / "integrity.json")
    unchanged = integrity.get("original_data_unchanged")
    if (integrity.get("complete") is not True or not isinstance(unchanged, dict)
            or not unchanged or any(value is not True for value in unchanged.values())):
        raise ValueError("최종 평가 완료와 원본 데이터 불변성이 확인되지 않았습니다.")
    if not re.fullmatch(r"[\w.\-]+", run_dir.name, re.UNICODE):
        raise ValueError("최종 실행 폴더 이름에는 문자·숫자·밑줄·점·하이픈만 허용합니다.")
    destination = root / ("github_deploy_" + run_dir.name)
    archive = destination.with_name(destination.name + ".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError(f"기존 전달 패키지를 덮어쓰지 않습니다: {destination}")
    dashboard = _dashboard_contract(run_dir)
    portable = _portable_config(_read_json(run_dir / "config.json"))
    metrics = _read_json(run_dir / "metrics.json")
    with tempfile.TemporaryDirectory(prefix=".package_build_", dir=root) as scratch_name:
        scratch = Path(scratch_name)
        # 압축 시계열의 긴 SHA-256 이름을 위해 Windows 임시 경로를 짧게 유지한다.
        staged = scratch / "p"
        staged.mkdir()
        shutil.copy2(root / "mode.py", staged / "mode.py")
        _copy_tree(root / "src", staged / "src")
        _copy_tree(root / "docs", staged / "docs")
        (staged / "docs").mkdir(exist_ok=True)
        _write_json(staged / "selected_config.json", portable)
        mandatory = _copy_mandatory_case(root, staged)
        (staged / "README.md").write_text(_readme((root / "README.md").read_text(encoding="utf-8"),
                                                  run_dir.name, metrics, dashboard, mandatory), encoding="utf-8")
        (staged / ".gitignore").write_text("data/\nresults/\n.ralph/\n__pycache__/\n*.py[cod]\n"
                                         ".ipynb_checkpoints/\n.venv/\nvenv/\n.env\n.env.*\n"
                                         "*.pt\n*.pth\n*.pkl\n*.npz\ngithub_deploy_*/\n*.zip\n",
                                         encoding="utf-8")
        copied_artifacts = []
        for name in ("dashboard.html", *_SUMMARY_FILES):
            source = run_dir / name
            if source.is_file():
                shutil.copy2(source, staged / "docs" / name)
                copied_artifacts.append(source)
        for source in sorted((run_dir / "dashboard_data").glob("*.js")):
            if source.is_symlink():
                raise ValueError(f"대시보드 symlink는 허용하지 않습니다: {source}")
            if source.name != "common.js" and not re.fullmatch(r"tag_\d{4,}\.js", source.name):
                raise ValueError(f"알 수 없는 대시보드 JS 파일: {source.name}")
            (staged / "docs/dashboard_data").mkdir(exist_ok=True)
            shutil.copy2(source, staged / "docs/dashboard_data" / source.name)
            copied_artifacts.append(source)
        for source in sorted((run_dir / "plot").glob("*.png")):
            (staged / "docs/plot").mkdir(exist_ok=True)
            shutil.copy2(source, staged / "docs/plot" / source.name)
            copied_artifacts.append(source)
        # 실제 복사본의 byte hash까지 확인한다. dashboard_data를 다시 요약하지 않는다.
        source_hashes = {}
        for source in copied_artifacts:
            relative = source.relative_to(run_dir)
            source_hashes[relative.as_posix()] = _sha256(source)
            if _sha256(staged / "docs" / relative) != source_hashes[relative.as_posix()]:
                raise RuntimeError(f"산출물 복사 hash 불일치: {relative}")
        files = {path.relative_to(staged).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
                 for path in sorted(staged.rglob("*")) if path.is_file()}
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "final_run": run_dir.name,
                    "uploaded_to_github": False, "dashboard": dashboard,
                    "mandatory_epoch20": mandatory,
                    "source_config_sha256": _sha256(run_dir / "config.json"),
                    "source_artifact_sha256": source_hashes, "files": files,
                    "excluded": ["raw train/test CSV", "model checkpoints", "scaler", ".ralph raw evidence", "cache"],
                    "includes_real_sensor_and_truth_values": True,
                    "portable_config_changes": ["paths override removed", "train.resume_from=None",
                         "deadline_utc=None", "threshold.spot_source_path override removed", "search.deadline_utc=None"],
                    "verification": "배포 파일 생성 및 byte hash 검증. 새 환경 GPU 재학습·웹 게시 검증을 의미하지 않음."}
        _write_json(staged / "package_manifest.json", manifest)
        staged_zip = scratch / archive.name
        with zipfile.ZipFile(staged_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(staged.rglob("*")):
                if path.is_file():
                    bundle.write(path, arcname=(Path(destination.name) / path.relative_to(staged)).as_posix())
        # 같은 파일시스템의 새 경로로만 이동. 기존 폴더/ZIP은 대체하지 않는다.
        if destination.exists() or archive.exists():
            raise FileExistsError("패키지 생성 도중 같은 이름의 대상이 생성되었습니다.")
        staged.rename(destination)
        staged_zip.rename(archive)
    return {"directory": str(destination), "zip": str(archive),
            "manifest": str(destination / "package_manifest.json"), "file_count": len(files) + 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--final-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_package(args.root, args.final_run), ensure_ascii=False, indent=2))
