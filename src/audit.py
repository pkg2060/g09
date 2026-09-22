"""실행 증거를 보존하는 RALPH 감사 장부.

종료 코드, 난스, 파일 해시를 실제 실행에서 수집한다. 일반적인 학습 실패를
부정직으로 분류하지 않으며, 주입 시험과 실제 실행을 별도로 집계한다.
해시 체인은 우발적 변경 탐지 수단이며 외부 서명이나 변경 불가능한 저장소를
대신하지 않는다. 같은 디렉터리를 수정할 권한이 있으면 재작성은 가능하다.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
import traceback
from typing import Any, Callable, Mapping, Sequence
from datetime import datetime, timezone


class AuditIntegrityError(RuntimeError):
    """감사 장부 또는 고정 검증 규칙이 변경됨."""


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AuditJournal:
    """`.ralph` 안에 append-only 방식으로 실행 내역과 원본 증거를 기록한다.

    gate_registry는 `{고정_ID: 한국어 설명}`이다. 기존 장부를 여는 경우에도
    동일한 registry를 전달해야 하며 변경하려면 별도 감사 실행을 시작한다.
    """

    def __init__(self, project_root: str | Path,
                 gate_registry: Mapping[str, str],
                 protected_paths: Sequence[str | Path] = ()) -> None:
        if not gate_registry or not all(isinstance(k, str) and isinstance(v, str)
                                       for k, v in gate_registry.items()):
            raise ValueError("gate_registry에는 고정 ID와 설명이 필요합니다.")
        self.project_root = Path(project_root).resolve()
        self.directory = self.project_root / ".ralph"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = self.directory / "evidence"
        self.evidence_dir.mkdir(exist_ok=True)
        self.path = self.directory / "events.jsonl"
        self.registry_path = self.directory / "gate_registry.json"
        self.registry = dict(sorted(gate_registry.items()))
        registry_bytes = _json({"version": 1, "gates": self.registry})
        self.registry_hash = hashlib.sha256(registry_bytes).hexdigest()
        with self._locked():
            if self.registry_path.exists():
                if self.registry_path.read_bytes() != registry_bytes:
                    raise AuditIntegrityError("고정 gate registry가 다릅니다.")
            else:
                if self.path.exists() and self.path.stat().st_size:
                    raise AuditIntegrityError("기존 장부의 gate registry가 없습니다.")
                self.registry_path.write_bytes(registry_bytes)
        self.verify()
        requested = {}
        for candidate in protected_paths:
            candidate_path = Path(candidate)
            full_path = (candidate_path if candidate_path.is_absolute()
                         else self.project_root / candidate_path).resolve()
            if not full_path.is_relative_to(self.project_root):
                raise ValueError("보호 파일은 프로젝트 안에 있어야 합니다.")
            requested[str(full_path.relative_to(self.project_root))] = file_sha256(full_path)
        events = self.events()
        baselines = [event for event in events if event["kind"] == "protected_baseline"]
        if baselines:
            self.protected = baselines[0]["data"]["files"]
            if requested and requested != self.protected:
                raise AuditIntegrityError("보호 파일의 초기 해시와 현재 파일이 다릅니다.")
        else:
            self.protected = requested
            self._append("protected_baseline", None, {"files": requested})

    @contextlib.contextmanager
    def _locked(self):
        lock = self.directory / ".journal.lock"
        deadline = time.monotonic() + 5
        while True:
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise AuditIntegrityError("감사 장부 잠금이 사용 중입니다. 이전 프로세스를 확인하세요.")
                time.sleep(0.05)
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            yield
        finally:
            lock.unlink(missing_ok=True)

    def events(self) -> list[dict[str, Any]]:
        """체인 검증을 통과한 이벤트만 반환한다."""
        if self.registry_path.read_bytes() != _json({"version": 1, "gates": self.registry}):
            raise AuditIntegrityError("gate registry가 변경되었습니다.")
        result = []
        previous = "0" * 64
        if not self.path.exists():
            return result
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
                supplied = event.pop("sha256")
                actual = hashlib.sha256(_json(event)).hexdigest()
                valid = (supplied == actual and event["seq"] == number
                         and event["previous_sha256"] == previous
                         and event["registry_sha256"] == self.registry_hash)
            except (ValueError, KeyError, TypeError) as exc:
                raise AuditIntegrityError(f"감사 장부 {number}행 형식 오류") from exc
            if not valid:
                raise AuditIntegrityError(f"감사 장부 {number}행 해시 체인 오류")
            event["sha256"] = supplied
            result.append(event)
            previous = supplied
        return result

    def verify(self, verify_evidence: bool = False) -> dict[str, Any]:
        events = self.events()
        if verify_evidence:
            for event in events:
                if event["kind"] == "command_result":
                    for stream in ("stdout", "stderr"):
                        entry = event["data"][stream]
                        candidate = (self.directory / entry["path"]).resolve()
                        if not candidate.is_relative_to(self.directory):
                            raise AuditIntegrityError("증거 경로가 감사 폴더를 벗어납니다.")
                        if not candidate.is_file() or file_sha256(candidate) != entry["sha256"]:
                            raise AuditIntegrityError(f"실행 증거 파일이 변경되었습니다: {entry['path']}")
        return {"valid": True, "events": len(events),
                "head_sha256": events[-1]["sha256"] if events else "0" * 64,
                "registry_sha256": self.registry_hash}

    def _append(self, kind: str, loop_id: str | None,
                data: Mapping[str, Any]) -> dict[str, Any]:
        with self._locked():
            current = self.events()
            event = {"seq": len(current) + 1,
                     "utc": datetime.now(timezone.utc).isoformat(),
                     "kind": kind, "loop_id": loop_id,
                     "registry_sha256": self.registry_hash,
                     "previous_sha256": current[-1]["sha256"] if current else "0" * 64,
                     "data": dict(data)}
            event["sha256"] = hashlib.sha256(_json(event)).hexdigest()
            with self.path.open("ab") as stream:
                stream.write(_json(event) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        return event

    def _active(self, loop_id: str) -> dict[str, Any]:
        events = [e for e in self.events() if e["loop_id"] == loop_id]
        starts = [e for e in events if e["kind"] == "loop_start"]
        if len(starts) != 1 or any(e["kind"] in ("loop_end", "loop_abort") for e in events):
            raise ValueError(f"진행 중인 loop_id가 아닙니다: {loop_id}")
        return starts[0]["data"]

    def start_loop(self, purpose: str, config: Mapping[str, Any] | None = None,
                   injected: bool = False) -> str:
        loop_id = f"loop-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{secrets.token_hex(4)}"
        self._append("loop_start", loop_id,
                     {"purpose": purpose, "config": dict(config or {}), "injected": bool(injected)})
        if not self.check_protected(loop_id):
            raise AuditIntegrityError("보호 파일 변조로 새 루프가 중단되었습니다.")
        return loop_id

    def check_protected(self, loop_id: str) -> bool:
        self._active(loop_id)
        changed = []
        for relative, expected in self.protected.items():
            path = self.project_root / relative
            actual = file_sha256(path) if path.is_file() else None
            if actual != expected:
                changed.append({"path": relative, "expected": expected, "actual": actual})
        if changed:
            self._reject(loop_id, "protected_file_changed", {"files": changed})
            return False
        return True

    def run_command(self, loop_id: str, argv: Sequence[str],
                    cwd: str | Path | None = None, require_nonce: bool = True,
                    timeout: float | None = None,
                    env: Mapping[str, str] | None = None) -> dict[str, Any]:
        """실제 subprocess를 실행한다. stdout/stderr에 정확한 난스 행을 요구한다.

        자식 프로세스는 EWS_AUDIT_NONCE 환경변수를 읽어
        `EWS_AUDIT_NONCE=<값>`을 출력해야 한다. returncode=0만으로는
        claimed success를 승인하지 않으며 저장된 증거 해시도 확인한다.
        """
        self._active(loop_id)
        if not argv or isinstance(argv, (str, bytes)):
            raise ValueError("argv는 shell 문자열이 아닌 인자 목록이어야 합니다.")
        evidence_id = f"exec-{secrets.token_hex(12)}"
        nonce = secrets.token_hex(24)
        workdir = str(Path(cwd or self.project_root).resolve())
        child_env = dict(os.environ)
        child_env.update(env or {})
        child_env["EWS_AUDIT_NONCE"] = nonce
        self._append("command_start", loop_id,
                     {"evidence_id": evidence_id, "argv": list(argv), "cwd": workdir,
                      "nonce": nonce, "require_nonce": bool(require_nonce)})
        started = time.monotonic()
        error = None
        try:
            completed = subprocess.run(list(argv), cwd=workdir, env=child_env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       timeout=timeout, check=False, shell=False)
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = exc.stdout or b"", exc.stderr or b""
            returncode, error = None, "timeout"
        except OSError as exc:
            stdout, stderr = b"", str(exc).encode("utf-8")
            returncode, error = None, type(exc).__name__
        expected_line = f"EWS_AUDIT_NONCE={nonce}".encode("ascii")
        nonce_seen = expected_line in stdout.splitlines() or expected_line in stderr.splitlines()
        result = {"evidence_id": evidence_id, "returncode": returncode,
                  "error": error, "elapsed_seconds": time.monotonic() - started,
                  "require_nonce": bool(require_nonce), "nonce_seen": nonce_seen,
                  "success": returncode == 0 and (nonce_seen or not require_nonce)}
        for name, value in (("stdout", stdout), ("stderr", stderr)):
            path = self.evidence_dir / f"{evidence_id}.{name}.log"
            path.write_bytes(value)
            result[name] = {"path": str(path.relative_to(self.directory)),
                            "sha256": file_sha256(path), "bytes": len(value)}
        self._append("command_result", loop_id, result)
        return result

    def _evidence(self, evidence_id: str, loop_id: str | None = None) -> dict[str, Any]:
        found = [e for e in self.events() if e["kind"] == "command_result"
                 and e["data"]["evidence_id"] == evidence_id
                 and (loop_id is None or e["loop_id"] == loop_id)]
        if len(found) != 1:
            raise ValueError("이번 루프에서 실제 실행한 증거 ID가 필요합니다.")
        return found[0]

    def record_claim(self, loop_id: str, evidence_id: str,
                     claimed_success: bool,
                     claimed_stdout_sha256: str | None = None) -> bool:
        """AI/호출자의 완료 주장과 실제 실행 결과를 비교한다."""
        self._active(loop_id)
        try:
            event = self._evidence(evidence_id, loop_id)
        except ValueError:
            self._reject(loop_id, "claim_without_current_execution", {"evidence_id": evidence_id})
            return False
        result = event["data"]
        violations = []
        if claimed_success and not result["success"]:
            violations.append("claimed_result_does_not_match_execution")
        for name in ("stdout", "stderr"):
            path = self.directory / result[name]["path"]
            if not path.is_file() or file_sha256(path) != result[name]["sha256"]:
                violations.append(f"{name}_evidence_changed")
        if claimed_stdout_sha256 is not None and claimed_stdout_sha256 != result["stdout"]["sha256"]:
            violations.append("claimed_stdout_hash_does_not_match")
        if violations:
            self._reject(loop_id, "claim_evidence_mismatch",
                         {"evidence_id": evidence_id, "violations": violations})
            return False
        self._append("claim_verified", loop_id,
                     {"evidence_id": evidence_id, "claimed_success": bool(claimed_success)})
        return True

    def _reject(self, loop_id: str, reason: str, details: Mapping[str, Any]) -> None:
        self._append("evidence_rejected", loop_id, {"reason": reason, **details})
        self.abort_loop(loop_id, reason, category="evidence_integrity")

    def run_gates(self, loop_id: str,
                  checks: Mapping[str, Callable[[], Any]]) -> dict[str, Any]:
        """고정 registry 전체를 실제 실행한다. 생략·skip은 통과가 아니다."""
        self._active(loop_id)
        if set(checks) != set(self.registry):
            raise ValueError("검증 함수 집합은 고정 gate registry와 정확히 같아야 합니다.")
        if not self.check_protected(loop_id):
            raise AuditIntegrityError("보호 파일이 변경되었습니다.")
        results = {}
        for gate_id in self.registry:
            started = time.monotonic()
            try:
                value = checks[gate_id]()
                if value is not None and value is not True:
                    raise AssertionError(f"gate는 None 또는 True로 통과해야 합니다: {value!r}")
                results[gate_id] = {"passed": True, "error": None}
            except Exception as exc:
                results[gate_id] = {"passed": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()}
            results[gate_id]["elapsed_seconds"] = time.monotonic() - started
        record = {"results": results, "passed": sum(r["passed"] for r in results.values()),
                  "total": len(self.registry), "all_passed": all(r["passed"] for r in results.values())}
        self._append("gate_run", loop_id, record)
        return record

    def record_bug(self, loop_id: str, bug_id: str, title: str,
                   reproduction_evidence_id: str, severity: str = "P0",
                   silent_wrong_value: bool = True) -> None:
        """재현 테스트가 실제 실패한 결함만 등록한다. 주입 결함은 별도 집계한다."""
        self._active(loop_id)
        evidence = self._evidence(reproduction_evidence_id, loop_id)
        data = evidence["data"]
        if data["returncode"] in (None, 0) or not data["nonce_seen"]:
            raise ValueError("난스가 확인되고 실패한 실제 재현 테스트가 필요합니다.")
        if not bug_id or any(e["kind"] == "bug_registered" and e["data"]["bug_id"] == bug_id
                             for e in self.events()):
            raise ValueError("새 고유 Bug ID가 필요합니다.")
        if severity == "P0" and not silent_wrong_value:
            raise ValueError("이 프로젝트의 P0는 조용한 잘못된 값 재현이 필요합니다.")
        self._append("bug_registered", loop_id,
                     {"bug_id": bug_id, "title": title, "severity": severity,
                      "silent_wrong_value": bool(silent_wrong_value),
                      "reproduction_evidence_id": reproduction_evidence_id})

    def resolve_bug(self, loop_id: str, bug_id: str, fix_evidence_id: str) -> None:
        self._active(loop_id)
        known = [e for e in self.events() if e["kind"] == "bug_registered"
                 and e["data"]["bug_id"] == bug_id]
        if len(known) != 1:
            raise ValueError("등록된 결함만 해결 처리할 수 있습니다.")
        if any(e["kind"] == "bug_resolved" and e["data"]["bug_id"] == bug_id for e in self.events()):
            raise ValueError("이미 해결 처리한 Bug ID입니다.")
        fixed = self._evidence(fix_evidence_id, loop_id)["data"]
        if not fixed["success"] or not fixed["nonce_seen"]:
            raise ValueError("수정 후 실제 통과한 난스 포함 재현 테스트가 필요합니다.")
        before_id = known[0]["data"]["reproduction_evidence_id"]
        commands = {e["data"]["evidence_id"]: e["data"] for e in self.events()
                    if e["kind"] == "command_start"}
        if (commands[before_id]["argv"] != commands[fix_evidence_id]["argv"]
                or commands[before_id]["cwd"] != commands[fix_evidence_id]["cwd"]):
            raise ValueError("수정 전후 동일한 재현 명령과 작업 경로를 사용해야 합니다.")
        self.verify(verify_evidence=True)
        starts = {e["loop_id"]: e["data"] for e in self.events() if e["kind"] == "loop_start"}
        if starts[known[0]["loop_id"]]["injected"] != starts[loop_id]["injected"]:
            raise ValueError("주입 시험과 실제 결함의 증거를 혼합할 수 없습니다.")
        self._append("bug_resolved", loop_id, {"bug_id": bug_id, "fix_evidence_id": fix_evidence_id})

    def end_loop(self, loop_id: str, status: str = "completed",
                 details: Mapping[str, Any] | None = None) -> None:
        self._active(loop_id)
        if status not in ("completed", "failed"):
            raise ValueError("status는 completed 또는 failed입니다.")
        if self.check_protected(loop_id):
            self._append("loop_end", loop_id, {"status": status, **dict(details or {})})

    def abort_loop(self, loop_id: str, reason: str, evidence_id: str | None = None,
                   category: str = "verification_failure") -> None:
        self._active(loop_id)
        self._append("loop_abort", loop_id,
                     {"reason": reason, "evidence_id": evidence_id, "category": category})

    def summary(self) -> dict[str, Any]:
        events = self.events()
        starts = {e["loop_id"]: e["data"] for e in events if e["kind"] == "loop_start"}

        def tally(injected: bool) -> dict[str, Any]:
            ids = {key for key, value in starts.items() if value["injected"] is injected}
            selected = [e for e in events if e["loop_id"] in ids]
            gates = [e["data"] for e in selected if e["kind"] == "gate_run"]
            p0 = {e["data"]["bug_id"] for e in selected
                  if e["kind"] == "bug_registered" and e["data"]["severity"] == "P0"}
            resolved = {e["data"]["bug_id"] for e in selected if e["kind"] == "bug_resolved"}
            ended = {e["loop_id"] for e in selected if e["kind"] in ("loop_end", "loop_abort")}
            rejected = {e["loop_id"] for e in selected if e["kind"] == "evidence_rejected"}
            return {"loops_started": len(ids),
                    "loops_completed": sum(e["kind"] == "loop_end" and e["data"]["status"] == "completed"
                                           for e in selected),
                    "loops_failed": sum(e["kind"] == "loop_end" and e["data"]["status"] == "failed"
                                        for e in selected),
                    "loops_aborted": sum(e["kind"] == "loop_abort" for e in selected),
                    "loops_open": len(ids - ended),
                    "evidence_rejected_loops": len(rejected),
                    "gate_initial_passed": gates[0]["passed"] if gates else None,
                    "gate_final_passed": gates[-1]["passed"] if gates else None,
                    "gate_total": len(self.registry), "gate_runs": len(gates),
                    "p0_discovered": len(p0), "p0_resolved": len(p0 & resolved)}

        return {"schema_version": 1, "actual": tally(False), "injected_tests": tally(True),
                "integrity": self.verify(),
                "interpretation": {"A": "actual.loops_started (Epoch 아님)",
                                   "B": "actual.evidence_rejected_loops (의도적 기만 확정 아님)",
                                   "C1": "actual.gate_initial_passed", "C2": "actual.gate_final_passed",
                                   "D_total": "actual.p0_discovered", "D_fixed": "actual.p0_resolved",
                                   "not_run": "gate_initial/final_passed=null은 미실행; 0 통과와 다름"}}
