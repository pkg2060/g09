"""요구사항 계약시험. 실제 GPU 성능 평가와 합성/모의 계약시험을 구분한다.

python -m unittest src.test_contracts
python -m unittest src.test_contracts.AuditTests  # 감사 자체는 표준 라이브러리만

ML 라이브러리가 없으면 해당 시험은 실패한다. skip을 통과로 집계하지 않는다.
GATE_REGISTRY와 gate_checks()를 AuditJournal.run_gates에 전달할 수 있다.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from .audit import AuditIntegrityError, AuditJournal, file_sha256


GATE_REGISTRY = {
    "G01": "정규화 fit은 시간순 Train 앞 80%에만 적용",
    "G02": "Eval 센서 정규화는 정확히 한 번이며 재fit 없음",
    "G03": "모든 현재 시점 복원·시작행 보존·분할 및 시간 gap 경계 격리",
    "G04": "단일 Tag/시점 strict 초과 즉시 경보 및 Overall OR",
    "G05": "원단위 공정 정지 OR/AND·경계값·억제 동작",
    "G06": "정답 Label/Inhibit 변조가 실제 고정 모델 예측을 바꾸지 않음",
    "G07": "Eval 모델 상태와 checkpoint/scaler 파일 불변",
    "G08": "전체 Test 행·Tag 분모와 미확인 Label 집계 정직성",
    "G09": "정답을 이용한 경보 구간 보정 금지",
    "G10": "잘못된 데이터에서 원본 행을 임의삭제하지 않고 명시적 실패",
    "G11": "센서 기반 정지 판단과 채점용 Inhibit의 분리",
    "G12": "감사 체인·실행 난스·주장 대조·주입 시험 분리",
}


class AuditTests(unittest.TestCase):
    """표준 라이브러리만 사용하는 감사 기능 회귀시험."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.journal = AuditJournal(self.root, {"G1": "실행한 검사만 통과"})

    def command(self, exit_code=0, nonce=True):
        program = "import os,sys; "
        if nonce:
            program += "print('EWS_AUDIT_NONCE='+os.environ['EWS_AUDIT_NONCE']); "
        return [sys.executable, "-c", program + f"sys.exit({exit_code})"]

    def test_actual_subprocess_nonce_and_claim(self):
        loop = self.journal.start_loop("실제 subprocess 검증")
        evidence = self.journal.run_command(loop, self.command())
        self.assertTrue(evidence["success"])
        self.assertTrue(self.journal.record_claim(loop, evidence["evidence_id"], True))
        self.journal.end_loop(loop)
        summary = self.journal.summary()["actual"]
        self.assertEqual(summary["loops_completed"], 1)
        self.assertEqual(summary["evidence_rejected_loops"], 0)
        self.assertIsNone(summary["gate_initial_passed"])
        self.assertEqual(summary["p0_discovered"], 0)
        self.assertTrue(self.journal.verify(verify_evidence=True)["valid"])

    def test_ordinary_failure_is_not_a_dishonesty_signal(self):
        loop = self.journal.start_loop("일반 검증 실패")
        evidence = self.journal.run_command(loop, self.command(3))
        self.assertTrue(self.journal.record_claim(loop, evidence["evidence_id"], False))
        self.journal.end_loop(loop, "failed")
        self.assertEqual(self.journal.summary()["actual"]["evidence_rejected_loops"], 0)

    def test_false_success_claim_aborts_loop(self):
        loop = self.journal.start_loop("실패를 성공이라고 주장한 시험", injected=True)
        evidence = self.journal.run_command(loop, self.command(3))
        self.assertFalse(self.journal.record_claim(loop, evidence["evidence_id"], True))
        summary = self.journal.summary()
        self.assertEqual(summary["actual"]["evidence_rejected_loops"], 0)
        self.assertEqual(summary["injected_tests"]["evidence_rejected_loops"], 1)
        self.assertEqual(summary["injected_tests"]["loops_aborted"], 1)
        with self.assertRaises(ValueError):
            self.journal.end_loop(loop)

    def test_missing_nonce_is_not_success_evidence(self):
        loop = self.journal.start_loop("난스 누락 주입", injected=True)
        evidence = self.journal.run_command(loop, self.command(nonce=False))
        self.assertEqual(evidence["returncode"], 0)
        self.assertFalse(evidence["success"])
        self.assertFalse(self.journal.record_claim(loop, evidence["evidence_id"], True))

    def test_old_loop_evidence_cannot_support_current_claim(self):
        first = self.journal.start_loop("이전 증거", injected=True)
        evidence = self.journal.run_command(first, self.command())
        self.journal.end_loop(first)
        second = self.journal.start_loop("과거 로그 재사용 주입", injected=True)
        self.assertFalse(self.journal.record_claim(second, evidence["evidence_id"], True))
        self.assertEqual(self.journal.summary()["injected_tests"]["evidence_rejected_loops"], 1)

    def test_modified_log_and_chain_are_detected(self):
        loop = self.journal.start_loop("증거 파일 변경 주입", injected=True)
        evidence = self.journal.run_command(loop, self.command())
        log = self.journal.directory / evidence["stdout"]["path"]
        log.write_text("invented output", encoding="utf-8")
        with self.assertRaises(AuditIntegrityError):
            self.journal.verify(verify_evidence=True)
        self.assertFalse(self.journal.record_claim(loop, evidence["evidence_id"], True))
        raw = self.journal.path.read_text(encoding="utf-8")
        self.journal.path.write_text(raw.replace('"injected":true', '"injected":false', 1), encoding="utf-8")
        with self.assertRaises(AuditIntegrityError):
            self.journal.verify()

    def test_registry_is_fixed_and_all_gates_must_execute(self):
        with self.assertRaises(AuditIntegrityError):
            AuditJournal(self.root, {"G1": "완화한 검사"})
        loop = self.journal.start_loop("실행 게이트")
        with self.assertRaises(ValueError):
            self.journal.run_gates(loop, {})
        before = self.journal.run_gates(loop, {"G1": lambda: False})
        self.assertEqual(before["passed"], 0)
        after = self.journal.run_gates(loop, {"G1": lambda: True})
        self.assertEqual(after["passed"], 1)
        self.journal.end_loop(loop)
        summary = self.journal.summary()["actual"]
        self.assertEqual((summary["gate_initial_passed"], summary["gate_final_passed"]), (0, 1))

    def test_protected_file_change_stops_loop(self):
        root = self.root / "protected_case"
        root.mkdir()
        guard = root / "guard.txt"
        guard.write_text("fixed rules", encoding="utf-8")
        journal = AuditJournal(root, {"G1": "보호 검사"}, [guard])
        loop = journal.start_loop("보호 파일 변경 주입", injected=True)
        guard.write_text("weakened rules", encoding="utf-8")
        self.assertFalse(journal.check_protected(loop))
        self.assertEqual(journal.summary()["injected_tests"]["loops_aborted"], 1)

    def test_bug_counts_require_same_real_reproduction_command(self):
        loop = self.journal.start_loop("P0 재현/해결 주입", injected=True)
        with self.assertRaises(ValueError):
            self.journal.record_bug(loop, "P0-MISSING", "증거 없음", "invented")
        state = self.root / "fixture_state.txt"
        state.write_text("broken", encoding="utf-8")
        command = [sys.executable, "-c", "import os,pathlib,sys; "
                   "print('EWS_AUDIT_NONCE='+os.environ['EWS_AUDIT_NONCE']); "
                   "sys.exit(0 if pathlib.Path('fixture_state.txt').read_text()=='fixed' else 1)"]
        failed = self.journal.run_command(loop, command)
        self.journal.record_bug(loop, "P0-INJECTED", "주입한 잘못된 값", failed["evidence_id"])
        unrelated = self.journal.run_command(loop, self.command())
        with self.assertRaises(ValueError):
            self.journal.resolve_bug(loop, "P0-INJECTED", unrelated["evidence_id"])
        state.write_text("fixed", encoding="utf-8")
        passed = self.journal.run_command(loop, command)
        self.journal.resolve_bug(loop, "P0-INJECTED", passed["evidence_id"])
        self.journal.end_loop(loop)
        summary = self.journal.summary()
        self.assertEqual(summary["actual"]["p0_discovered"], 0)
        self.assertEqual(summary["injected_tests"]["p0_discovered"], 1)
        self.assertEqual(summary["injected_tests"]["p0_resolved"], 1)

    def test_verify_wrapper_propagates_failed_gate_exit_code(self):
        """결과 dict가 비어 있지 않아도 all_passed=False이면 반드시 exit 1."""
        from . import verify as verification
        journal = Mock()
        journal.start_loop.return_value = "mocked-wrapper-contract"
        journal.run_gates.return_value = {
            "all_passed": False, "passed": 11, "total": 12,
            "results": {"G03": {"passed": False, "error": "injected failure"}},
        }
        with patch.object(verification, "AuditJournal", return_value=journal), \
                patch.object(verification, "gate_checks", return_value={"G03": Mock()}), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                verification.verify()
        self.assertEqual(raised.exception.code, 1)
        journal.abort_loop.assert_called_once()
        journal.end_loop.assert_not_called()
        journal.summary.assert_not_called()

    def test_verify_protects_static_sources_but_excludes_notebook_autosaves(self):
        from . import verify as verification
        project = self.root / "verify_source_fixture"
        source = project / "src"
        vendor = source / "vendor"
        vendor.mkdir(parents=True)
        entry = project / "mode.py"
        verifier = source / "verify.py"
        worker = source / "worker.py"
        external = vendor / "spot.py"
        license_file = vendor / "LICENSE.GPL-3.0"
        assets = source / "dashboard_assets"
        assets.mkdir()
        asset = assets / "dashboard.js"
        requirements = source / "requirements.txt"
        protected = (entry, verifier, worker, external, license_file, asset, requirements)
        for path in protected:
            path.write_text("# unchanged source\n", encoding="utf-8")
        notebook = source / "실행.ipynb"
        notebook.write_text("{}", encoding="utf-8")
        cache = source / "__pycache__"
        cache.mkdir()
        bytecode = cache / "worker.pyc"
        bytecode.write_bytes(b"cached code")
        autosave = source / ".ipynb_checkpoints"
        autosave.mkdir()
        checkpoint_python = autosave / "worker-checkpoint.py"
        checkpoint_python.write_text("# autosave", encoding="utf-8")
        registry = {"G1": "고정 소스 계약"}
        with patch.object(verification, "__file__", str(verifier)), \
                patch.object(verification, "GATE_REGISTRY", registry), \
                patch.object(verification, "gate_checks", return_value={"G1": lambda: True}), \
                contextlib.redirect_stdout(io.StringIO()):
            result = verification.verify()
            self.assertTrue(result["all_passed"])
            # Optimizer-style reopening without a new list must retain protection.
            reopened = AuditJournal(project, registry)
            expected = {str(path.relative_to(project)) for path in protected}
            self.assertEqual(set(reopened.protected), expected)
            notebook.write_text('{"outputs": ["new cell result"]}', encoding="utf-8")
            bytecode.write_bytes(b"updated cache")
            checkpoint_python.write_text("# updated autosave", encoding="utf-8")
            self.assertTrue(verification.verify()["all_passed"])
            external.write_text("# changed source\n", encoding="utf-8")
            with self.assertRaises(AuditIntegrityError):
                verification.verify()
            loop = reopened.start_loop  # mutation must also be rejected on loop entry
            with self.assertRaises(AuditIntegrityError):
                loop("source mutation injection", injected=True)


class PipelineContracts(unittest.TestCase):
    """작은 합성 데이터로 데이터/점수/모델/평가의 연결 계약을 확인한다."""

    def setUp(self):
        import numpy as np
        import pandas as pd
        import torch
        from .config import get_config
        self.np, self.pd, self.torch = np, pd, torch
        self.config = get_config()
        self.config["train"]["num_workers"] = 0
        self.config["model"].update(window=3, dropout=0.0)
        self.config["report"].update(export_database=False, dashboard=False, static_plots=False)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.features = ["PDC4303_7.PV", "PPVI43032A.PV"]

    def times(self, n, start="2026-01-01"):
        return self.pd.date_range(start, periods=n, freq="min").to_numpy()

    def fixture(self):
        from .data import fit_training_scaler
        np, pd = self.np, self.pd
        train_values = np.array([[0, 0], [10, 20], [2, 4], [8, 16], [100, 200]], dtype=np.float32)
        scaler, cut = fit_training_scaler(train_values, .2)
        raw = np.array([[3, 6], [.2, 6], [3, 2.5], [8, 16]], dtype=np.float32)
        frame = pd.DataFrame(raw, columns=self.features)
        frame["Inhibit"] = [1, 0, 0, 1]  # intentionally differs from sensor truth
        for tag in self.features:
            frame["Alarm_" + tag] = [1, 0, 1, 0]
        train = {"values": train_values, "timestamps": self.times(5), "features": self.features}
        test = {"values": raw, "timestamps": self.times(4, "2026-02-01"),
                "features": self.features, "frame": frame,
                "labels": np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float),
                "inhibit": frame["Inhibit"].to_numpy(dtype=float)}
        return train, test, {"scaler": scaler, "features": self.features, "training_rows": cut}

    def evaluate_case(self, model=None, fixture_data=None):
        """실제 전처리/채점을 사용하며 GPU 추론과 그래프만 명시적으로 대체한다."""
        from . import pipeline
        from sklearn.preprocessing import MinMaxScaler
        np = self.np
        train, test, payload = self.fixture() if fixture_data is None else fixture_data
        run = self.root / "eval"
        run.mkdir()
        calls = []

        def fake_reconstruct(model_argument, loader, device):
            if model_argument is not model:
                raise AssertionError("Unexpected model instance")
            values = np.concatenate([targets.numpy() for _, targets in loader], axis=0)
            calls.append(values.copy())
            return np.zeros_like(values)

        transform = Mock(wraps=payload["scaler"].transform)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(MinMaxScaler, "fit", side_effect=AssertionError("Eval refit")))
            stack.enter_context(patch.object(MinMaxScaler, "partial_fit", side_effect=AssertionError("Eval partial_fit")))
            stack.enter_context(patch.object(payload["scaler"], "transform", transform))
            stack.enter_context(patch.object(pipeline, "reconstruct", side_effect=fake_reconstruct))
            stack.enter_context(patch("src.reports.create_reports", return_value=None))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            metrics = pipeline.evaluate(model, payload, train, test, self.config, run, self.torch.device("cpu"))
        return train, test, payload, metrics, calls, transform, run

    def test_G01_scaler_training_partition_only(self):
        train, test, payload = self.fixture()
        scaler = payload["scaler"]
        self.assertEqual(payload["training_rows"], 4)
        self.np.testing.assert_array_equal(scaler.data_max_, [10, 20])
        self.np.testing.assert_array_equal(scaler.transform(train["values"][-1:]), [[10, 10]])
        self.assertFalse(scaler.clip)

    def test_G02_eval_normalizes_exactly_once(self):
        _, test, payload, _, calls, spy, _ = self.evaluate_case()
        self.assertEqual(spy.call_count, 1)
        self.np.testing.assert_array_equal(spy.call_args.args[0], test["values"])
        self.np.testing.assert_allclose(calls[0], test["values"] / [10, 20], atol=1e-7)

    def test_G03_current_timestamp_and_segment_boundaries(self):
        from .data import WindowDataset
        np = self.np
        values = np.arange(12, dtype=np.float32).reshape(6, 2)
        times = self.times(6).copy()  # pandas may expose a read-only datetime array
        times[3:] += np.timedelta64(60, "m")
        dataset = WindowDataset(values, times, window=3)
        self.assertEqual(len(dataset), len(values))
        for i in range(len(dataset)):
            window, target = dataset[i]
            np.testing.assert_array_equal(window[-1].numpy(), values[i])
            np.testing.assert_array_equal(target.numpy(), values[i])
            self.assertFalse(np.any(window.numpy() > values[i]))
        np.testing.assert_array_equal(dataset[0][0], np.repeat(values[:1], 3, axis=0))
        np.testing.assert_array_equal(dataset[3][0], np.repeat(values[3:4], 3, axis=0))
        validation = WindowDataset(values[4:], times[4:], window=3)
        np.testing.assert_array_equal(validation[0][0], np.repeat(values[4:5], 3, axis=0))

    def test_G04_threshold_strict_and_single_point(self):
        from .detection import generate_alarms
        np = self.np
        limit = .2
        scores = np.array([[limit, limit], [np.nextafter(limit, np.inf), 0], [0, .3], [.4, .4]])
        alarms, overall = generate_alarms(scores, np.array([limit, limit]), np.array([False, False, False, True]))
        np.testing.assert_array_equal(alarms, [[False, False], [True, False], [False, True], [False, False]])
        np.testing.assert_array_equal(overall, [False, True, True, False])

    def test_G05_stop_or_and_raw_unit_boundaries(self):
        from .detection import process_stop, generate_alarms
        np, pd = self.np, self.pd
        raw = pd.DataFrame({self.features[0]: [1, .2, 1, .2], self.features[1]: [3, 3, 2.5, 2.5]})
        expected = {"or": [False, True, True, True], "and": [False, False, False, True]}
        for mode in expected:
            config = copy.deepcopy(self.config["alarm"])
            config["mode"] = mode
            config["release_hold_minutes"] = 0  # original raw-condition boundary contract
            stop = process_stop(raw, config)
            np.testing.assert_array_equal(stop, expected[mode])
            alarms, _ = generate_alarms(np.ones((4, 2)), np.array([.2, .2]), stop)
            self.assertEqual(int(alarms[stop].sum()), 0)

    def test_G05_release_hold_is_causal_and_stops_at_time_gaps(self):
        from .detection import process_stop
        np, pd = self.np, self.pd
        cfg = {'enabled': True, 'mode': 'or', 'release_hold_minutes': 1,
               'conditions': [{'tag': 'RUN.PV', 'op': '<=', 'value': 0}]}
        raw = pd.DataFrame({'RUN.PV': [1., 0., 1., 1., 1., 0., 1., 1.]})
        times = self.times(8).copy()
        times[6:] += np.timedelta64(10, 'm')
        stop = process_stop(raw, cfg, times)
        np.testing.assert_array_equal(stop, [False, True, True, False, False, True, False, False])
        self.assertFalse(stop[0])  # the future stop at minute1 cannot suppress minute0
        self.assertFalse(stop[4])  # likewise before the later stop transition
        self.assertFalse(stop[6])  # previous row stopped, but it is eleven minutes ago
        np.testing.assert_array_equal(process_stop(raw.iloc[:5], cfg, times[:5]), stop[:5])
        raw_only = process_stop(raw, {**cfg, 'release_hold_minutes': 0})
        np.testing.assert_array_equal(raw_only, raw['RUN.PV'].to_numpy() <= 0)

    def test_G05_release_hold_requires_actual_ordered_one_minute_predecessor(self):
        from .detection import apply_release_hold
        np = self.np
        raw = np.array([True, False, False, False])
        times = self.times(4)
        np.testing.assert_array_equal(apply_release_hold(raw, times, 2), [True, True, True, False])
        subminute = np.array(['2026-01-01T00:00:00', '2026-01-01T00:00:30',
                              '2026-01-01T00:01:30'], dtype='datetime64[us]')
        np.testing.assert_array_equal(apply_release_hold(raw[:3], subminute, 1), [True, False, False])
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            apply_release_hold(raw, None, 1)
        for invalid in (times[[0, 0, 2, 3]], times[::-1], np.array(['NaT']*4, dtype='datetime64[ns]')):
            with self.assertRaises(ValueError):
                apply_release_hold(raw, invalid, 1)
        for invalid_hold in (-1, .5, True, float('nan')):
            with self.assertRaises(ValueError):
                apply_release_hold(raw, times, invalid_hold)

    def test_G06_truth_shuffle_leaves_real_model_outputs_unchanged(self):
        from .data import read_sensor_csv, WindowDataset
        from .model import build_model
        from .detection import point_scores, process_stop, generate_alarms
        np, torch = self.np, self.torch
        _, source, payload = self.fixture()
        original = source["frame"].copy()
        original.insert(0, "DateTime", source["timestamps"])
        changed = original.copy()
        for column in ["Inhibit"] + ["Alarm_" + tag for tag in self.features]:
            changed[column] = 1 - changed[column].to_numpy()
            self.assertFalse(np.array_equal(changed[column], original[column]))
        first, second = self.root / "original.csv", self.root / "changed.csv"
        original.to_csv(first, index=False)
        changed.to_csv(second, index=False)
        model = build_model(2, self.config["model"]).eval()
        snapshots = []
        for path in (first, second):
            data = read_sensor_csv(path, self.config["data"])
            self.assertEqual(data["features"], self.features)
            values = payload["scaler"].transform(data["values"]).astype(np.float32)
            dataset = WindowDataset(values, data["timestamps"], 3)
            inputs = torch.stack([dataset[i][0] for i in range(len(dataset))])
            with torch.no_grad():
                reconstruction = model(inputs)[1].numpy()
            scores = point_scores(values, reconstruction)
            stop = process_stop(data["frame"][self.features], self.config["alarm"], data["timestamps"])
            alarms, overall = generate_alarms(scores, np.array([.2, .2]), stop)
            snapshots.append((reconstruction, scores, stop, alarms, overall))
        for first_result, second_result in zip(*snapshots):
            np.testing.assert_array_equal(first_result, second_result)

    def test_G07_eval_model_checkpoint_scaler_immutable(self):
        from .model import build_model
        from .data import save_scaler, load_scaler
        from .trainer import load_trusted_checkpoint
        model = build_model(2, self.config["model"]).eval()
        before_parameters = {name: value.clone() for name, value in model.state_dict().items()}
        checkpoint = self.root / "best.pt"
        self.torch.save({"model_state_dict": model.state_dict()}, checkpoint)
        train, test, payload = self.fixture()
        scaler_file = self.root / "scaler.pkl"
        save_scaler(payload["scaler"], self.features, payload["training_rows"], scaler_file)
        hashes = (file_sha256(checkpoint), file_sha256(scaler_file))
        restored = load_scaler(scaler_file)
        loaded = load_trusted_checkpoint(checkpoint)
        model.load_state_dict(loaded["model_state_dict"])
        scaler_state = pickle.dumps(restored["scaler"].__dict__)
        # This is a component Eval contract. Full CLI/GPU evidence is separate.
        self.evaluate_case(model, (train, test, restored))
        self.assertEqual((file_sha256(checkpoint), file_sha256(scaler_file)), hashes)
        for name, before in before_parameters.items():
            self.assertTrue(self.torch.equal(model.state_dict()[name], before))
        self.assertEqual(pickle.dumps(restored["scaler"].__dict__), scaler_state)
        self.np.testing.assert_array_equal(restored["scaler"].data_max_, [10, 20])

    def test_G08_all_rows_and_missing_labels_denominators(self):
        from .detection import regression_metrics, classification_metrics
        np = self.np
        raw = np.arange(8, dtype=float).reshape(4, 2)
        recon = raw + .01
        result = regression_metrics(raw, recon, self.features)
        self.assertEqual(result["overall"]["n"], 8)
        self.assertAlmostEqual(result["overall"]["mae"], .01)
        labels = np.array([[0, 1], [1, np.nan], [0, 0], [1, 1]])
        alarms = np.zeros((4, 2), dtype=bool)
        result = classification_metrics(labels, alarms, self.times(4), self.features)
        self.assertEqual(result["overall"]["n"], 7)
        self.assertEqual(result["per_tag"][self.features[0]]["n"], 4)
        self.assertEqual(result["per_tag"][self.features[1]]["n"], 3)
        _, test, _, metrics, _, _, run = self.evaluate_case()
        self.assertEqual(metrics["denominators"]["test_rows"], 4)
        with np.load(run / "predictions.npz", allow_pickle=False) as artifact:
            self.assertEqual(len(artifact["scores"]), len(test["values"]))

    def test_G09_no_label_point_adjustment(self):
        from .detection import classification_metrics
        np = self.np
        labels = np.array([[1], [1], [1]], dtype=float)
        alarms = np.array([[False], [True], [False]])
        before = alarms.copy()
        metrics = classification_metrics(labels, alarms, self.times(3), ["A.PV"])
        self.assertEqual(metrics["overall"]["tp"], 1)
        self.assertEqual(metrics["overall"]["fn"], 2)
        self.assertEqual(metrics["per_tag"]["A.PV"]["mean_detection_delay_seconds"], 60)
        np.testing.assert_array_equal(alarms, before)

    def test_G10_invalid_input_fails_without_deletion(self):
        from .data import read_sensor_csv
        _, source, _ = self.fixture()
        frame = source["frame"].copy()
        frame.insert(0, "DateTime", source["timestamps"])
        frame.loc[1, "DateTime"] = frame.loc[0, "DateTime"]
        path = self.root / "duplicate.csv"
        frame.to_csv(path, index=False)
        digest = file_sha256(path)
        with self.assertRaises(ValueError):
            read_sensor_csv(path, self.config["data"])
        self.assertEqual(file_sha256(path), digest)
        frame["DateTime"] = source["timestamps"]
        frame.loc[1, self.features[0]] = float("nan")
        frame.to_csv(path, index=False)
        digest = file_sha256(path)
        with self.assertRaises(ValueError):
            read_sensor_csv(path, self.config["data"])
        self.assertEqual(file_sha256(path), digest)

    def test_G11_stop_groundtruth_is_scoring_only(self):
        from .detection import process_stop
        _, test, _, metrics, _, _, run = self.evaluate_case()
        self.assertFalse(metrics["stop_safety"]["truth_used_for_prediction"])
        self.assertEqual(metrics["stop_safety"]["predicted_stop_alarm_count"], 0)
        self.assertGreater(metrics["stop_safety"]["truth_stop_missed_minutes"], 0)
        with self.np.load(run / "predictions.npz", allow_pickle=False) as artifact:
            self.assertTrue(artifact["alarms"][0].any())  # Inhibit truth=1, sensors=operating
            self.assertFalse(artifact["alarms"][2].any())  # Inhibit truth=0, sensor=stopped
        cfg = {"enabled": True, "mode": "or", "conditions": [{"tag": "Inhibit", "op": "==", "value": 1}]}
        with self.assertRaises(ValueError):
            process_stop(test["frame"], cfg)

    def test_G11_unlabelled_new_tags_still_produce_scores_and_alarms(self):
        """정답/기존 정지 센서가 모두 없는 새 설비도 실제 평가 경로를 통과한다."""
        np = self.np
        train, test, payload = self.fixture()
        features = ["OTHER_TEMP.pv", "OTHER_VIB.PV"]
        test["frame"] = test["frame"][self.features].rename(columns=dict(zip(self.features, features)))
        train["features"] = test["features"] = payload["features"] = features
        test["labels"] = np.full((4, 2), np.nan)
        test["inhibit"] = np.full(4, np.nan)
        _, _, _, metrics, _, _, run = self.evaluate_case(fixture_data=(train, test, payload))
        self.assertFalse(metrics["stop_rule_status"]["applicable"])
        self.assertEqual(set(metrics["stop_rule_status"]["missing_sensor_tags"]), set(self.features))
        self.assertEqual(metrics["classification"]["overall"]["n"], 0)
        self.assertEqual(metrics["classification"]["overall"]["missing_label_count"], 8)
        for row in metrics["classification"]["per_tag"].values():
            self.assertEqual(row["n"], 0)
            self.assertIsNone(row["accuracy"])
        self.assertFalse(metrics["gate"]["classification_pass"])
        self.assertFalse(metrics["gate"]["phase2_numeric_pass"])
        self.assertTrue(all(value is None for value in metrics["gate"]["per_tag_accuracy"].values()))
        self.assertEqual(metrics["stop_validation"]["overall"]["n"], 0)
        self.assertEqual(metrics["stop_safety"]["truth_known_minutes"], 0)
        self.assertEqual(metrics["normalized_regression"]["overall"]["n"], 8)
        with np.load(run/"predictions.npz", allow_pickle=False) as artifact:
            self.assertEqual(artifact["scores"].shape, (4, 2))
            self.assertTrue(np.isfinite(artifact["scores"]).all())
            self.assertTrue(np.isnan(artifact["labels"]).all())
            self.assertTrue(np.isnan(artifact["inhibit"]).all())
            self.assertFalse(artifact["stop"].any())
            self.assertTrue(artifact["alarms"].any())
            np.testing.assert_array_equal(artifact["alarms"], artifact["scores"] > artifact["thresholds"])
        persisted = json.loads((run/"metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["classification"]["overall"]["n"], 0)

    def test_G11_missing_stop_sensors_exclude_known_inhibit_from_stop_validation(self):
        """규칙 미적용을 정상 운전 예측의 채점 성공으로 오인하지 않는다."""
        np = self.np
        train, test, payload = self.fixture()
        features = ["NEW_TEMP.PV", "NEW_VIB.PV"]
        test["frame"] = test["frame"][self.features].rename(columns=dict(zip(self.features, features)))
        train["features"] = test["features"] = payload["features"] = features
        test["inhibit"] = np.array([1., 0., 1., 0.])
        _, _, _, metrics, _, _, run = self.evaluate_case(fixture_data=(train, test, payload))
        self.assertFalse(metrics["stop_rule_status"]["applicable"])
        self.assertEqual(metrics["stop_safety"]["truth_known_minutes"], 4)
        stop_metrics = metrics["stop_validation"]["overall"]
        self.assertEqual(stop_metrics["n"], 0)
        self.assertEqual(stop_metrics["missing_label_count"], 0)
        self.assertEqual(stop_metrics["excluded_count"], 4)
        self.assertIsNone(stop_metrics["accuracy"])
        with np.load(run/"predictions.npz", allow_pickle=False) as artifact:
            self.assertFalse(artifact["stop"].any())
            self.assertTrue(artifact["alarms"][test["inhibit"] == 1].any())

    def test_G11_lowercase_sensor_names_apply_uppercase_stop_rules(self):
        np = self.np
        self.config['alarm']['release_hold_minutes'] = 0  # isolate raw-name mapping contract
        train, test, payload = self.fixture()
        features = [tag.lower() for tag in self.features]
        test["frame"] = test["frame"].rename(columns=dict(zip(self.features, features)))
        # CSV의 원단위 DataFrame처럼 float64로 유지한다. float32의 .2를
        # float64로 넓히면 .20000000298이므로 exact .2 경계와 다른 값이다.
        test["frame"][features] = test["frame"][features].astype(float)
        test["frame"].loc[1, features[0]] = .2
        train["features"] = test["features"] = payload["features"] = features
        _, _, _, metrics, _, _, run = self.evaluate_case(fixture_data=(train, test, payload))
        status = metrics["stop_rule_status"]
        self.assertTrue(status["applicable"])
        self.assertEqual(status["missing_sensor_tags"], [])
        self.assertEqual([condition["tag"] for condition in status["resolved_conditions"]], features)
        self.assertEqual(metrics["stop_validation"]["overall"]["n"], 4)
        with np.load(run/"predictions.npz", allow_pickle=False) as artifact:
            np.testing.assert_array_equal(artifact["stop"], [False, True, True, False])
            self.assertFalse(artifact["alarms"][[1, 2]].any())
            self.assertTrue(artifact["alarms"][[0, 3]].any())

    def test_G11_default_release_hold_is_logged_and_uses_only_sensor_history(self):
        np = self.np
        self.assertEqual(self.config['alarm']['release_hold_minutes'], 1)
        train, test, payload = self.fixture()
        test['frame'][self.features] = test['frame'][self.features].astype(float)
        test['frame'].loc[1, self.features[0]] = .2
        _, _, _, metrics, _, _, run = self.evaluate_case(fixture_data=(train, test, payload))
        status = metrics['stop_rule_status']
        self.assertEqual(status['release_hold_minutes'], 1)
        self.assertTrue(status['release_hold_applied'])
        self.assertEqual(status['raw_sensor_stop_minutes'], 2)
        self.assertEqual(status['additional_release_hold_minutes'], 1)
        self.assertFalse(status['hold_propagates_across_gap'])
        self.assertFalse(status['uses_future_values'])
        self.assertFalse(status['truth_used'])
        with np.load(run/'predictions.npz', allow_pickle=False) as artifact:
            np.testing.assert_array_equal(artifact['stop'], [False, True, True, True])
            self.assertTrue(artifact['alarms'][0].any())  # Inhibit=1 is still not an input
            self.assertFalse(artifact['alarms'][1:].any())


def _run_case(case_type, method=None):
    if isinstance(method, (tuple, list)):
        suite = unittest.TestSuite([case_type(name) for name in method])
    else:
        suite = (unittest.TestSuite([case_type(method)]) if method else
                 unittest.defaultTestLoader.loadTestsFromTestCase(case_type))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise AssertionError(stream.getvalue())


def gate_checks():
    """고정 registry에 대응하는 실제 실행 함수를 반환한다."""
    methods = {}
    for name in sorted(dir(PipelineContracts)):
        if name.startswith("test_G"):
            methods.setdefault(name.split("_")[1], []).append(name)
    checks = {gate: (lambda grouped=tuple(grouped): _run_case(PipelineContracts, grouped))
              for gate, grouped in methods.items()}
    checks["G12"] = lambda: _run_case(AuditTests)
    return checks


if __name__ == "__main__":
    nonce = os.environ.get("EWS_AUDIT_NONCE")
    if nonce:
        print(f"EWS_AUDIT_NONCE={nonce}", flush=True)
    unittest.main()
