"""새 CSV 관측값만 추론하고 윈도우·점수·알람 상태와 결과를 원자적으로 저장한다."""

import csv
import fcntl
import json
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from models.base import checkpoint_value

from .alarm import AlarmGenerator
from .dataloader import build_prediction_data
from .evaluation import evaluate_predictions
from .preprocessing import OnlinePreprocessor
from .save import ALARM_COLUMNS, save_report
from .scoring import calibration_thresholds
from .udf.postprocessing import ScoreProcessor


def encode(value):
    """배열·시각을 코드 실행 없이 복원할 수 있는 JSON으로 직렬화한다."""
    return json.dumps(
        checkpoint_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        default=lambda item: [timestamp.isoformat() for timestamp in item]
        if isinstance(item, pd.DatetimeIndex)
        else str(item),
    )


@contextmanager
def output_lock(directory):
    """같은 출력 폴더에 두 온라인 실행이 동시에 쓰지 못하게 한다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / ".online.lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("같은 출력 폴더에서 online이 이미 실행 중입니다") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class OnlineSession:
    """제한된 길이의 입력 윈도우와 온라인 판정 상태를 유지한다."""

    def __init__(self, model, calibration, settings, state=None):
        """SQLite에 확정된 상태로부터 이전 실행을 이어간다."""
        self.model, self.calibration = model, calibration
        self.settings = deepcopy(settings)
        # 지표는 저장된 전체 결과에서 한 번 계산하고 모델은 새 시점만 추론한다.
        self.settings["evaluation"]["enabled"] = False
        pipeline = model.data_state["metadata"]["pipeline"]
        self.timestamp_col = pipeline["data"]["timestamp_column"]
        interval = pipeline["data"]["expected_interval_seconds"]
        if interval is None or interval <= 0:
            raise ValueError("온라인에는 양의 수집 간격이 필요합니다")
        self.max_gap = interval * pipeline["data"]["max_gap_multiplier"]
        self.window_size = model.n_window
        self.pad_start = pipeline["dataloader"]["pad_start"]
        state = state or {}
        self.preprocessor = OnlinePreprocessor(pipeline, model.data_state["feature_names"], state.get("preprocessing"))
        self.buffer = state.get("buffer", [])
        self.processed_rows = state.get("processed_rows", 0)
        self.segment = state.get("segment", 0)
        self.last_timestamp = state.get("last_timestamp")
        self.processor = ScoreProcessor(**calibration["score_processor"])
        for row in state.get("score_history", []):
            values = np.asarray(row, dtype=float)
            if values.shape != (model.n_feats,) or not np.isfinite(values).all():
                raise ValueError("온라인 점수 이력이 특성 수와 맞지 않습니다")
            self.processor.history.append(values)
        self.processor.previous_segment = state.get("score_segment")
        feature_names = model.data_state["feature_names"]
        self.alarms = AlarmGenerator(
            calibration_thresholds(calibration, feature_names), feature_names, settings["alarm"]
        )
        if state.get("alarms"):
            self.alarms.load_state_dict(state["alarms"])
        if len(self.buffer) > self.window_size or self.processed_rows < len(self.buffer):
            raise ValueError("온라인 윈도우 상태가 잘못되었습니다")

    def instrument_fault_events(self, row, timestamp):
        """운전 중 원본 0·결측값을 계기 이상 이벤트로 변환한다."""
        raw = pd.to_numeric(pd.Series([row[name] for name in self.model.data_state["feature_names"]]), errors="coerce")
        return self.alarms.instrument_fault_events(raw.to_numpy(dtype=float), timestamp, include_zero=True)

    def consume(self, row, process_data):
        """계기 이상을 우선 판정하고 유효한 관측만 공통 Lightning 추론 경로로 전달한다."""
        timestamp = pd.Timestamp(row[self.timestamp_col])
        if pd.isna(timestamp) or (self.last_timestamp is not None and timestamp <= pd.Timestamp(self.last_timestamp)):
            raise ValueError("온라인 CSV는 유효한 시각을 중복 없이 오름차순으로 추가해야 합니다")
        self.last_timestamp = timestamp.isoformat()
        events = self.instrument_fault_events(row, timestamp)
        if events:
            self.buffer.clear()
            self.segment += 1
            self.preprocessor.reset()
            self.processor.reset()
            self.alarms.reset()
            return None, events
        prepared = self.preprocessor.process(row)
        if prepared is None:
            self.buffer.clear()
            self.segment += 1
            self.processor.reset()
            self.alarms.reset()
            return None, []
        if (
            self.buffer
            and (timestamp - pd.Timestamp(self.buffer[-1][self.timestamp_col])).total_seconds() > self.max_gap
        ):
            self.buffer.clear()
            self.segment += 1
        prepared[self.timestamp_col] = timestamp.isoformat()
        self.buffer.append(prepared)
        self.buffer = self.buffer[-self.window_size :]
        self.processed_rows += 1
        if not self.pad_start and len(self.buffer) < self.window_size:
            return None, []
        frame = pd.DataFrame(self.buffer)
        frame[self.timestamp_col] = pd.to_datetime(frame[self.timestamp_col])
        data = build_prediction_data(
            frame, self.model.data_state, self.model.model_config, preprocessed=True, last_only=True
        )
        return process_data(
            self.model,
            data,
            self.calibration,
            self.settings,
            processor=self.processor,
            alarm_generator=self.alarms,
            index_offset=self.processed_rows - len(self.buffer),
            segment=self.segment,
        ), []

    def state_dict(self):
        """학습 데이터나 전체 입력 대신 다음 관측 처리에 필요한 상태만 반환한다."""
        return {
            "buffer": self.buffer,
            "processed_rows": self.processed_rows,
            "segment": self.segment,
            "last_timestamp": self.last_timestamp,
            "preprocessing": self.preprocessor.state_dict(),
            "score_history": list(self.processor.history),
            "score_segment": self.processor.previous_segment,
            "alarms": self.alarms.state_dict(),
        }


def merge_results(rows):
    """확정된 시점별 결과를 결합하며 모델 추론과 점수·알람 판정을 반복하지 않는다."""
    result = dict(rows[0])
    for key in ("prediction", "target", "raw_score", "score", "label", "index", "segments", "quantiles"):
        result[key] = None if rows[0].get(key) is None else np.concatenate([row[key] for row in rows])
    result["timestamp"] = pd.to_datetime(np.concatenate([row["timestamp"] for row in rows]))
    result["thresholds"] = np.asarray(result["thresholds"])
    for key in ("all_scores", "source_columns"):
        result[key] = {name: np.concatenate([row[key][name] for row in rows]) for name in rows[0][key]}
    result["alarms"] = {
        key: np.concatenate([row["alarms"][key] for row in rows])
        for key in (
            "per_variable",
            "overall",
            "levels",
            "inhibited",
            "crv_per_variable",
            "crv_overall",
            "crv_directions",
        )
    }
    result["alarms"]["events"] = [event for row in rows for event in row["alarms"]["events"]]
    return result


def export_results(connection, settings, directory):
    """확정된 결과에서 기존 보고서 형식을 갱신하며 중단 시 다음 실행에서 복구한다."""
    rows = [json.loads(value) for (value,) in connection.execute("SELECT payload FROM predictions ORDER BY row_index")]
    instrument_events = [
        json.loads(value)
        for (value,) in connection.execute("SELECT payload FROM instrument_alarms ORDER BY row_index, variable")
    ]
    if not rows:
        if instrument_events:
            with TemporaryDirectory(prefix=".report-", dir=directory) as temporary:
                source = Path(temporary) / "alarms.csv"
                pd.DataFrame(instrument_events).reindex(columns=ALARM_COLUMNS).to_csv(source, index=False)
                source.replace(Path(directory) / "alarms.csv")
        return
    result = merge_results(rows)
    result["instrument_events"] = instrument_events
    result["evaluation"] = evaluate_predictions(result) if settings["evaluation"]["enabled"] else None
    # SQLite가 원본이며 보고서는 재생성 가능하다. 완성한 파일만 하나씩 원자 교체한다.
    with TemporaryDirectory(prefix=".report-", dir=directory) as temporary:
        save_report(result, settings["output"], directory=temporary)
        for source in Path(temporary).rglob("*"):
            if source.is_file():
                target = Path(directory) / source.relative_to(temporary)
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)


def process_snapshot(model, csv_path, calibration, settings, directory, *, process_data):
    """CSV의 확정된 앞부분을 확인하고 새 행의 결과·상태를 같은 트랜잭션으로 저장한다."""
    directory = Path(directory)
    with output_lock(directory):
        filename = settings["online"]["database_filename"]
        if not isinstance(filename, str) or filename in {"", ".", ".."} or Path(filename).name != filename:
            raise ValueError("온라인 DB 이름은 출력 폴더 안의 파일명이어야 합니다")
        database = directory / filename
        if database.is_symlink():
            raise ValueError("온라인 상태 DB에는 심볼릭 링크를 사용할 수 없습니다")
        descriptor = os.open(database, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        os.close(descriptor)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS predictions (row_index INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS instrument_alarms (
                    row_index INTEGER NOT NULL, variable TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (row_index, variable)
                )"""
            )
            saved = connection.execute("SELECT payload FROM state WHERE id = 1").fetchone()
            state = json.loads(saved[0]) if saved else {}
            identity = sha256(
                encode(
                    {
                        "calibration": calibration,
                        "alarm": settings["alarm"],
                        "data_state": model.data_state,
                        "input": str(Path(csv_path).resolve()),
                    }
                ).encode()
            ).hexdigest()
            if state and state["identity"] != identity:
                raise ValueError("온라인 모델·보정·알람·입력이 바뀌었습니다. 새 출력 폴더를 사용하세요")
            session = OnlineSession(model, calibration, settings, state.get("session"))
            previous_rows = state.get("rows", 0)
            count = 0
            with connection, Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                required = {session.timestamp_col, *model.data_state["feature_names"]}
                if not columns or len(set(columns)) != len(columns) or not required <= set(columns):
                    raise ValueError("온라인 CSV의 시각·특성 컬럼이 누락되었거나 중복되었습니다")
                if state and columns != state["columns"]:
                    raise ValueError("온라인 CSV의 기존 컬럼이 변경되었습니다")
                digest = sha256(encode(columns).encode())
                for count, row in enumerate(reader, 1):
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("온라인 CSV에 열 개수가 맞지 않는 미완성 행이 있습니다")
                    digest.update(encode(row).encode())
                    if count <= previous_rows:
                        if count == previous_rows and digest.hexdigest() != state["input_digest"]:
                            raise ValueError("이미 처리한 입력 행이 변경·삭제되었습니다. 새 출력 폴더를 사용하세요")
                        continue
                    result, instrument_events = session.consume(row, process_data)
                    if result is not None:
                        connection.execute(
                            "INSERT INTO predictions VALUES (?, ?)", (int(result["index"][0]), encode(result))
                        )
                    for event in instrument_events:
                        connection.execute(
                            "INSERT INTO instrument_alarms VALUES (?, ?, ?)",
                            (count - 1, event["variable"], encode(event)),
                        )
                if count < previous_rows:
                    raise ValueError("이미 처리한 입력 행이 변경·삭제되었습니다. 새 출력 폴더를 사용하세요")
                changed = count > previous_rows
                if changed:
                    committed = {
                        "identity": identity,
                        "columns": columns,
                        "rows": count,
                        "input_digest": digest.hexdigest(),
                        "session": session.state_dict(),
                    }
                    connection.execute("INSERT OR REPLACE INTO state VALUES (1, ?)", (encode(committed),))
                    state = committed
            # 상태와 결과는 함께 커밋되어 재시작 때 이미 처리한 행을 추론하지 않는다.
            # ponytail: CSV 앞부분 확인과 누적 보고서는 O(N)이다. 대용량에서는 입력 커서·DB 조회로 교체한다.
            export_signature = sha256(
                encode({"rows": count, "output": settings["output"], "evaluation": settings["evaluation"]}).encode()
            ).hexdigest()
            if count and state.get("export_signature") != export_signature:
                export_results(connection, settings, directory)
                state["export_signature"] = export_signature
                with connection:
                    connection.execute("UPDATE state SET payload = ? WHERE id = 1", (encode(state),))
            return changed


def watch_csv(model, csv_path, calibration, settings, directory, *, process_data):
    """모델을 유지한 채 새 관측값을 처리하고 오류에서는 중단한다."""
    try:
        while True:
            if process_snapshot(model, csv_path, calibration, settings, directory, process_data=process_data):
                print("온라인 추론 결과 갱신", flush=True)  # noqa: T201
            time.sleep(settings["online"]["poll_interval_seconds"])
    except KeyboardInterrupt:
        print("온라인 추론 종료")  # noqa: T201
