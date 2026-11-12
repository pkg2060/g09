"""다변량 시계열 학습에 필요한 핵심 전처리 함수.

이 모듈은 모델이나 PyTorch 코드와 분리되어 있으며 다음 작업만 담당한다.

1. 타임스탬프 변환, 정렬 및 중복 제거
2. 모델 특성의 수치형 변환
3. 학습용 결측 처리와 추론용 계기 이상 행 분리

실행할 함수와 순서는 ``configs/config.yaml``의 ``preprocessing.steps``에서 관리한다.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

_RESERVED_COLUMNS = {"__row_id__", "__segment_id__"}


def _resolve_feature_columns(
    frame: pd.DataFrame,
    timestamp_col: str | None,
    feature_columns: Sequence[str] | None,
) -> list[str]:
    """모델 입력에 사용할 특성 컬럼과 순서를 확정한다.

    명시적인 목록이 없으면 타임스탬프와 내부 예약 컬럼을 제외한 모든 컬럼을
    선택한다. 존재하지 않는 컬럼을 요청하거나 선택 결과가 비어 있으면 예외를 낸다.
    """

    if feature_columns is None:
        columns = [column for column in frame.columns if column != timestamp_col and column not in _RESERVED_COLUMNS]
    else:
        columns = list(feature_columns)

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"입력 데이터에 없는 특성 컬럼입니다: {missing}")
    if not columns:
        raise ValueError("모델 입력으로 선택된 특성 컬럼이 없습니다")
    return columns


def prepare_time_index(
    frame: pd.DataFrame,
    *,
    timestamp_col: str | None,
    feature_columns: Sequence[str] | None = None,
    drop_invalid: bool = True,
    sort: bool = True,
    duplicate_keep: str | bool = "last",
    **_: Any,
) -> pd.DataFrame:
    """타임스탬프를 변환하고 정렬한 뒤 중복 시점을 제거한다.

    타임스탬프 컬럼이 ``None``이면 원본 행 순서를 그대로 사용한다. 잘못된
    타임스탬프를 삭제하지 않도록 설정한 경우에는 발견 즉시 예외를 낸다.
    """

    del feature_columns
    result = frame.copy()
    if timestamp_col is None:
        return result
    if timestamp_col not in result.columns:
        raise ValueError(f"타임스탬프 컬럼이 없습니다: {timestamp_col}")

    result[timestamp_col] = pd.to_datetime(result[timestamp_col], errors="coerce")
    invalid_count = int(result[timestamp_col].isna().sum())
    if invalid_count:
        if not drop_invalid:
            raise ValueError(f"잘못된 타임스탬프가 {invalid_count}개 있습니다")
        result = result.loc[result[timestamp_col].notna()].copy()

    if sort:
        result = result.sort_values(timestamp_col, kind="stable")
    if duplicate_keep is not False:
        if duplicate_keep not in {"first", "last"}:
            raise ValueError("duplicate_keep은 first, last 또는 false여야 합니다")
        result = result.drop_duplicates(timestamp_col, keep=duplicate_keep)
    return result.reset_index(drop=True)


def coerce_numeric(
    frame: pd.DataFrame,
    *,
    timestamp_col: str | None,
    feature_columns: Sequence[str] | None = None,
    replace_infinite: bool = True,
    **_: Any,
) -> pd.DataFrame:
    """선택한 특성을 수치형으로 변환하고 잘못된 값을 NaN으로 치환한다."""

    result = frame.copy()
    columns = _resolve_feature_columns(result, timestamp_col, feature_columns)
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if replace_infinite:
        result[columns] = result[columns].replace([np.inf, -np.inf], np.nan)
    return result


def drop_missing(
    frame: pd.DataFrame,
    *,
    timestamp_col: str | None,
    feature_columns: Sequence[str] | None = None,
    **_: Any,
) -> pd.DataFrame:
    """모델 특성 중 하나라도 결측인 행을 보간하지 않고 제외한다.

    결측 행의 시각은 남은 행과의 시간 간격으로 나타나므로 DataLoader가
    해당 지점을 넘는 Window를 생성하지 않는다.
    """

    columns = _resolve_feature_columns(frame, timestamp_col, feature_columns)
    return frame.dropna(subset=columns).reset_index(drop=True)


def drop_stopped(
    frame: pd.DataFrame,
    *,
    timestamp_col: str | None,
    feature_columns: Sequence[str] | None = None,
    conditions: Sequence[Mapping[str, Any]] | None = None,
    logic: str = "any",
    **_: Any,
) -> pd.DataFrame:
    """운전 tag별 조건을 결합해 공정 정지로 판정한 행을 제거한다."""
    if logic not in {"all", "any"}:
        raise ValueError("drop_stopped.logic은 all 또는 any여야 합니다")
    if not conditions:
        raise ValueError("drop_stopped.conditions에는 하나 이상의 운전 조건이 필요합니다")
    columns = _resolve_feature_columns(frame, timestamp_col, feature_columns)
    operations = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
    results = []
    for condition in conditions:
        if not isinstance(condition, Mapping) or set(condition) != {"variable", "operator", "value"}:
            raise ValueError("공정 정지 조건에는 variable, operator, value가 필요합니다")
        variable, comparison, value = condition["variable"], condition["operator"], condition["value"]
        if not isinstance(variable, str) or variable not in columns:
            raise ValueError(f"공정 정지 조건에 없는 특성이 있습니다: {variable}")
        if not isinstance(comparison, str) or comparison not in operations:
            raise ValueError("공정 정지 조건의 operator는 <, <=, >, >= 중 하나여야 합니다")
        if isinstance(value, bool) or not isinstance(value, int | float | np.number) or not np.isfinite(value):
            raise ValueError("공정 정지 조건의 value는 유효한 수치여야 합니다")
        results.append(operations[comparison](frame[variable].to_numpy(dtype=float), float(value)))
    stopped = np.logical_and.reduce(results) if logic == "all" else np.logical_or.reduce(results)
    return frame.loc[~stopped].reset_index(drop=True)


def _continuity_groups(
    timestamps: pd.Series,
    *,
    expected_interval_seconds: float,
    max_gap_multiplier: float,
) -> np.ndarray:
    """허용된 시간 간격을 넘는 지점마다 새로운 연속 구간 번호를 생성한다."""

    seconds = timestamps.diff().dt.total_seconds().to_numpy()
    boundary = np.zeros(len(timestamps), dtype=bool)
    if not len(timestamps):
        return np.empty(0, dtype=np.int64)
    boundary[0] = True
    boundary[1:] = (
        ~np.isfinite(seconds[1:]) | (seconds[1:] <= 0) | (seconds[1:] > expected_interval_seconds * max_gap_multiplier)
    )
    return np.cumsum(boundary, dtype=np.int64) - 1


def interpolate_missing(
    frame: pd.DataFrame,
    *,
    timestamp_col: str | None,
    feature_columns: Sequence[str] | None = None,
    method: str = "time",
    limit: int | None = 10,
    limit_direction: str = "both",
    drop_remaining: bool = True,
    expected_interval_seconds: float = 60.0,
    max_gap_multiplier: float = 1.5,
    **_: Any,
) -> pd.DataFrame:
    """연속된 시간 구간 안에서만 특성의 결측값을 제한적으로 보간한다.

    Args:
        frame: 시간 정렬과 수치형 변환이 끝난 데이터.
        timestamp_col: 시간축 컬럼 이름. ``time`` 보간에는 반드시 필요하다.
        feature_columns: 보간할 모델 특성 컬럼.
        method: ``time`` 또는 ``linear`` 보간 방식.
        limit: 연속해서 보간할 최대 결측 행 수.
        limit_direction: 보간 방향.
        drop_remaining: 보간 후에도 남은 결측 행을 삭제할지 여부.
        expected_interval_seconds: 데이터의 정상 수집 간격.
        max_gap_multiplier: 서로 다른 연속 구간으로 분리할 최대 간격 배수.

    Returns:
        원래 시간 순서를 유지하며 결측값을 처리한 데이터.
    """

    if method not in {"time", "linear"}:
        raise ValueError("보간 방식은 time 또는 linear여야 합니다")

    result = frame.copy()
    columns = _resolve_feature_columns(result, timestamp_col, feature_columns)

    if timestamp_col is None:
        if method == "time":
            raise ValueError("time 보간에는 타임스탬프 컬럼이 필요합니다")
        groups = np.zeros(len(result), dtype=np.int64)
    else:
        if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
            raise ValueError("보간 전에 prepare_time_index를 실행해야 합니다")
        groups = _continuity_groups(
            result[timestamp_col],
            expected_interval_seconds=expected_interval_seconds,
            max_gap_multiplier=max_gap_multiplier,
        )

    for group_id in np.unique(groups):
        row_mask = groups == group_id
        group = result.loc[row_mask, [timestamp_col, *columns]] if timestamp_col else result.loc[row_mask, columns]
        if method == "time":
            values = group.set_index(timestamp_col)[columns].interpolate(
                method="time",
                limit=limit,
                limit_direction=limit_direction,
            )
            result.loc[row_mask, columns] = values.to_numpy()
        else:
            result.loc[row_mask, columns] = (
                group[columns]
                .interpolate(
                    method="linear",
                    limit=limit,
                    limit_direction=limit_direction,
                )
                .to_numpy()
            )

    if drop_remaining:
        result = result.dropna(subset=columns)
    return result.reset_index(drop=True)


class ForwardFillState:
    """과거의 마지막 유효값만 사용하며 단절과 결측 연속 횟수를 유지한다."""

    def __init__(self, width, *, limit=10, expected_interval_seconds=60, max_gap_multiplier=1.5, state=None):
        """배치와 온라인에 같은 제한적 전방 채움 규칙을 적용한다."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("forward_fill.limit은 양의 정수여야 합니다")
        if (
            expected_interval_seconds is None
            or not np.isfinite(expected_interval_seconds)
            or expected_interval_seconds <= 0
        ):
            raise ValueError("전방 채움에는 양의 수집 간격이 필요합니다")
        if not np.isfinite(max_gap_multiplier) or max_gap_multiplier < 1:
            raise ValueError("단절 배수는 1 이상의 유한한 값이어야 합니다")
        self.limit = limit
        self.max_gap = expected_interval_seconds * max_gap_multiplier
        state = state or {}
        self.values = np.asarray(state.get("values", [None] * width), dtype=float)
        self.missing = np.asarray(state.get("missing", [0] * width), dtype=int)
        self.previous_time = pd.Timestamp(state["previous_time"]) if state.get("previous_time") else None
        if self.values.shape != (width,) or self.missing.shape != (width,) or (self.missing < 0).any():
            raise ValueError("전방 채움 상태의 특성 수 또는 결측 횟수가 잘못되었습니다")

    def update(self, values, timestamp):
        """새 관측값 하나를 채우고 실제 관측값과 결측 횟수를 갱신한다."""
        timestamp = pd.Timestamp(timestamp)
        if pd.isna(timestamp):
            raise ValueError("전방 채움에는 유효한 시각이 필요합니다")
        if self.previous_time is not None:
            seconds = (timestamp - self.previous_time).total_seconds()
            if seconds <= 0:
                raise ValueError("온라인 데이터는 중복 없이 시간순이어야 합니다")
            if seconds > self.max_gap:
                self.values[:] = np.nan
                self.missing[:] = 0
        self.previous_time = timestamp
        values = np.asarray(values, dtype=float)
        present = np.isfinite(values)
        self.values[present] = values[present]
        self.missing = np.where(present, 0, self.missing + 1)
        return np.where(present, values, np.where(self.missing <= self.limit, self.values, np.nan))

    def state_dict(self):
        """JSON으로 저장할 수 있는 최소 상태를 반환한다."""
        return {
            "values": [float(v) if np.isfinite(v) else None for v in self.values],
            "missing": self.missing.tolist(),
            "previous_time": None if self.previous_time is None else self.previous_time.isoformat(),
        }


def forward_fill_missing(
    frame,
    *,
    timestamp_col,
    feature_columns=None,
    limit=10,
    expected_interval_seconds=60,
    max_gap_multiplier=1.5,
    drop_remaining=True,
):
    """학습·평가에서도 온라인과 같은 과거값 기반 결측 처리를 적용한다."""
    columns = _resolve_feature_columns(frame, timestamp_col, feature_columns)
    if timestamp_col is None:
        raise ValueError("forward_fill에는 타임스탬프 컬럼이 필요합니다")
    result = frame.copy()
    filler = ForwardFillState(
        len(columns),
        limit=limit,
        expected_interval_seconds=expected_interval_seconds,
        max_gap_multiplier=max_gap_multiplier,
    )
    filled = [
        filler.update(values, timestamp)
        for values, timestamp in zip(result[columns].to_numpy(dtype=float), result[timestamp_col], strict=False)
    ]
    if filled:
        result[columns] = np.asarray(filled)
    if drop_remaining:
        result = result.dropna(subset=columns)
    return result.reset_index(drop=True)


def apply_missing_strategy(
    frame,
    *,
    timestamp_col,
    feature_columns=None,
    strategy="drop",
    max_consecutive=10,
    interpolation_method="time",
    interpolation_direction="both",
    expected_interval_seconds=60,
    max_gap_multiplier=1.5,
):
    """학습·검증 데이터에 선택한 결측 처리 전략 하나만 적용한다."""
    common = {
        "timestamp_col": timestamp_col,
        "feature_columns": feature_columns,
        "expected_interval_seconds": expected_interval_seconds,
        "max_gap_multiplier": max_gap_multiplier,
    }
    if strategy == "drop":
        return drop_missing(frame, timestamp_col=timestamp_col, feature_columns=feature_columns)
    if strategy == "forward_fill":
        return forward_fill_missing(frame, limit=max_consecutive, drop_remaining=True, **common)
    if strategy == "interpolate":
        return interpolate_missing(
            frame,
            method=interpolation_method,
            limit=max_consecutive,
            limit_direction=interpolation_direction,
            drop_remaining=True,
            **common,
        )
    raise ValueError("결측 처리 strategy는 drop, forward_fill, interpolate 중 하나여야 합니다")


class OnlinePreprocessor:
    """저장된 인과적 전처리만 허용하고 새 행 하나씩 같은 규칙으로 처리한다."""

    def __init__(self, pipeline, features, state=None):
        """미래값을 쓰는 체크포인트를 조용히 다른 전처리로 바꾸지 않는다."""
        self.timestamp_col = pipeline["data"]["timestamp_column"]
        self.features = list(features)
        steps = [
            ({"name": step} if isinstance(step, str) else step)
            for step in pipeline["preprocessing"]["steps"]
            if isinstance(step, str) or step.get("enabled", True)
        ]
        names = [step["name"] for step in steps]
        missing = pipeline["preprocessing"].get("missing")
        if missing is not None:
            missing_strategy = missing.get("strategy")
        elif "interpolate" in names:
            missing_strategy = "interpolate"
        elif "forward_fill" in names:
            missing_strategy = "forward_fill"
        else:
            missing_strategy = "drop"
        if missing_strategy != "drop":
            raise ValueError("online은 preprocessing.missing.strategy=drop으로 학습한 모델만 지원합니다")
        allowed = {
            ("prepare_time_index", "coerce_numeric"),
            ("prepare_time_index", "coerce_numeric", "forward_fill"),
            ("prepare_time_index", "coerce_numeric", "drop_missing"),
            ("prepare_time_index", "coerce_numeric", "drop_stopped"),
            ("prepare_time_index", "coerce_numeric", "drop_stopped", "forward_fill"),
            ("prepare_time_index", "coerce_numeric", "drop_stopped", "drop_missing"),
        }
        if not self.timestamp_col or tuple(names) not in allowed:
            raise ValueError(
                "online은 prepare_time_index → coerce_numeric → 선택적 drop_stopped·forward_fill만 지원합니다. "
                "미래값 보간 모델은 같은 인과적 전처리로 다시 학습하세요"
            )
        self.numeric_options = steps[1].get("params", {})
        self.stop_options = None
        for step in steps[2:]:
            if step["name"] == "drop_stopped":
                self.stop_options = dict(step.get("params", {}))
        self.drop_remaining = False
        self.filler = None
        if "forward_fill" in names:
            step = steps[names.index("forward_fill")]
            options = dict(step.get("params", {}))
            self.drop_remaining = options.pop("drop_remaining", True)
            self.filler = ForwardFillState(len(features), state=state, **options)
        elif "drop_missing" in names:
            # OnlineSession이 계기 이상 알람을 먼저 만들고, 보험으로 남은 결측 행도 추론에서 제외한다.
            self.drop_remaining = True

    def process(self, row):
        """시각·특성을 검증하고 채울 수 없는 행은 설정에 따라 건너뛴다."""
        frame = prepare_time_index(
            pd.DataFrame([row]), timestamp_col=self.timestamp_col, drop_invalid=False, sort=False, duplicate_keep=False
        )
        frame = coerce_numeric(
            frame, timestamp_col=self.timestamp_col, feature_columns=self.features, **self.numeric_options
        )
        if self.stop_options is not None:
            frame = drop_stopped(
                frame,
                timestamp_col=self.timestamp_col,
                feature_columns=self.features,
                **self.stop_options,
            )
            if frame.empty:
                return None
        values = frame[self.features].iloc[0].to_numpy(dtype=float)
        if self.filler is not None:
            values = self.filler.update(values, frame[self.timestamp_col].iloc[0])
        if not np.isfinite(values).all():
            if self.drop_remaining:
                return None
            raise ValueError("온라인 입력의 특성에 처리할 수 없는 결측값·무한대가 있습니다")
        frame.loc[frame.index[0], self.features] = values
        return frame.iloc[0].to_dict()

    def reset(self):
        """계기 이상 뒤 과거 결측 채움 상태가 다음 운전 구간으로 넘어가지 않게 초기화한다."""
        if self.filler is not None:
            self.filler.values[:] = np.nan
            self.filler.missing[:] = 0
            self.filler.previous_time = None

    def state_dict(self):
        """재시작 시 이어갈 결측 처리 상태를 반환한다."""
        return self.filler.state_dict() if self.filler is not None else None


PREPROCESS_FUNCTIONS = {
    "prepare_time_index": prepare_time_index,
    "coerce_numeric": coerce_numeric,
    "drop_missing": drop_missing,
    "drop_stopped": drop_stopped,
    "interpolate": interpolate_missing,
    "forward_fill": forward_fill_missing,
}


@dataclass
class PreprocessResult:
    """전처리된 데이터와 실제 실행 이력을 함께 전달하는 결과 객체.

    Attributes:
        frame: 전처리가 끝난 시계열 데이터.
        feature_columns: 모델 입력으로 사용할 특성 컬럼 순서.
        applied_steps: 실제 실행된 단계와 단계별 행 수.
        rows_before: 전처리 전 행 수.
        rows_after: 전처리 후 행 수.
        instrument_fault_rows: 보간하지 않고 제외한 계기 결측 행.
    """

    frame: pd.DataFrame
    feature_columns: list[str]
    applied_steps: list[dict[str, Any]] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    instrument_fault_rows: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_preprocessing(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None,
    *,
    timestamp_col: str | None,
    feature_columns: Sequence[str] | None = None,
    inference: bool = False,
    expected_interval_seconds: float = 60.0,
    max_gap_multiplier: float = 1.5,
) -> PreprocessResult:
    """설정에 선언된 핵심 전처리 함수를 순서대로 실행한다.

    학습·검증에는 설정한 결측 전략을 적용한다. 일괄 추론에서는 원본 결측 행을
    계기 이상으로 분리하고 모델 입력에서 제외한다. 실행이 끝나면 모든 모델
    특성이 수치형이고 결측값이 없는지 검사한다.
    ``src/dataloader.py``는 이 함수만 호출하므로 전처리 구현과 로더가 분리된다.
    """

    result = frame.copy()
    rows_before = len(result)
    source_features = _resolve_feature_columns(result, timestamp_col, feature_columns)
    steps = list(dict(config or {}).get("steps", []))
    applied: list[dict[str, Any]] = []
    instrument_fault_rows: list[pd.DataFrame] = []
    missing_step_names = {"drop_missing", "forward_fill", "interpolate"}

    for raw_step in steps:
        if isinstance(raw_step, str):
            name, enabled, params = raw_step, True, {}
        elif isinstance(raw_step, Mapping):
            name = str(raw_step.get("name", ""))
            enabled = bool(raw_step.get("enabled", True))
            params = dict(raw_step.get("params", {}))
        else:
            raise TypeError(f"잘못된 전처리 단계 설정입니다: {raw_step!r}")

        if not enabled:
            continue
        if inference and name in missing_step_names:
            # 이전 체크포인트의 결측 단계도 일괄 추론에서는 계기 이상 우선 정책으로 대체한다.
            continue
        if name not in PREPROCESS_FUNCTIONS:
            available = ", ".join(PREPROCESS_FUNCTIONS)
            raise ValueError(f"지원하지 않는 전처리 단계입니다: {name}. 사용 가능: {available}")

        before = len(result)
        if name == "drop_missing":
            missing = result[source_features].isna().any(axis=1)
            if missing.any():
                instrument_fault_rows.append(result.loc[missing].copy())
        result = PREPROCESS_FUNCTIONS[name](
            result,
            timestamp_col=timestamp_col,
            feature_columns=source_features,
            **params,
        )
        applied.append({"name": name, "rows_before": before, "rows_after": len(result)})

    if inference:
        invalid = ~np.isfinite(result[source_features].to_numpy(dtype=float)).all(axis=1)
        if invalid.any():
            instrument_fault_rows.append(result.loc[invalid].copy())
            result = result.loc[~invalid].reset_index(drop=True)
        applied.append(
            {
                "name": "instrument_fault_drop",
                "rows_before": int(len(result) + invalid.sum()),
                "rows_after": len(result),
            }
        )
    elif "missing" in dict(config or {}):
        options = dict(config["missing"])
        before = len(result)
        result = apply_missing_strategy(
            result,
            timestamp_col=timestamp_col,
            feature_columns=source_features,
            expected_interval_seconds=expected_interval_seconds,
            max_gap_multiplier=max_gap_multiplier,
            **options,
        )
        applied.append(
            {
                "name": f"missing_{options['strategy']}",
                "rows_before": before,
                "rows_after": len(result),
            }
        )

    non_numeric = [column for column in source_features if not pd.api.types.is_numeric_dtype(result[column])]
    if non_numeric:
        raise TypeError(f"수치형으로 변환되지 않은 모델 특성이 있습니다: {non_numeric}")

    missing = result[source_features].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        details = ", ".join(f"{column}={count}" for column, count in missing.items())
        raise ValueError(f"전처리 후에도 결측값이 남아 있습니다: {details}")
    if result.empty or not np.isfinite(result[source_features].to_numpy(dtype=float)).all():
        raise ValueError("전처리 결과는 비어 있지 않은 유한한 수치여야 합니다")

    return PreprocessResult(
        frame=result.reset_index(drop=True),
        feature_columns=source_features,
        applied_steps=applied,
        rows_before=rows_before,
        rows_after=len(result),
        instrument_fault_rows=(
            pd.concat(instrument_fault_rows, ignore_index=True)
            if instrument_fault_rows
            else pd.DataFrame(columns=frame.columns)
        ),
    )


def preprocess_time_series_splits(
    frame,
    config,
    *,
    timestamp_col,
    feature_columns,
    n_splits,
    enabled=True,
    validation_ratio=0.2,
    expected_interval_seconds=60.0,
    max_gap_multiplier=1.5,
):
    """시간 순서를 유지하는 Fold 또는 단일 Holdout 분할을 전처리한다.

    시간축 정렬은 전체 입력에서 한 번만 수행하고, 결측 처리와 정지구간 제거는 각
    분할 안에서 독립 실행해 미래 Validation 값이 Train 전처리에 섞이지 않게 한다.
    """
    if not isinstance(enabled, bool):
        raise ValueError("time_series_split_enabled는 불리언이어야 합니다")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("time_series_splits는 2 이상의 정수여야 합니다")
    if (
        isinstance(validation_ratio, bool)
        or not isinstance(validation_ratio, int | float)
        or not 0 < validation_ratio < 1
    ):
        raise ValueError("validation_ratio는 0보다 크고 1보다 작은 수여야 합니다")
    steps = [step for step in config.get("steps", []) if isinstance(step, str) or step.get("enabled", True)]
    prepared = frame.copy()
    remaining = list(steps)
    if timestamp_col:
        first = steps[0] if steps else None
        name = first if isinstance(first, str) else (first or {}).get("name")
        if name != "prepare_time_index":
            raise ValueError("시간축이 있으면 첫 전처리는 prepare_time_index여야 합니다")
        params = {} if isinstance(first, str) else first.get("params", {})
        prepared = prepare_time_index(prepared, timestamp_col=timestamp_col, **params)
        remaining = steps[1:]
    if enabled:
        if len(prepared) <= n_splits:
            raise ValueError("TimeSeriesSplit에 필요한 학습 행이 부족합니다")
        split_indices = TimeSeriesSplit(n_splits=n_splits).split(prepared)
    else:
        split_at = int(len(prepared) * (1 - validation_ratio))
        if split_at < 2 or len(prepared) - split_at < 2:
            raise ValueError("단일 Holdout 분할 후 Train과 Validation에 각각 최소 두 행이 필요합니다")
        split_indices = [(np.arange(split_at), np.arange(split_at, len(prepared)))]
    fold_config = {"steps": remaining}
    if "missing" in config:
        fold_config["missing"] = config["missing"]
    results = []
    for train_indices, validation_indices in split_indices:
        train = run_preprocessing(
            prepared.iloc[train_indices],
            fold_config,
            timestamp_col=timestamp_col,
            feature_columns=feature_columns,
            expected_interval_seconds=expected_interval_seconds,
            max_gap_multiplier=max_gap_multiplier,
        )
        validation = run_preprocessing(
            prepared.iloc[validation_indices],
            fold_config,
            timestamp_col=timestamp_col,
            feature_columns=feature_columns,
            expected_interval_seconds=expected_interval_seconds,
            max_gap_multiplier=max_gap_multiplier,
        )
        if min(len(train.frame), len(validation.frame)) < 2:
            raise ValueError("전처리 후 각 TimeSeriesSplit fold에 최소 두 행이 필요합니다")
        results.append((train, validation))
    return results
