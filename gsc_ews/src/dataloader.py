"""두 모델에서 사용하는 설정 기반 CSV 및 DataLoader 생성 모듈.

이 모듈은 구체적인 딥러닝 모델과 독립적으로 동작한다. 다음 키를 갖는 표준 배치를
생성한다.

    context:   [batch, window, feature]
    target:    [batch, feature] (복원 구간의 마지막 시점)
    label:     [batch, feature]
    index:     전처리된 데이터 분할 내 원본 행 인덱스
    timestamp: 유닉스 기준 나노초 단위 int64 값, 값이 없으면 -1

Lightning 학습·평가·추론이 이 로더를 함께 사용한다. Dataset은
필요한 시점에 구간을 잘라 반환하므로, 대용량 CSV를 N x window x feature 텐서로
미리 만들어 메모리에 올리지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import preprocess_time_series_splits, run_preprocessing


def _torch_dtype(name: str) -> torch.dtype:
    """설정의 dtype 문자열을 대응하는 PyTorch dtype으로 변환한다."""

    normalized = str(name).lower()
    if normalized in {"float32", "fp32", "single"}:
        return torch.float32
    if normalized in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


class FeatureScaler:
    """학습 데이터에만 맞추며 상태를 체크포인트에 저장할 수 있는 스케일러."""

    def __init__(
        self,
        method: str = "minmax",
        epsilon: float = 1e-8,
        clip: bool = False,
        quantile_range: Sequence[float] = (25.0, 75.0),
    ) -> None:
        """정규화 방법과 수치 안정성 및 범위 제한 설정을 초기화한다.

        Args:
            method: ``minmax``, ``standard``, ``robust``, ``none`` 중 하나.
            epsilon: 0에 가까운 스케일을 판정할 최소값.
            clip: Min-Max 변환 결과를 0~1 범위로 제한할지 여부.
            quantile_range: Robust 변환에 사용할 하위·상위 백분위수.
        """

        method = str(method).lower()
        if method not in {"minmax", "standard", "robust", "none"}:
            raise ValueError("normalization.method must be minmax, standard, robust, or none")
        if len(quantile_range) != 2:
            raise ValueError("normalization.quantile_range must contain two values")
        quantile_low, quantile_high = map(float, quantile_range)
        if not 0.0 <= quantile_low < quantile_high <= 100.0:
            raise ValueError("normalization.quantile_range must satisfy 0 <= low < high <= 100")

        self.method = method
        self.epsilon = float(epsilon)
        self.clip = bool(clip)
        self.quantile_range = (quantile_low, quantile_high)
        self.location: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> FeatureScaler:
        """학습 구간에서 특성별 위치와 스케일 통계를 계산한다.

        검증 및 테스트 정보가 통계에 섞이지 않도록 학습 구간 배열만 전달해야 한다.
        """

        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("Scaler expects a non-empty [time, feature] array")
        if not np.isfinite(values).all():
            raise ValueError("Scaler input contains NaN or infinite values")

        if self.method == "minmax":
            self.location = values.min(axis=0)
            self.scale = values.max(axis=0) - self.location
        elif self.method == "standard":
            self.location = values.mean(axis=0)
            self.scale = values.std(axis=0)
        elif self.method == "robust":
            quantile_low, quantile_high = np.percentile(
                values,
                self.quantile_range,
                axis=0,
            )
            self.location = np.median(values, axis=0)
            self.scale = quantile_high - quantile_low
        elif self.method == "none":
            self.location = np.zeros(values.shape[1], dtype=np.float64)
            self.scale = np.ones(values.shape[1], dtype=np.float64)
        self.scale = np.where(np.abs(self.scale) < self.epsilon, 1.0, self.scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """학습 구간에서 계산한 통계로 입력 배열을 정규화한다."""

        if self.location is None or self.scale is None:
            raise RuntimeError("FeatureScaler must be fitted before transform")
        transformed = (np.asarray(values, dtype=np.float64) - self.location) / self.scale
        if self.method == "minmax" and self.clip:
            transformed = np.clip(transformed, 0.0, 1.0)
        return transformed

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """정규화된 배열을 원래 특성 단위로 복원한다."""

        if self.location is None or self.scale is None:
            raise RuntimeError("FeatureScaler must be fitted before inverse_transform")
        return np.asarray(values) * self.scale + self.location

    def state_dict(self) -> dict[str, Any]:
        """체크포인트에 저장할 수 있는 스케일러 상태를 반환한다."""

        return {
            "method": self.method,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "quantile_range": self.quantile_range,
            "location": None if self.location is None else self.location.copy(),
            "scale": None if self.scale is None else self.scale.copy(),
        }

    @classmethod
    def from_state_dict(cls, state):
        """체크포인트 통계를 복원하며 추론 데이터로 다시 맞추지 않는다."""
        scaler = cls(**{key: state[key] for key in ("method", "epsilon", "clip", "quantile_range")})
        scaler.location = np.asarray(state["location"], dtype=np.float64)
        scaler.scale = np.asarray(state["scale"], dtype=np.float64)
        if (
            scaler.location.ndim != 1
            or scaler.location.shape != scaler.scale.shape
            or not np.isfinite(scaler.location).all()
            or not np.isfinite(scaler.scale).all()
            or np.any(scaler.scale <= 0)
        ):
            raise ValueError("Invalid checkpoint scaler state")
        return scaler


def _segment_ids(
    timestamps: pd.Series | None,
    length: int,
    *,
    expected_interval_seconds: float | None,
    max_gap_multiplier: float,
) -> np.ndarray:
    """타임스탬프 간격을 검사하여 연속 시계열별 구간 번호를 생성한다.

    시간이 역행하거나 중복되거나 허용된 최대 간격을 넘으면 새 구간을 시작한다.
    타임스탬프 또는 예상 간격이 없으면 전체를 하나의 구간으로 취급한다.
    """

    if timestamps is None or expected_interval_seconds is None:
        return np.zeros(length, dtype=np.int64)
    seconds = pd.to_datetime(timestamps).diff().dt.total_seconds().to_numpy()
    boundary = np.zeros(length, dtype=bool)
    if length == 0:
        return np.empty(0, dtype=np.int64)
    boundary[0] = True
    boundary[1:] = (
        ~np.isfinite(seconds[1:]) | (seconds[1:] <= 0) | (seconds[1:] > expected_interval_seconds * max_gap_multiplier)
    )
    return np.cumsum(boundary, dtype=np.int64) - 1


class TimeSeriesWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """타임스탬프 단절 구간을 넘지 않는 지연 생성 방식의 인과적 윈도우."""

    def __init__(
        self,
        values: np.ndarray,
        *,
        window_size: int,
        labels: np.ndarray | None = None,
        timestamps: pd.Series | Sequence[Any] | None = None,
        segment_ids: np.ndarray | None = None,
        pad_start: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """시계열 배열과 window 생성 규칙을 초기화한다.

        Args:
            values: ``[시점, 특성]`` 형태의 정규화된 값.
            window_size: 모델에 입력할 시계열 길이.
            labels: 값과 같은 shape의 변수별 이상 라벨.
            timestamps: 각 행의 타임스탬프.
            segment_ids: 시간 단절을 반영한 행별 구간 번호.
            pad_start: 구간 시작부의 부족한 길이를 첫 값으로 채울지 여부.
            dtype: 모델 입력 텐서 자료형.
        """

        values = np.asarray(values)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("values must be a non-empty [time, feature] array")
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.values = torch.as_tensor(values, dtype=dtype)
        self.window_size = int(window_size)
        self.pad_start = bool(pad_start)
        self.labels = torch.as_tensor(
            np.zeros_like(values) if labels is None else labels,
            dtype=dtype,
        )
        if self.labels.shape != self.values.shape:
            raise ValueError(f"labels shape {tuple(self.labels.shape)} != values shape {tuple(self.values.shape)}")

        self.segment_ids = np.asarray(
            np.zeros(len(values), dtype=np.int64) if segment_ids is None else segment_ids,
            dtype=np.int64,
        )
        if len(self.segment_ids) != len(values):
            raise ValueError("segment_ids length must match values")

        starts = np.zeros(len(values), dtype=np.int64)
        starts[0] = 0
        for index in range(1, len(values)):
            starts[index] = index if self.segment_ids[index] != self.segment_ids[index - 1] else starts[index - 1]
        self.segment_starts = starts

        if pad_start:
            self.end_indices = np.arange(len(values), dtype=np.int64)
        else:
            enough_history = np.arange(len(values)) - starts + 1 >= self.window_size
            self.end_indices = np.flatnonzero(enough_history).astype(np.int64)
        if len(self.end_indices) == 0:
            raise ValueError("No valid windows; reduce window_size or enable pad_start")

        if timestamps is None:
            self.timestamps_ns = np.full(len(values), -1, dtype=np.int64)
        else:
            parsed = pd.to_datetime(pd.Series(timestamps), errors="coerce")
            if len(parsed) != len(values):
                raise ValueError("timestamps length must match values")
            self.timestamps_ns = parsed.astype("int64").to_numpy(copy=True)
            self.timestamps_ns[parsed.isna().to_numpy()] = -1

    def __len__(self) -> int:
        """현재 설정에서 생성 가능한 window 개수를 반환한다."""

        return len(self.end_indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        """지정한 위치의 인과적 window와 정렬된 target·label을 반환한다.

        반환 target은 context의 마지막 시점이며 TranAD 복원 학습 기준과 일치한다.
        시간 단절 직후에는 이전 구간의 값이 context에 포함되지 않는다.
        """

        end = int(self.end_indices[item])
        segment_start = int(self.segment_starts[end])
        start = max(segment_start, end - self.window_size + 1)
        context = self.values[start : end + 1]
        if len(context) < self.window_size:
            padding = self.values[segment_start : segment_start + 1].repeat(self.window_size - len(context), 1)
            context = torch.cat([padding, context], dim=0)
        return {
            "context": context,
            "target": self.values[end],
            "label": self.labels[end],
            "index": torch.tensor(end, dtype=torch.long),
            "timestamp": torch.tensor(self.timestamps_ns[end], dtype=torch.long),
        }


@dataclass
class DataBundle:
    """학습에 필요한 DataLoader와 데이터 처리 상태를 묶은 결과 객체.

    스케일러, 최종 특성 순서, 전처리된 DataFrame 및 실행 metadata를 함께 보관하여
    학습·평가·체크포인트 저장 단계가 동일한 데이터 계약을 공유하게 한다.
    """

    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader | None
    scaler: FeatureScaler
    feature_names: list[str]
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    instrument_fault_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)
    folds: list[DataBundle] = field(default_factory=list)
    fold_index: int = 0

    @property
    def n_features(self) -> int:
        """모델에 입력되는 최종 특성 개수를 반환한다."""

        return len(self.feature_names)


# UTF-8(BOM 포함)을 먼저 시도하고, 한글 Windows Excel 기본 저장 형식인 CP949로 재시도한다.
CSV_ENCODINGS = ("utf-8-sig", "cp949")


def _read_csv(path: str | Path, *, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    """필요한 컬럼만 CSV에서 읽고 UTF-8·CP949 인코딩을 순서대로 시도한다."""

    path = Path(path).expanduser()
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, usecols=usecols, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(
        f"{path}: CSV 인코딩을 해석할 수 없습니다. UTF-8 또는 CP949(EUC-KR)로 저장하세요"
    ) from last_error


def load_tag_descriptions(path, equipment_id, feature_names):
    """설비와 RTDB Tag ID가 일치하는 임시 CSV 설명을 Plot용 매핑으로 읽는다."""
    if path is None:
        return {}
    required = ["Equipment", "RTDB Tag ID", "TagDescription"]
    frame = _read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Tag Description CSV에 필수 컬럼이 없습니다: {sorted(missing)}")
    frame = frame[required]
    if frame.isna().any().any():
        raise ValueError("Tag Description CSV의 필수값은 비어 있을 수 없습니다")
    for column in required:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Tag Description CSV의 {column} 값은 비어 있을 수 없습니다")
    duplicated = frame.duplicated(["Equipment", "RTDB Tag ID"], keep=False)
    if duplicated.any():
        pairs = frame.loc[duplicated, ["Equipment", "RTDB Tag ID"]].drop_duplicates().values.tolist()
        raise ValueError(f"Tag Description CSV에 중복된 설비·Tag가 있습니다: {pairs}")
    selected = frame.loc[frame["Equipment"] == str(equipment_id)]
    descriptions = dict(zip(selected["RTDB Tag ID"], selected["TagDescription"], strict=True))
    return {name: descriptions[name] for name in feature_names if name in descriptions}


def _load_aligned_labels(
    path: str | Path | None,
    *,
    test_frame: pd.DataFrame,
    timestamp_col: str | None,
    feature_names: Sequence[str],
    label_columns: Sequence[str] | None,
) -> tuple[np.ndarray, bool]:
    """테스트 데이터의 시점과 특성 순서에 맞춰 라벨을 정렬한다.

    타임스탬프가 있으면 시간 기준으로 결합하고, 없으면 전처리된 테스트 데이터와
    행 수가 정확히 같아야 한다. 단일 라벨 컬럼은 모든 특성에 복제한다.

    Returns:
        정렬된 변수별 이진 라벨과 실제 라벨 파일 제공 여부.
    """

    if path is None:
        return np.zeros((len(test_frame), len(feature_names)), dtype=np.float64), False

    labels = _read_csv(path)
    available = [column for column in labels.columns if column != timestamp_col]
    # 같은 특성명이 있으면 CSV 저장 순서와 무관하게 모델 특성 순서로 맞춘다.
    requested = list(label_columns or (feature_names if set(feature_names) <= set(available) else available))
    if timestamp_col and timestamp_col in labels.columns and timestamp_col in test_frame.columns:
        labels[timestamp_col] = pd.to_datetime(labels[timestamp_col], errors="coerce")
        reference = test_frame[[timestamp_col]].copy()
        reference[timestamp_col] = pd.to_datetime(reference[timestamp_col], errors="coerce")
        if labels[timestamp_col].isna().any() or labels[timestamp_col].duplicated().any():
            raise ValueError("라벨 타임스탬프는 유효하고 중복이 없어야 합니다")
        if reference[timestamp_col].isna().any():
            raise ValueError("데이터 타임스탬프는 라벨 정렬 전에 유효해야 합니다")
        aligned = reference.merge(labels[[timestamp_col, *requested]], on=timestamp_col, how="left", sort=False)
        values = aligned[requested].to_numpy(dtype=np.float64)
    else:
        values = labels[requested].to_numpy(dtype=np.float64)
        if len(values) != len(test_frame):
            raise ValueError("A label CSV without timestamps must have the same row count as processed test data")

    if values.shape[1] == 1:
        values = np.repeat(values, len(feature_names), axis=1)
    if values.shape != (len(test_frame), len(feature_names)):
        raise ValueError(f"Label shape {values.shape} must be {(len(test_frame), len(feature_names))}")
    if not np.isfinite(values).all():
        raise ValueError("Labels contain missing or infinite values after alignment")
    return (values > 0).astype(np.float64), True


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    """공통 옵션으로 PyTorch DataLoader 인스턴스를 생성한다."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def _build_fold_bundle(config, train_result, val_result, test_result, *, fold_index, fold_count, train_only):
    """한 fold 또는 전체 정상 학습 데이터의 스케일러와 DataLoader를 구성한다."""
    data_cfg = config["data"]
    loader_cfg = config["dataloader"]
    model_name = config["active_model"]
    model_cfg = config["models"][model_name]
    timestamp_col = data_cfg["timestamp_column"]
    feature_names = train_result.feature_columns
    if test_result is not None and feature_names != test_result.feature_columns:
        raise ValueError("Train/test feature schemas differ after preprocessing")
    train_frame = (
        train_result.frame.copy()
        if val_result is None
        else pd.concat([train_result.frame, val_result.frame], ignore_index=True)
    )
    test_frame = train_frame.iloc[:0].copy() if train_only else test_result.frame
    split_index = len(train_result.frame)
    norm_cfg = data_cfg["normalization"]
    scaler = FeatureScaler(
        method=norm_cfg["method"],
        epsilon=norm_cfg["epsilon"],
        clip=norm_cfg["clip"],
        quantile_range=norm_cfg["quantile_range"],
    ).fit(train_frame.iloc[:split_index][feature_names].to_numpy())
    train_values = scaler.transform(train_frame[feature_names].to_numpy())
    test_values = scaler.transform(test_frame[feature_names].to_numpy())
    labels, labels_provided = _load_aligned_labels(
        None if train_only else data_cfg["labels_path"],
        test_frame=test_frame,
        timestamp_col=timestamp_col,
        feature_names=feature_names,
        label_columns=data_cfg["label_columns"],
    )
    expected_interval = data_cfg["expected_interval_seconds"]
    gap_multiplier = data_cfg["max_gap_multiplier"]
    train_timestamps = train_frame[timestamp_col] if timestamp_col else None
    test_timestamps = test_frame[timestamp_col] if timestamp_col else None
    train_segments = _segment_ids(
        train_timestamps,
        len(train_frame),
        expected_interval_seconds=expected_interval,
        max_gap_multiplier=gap_multiplier,
    )
    test_segments = _segment_ids(
        test_timestamps,
        len(test_frame),
        expected_interval_seconds=expected_interval,
        max_gap_multiplier=gap_multiplier,
    )
    window_size = model_cfg["window_size"]
    pad_start = loader_cfg["pad_start"]
    dtype = _torch_dtype(model_cfg["dtype"])
    train_dataset = TimeSeriesWindowDataset(
        train_values[:split_index],
        window_size=window_size,
        timestamps=None if train_timestamps is None else train_timestamps.iloc[:split_index],
        segment_ids=train_segments[:split_index],
        pad_start=pad_start,
        dtype=dtype,
    )
    val_dataset = (
        None
        if val_result is None
        else TimeSeriesWindowDataset(
            train_values[split_index:],
            window_size=window_size,
            timestamps=None if train_timestamps is None else train_timestamps.iloc[split_index:],
            segment_ids=train_segments[split_index:],
            pad_start=pad_start,
            dtype=dtype,
        )
    )
    test_dataset = (
        None
        if train_only
        else TimeSeriesWindowDataset(
            test_values,
            window_size=window_size,
            labels=labels,
            timestamps=test_timestamps,
            segment_ids=test_segments,
            pad_start=pad_start,
            dtype=dtype,
        )
    )
    worker_options = {"num_workers": loader_cfg["num_workers"], "pin_memory": loader_cfg["pin_memory"]}
    return DataBundle(
        train_loader=_loader(
            train_dataset,
            batch_size=loader_cfg["batch_size"],
            shuffle=loader_cfg["shuffle_train"],
            drop_last=loader_cfg["drop_last"] and len(train_dataset) >= loader_cfg["batch_size"],
            **worker_options,
        ),
        val_loader=None
        if val_dataset is None
        else _loader(
            val_dataset,
            batch_size=loader_cfg["eval_batch_size"],
            shuffle=False,
            drop_last=False,
            **worker_options,
        ),
        test_loader=None
        if train_only
        else _loader(
            test_dataset,
            batch_size=loader_cfg["eval_batch_size"],
            shuffle=False,
            drop_last=False,
            **worker_options,
        ),
        scaler=scaler,
        feature_names=feature_names,
        train_frame=train_frame,
        test_frame=test_frame,
        instrument_fault_rows=(pd.DataFrame() if train_only else test_result.instrument_fault_rows.copy()),
        fold_index=fold_index,
        metadata={
            "pipeline": {
                "data": {
                    key: data_cfg[key]
                    for key in ("timestamp_column", "expected_interval_seconds", "max_gap_multiplier")
                },
                "preprocessing": config["preprocessing"],
                "dataloader": loader_cfg,
            },
            # Calibration 파일에서 후처리 설정을 임의로 바꾸지 못하도록 학습 시점 설정을 보관한다.
            "score_processor_settings": config["calibration_rules"]["postprocessing"],
            "active_model": model_name,
            "model_version": config["model_version"],
            "equipment_id": config["equipment_id"],
            "run_id": config["run"]["run_id"],
            "fold_index": fold_index,
            "fold_count": fold_count,
            "window_size": window_size,
            "dtype": str(dtype),
            "train_rows": split_index,
            "validation_rows": len(train_frame) - split_index,
            "test_rows": len(test_frame),
            "test_instrument_fault_rows": 0 if train_only else len(test_result.instrument_fault_rows),
            "train_segments": int(train_segments.max() + 1),
            "test_segments": int(test_segments.max() + 1) if len(test_segments) else 0,
            "labels_provided": labels_provided,
            "normalization": {
                "method": scaler.method,
                "clip": scaler.clip,
                "quantile_range": scaler.quantile_range,
            },
            "preprocessing": {
                "train": train_result.applied_steps,
                "validation": [] if val_result is None else val_result.applied_steps,
                "test": [] if train_only else test_result.applied_steps,
            },
        },
    )


def build_dataloaders(config: Mapping[str, Any]) -> DataBundle:
    """설정에 따라 TimeSeriesSplit 또는 단일 시간순 Holdout DataLoader를 구성한다."""
    data_cfg = config["data"]
    loader_cfg = config["dataloader"]
    timestamp_col = data_cfg["timestamp_column"]
    requested_features = data_cfg["feature_columns"]
    usecols = None if not requested_features else [*([timestamp_col] if timestamp_col else []), *requested_features]
    train_raw = _read_csv(data_cfg["train_path"], usecols=usecols)
    excluded_anomaly_rows = 0
    if data_cfg["train_labels_path"]:
        feature_names = list(requested_features or train_raw.columns.difference([timestamp_col], sort=False))
        train_labels, _ = _load_aligned_labels(
            data_cfg["train_labels_path"],
            test_frame=train_raw,
            timestamp_col=timestamp_col,
            feature_names=feature_names,
            label_columns=data_cfg["label_columns"],
        )
        normal = ~train_labels.astype(bool).any(axis=1)
        excluded_anomaly_rows = int((~normal).sum())
        train_raw = train_raw.loc[normal].reset_index(drop=True)
        if train_raw.empty:
            raise ValueError("이상 라벨 제외 후 정상 학습 데이터가 없습니다")
    train_only = config["run"]["mode"] == "train"
    test_raw = None if train_only else _read_csv(data_cfg["test_path"], usecols=usecols)
    fold_results = preprocess_time_series_splits(
        train_raw,
        config["preprocessing"],
        timestamp_col=timestamp_col,
        feature_columns=requested_features,
        n_splits=loader_cfg["time_series_splits"],
        enabled=loader_cfg["time_series_split_enabled"],
        validation_ratio=loader_cfg["validation_ratio"],
        expected_interval_seconds=data_cfg["expected_interval_seconds"],
        max_gap_multiplier=data_cfg["max_gap_multiplier"],
    )
    test_result = (
        None
        if train_only
        else run_preprocessing(
            test_raw,
            config["preprocessing"],
            timestamp_col=timestamp_col,
            feature_columns=requested_features,
            inference=True,
            expected_interval_seconds=data_cfg["expected_interval_seconds"],
            max_gap_multiplier=data_cfg["max_gap_multiplier"],
        )
    )
    folds = [
        _build_fold_bundle(
            config,
            train_result,
            val_result,
            test_result,
            fold_index=index,
            fold_count=len(fold_results),
            train_only=train_only,
        )
        for index, (train_result, val_result) in enumerate(fold_results, 1)
    ]
    if not loader_cfg["time_series_split_enabled"]:
        holdout = folds[0]
        holdout.folds = []
        holdout.metadata["excluded_anomaly_rows"] = excluded_anomaly_rows
        holdout.metadata["training_scope"] = "time_holdout_train_rows"
        holdout.metadata["validation_strategy"] = "time_holdout"
        return holdout

    # TimeSeriesSplit을 사용할 때만 Fold 결과로 Epoch를 정한 뒤 전체 정상 데이터를 재학습한다.
    full_result = run_preprocessing(
        train_raw,
        config["preprocessing"],
        timestamp_col=timestamp_col,
        feature_columns=requested_features,
        expected_interval_seconds=data_cfg["expected_interval_seconds"],
        max_gap_multiplier=data_cfg["max_gap_multiplier"],
    )
    final = _build_fold_bundle(
        config,
        full_result,
        None,
        test_result,
        fold_index=0,
        fold_count=len(folds),
        train_only=train_only,
    )
    final.folds = folds
    final.metadata["excluded_anomaly_rows"] = excluded_anomaly_rows
    final.metadata["training_scope"] = "all_normal_rows"
    final.metadata["validation_strategy"] = "time_series_split"
    return final


@dataclass
class PredictionData:
    """학습 시 저장한 처리 규칙으로 만든 추론 입력과 원본 단위 데이터."""

    loader: DataLoader
    frame: pd.DataFrame
    scaler: FeatureScaler
    feature_names: list[str]
    labels_provided: bool = False
    instrument_fault_rows: pd.DataFrame = field(default_factory=pd.DataFrame)


def evaluation_loader(dataset, loader_config):
    """셔플·마지막 배치 삭제 없이 모든 윈도우를 시간순으로 평가한다."""
    return _loader(
        dataset,
        batch_size=loader_config["eval_batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
    )


def _slice_time_range(frame, timestamp_col, input_start, input_end):
    """양쪽 경계를 포함해 분석 구간만 남기고 잘못된 시각 비교를 명확히 보고한다."""
    if input_start is None and input_end is None:
        return frame
    if timestamp_col is None:
        raise ValueError("분석 시간 범위를 사용하려면 data.timestamp_column이 필요합니다")
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    try:
        mask = pd.Series(True, index=frame.index)
        if input_start is not None:
            mask &= timestamps >= pd.Timestamp(input_start)
        if input_end is not None:
            mask &= timestamps <= pd.Timestamp(input_end)
    except (TypeError, ValueError) as exc:
        raise ValueError("분석 시간 범위와 입력 타임스탬프의 형식 또는 시간대가 일치하지 않습니다") from exc
    selected = frame.loc[mask].reset_index(drop=True)
    if selected.empty:
        raise ValueError("지정한 분석 시간 범위에 모델 입력 데이터가 없습니다")
    return selected


# save_report()가 기록하는 predictions.csv의 시각 컬럼과 원본 단위 입력값 접미사.
PREDICTION_REPORT_TIMESTAMP = "timestamp"
PREDICTION_REPORT_ACTUAL_SUFFIX = "__actual"


def prediction_report_to_input(frame, timestamp_col, feature_names):
    """predictions.csv(full·evaluate·online 결과)를 학습 CSV와 같은 시각·피처 형식으로 되돌린다.

    ``<피처>__actual``은 모델에 입력된 원본 단위 값이므로 Checkpoint Scaler로 다시 정규화할 수 있다.
    계기 이상 행은 ``__actual``이 비어 있으므로 일반 입력과 같이 추론 대상에서 제외된다.
    이미 원본 입력 형식이거나 결과 파일 형식이 아니면 그대로 반환한다.
    """
    actual = {f"{name}{PREDICTION_REPORT_ACTUAL_SUFFIX}": name for name in feature_names}
    if set(feature_names) <= set(frame.columns) or not set(actual) <= set(frame.columns):
        return frame
    if timestamp_col is None:
        raise ValueError("predictions.csv를 입력으로 사용하려면 data.timestamp_column이 필요합니다")
    if PREDICTION_REPORT_TIMESTAMP not in frame.columns:
        raise ValueError(f"predictions.csv에 {PREDICTION_REPORT_TIMESTAMP} 컬럼이 없습니다")
    columns = {PREDICTION_REPORT_TIMESTAMP: timestamp_col, **actual}
    return frame[list(columns)].rename(columns=columns)


def build_prediction_data(
    frame,
    data_state,
    model_config,
    *,
    labels_path=None,
    label_columns=None,
    preprocessed=False,
    last_only=False,
    input_start=None,
    input_end=None,
):
    """체크포인트의 스케일러·특성 순서·전처리로 CSV 또는 DataFrame을 변환한다.

    학습 CSV 없이도 추론할 수 있으며 통계를 새로 학습하지 않는다. 분석 시간 범위는
    전처리와 Window 생성 전에 적용하여 범위 밖 관측값을 처리하거나 섞지 않는다.
    """
    pipeline = data_state["metadata"]["pipeline"]
    data = pipeline["data"]
    features = list(data_state["feature_names"])
    scaler = FeatureScaler.from_state_dict(data_state["scaler"])
    if len(scaler.scale) != len(features):
        raise ValueError("Scaler and feature schema sizes differ")
    raw = frame if isinstance(frame, pd.DataFrame) else _read_csv(frame)
    raw = prediction_report_to_input(raw, data["timestamp_column"], features)
    raw = _slice_time_range(raw, data["timestamp_column"], input_start, input_end)
    result = run_preprocessing(
        raw,
        {"steps": []} if preprocessed else pipeline["preprocessing"],
        timestamp_col=data["timestamp_column"],
        feature_columns=features,
        inference=not preprocessed,
        expected_interval_seconds=data["expected_interval_seconds"],
        max_gap_multiplier=data["max_gap_multiplier"],
    )
    timestamp_col = data["timestamp_column"]
    selected_frame = result.frame
    instrument_fault_rows = result.instrument_fault_rows
    values = scaler.transform(selected_frame[features].to_numpy())
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("추론 데이터는 비어 있지 않은 유한한 수치여야 합니다")
    timestamps = selected_frame[timestamp_col] if timestamp_col else None
    if timestamps is not None and not timestamps.is_monotonic_increasing:
        raise ValueError("추론 데이터는 시간 오름차순이어야 합니다")
    labels, provided = _load_aligned_labels(
        labels_path,
        test_frame=selected_frame,
        timestamp_col=timestamp_col,
        feature_names=features,
        label_columns=label_columns,
    )
    dataset = TimeSeriesWindowDataset(
        values,
        window_size=model_config["window_size"],
        labels=labels,
        timestamps=timestamps,
        segment_ids=_segment_ids(
            timestamps,
            len(values),
            expected_interval_seconds=data["expected_interval_seconds"],
            max_gap_multiplier=data["max_gap_multiplier"],
        ),
        pad_start=pipeline["dataloader"]["pad_start"],
        dtype=_torch_dtype(model_config["dtype"]),
    )
    if last_only:
        dataset.end_indices = dataset.end_indices[-1:]
    return PredictionData(
        evaluation_loader(dataset, pipeline["dataloader"]),
        selected_frame,
        scaler,
        features,
        provided,
        instrument_fault_rows,
    )
