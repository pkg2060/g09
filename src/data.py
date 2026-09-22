"""원본 행을 보존하는 CSV 검증, 시간순 분할·정규화 및 인과적 윈도우.

Alarm_*와 Inhibit는 별도 채점 배열로만 반환하며 values/WindowDataset에
포함하지 않는다. 시각 불연속과 Train/Validation 경계에서는 새 구간의 첫
측정값만 왼쪽 패딩하므로 미래값이나 다른 분할의 행을 사용하지 않는다.
"""
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, RandomSampler

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()

def read_sensor_csv(path, data_cfg, features=None):
    """원본 행·센서 이름을 보존하고 선택적인 채점 열을 분리한다.

    센서 suffix, Alarm prefix, Inhibit, timestamp 및 Train/Test 대응은
    대소문자를 구분하지 않는다. 모호한 중복 열은 실패시킨다. Test의 표시명은
    Train feature 순서에 맞추고 실제 CSV 열 이름은 source_frame/profile의
    column_mapping에 보존한다. Label/Inhibit가 없으면 NaN이며 정상0이 아니다.
    """
    with open(path, encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    folded_header = [str(column).casefold() for column in header]
    if len(folded_header) != len(set(folded_header)):
        raise ValueError('Ambiguous duplicate CSV columns (case-insensitive); original file preserved')
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f'CSV is empty: {path}')
    columns = {str(column).casefold(): column for column in frame.columns}
    stamp_requested = data_cfg.get('timestamp', 'DateTime')
    stamp = columns.get(stamp_requested.casefold())
    if stamp is None:
        raise ValueError(f'Missing timestamp column {stamp_requested}: {path}')
    frame[stamp] = pd.to_datetime(frame[stamp], errors='raise')
    if frame[stamp].isna().any() or frame[stamp].duplicated().any():
        raise ValueError('Missing/duplicate timestamps require explicit data correction; no rows deleted')
    if not frame[stamp].is_monotonic_increasing:
        raise ValueError('CSV must be chronological; original order is preserved')
    suffix = data_cfg.get('sensor_suffix', '.PV').casefold()
    label_prefix = data_cfg.get('label_prefix', 'Alarm_')
    stop_requested = data_cfg.get('stop_label', 'Inhibit')
    candidates = [c for c in frame if c.casefold().endswith(suffix)
                  and not c.casefold().startswith(label_prefix.casefold())
                  and c.casefold() != stop_requested.casefold()]
    if features is None:
        features = candidates
    features = list(features)
    feature_keys = [tag.casefold() for tag in features]
    candidate_keys = {tag.casefold() for tag in candidates}
    if not features or len(set(feature_keys)) != len(features) or set(feature_keys) != candidate_keys:
        raise ValueError(f'Sensor schema mismatch: expected {features}, found {candidates}')
    feature_columns = {tag: columns[tag.casefold()] for tag in features}
    values = frame[list(feature_columns.values())].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite sensor measurements; no arbitrary row deletion/imputation is allowed')
    labels = np.full(values.shape, np.nan, dtype=np.float64)
    label_columns = {}
    for j, tag in enumerate(features):
        col = columns.get((label_prefix + tag).casefold())
        label_columns[tag] = col
        if col is not None:
            label = pd.to_numeric(frame[col], errors='raise').to_numpy(dtype=float)
            if np.isinf(label).any() or not np.isin(label[np.isfinite(label)], [0, 1]).all():
                raise ValueError(f'Non-binary label in {col}')
            labels[:, j] = label
    stop_col = columns.get(stop_requested.casefold())
    if stop_col is not None:
        inhibit = pd.to_numeric(frame[stop_col], errors='raise').to_numpy(dtype=float)
        if np.isinf(inhibit).any() or not np.isin(inhibit[np.isfinite(inhibit)], [0, 1]).all():
            raise ValueError('Inhibit must be 0=operating, 1=stopped or missing')
    else:
        inhibit = np.full(len(frame), np.nan)
    delta = frame[stamp].diff().dt.total_seconds().to_numpy()
    expected = data_cfg.get('sample_minutes', 1)*60
    profile = {'path': str(Path(path).resolve()), 'sha256': file_sha256(path),
               'rows': len(frame), 'columns': list(frame.columns), 'features': features,
               'column_mapping': {'timestamp': stamp, 'sensors': feature_columns,
                                  'labels': label_columns, 'inhibit': stop_col},
               'truth_columns_optional': True,
               'missing_truth_means': 'unavailable_not_normal',
               'start': str(frame[stamp].iloc[0]), 'end': str(frame[stamp].iloc[-1]),
               'timestamp_gaps': int(np.sum(delta[1:] != expected)),
               'sensor_missing': int(np.isnan(values).sum()), 'rows_deleted': 0,
               'labels_present': {tag: int(np.isfinite(labels[:,j]).sum()) for j,tag in enumerate(features)},
               'positive_labels': {tag: int(np.nansum(labels[:,j])) for j,tag in enumerate(features)},
               'inhibit_counts': {str(i): int(np.sum(inhibit == i)) for i in [0,1]},
               'minimum': dict(zip(features, map(float, values.min(axis=0)))),
               'maximum': dict(zip(features, map(float, values.max(axis=0))))}
    aliases = {original: expected for expected, original in feature_columns.items()
               if original != expected}
    if stamp != stamp_requested:
        aliases[stamp] = stamp_requested
    aligned_frame = frame.copy(deep=False)
    aligned_frame.columns = [aliases.get(column, column) for column in frame.columns]
    return {'frame': aligned_frame, 'source_frame': frame, 'values': values, 'timestamps': frame[stamp].to_numpy(),
            'features': features, 'labels': labels, 'inhibit': inhibit, 'profile': profile}

def fit_training_scaler(values, validation_fraction=.2):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError('Scaler requires a finite two-dimensional sensor matrix')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must lie strictly between zero and one')
    cut = int(len(values)*(1-validation_fraction))
    if cut < 2 or cut >= len(values):
        raise ValueError('Train and validation must both contain rows')
    scaler = MinMaxScaler(clip=False)
    scaler.fit(values[:cut])
    return scaler, cut

def save_scaler(scaler, features, cut, path):
    payload = {'scaler': scaler, 'features': features, 'training_rows': cut,
               'fit_partition': 'first_chronological_training_partition_only', 'clip': False}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    return payload

def load_scaler(path):
    # Only load this project's own saved artifact, never untrusted pickle files.
    with open(path, 'rb') as f:
        return pickle.load(f)

class WindowDataset(Dataset):
    """One sample per original timestamp, including the start of every segment.

    Current-point reconstruction: window[-1] == target. A timestamp gap starts
    a fresh segment; only its first value pads unavailable left history. Thus
    neither split leakage nor gap-crossing nor deletion of initial rows occurs.
    """
    def __init__(self, values, timestamps, window=10, sample_minutes=1):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError('Windows require a finite two-dimensional sensor matrix')
        self.values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
        self.window = int(window)
        if self.window < 1 or not len(values):
            raise ValueError('Empty data or invalid window')
        times = np.asarray(timestamps, dtype='datetime64[ns]')
        if times.ndim != 1 or len(times) != len(values) or np.isnat(times).any():
            raise ValueError('Each sensor row must have exactly one valid timestamp')
        if np.any(np.diff(times) <= np.timedelta64(0, 'ns')):
            raise ValueError('Window timestamps must be strictly increasing')
        interval_ns = round(float(sample_minutes) * 60 * 1e9)
        if interval_ns <= 0:
            raise ValueError('sample_minutes must be positive')
        starts = np.r_[True, np.diff(times) != np.timedelta64(interval_ns, 'ns')]
        self.segment_start = np.maximum.accumulate(np.where(starts, np.arange(len(times)), 0))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        first = max(int(self.segment_start[index]), index-self.window+1)
        window = self.values[first:index+1]
        if len(window) < self.window:
            window = torch.cat((window[0:1].expand(self.window-len(window), -1), window), 0)
        return window, self.values[index]

def make_loader(values, timestamps, config, training=False):
    dataset = WindowDataset(values, timestamps, config['model']['window'], config['data']['sample_minutes'])
    workers = int(config['train'].get('num_workers', 0))
    generator = torch.Generator().manual_seed(config['train']['seed'])
    # Worker startup must not consume the sampler's random stream: otherwise a
    # resumed run with new workers uses a different row permutation.
    sampler = (RandomSampler(dataset, generator=torch.Generator().manual_seed(config['train']['seed']))
               if training and config['train'].get('shuffle', True) else None)
    batch_size = int(config['train']['batch_size'] if training else
                     config['train'].get('inference_batch_size', config['train']['batch_size']))
    if workers < 0 or batch_size < 1:
        raise ValueError('num_workers must be nonnegative and batch_size positive')
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, sampler=sampler,
                      drop_last=False, pin_memory=torch.cuda.is_available(), num_workers=workers,
                      persistent_workers=workers>0, generator=generator)
