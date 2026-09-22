"""빠른 Tag 그룹 SHAP: 작은 센서 집합은 정확 열거, 큰 집합은 순열 근사.

설명 대상은 f(x)=abs(x[-1]-TranAD_decoder2(x))의 선택 Tag 또는 전체 평균이다.
Tag 하나가 하나의 player이며 해당 Tag의 전체 과거·현재 window를 함께
Train80 중앙값으로 치환한다. 치환한 현재값을 복원오차의 target에도 사용한다.
따라서 서로 다른 함수를 혼합한 단순 perturbation 순위가 아닌 Shapley 값이다.

단일 중앙값 기준점에 대한 interventional group SHAP이며 인과 효과나 조건부
분포 추정이 아니다. 8개 이하에서는 모든 coalition의 정확 가중합, 그 이상은
고정 seed 순열과 역순열의 주변 기여도 평균을 사용한다. 외부 shap 패키지를
호출하지 않고 동일한 Shapley 정의를 직접 계산한다. 공식 개념 참고:
https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html
https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import torch

from .config import get_config
from .data import WindowDataset, file_sha256, load_scaler, read_sensor_csv
from .model import build_model
from .trainer import canonical_scaler_state, load_trusted_checkpoint


def coalition_plan(n_features, exact_max_features=8, permutations=16, seed=42):
    """반환: mask[M,F], 순열별(feature 순서, coalition index 경로), exact 여부."""
    if n_features < 1 or exact_max_features < 1 or exact_max_features > 8 or permutations < 1:
        raise ValueError('유효한 Tag 수, exact 상한1~8, 순열 반복수가 필요합니다.')
    if n_features <= exact_max_features:
        ids = list(range(1 << n_features))
        paths = []
        exact = True
    else:
        rng = np.random.default_rng(seed)
        ids = [0, (1 << n_features)-1]
        seen = set(ids)
        unindexed = []
        for _ in range(permutations):
            order = rng.permutation(n_features).tolist()
            for direction in (order, list(reversed(order))):
                active, path = 0, [0]
                for feature in direction:
                    active |= 1 << feature
                    path.append(active)
                    if active not in seen:
                        seen.add(active)
                        ids.append(active)
                unindexed.append((direction, path))
        lookup = {coalition: index for index, coalition in enumerate(ids)}
        paths = [(order, [lookup[coalition] for coalition in path]) for order, path in unindexed]
        exact = False
    masks = np.array([[(coalition >> feature) & 1 for feature in range(n_features)]
                      for coalition in ids], dtype=bool)
    return masks, paths, exact


def exact_shapley(coalition_values, n_features):
    """binary mask 순서의 f(S)[N,2^F]에서 정확 Shapley 값을 계산한다."""
    values = np.asarray(coalition_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 1 << n_features or not np.isfinite(values).all():
        raise ValueError('coalition 값은 유한한 [N,2^F] 행렬이어야 합니다.')
    result = np.zeros((len(values), n_features), dtype=np.float64)
    for feature in range(n_features):
        absent = [mask for mask in range(1 << n_features) if not mask & (1 << feature)]
        weights = np.array([1./(n_features*math.comb(n_features-1, mask.bit_count()))
                            for mask in absent])
        present = [mask | (1 << feature) for mask in absent]
        result[:, feature] = ((values[:, present]-values[:, absent])*weights).sum(axis=1)
    return result


def permutation_shapley(coalition_values, paths, n_features):
    """완전한 forward/reverse 순열의 주변 기여도 평균. 근사임을 명시한다."""
    values = np.asarray(coalition_values, dtype=np.float64)
    if values.ndim != 2 or not paths or not np.isfinite(values).all():
        raise ValueError('유효한 coalition 값과 순열 경로가 필요합니다.')
    result = np.zeros((len(values), n_features), dtype=np.float64)
    for order, path in paths:
        if sorted(order) != list(range(n_features)) or len(path) != n_features+1:
            raise ValueError('모든 Tag를 정확히 한 번 포함한 전체 순열이 필요합니다.')
        for step, feature in enumerate(order):
            result[:, feature] += values[:, path[step+1]]-values[:, path[step]]
    return result/len(paths)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class ExplanationEngine:
    """신뢰하는 저장 모델의 CUDA 추론만 수행한다. 학습·재fit은 수행하지 않는다.

    CSV 시간이 timezone-naive이면 같은 현지 시각으로 해석하며 timezone이
    명시된 사용자 입력은 Asia/Seoul로 변환한다. timezone-aware CSV라면 입력도
    그 timezone으로 정렬한다. [start, start+minutes) 원본 시점만 설명한다.
    """
    def __init__(self, run_dir, checkpoint_path):
        if not torch.cuda.is_available():
            raise RuntimeError('SHAP 모델 평가는 CUDA가 필요합니다. CPU 추론으로 대체하지 않습니다.')
        self.run_dir = Path(run_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.predictions_path = self.run_dir/'predictions.npz'
        if not self.predictions_path.is_file():
            raise ValueError('먼저 -eval/-full로 predictions.npz를 생성해야 합니다.')
        self.config = get_config()
        saved_config = _read_json(self.run_dir/'config.json')
        self.options = {**self.config['explain'], **saved_config.get('explain', {})}
        self.config.update(saved_config)
        self.device = torch.device('cuda:0')
        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)
        self.predictions_sha256 = file_sha256(self.predictions_path)
        checkpoint = load_trusted_checkpoint(self.checkpoint_path, map_location='cpu')
        self.scaler_payload = load_scaler(self.checkpoint_path.parent/'scaler.pkl')
        if canonical_scaler_state(checkpoint['scaler_state']) != canonical_scaler_state(self.scaler_payload):
            raise ValueError('Checkpoint와 저장 scaler의 학습 상태가 다릅니다.')
        self.features = list(checkpoint['feature_names'])
        if self.features != self.scaler_payload['features']:
            raise ValueError('Checkpoint와 scaler의 Tag 순서가 다릅니다.')
        if any(name.casefold().startswith('alarm_') or name.casefold() == 'inhibit' for name in self.features):
            raise ValueError('채점용 Label/Inhibit는 모델 feature에 포함할 수 없습니다.')
        self.model = build_model(len(self.features), checkpoint['model_cfg']).to(self.device).eval()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.requires_grad_(False)
        self.dtype = next(self.model.parameters()).dtype
        self.window = int(checkpoint['model_cfg']['window'])
        with np.load(self.predictions_path, allow_pickle=False) as data:
            if data['features'].tolist() != self.features:
                raise ValueError('평가 데이터와 checkpoint의 Tag 순서가 다릅니다.')
            # 이미 저장된 정규화 입력이다. 여기서 scaler.transform을 다시 하지 않는다.
            self.actual_scaled = np.asarray(data['actual_scaled'], dtype=np.float32).copy()
            self.saved_scores = np.asarray(data['scores'], dtype=np.float64).copy()
            timestamps = data['timestamps'].astype(str)
        if (self.actual_scaled.shape != self.saved_scores.shape or
                self.actual_scaled.ndim != 2 or self.actual_scaled.shape[1] != len(self.features) or
                not np.isfinite(self.actual_scaled).all() or not np.isfinite(self.saved_scores).all()):
            raise ValueError('저장된 정규화 입력/Score shape 또는 수치가 유효하지 않습니다.')
        self.times = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if len(self.times) != len(self.actual_scaled) or self.times.hasnans or not self.times.is_unique or not self.times.is_monotonic_increasing:
            raise ValueError('평가 시각은 유일하고 증가하는 원본 시각이어야 합니다.')
        # Pandas3는 datetime64[us]도 유지한다. asi8의 정수 단위를 ns로
        # 재해석하면 60초가 0.06초가 되어 모든 행을 gap으로 오인한다.
        times_np = self.times.to_numpy(dtype='datetime64[ns]')
        self.dataset = WindowDataset(self.actual_scaled, times_np, self.window,
                                     self.config['data']['sample_minutes'])
        provenance_path = self.run_dir/'eval_provenance.json'
        if provenance_path.is_file() and _read_json(provenance_path)['checkpoint_sha256'] != self.checkpoint_sha256:
            raise ValueError('선택 checkpoint는 해당 평가 결과를 만든 모델과 다릅니다.')
        profile = _read_json(self.run_dir/'data_profile.json')['train']
        train_path = Path(profile['path'])
        if not train_path.is_file() or file_sha256(train_path) != profile['sha256']:
            raise ValueError('SHAP 기준점에 사용할 원본 Train 파일 또는 hash가 다릅니다.')
        train = read_sensor_csv(train_path, self.config['data'], self.features)
        cut = int(self.scaler_payload['training_rows'])
        if not 1 <= cut < len(train['values']):
            raise ValueError('Checkpoint의 Train80 분할 위치가 유효하지 않습니다.')
        self.background_raw = np.median(train['values'][:cut], axis=0)
        self.background_scaled = self.scaler_payload['scaler'].transform(self.background_raw[None, :])[0].astype(np.float32)
        self.background_info = {
            'method': 'Train 앞80% Tag별 원단위 중앙값 → 저장 scaler로 한 번 변환',
            'training_rows': cut, 'train_sha256': profile['sha256'],
            'raw_per_tag': dict(zip(self.features, map(float, self.background_raw))),
            'normalized_per_tag': dict(zip(self.features, map(float, self.background_scaled))),
            'one_reference_vector_repeated_across_window': True,
            'fit_performed': False, 'labels_used': False, 'inhibit_used': False,
            'interpretation': '단일 중앙값 기준점의 interventional group SHAP; 인과·조건부 SHAP 아님',
        }
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def _interval(self, start_time, duration_minutes):
        if isinstance(duration_minutes, bool):
            raise ValueError('기간은 정수 분으로 입력하세요.')
        duration = int(duration_minutes)
        if float(duration_minutes) != duration or not 1 <= duration <= int(self.options['max_duration_minutes']):
            raise ValueError(f'기간은 1~{self.options["max_duration_minutes"]}분의 정수여야 합니다.')
        start = pd.Timestamp(start_time)
        if pd.isna(start):
            raise ValueError('유효한 시작 시각이 필요합니다.')
        if self.times.tz is None:
            if start.tzinfo is not None:
                start = start.tz_convert('Asia/Seoul').tz_localize(None)
        else:
            start = start.tz_localize(self.times.tz) if start.tzinfo is None else start.tz_convert(self.times.tz)
        end = start+pd.Timedelta(minutes=duration)
        indices = np.flatnonzero((self.times >= start) & (self.times < end))
        if not len(indices):
            raise ValueError('선택 구간에 원본 데이터가 없습니다. 데이터 시각 범위 안에서 선택하세요.')
        if len(indices) > int(self.options['max_duration_minutes']):
            raise ValueError('1분 주기 기준 최대1440개 원본 시점만 한 번에 설명할 수 있습니다.')
        return start, end, indices

    def _coalition_scores(self, windows, masks, target_index):
        """GPU에 전체 coalition 텐서를 만들지 않고 제한된 minibatch로 평가한다."""
        n_rows, n_masks = len(windows), len(masks)
        output = np.empty((n_rows*n_masks,), dtype=np.float64)
        source = windows.to(device=self.device, dtype=self.dtype)
        reference = torch.as_tensor(self.background_scaled, device=self.device, dtype=self.dtype)
        mask_tensor = torch.as_tensor(masks, device=self.device, dtype=torch.bool)
        batch_size = int(self.options['inference_batch_size'])
        if batch_size < 1:
            raise ValueError('SHAP inference_batch_size는 양수여야 합니다.')
        self.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(output), batch_size):
                stop = min(offset+batch_size, len(output))
                flat = torch.arange(offset, stop, device=self.device)
                masked = torch.where(mask_tensor[flat % n_masks, None, :],
                                     source[torch.div(flat, n_masks, rounding_mode='floor')],
                                     reference[None, None, :])
                _, reconstruction = self.model(masked)
                # f(masked)와 f(original)의 정의가 같다. original target을 여기
                # 재사용하면 다른 게임을 설명하게 되어 Shapley 해석이 틀어진다.
                error = (masked[:, -1, :]-reconstruction).abs()
                values = error.mean(dim=1) if target_index is None else error[:, target_index]
                if not torch.isfinite(values).all():
                    raise FloatingPointError('SHAP coalition 추론에서 유효하지 않은 값이 나왔습니다.')
                output[offset:stop] = values.cpu().numpy()
        return output.reshape(n_rows, n_masks)

    def explain(self, start_time, duration_minutes, target_tag=None):
        started = time.perf_counter()
        start, end, indices = self._interval(start_time, duration_minutes)
        lookup = {tag.casefold(): index for index, tag in enumerate(self.features)}
        if target_tag is None or target_tag == '':
            target_index = None
        elif str(target_tag).casefold() not in lookup:
            raise ValueError(f'존재하지 않는 설명 대상 Tag: {target_tag}')
        else:
            target_index = lookup[str(target_tag).casefold()]
        key = (start.isoformat(), end.isoformat(), target_index)
        with self._lock:
            if key in self._cache:
                cached = copy.deepcopy(self._cache.pop(key))
                self._cache[key] = copy.deepcopy(cached)
                cached.update(cache_hit=True, elapsed_seconds=time.perf_counter()-started)
                return cached
            masks, paths, exact = coalition_plan(len(self.features),
                int(self.options['exact_max_features']), int(self.options['permutations']),
                int(self.config['train'].get('seed', 42)))
            windows = torch.stack([self.dataset[int(index)][0] for index in indices])
            scores = self._coalition_scores(windows, masks, target_index)
            shap_values = (exact_shapley(scores, len(self.features)) if exact else
                           permutation_shapley(scores, paths, len(self.features)))
            empty_index = int(np.flatnonzero(~masks.any(axis=1))[0])
            full_index = int(np.flatnonzero(masks.all(axis=1))[0])
            baseline, model_output = scores[:, empty_index], scores[:, full_index]
            expected = (self.saved_scores[indices].mean(axis=1) if target_index is None else
                        self.saved_scores[indices, target_index])
            if not np.allclose(model_output, expected, rtol=1e-4, atol=5e-5):
                raise ValueError('SHAP 실제 입력 점수와 저장된 평가 Score가 다릅니다. 모델/입력/정규화를 확인하세요.')
            residual = model_output-baseline-shap_values.sum(axis=1)
            if not np.isfinite(shap_values).all() or np.max(np.abs(residual)) > 1e-7*max(1., float(np.max(np.abs(model_output)))):
                raise FloatingPointError('SHAP 가산성 검증에 실패했습니다.')
            signed, absolute = shap_values.mean(axis=0), np.abs(shap_values).mean(axis=0)
            result = {
                'schema_version': 1, 'features': self.features, 'feature_names': self.features,
                'method': 'exact_group_shapley_all_coalitions' if exact else 'permutation_group_shapley_antithetic',
                'exact': exact, 'n_samples': len(indices), 'coalitions_per_sample': len(masks),
                'permutation_paths': 0 if exact else len(paths),
                'seed': int(self.config['train'].get('seed', 42)),
                'start_time': start.isoformat(), 'end_time_exclusive': end.isoformat(),
                'duration_minutes': int(duration_minutes),
                'target_tag': None if target_index is None else self.features[target_index],
                'score_definition': 'abs(masked_current - TranAD_decoder2(masked_window)); 전체 Tag 평균 또는 선택 Tag',
                'prediction': float(model_output.mean()), 'base_value': float(baseline.mean()),
                'sum_contributions': float(signed.sum()), 'additivity_residual': float(residual.mean()),
                'max_abs_additivity_residual': float(np.abs(residual).max()),
                'saved_score_max_abs_difference': float(np.abs(model_output-expected).max()),
                'mean_signed_contributions': signed.tolist(), 'mean_abs_contributions': absolute.tolist(),
                'contributions': [{'tag': tag, 'mean_signed': float(signed[j]), 'mean_abs': float(absolute[j])}
                                  for j, tag in enumerate(self.features)],
                'per_minute': [{'timestamp': self.times[index].isoformat(), 'baseline': float(baseline[j]),
                               'model_output': float(model_output[j]), 'shap_values': shap_values[j].tolist(),
                               'additivity_residual': float(residual[j])} for j, index in enumerate(indices)],
                'background': self.background_info, 'checkpoint_sha256': self.checkpoint_sha256,
                'predictions_sha256': self.predictions_sha256, 'device': str(self.device),
                'cache_hit': False, 'elapsed_seconds': time.perf_counter()-started,
                'notes': ['각 player는 Tag 전체 window이며 시차별 기여도를 별도로 분해하지 않습니다.',
                          'MAE 이상 점수의 설명입니다. Threshold 초과·정지 억제를 포함한 최종 Alarm 자체의 설명은 아닙니다.',
                          '양수는 기준점 대비 점수 증가, 음수는 점수 감소 기여입니다.',
                          '순열 근사는 가산성이 맞아도 정확한 Shapley 값으로 수렴했음을 보장하지 않습니다.'],
                'references': ['https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html',
                               'https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html'],
            }
            self._cache[key] = copy.deepcopy(result)
            while len(self._cache) > max(0, int(self.options['cache_size'])):
                self._cache.popitem(last=False)
            return result
