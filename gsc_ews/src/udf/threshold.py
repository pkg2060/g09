"""학습 점수에서 변수별 Manual·POT·Percentile·Sigma 임계값을 계산한다."""

import warnings

import numpy as np
from scipy.stats import genpareto


def compute_thresholds(train_scores, settings, feature_names):
    """테스트 점수를 받지 않는 인터페이스로 임계값 정보 누출을 막는다.

    POT는 학습 점수의 상위 초과량에 Generalized Pareto Distribution을 맞춘다.
    꼬리 표본이 부족하면 같은 학습 구간의 백분위수로 대체한다.
    """
    scores = np.asarray(train_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(feature_names) or not len(scores):
        raise ValueError("학습 점수와 특성 스키마가 일치해야 합니다")
    if not np.isfinite(scores).all():
        raise ValueError("학습 점수에 NaN 또는 무한대가 있습니다")
    method = settings["method"]
    if method == "manual":
        thresholds = np.full(scores.shape[1], settings["default"], dtype=float)
    elif method == "percentile":
        thresholds = np.percentile(scores, settings["percentile"], axis=0)
    elif method == "sigma":
        thresholds = scores.mean(axis=0) + settings["sigma_k"] * scores.std(axis=0)
    elif method == "pot":
        thresholds = np.percentile(scores, settings["percentile"], axis=0)
        for index, values in enumerate(scores.T):
            initial = np.quantile(values, settings["pot_init_quantile"])
            excess = values[values > initial] - initial
            if len(excess) < 5 or np.ptp(excess) == 0:
                warnings.warn(
                    f"{feature_names[index]}: POT 꼬리 표본 부족, 학습 백분위수 사용",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            try:
                shape, _, scale = genpareto.fit(excess, floc=0)
                # 전체 초과 확률 = 꼬리 표본 비율 × 조건부 꼬리 확률
                probability = settings["pot_risk"] * len(values) / len(excess)
                if not 0 < probability < 1:
                    raise ValueError("POT risk must be smaller than the tail fraction")
                threshold = initial + genpareto.isf(probability, shape, loc=0, scale=scale)
                threshold *= settings["pot_multiplier"]
                if not np.isfinite(threshold) or threshold < initial:
                    raise ValueError("POT produced an invalid threshold")
                thresholds[index] = threshold
            except (ValueError, FloatingPointError, RuntimeError) as error:
                warnings.warn(
                    f"{feature_names[index]}: POT 실패, 학습 백분위수 사용: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    # 변수별 수동 오버라이드 적용
    for feature, value in settings["per_variable"].items():
        if isinstance(feature, int) and not isinstance(feature, bool) and 0 <= feature < len(feature_names):
            index = feature
        elif feature in feature_names:
            index = feature_names.index(feature)
        else:
            raise ValueError(f"임계값 설정에 없는 특성이 있습니다: {feature}")
        thresholds[index] = value
    if not np.isfinite(thresholds).all() or np.any(thresholds < 0):
        raise ValueError("임계값은 0 이상의 유한한 값이어야 합니다")
    return thresholds
