"""학습 점수로만 보정하고 미래값을 사용하지 않는 공통 점수 후처리."""

from collections import deque

import numpy as np


class ScoreProcessor:
    """시간 단절마다 이동 평균을 초기화하고 선택한 점수 정규화 통계를 재사용한다."""

    def __init__(self, settings, location=None, scale=None, sample_scale=None):
        """설정과 선택적인 저장 통계로 처리기를 구성한다."""
        self.settings = dict(settings)
        method = self.settings.get("normalization_method")
        if method is None:
            # 기존 Calibration 파일은 평가·온라인에서 계속 사용할 수 있도록 한 번만 변환한다.
            if self.settings.pop("quantile_normalization", False):
                method = "quantile"
            elif self.settings.pop("robust_normalization", False):
                method = "robust"
            else:
                method = "none"
            self.settings["normalization_method"] = method
        if method not in {"none", "robust", "quantile"}:
            raise ValueError("normalization_method는 none, robust, quantile 중 하나여야 합니다")
        self.normalization_method = method
        self.location = None if location is None else np.asarray(location, dtype=float)
        self.scale = None if scale is None else np.asarray(scale, dtype=float)
        self.sample_scale = None if sample_scale is None else float(sample_scale)
        self.history = deque(maxlen=settings["smooth_window"])
        self.previous_segment = None

    def reset(self):
        """새 데이터 분할이나 스트림에서 이전 이동 평균 이력을 비운다."""
        self.history.clear()
        self.previous_segment = None

    def _smooth(self, scores, segments):
        """현재와 과거 점수만 평균하며 미래 점수가 과거 결과를 바꾸지 못하게 한다."""
        result = []
        for score, segment in zip(scores, segments, strict=False):
            if segment != self.previous_segment:
                self.history.clear()
            self.history.append(score)
            self.previous_segment = segment
            result.append(np.mean(self.history, axis=0))
        return np.asarray(result)

    def fit(self, train_scores, segments):
        """검증 점수에서 Robust 또는 Quantile 정규화 통계를 계산한다."""
        self.reset()
        scores = self._smooth(train_scores, segments)
        if scores.ndim != 2 or not len(scores) or not np.isfinite(scores).all():
            raise ValueError("학습 점수는 비어 있지 않은 유한한 2차원 배열이어야 합니다")
        self.location = np.zeros(scores.shape[1])
        self.scale = np.ones(scores.shape[1])
        self.sample_scale = 1.0
        if self.normalization_method == "quantile":
            level = float(self.settings.get("quantile_level", 0.99))
            self.scale = np.maximum(np.quantile(scores, level, axis=0), 1e-8)
            self.sample_scale = max(float(np.quantile(scores.mean(axis=1), level)), 1e-8)
        elif self.normalization_method == "robust":
            self.location = np.median(scores, axis=0)
            mad = np.median(np.abs(scores - self.location), axis=0)
            # 정규분포에서는 1.4826 * MAD가 표준편차에 근접한다.
            self.scale = 1.4826 * mad + 1e-8
        self.reset()
        return self

    def transform(self, scores, segments, *, reset=True):
        """학습 통계로 변환하며 음수 편차는 정상 영역인 0으로 제한한다."""
        if self.location is None or self.scale is None:
            raise RuntimeError("학습 점수로 먼저 fit해야 합니다")
        scores = np.asarray(scores, dtype=float)
        if (
            scores.ndim != 2
            or scores.shape[1] != len(self.scale)
            or len(scores) != len(segments)
            or not np.isfinite(scores).all()
        ):
            raise ValueError("점수·특성·구간 길이가 일치해야 합니다")
        if reset:
            self.reset()
        smoothed = self._smooth(scores, segments)
        normalized = np.maximum((smoothed - self.location) / self.scale, 0)
        if self.normalization_method == "quantile":
            weight = float(self.settings.get("mae_weight", 0.5))
            sample_mae = smoothed.mean(axis=1, keepdims=True) / self.sample_scale
            normalized = (1.0 - weight) * normalized + weight * sample_mae
        return normalized

    def state_dict(self):
        """데이터 분할 간에 공유할 설정과 학습 통계만 저장한다."""
        return {
            "settings": self.settings,
            "location": self.location.tolist(),
            "scale": self.scale.tolist(),
            "sample_scale": self.sample_scale,
        }
