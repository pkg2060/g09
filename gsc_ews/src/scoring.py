"""학습 점수로 후처리·임계값을 보정하고 저장된 통계를 새 점수에 적용한다."""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from numbers import Real

import numpy as np

from models.base import checkpoint_value

from .udf.configuration import validate_alarm_settings
from .udf.postprocessing import ScoreProcessor
from .udf.threshold import compute_thresholds


def checkpoint_digest(path):
    """보정 상태가 다른 모델 가중치에 적용되지 않도록 체크포인트를 식별한다."""
    digest = sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate(model, bundle, settings, checkpoint_path, *, trainer):
    """모든 fold의 정상 OOF 점수로 후처리 통계와 임계값을 보정한다."""
    state = model.data_state
    expected = {"scaler": checkpoint_value(bundle.scaler.state_dict()), "feature_names": bundle.feature_names}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"보정 데이터가 학습 체크포인트의 {key}와 다릅니다")
    if checkpoint_value(bundle.metadata["pipeline"]) != state["metadata"]["pipeline"]:
        raise ValueError("보정 데이터의 전처리·DataLoader 설정이 학습 시점과 다릅니다")
    raw_scores = np.asarray(getattr(trainer, "oof_raw_score", []), dtype=float)
    segments = np.asarray(getattr(trainer, "oof_segments", []), dtype=int)
    if raw_scores.ndim != 2 or raw_scores.shape[1:] != (bundle.n_features,) or len(raw_scores) != len(segments):
        raise ValueError("Calibration에는 fold별 정상 OOF 점수가 필요합니다")
    processor = ScoreProcessor(settings["postprocessing"]).fit(raw_scores, segments)
    scores = processor.transform(raw_scores, segments)
    thresholds = compute_thresholds(scores, settings["threshold"], bundle.feature_names)
    threshold_map = {name: float(value) for name, value in zip(bundle.feature_names, thresholds, strict=True)}
    threshold_default = float(settings["threshold"]["default"]) if settings["threshold"]["method"] == "manual" else None
    threshold_overrides = (
        {name: value for name, value in threshold_map.items() if value != threshold_default}
        if threshold_default is not None
        else threshold_map
    )
    alarm = deepcopy(settings["alarm"])
    alarm.pop("crv", None)
    return {
        "version": 7,
        "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        "model": bundle.metadata["active_model"],
        "feature_names": bundle.feature_names,
        "anomaly_score": model.model_config["anomaly_score"],
        "thresholds": {
            "default": threshold_default,
            "per_variable": threshold_overrides,
        },
        "alarm": alarm,
        "score_processor": processor.state_dict(),
    }


def calibration_thresholds(calibration, feature_names):
    """공통값과 피처별 예외를 모델 순서로 풀고 기존 Calibration도 읽는다."""
    version = calibration.get("version")
    values = calibration.get("thresholds")
    if version in {5, 6, 7}:
        if not isinstance(values, Mapping) or set(values) != {"default", "per_variable"}:
            raise ValueError("보정 임계값에는 default와 per_variable이 필요합니다")
        default = values["default"]
        overrides = values["per_variable"]
        if not isinstance(overrides, Mapping):
            raise ValueError("보정 임계값의 per_variable은 피처명 매핑이어야 합니다")
        unknown = set(overrides) - set(feature_names)
        if unknown:
            raise ValueError(f"보정 임계값에 모델에 없는 피처가 있습니다: {sorted(unknown)}")
        missing = set(feature_names) - set(overrides)
        if default is None and missing:
            raise ValueError("보정 임계값의 default가 null이면 모든 피처값을 지정해야 합니다")
        values = [overrides.get(name, default) for name in feature_names]
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError("보정 임계값은 피처별 숫자여야 합니다")
    elif version == 4:
        if not isinstance(values, Mapping) or set(values) != set(feature_names):
            raise ValueError("보정 임계값의 피처명이 모델 특성과 정확히 일치해야 합니다")
        values = [values[name] for name in feature_names]
    elif version != 3 or not isinstance(values, list):
        raise ValueError("지원하지 않는 Calibration 임계값 형식입니다")
    try:
        thresholds = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("보정 임계값은 피처별 숫자여야 합니다") from exc
    if thresholds.shape != (len(feature_names),) or not np.isfinite(thresholds).all() or np.any(thresholds < 0):
        raise ValueError("보정 임계값의 길이 또는 값이 잘못되었습니다")
    return thresholds


def validate_calibration(calibration, model, checkpoint_path):
    """모델·특성·스케일 정보가 일치하는 보정 결과만 허용한다."""
    if calibration.get("version") != 7 or calibration["checkpoint_sha256"] != checkpoint_digest(checkpoint_path):
        raise ValueError("보정 상태와 체크포인트가 다릅니다. 해당 모델로 다시 보정하세요")
    if calibration["feature_names"] != model.data_state["feature_names"]:
        raise ValueError("보정 상태의 특성 순서가 다릅니다")
    if calibration["anomaly_score"] != model.model_config["anomaly_score"]:
        raise ValueError("보정 상태의 이상 점수 정의가 다릅니다")
    saved_settings = model.data_state["metadata"].get("score_processor_settings")
    current_settings = calibration.get("score_processor", {}).get("settings")
    if saved_settings is None or current_settings != saved_settings:
        raise ValueError("Calibration 후처리 설정이 학습 시점과 다릅니다. 전체 학습·보정을 다시 실행하세요")
    validate_alarm_settings(calibration.get("alarm"), include_crv=False)
    feature_names = model.data_state["feature_names"]
    thresholds = calibration_thresholds(calibration, feature_names)
    width = model.n_feats
    for values in (
        calibration["score_processor"]["location"],
        calibration["score_processor"]["scale"],
    ):
        if np.asarray(values).shape != (width,) or not np.isfinite(values).all():
            raise ValueError("보정 배열의 길이 또는 값이 잘못되었습니다")
    if np.any(thresholds < 0) or np.any(np.asarray(calibration["score_processor"]["scale"]) <= 0):
        raise ValueError("보정 임계값·스케일의 범위가 잘못되었습니다")
    sample_scale = calibration["score_processor"].get("sample_scale", 1.0)
    if not np.isfinite(sample_scale) or sample_scale <= 0:
        raise ValueError("보정 샘플 MAE 스케일이 잘못되었습니다")


def score_predictions(prediction, calibration, *, processor=None):
    """원시 점수에 학습 통계와 시간 단절 규칙을 적용하고 임계값을 연결한다."""
    if prediction["feature_names"] != calibration["feature_names"]:
        raise ValueError("추론과 보정 상태의 특성 순서가 다릅니다")
    persistent = processor is not None
    processor = processor or ScoreProcessor(**calibration["score_processor"])
    scores = processor.transform(prediction["raw_score"], prediction["segments"], reset=not persistent)
    thresholds = calibration_thresholds(calibration, prediction["feature_names"])
    return {**prediction, "score": scores, "thresholds": thresholds}
