"""추론 결과의 복원 품질과 라벨이 있는 구간의 탐지 성능을 평가한다."""

import numpy as np
import pandas as pd

from .udf.metrics import detection_metrics, ranking_metrics
from .udf.quality import compute_reconstruction_quality


def evaluate_predictions(result):
    """파일을 저장하지 않고 전체·변수별 성능 지표를 반환한다."""
    names = result["feature_names"]
    quality = compute_reconstruction_quality(result["prediction"], result["target"], names)
    scores = np.asarray(result["score"], dtype=float)
    thresholds = np.asarray(result["thresholds"], dtype=float)
    alarms = np.asarray(result["alarms"]["per_variable"], dtype=bool)
    if scores.shape != alarms.shape or scores.shape[1:] != (len(names),) or thresholds.shape != (len(names),):
        raise ValueError("품질표의 점수·임계값·알람 배열이 특성 순서와 일치하지 않습니다")
    for index, values in enumerate(quality["per_variable"]):
        values.update(
            mean_anomaly_score=float(scores[:, index].mean()),
            threshold=float(thresholds[index]),
            alarm_rate_pct=float(alarms[:, index].mean() * 100),
        )
    quality["overall"].update(
        mean_anomaly_score=float(scores.mean()),
        mean_threshold=float(thresholds.mean()),
        alarm_rate_pct=float(alarms.any(axis=1).mean() * 100),
    )
    metrics = {"reconstruction": quality["overall"], "labels_provided": result["labels_provided"]}
    per_variable = pd.DataFrame(quality["per_variable"])
    if result["labels_provided"]:
        metrics["detection"] = detection_metrics(result["score"], result["label"], result["thresholds"])
        metrics["ranking"] = ranking_metrics(result["score"], result["label"])
        detection = [
            detection_metrics(result["score"][:, i], result["label"][:, i], result["thresholds"][i])
            for i in range(len(names))
        ]
        per_variable = pd.concat([per_variable, pd.DataFrame(detection)], axis=1)
    return {"metrics": metrics, "per_variable": per_variable, "quality": quality}
