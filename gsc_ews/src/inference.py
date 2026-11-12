"""DataLoader를 모델에 전달하고 시점별 예측·원시 점수를 원본 순서로 반환한다."""

import numpy as np
import pandas as pd
import torch


def predict_batches(model, loader, *, trainer):
    """Lightning으로 추론하고 CPU 결과를 원본 행 순서로 정렬한다."""
    batches = trainer.predict(model=model, dataloaders=loader, ckpt_path=None, return_predictions=True)
    if not batches:
        raise ValueError("추론할 윈도우가 없습니다")
    result = {key: torch.cat([batch[key] for batch in batches]).numpy() for key in batches[0]}
    order = np.argsort(result["index"], kind="stable")
    if len(np.unique(result["index"])) != len(order):
        raise ValueError("중복 추론 행이 있습니다")
    result = {key: value[order] for key, value in result.items()}
    if not np.isfinite(result["anomaly_score"]).all():
        raise ValueError("모델 점수에 NaN 또는 무한대가 있습니다")
    return result


def infer(model, data, *, trainer):
    """모델의 원시 이상 점수와 원본 단위 복원값을 생성한다."""
    prediction = predict_batches(model, data.loader, trainer=trainer)
    indices = prediction["index"]
    dataset = data.loader.dataset
    segments = dataset.segment_ids[indices]
    raw = data.frame.iloc[indices][data.feature_names].to_numpy(dtype=float)
    timestamps = prediction["timestamp"]
    if np.all(timestamps == -1):
        # 시간축이 없는 CSV는 행 순번과 정상 수집 간격으로 상대 시각을 사용한다.
        interval = model.data_state["metadata"]["pipeline"]["data"]["expected_interval_seconds"] or 1
        timestamps = pd.to_datetime(indices * interval, unit="s")
    else:
        timestamps = pd.to_datetime(timestamps, unit="ns")
    timestamp_col = model.data_state["metadata"]["pipeline"]["data"]["timestamp_column"]
    extras = data.frame.columns.difference([*data.feature_names, timestamp_col], sort=False)
    source_columns = {name: data.frame.iloc[indices][name].to_numpy() for name in extras}
    return {
        "source_columns": source_columns,
        "prediction": data.scaler.inverse_transform(prediction["prediction"]),
        "model_name": model.data_state["metadata"]["active_model"],
        "score_name": model.model_config["anomaly_score"],
        "all_scores": {
            key.removeprefix("score_"): value for key, value in prediction.items() if key.startswith("score_")
        },
        "target": raw,
        "raw_score": prediction["anomaly_score"],
        "label": prediction["label"],
        "timestamp": timestamps,
        "index": indices,
        "segments": segments,
        "feature_names": data.feature_names,
        "quantiles": prediction.get("quantile_predictions"),
        "labels_provided": data.labels_provided,
    }
