"""변수별·전체 복원 품질의 공통 계산식."""

import numpy as np


def _mae(pred, target):
    """Mean Absolute Error: mean(|pred - target|)
    값이 낮을수록 좋음 (0 = 완벽한 복원)"""
    return np.mean(np.abs(pred - target))


def _rmse(pred, target):
    """Root Mean Squared Error: sqrt(mean((pred - target)²))
    값이 낮을수록 좋음. 큰 오차에 MAE보다 더 민감하게 반응"""
    return np.sqrt(np.mean((pred - target) ** 2))


def _r2_score(pred, target):
    """R² Score (결정계수): 1 - SS_res / SS_tot
    1에 가까울수록 좋음 (1 = 완벽, 0 = 평균 수준, 음수 = 평균보다 나쁨)"""
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def _mape(pred, target, epsilon=1e-8):
    """Mean Absolute Percentage Error: mean(|pred - target| / |target|) * 100
    값이 낮을수록 좋음 (0% = 완벽).
    target이 0에 가까운 경우 epsilon으로 보정"""
    return np.mean(np.abs((target - pred) / (np.abs(target) + epsilon))) * 100.0


def _mae_timeseries(pred, target):
    """시점별 MAE: 각 시점에서의 절대 오차"""
    return np.abs(pred - target)


def _mse_timeseries(pred, target):
    """시점별 MSE: 각 시점에서의 제곱 오차"""
    return (pred - target) ** 2


# ==============================================================================
# 2. 종합 복원 품질 계산
# ==============================================================================


def compute_reconstruction_quality(predictions, targets, feature_names=None):
    """
    복원 품질 지표를 개별 변수 및 전체 종합으로 계산합니다.

    Args:
        predictions: (T, n_feats) 모델 복원 예측값
        targets:     (T, n_feats) 원본 데이터 (Ground Truth)
        feature_names: 변수 이름 리스트

    Returns:
        dict: {
            "per_variable": [   # 변수별 지표
                {"name": "Var_0", "MAE": ..., "RMSE": ..., "R2": ..., "MAPE": ...},
                ...
            ],
            "overall": {        # 전체 종합 지표
                "MAE": ..., "RMSE": ..., "R2": ..., "MAPE": ...
            },
            "timeseries": {     # 시계열별 오차 (시각화용)
                "mae_per_step":  (T,) 각 시점의 전체 MAE,
                "mae_per_var":   (T, n_feats) 변수별 시점별 절대오차,
            },
            "predictions": predictions,
            "targets": targets,
            "feature_names": feature_names,
        }
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    T, n_feats = targets.shape

    if feature_names is None:
        feature_names = [f"Var_{i}" for i in range(n_feats)]

    # ─── 변수별(Per-Variable) 계산 ───
    per_variable = []
    for i in range(n_feats):
        pred_i = predictions[:, i]
        tgt_i = targets[:, i]

        metrics = {
            "name": feature_names[i] if i < len(feature_names) else f"Var_{i}",
            "index": i,
            "MAE": float(_mae(pred_i, tgt_i)),
            "RMSE": float(_rmse(pred_i, tgt_i)),
            "R2": float(_r2_score(pred_i, tgt_i)),
            "MAPE": float(_mape(pred_i, tgt_i)),
        }
        per_variable.append(metrics)

    # ─── 전체 종합(Overall) 계산 ───
    # 방법 1: 전체 데이터를 하나로 펼쳐서 계산
    overall_flat = {
        "MAE": float(_mae(predictions, targets)),
        "RMSE": float(_rmse(predictions, targets)),
        "R2": float(_r2_score(predictions.flatten(), targets.flatten())),
        "MAPE": float(_mape(predictions, targets)),
    }

    # 방법 2: 변수별 평균 (보고용)
    overall_mean = {
        "MAE_mean": float(np.mean([m["MAE"] for m in per_variable])),
        "RMSE_mean": float(np.mean([m["RMSE"] for m in per_variable])),
        "R2_mean": float(np.mean([m["R2"] for m in per_variable])),
        "MAPE_mean": float(np.mean([m["MAPE"] for m in per_variable])),
    }

    overall = {**overall_flat, **overall_mean}

    # ─── 시계열(Timeseries) 오차 계산 (그래프용) ───
    mae_per_var = _mae_timeseries(predictions, targets)  # (T, n_feats)
    mae_per_step = np.mean(mae_per_var, axis=1)  # (T,)
    mse_per_var = _mse_timeseries(predictions, targets)  # (T, n_feats)
    mse_per_step = np.mean(mse_per_var, axis=1)  # (T,)

    timeseries = {
        "mae_per_var": mae_per_var,
        "mae_per_step": mae_per_step,
        "mse_per_var": mse_per_var,
        "mse_per_step": mse_per_step,
    }

    return {
        "per_variable": per_variable,
        "overall": overall,
        "timeseries": timeseries,
        "predictions": predictions,
        "targets": targets,
        "feature_names": feature_names,
        "n_feats": n_feats,
        "n_samples": T,
    }
