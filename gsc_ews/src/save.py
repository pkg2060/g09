"""두 모델의 예측·오차·알람·복원 품질을 동일한 형식으로 저장한다."""

import fcntl
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

ALARM_COLUMNS = [
    "timestamp",
    "variable",
    "level",
    "event_type",
    "raw_value",
    "crv_limit",
    "score",
    "threshold",
    "ratio",
    "consecutive_count",
    "anomalous_count",
    "score_overall",
    "message",
]

CALIBRATION_YAML_HEADER = """# ==============================================================================
# Calibration 운영 파일 수정 안내
# [직접 수정 가능]
# - thresholds.default, thresholds.per_variable: 태그별 운영 Threshold
# - alarm: 연속 횟수, 최소 이상 태그 수, Level, Inhibit 등 Alarm 규칙
#
# [직접 수정 금지: Calibration 재계산 필요]
# - score_processor.settings: smooth_window, normalization_method, quantile_level, mae_weight
# - score_processor.location, score_processor.scale, score_processor.sample_scale
# - version, checkpoint_sha256, model, feature_names, anomaly_score
#
# 파일 수정값은 실행 중 자동 반영되지 않으며 evaluate/online을 다시 시작할 때 로드됩니다.
# =============================================================================="""


def _save_yaml(values, path, *, header=None):
    """선택적인 안내 주석과 YAML을 임시 파일에 쓴 뒤 원자적으로 교체한다."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        content = yaml.safe_dump(values, allow_unicode=True, sort_keys=False)
        if header:
            content = f"{header.rstrip()}\n{content}"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_experiment(config, cross_validation, execution, metrics=None):
    """실행 설정·요약을 저장하고 전체 실험 CSV를 잠금 상태에서 갱신한다."""
    directory = Path(execution["output_directory"])
    root = Path(config["output"]["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    metrics = metrics or {}
    reconstruction = metrics.get("reconstruction", {})
    detection = metrics.get("detection", {})
    row = {
        "equipment": config["equipment_id"],
        "model": config["active_model"],
        "version": config["model_version"],
        "run_id": config["run"]["run_id"],
        "mode": config["run"]["mode"],
        "best_score": float(cross_validation["validation_loss_mean"]),
        "best_score_metric": "validation_loss_mean",
        "best_score_direction": "min",
        "best_score_std": float(cross_validation["validation_loss_std"]),
        "final_epochs": int(cross_validation["final_epochs"]),
        "mae": reconstruction.get("MAE"),
        "rmse": reconstruction.get("RMSE"),
        "r2": reconstruction.get("R2"),
        "mape": reconstruction.get("MAPE"),
        "precision": detection.get("precision"),
        "recall": detection.get("recall"),
        "f1": detection.get("f1"),
        "roc_auc": detection.get("ROC_AUC"),
        "checkpoint": execution.get("checkpoint"),
        "calibration": execution.get("calibration"),
        "output_directory": str(directory),
    }

    summary = {
        **row,
        "cross_validation": cross_validation,
        "calibration_rows": execution.get("calibration_rows"),
    }
    for name, values in (("config.yaml", config), ("summary.yaml", summary)):
        _save_yaml(values, directory / name)

    index_path = root / "experiments.csv"
    with (root / ".experiments.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        experiments = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame(columns=row)
        if list(experiments.columns) != list(row):
            raise ValueError("experiments.csv 컬럼이 현재 실험 요약 형식과 다릅니다")
        same_run = np.ones(len(experiments), dtype=bool)
        for key in ("equipment", "model", "version", "run_id"):
            same_run &= experiments[key].astype(str).eq(str(row[key])).to_numpy()
        experiments = experiments.loc[~same_run].reset_index(drop=True)
        experiments.loc[len(experiments)] = row
        temporary = root / f".experiments.{uuid4().hex}.tmp"
        try:
            experiments.to_csv(temporary, index=False)
            temporary.replace(index_path)
        finally:
            temporary.unlink(missing_ok=True)
    return row


def save_report(result, settings, *, directory=None):
    """실제 행 인덱스·타임스탬프에 맞춰 CSV·JSON·Excel·텍스트 보고서와 그래프를 기록한다."""
    directory = Path(directory or settings["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    names, alarms = result["feature_names"], result["alarms"]
    overall_score = result["score"].mean(axis=1)
    crv_per_variable = alarms.get("crv_per_variable", np.zeros_like(alarms["per_variable"], dtype=bool))
    crv_overall = alarms.get("crv_overall", np.zeros(len(overall_score), dtype=bool))
    crv_directions = alarms.get("crv_directions", np.full(alarms["levels"].shape, "NORMAL", dtype=object))
    columns = {
        "score_overall": overall_score,
        "index": result["index"],
        "timestamp": result["timestamp"],
        "alarm_overall": alarms["overall"].astype(int),
        "crv_alarm_overall": np.asarray(crv_overall).astype(int),
        "inhibited": alarms["inhibited"],
    }
    for index, name in enumerate(names):
        for suffix, values in {
            "actual": result["target"][:, index],
            "prediction": result["prediction"][:, index],
            "score": result["score"][:, index],
            "threshold": result["thresholds"][index],
            "alarm": alarms["per_variable"][:, index].astype(int),
            "level": alarms["levels"][:, index],
            "crv_alarm": np.asarray(crv_per_variable)[:, index].astype(int),
            "crv_level": np.asarray(crv_directions)[:, index],
        }.items():
            columns[f"{name}__{suffix}"] = values
    # 입력의 부가 컬럼도 예측과 같은 행 순서로 보존하며 생성 컬럼을 덮어쓰지 않는다.
    source_targets = {}
    for name, values in result.get("source_columns", {}).items():
        column = name
        while column in columns:
            column = f"input__{column}"
        columns[column] = values
        source_targets[name] = column
    output = pd.DataFrame(columns)
    fault_output = []
    fault_rows = result.get("instrument_fault_rows")
    if not fault_rows:
        fault_rows = [
            {
                "timestamp": timestamp,
                "raw_values": np.full(len(names), np.nan),
                "source_columns": {},
                "inhibited": False,
                "alarm_overall": True,
            }
            for timestamp in dict.fromkeys(
                event["timestamp"]
                for event in result.get("instrument_events", [])
                if event.get("event_type") == "INSTRUMENT_FAULT"
            )
        ]
    for fault in fault_rows:
        row = dict.fromkeys(output.columns, pd.NA)
        row.update(
            timestamp=fault["timestamp"],
            alarm_overall=int(fault["alarm_overall"]),
            crv_alarm_overall=0,
            inhibited=bool(fault["inhibited"]),
        )
        for index, value in enumerate(fault["raw_values"]):
            row[f"{names[index]}__actual"] = value
        for name, value in fault["source_columns"].items():
            if name in source_targets:
                row[source_targets[name]] = value
        fault_output.append(row)
    if fault_output:
        output = pd.DataFrame([*output.to_dict("records"), *fault_output], columns=output.columns)
        timestamps = pd.to_datetime(output["timestamp"], errors="raise", format="mixed")
        order = timestamps.sort_values(kind="stable").index
        output = output.loc[order].reset_index(drop=True)
        output["timestamp"] = timestamps.loc[order].reset_index(drop=True)
    output.to_csv(directory / "predictions.csv", index=False)
    events = alarm_events(result, overall_score)
    events.to_csv(directory / "alarms.csv", index=False)
    alarm_summary = summarize_alarm_events(events)
    pd.DataFrame({"feature": names, "threshold": result["thresholds"]}).to_csv(
        directory / "thresholds.csv", index=False
    )
    evaluation = result.get("evaluation")
    if evaluation is not None:
        metrics = evaluation["metrics"]
        per_variable = evaluation["per_variable"].assign(threshold=result["thresholds"])
        per_variable.to_csv(directory / "metrics.csv", index=False)
        (directory / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        save_quality_text(evaluation["quality"], directory, model_name=result["model_name"])
        save_overall_text(result, directory)
    if settings["excel"]:
        if evaluation is not None:
            with pd.ExcelWriter(directory / "reconstruction_quality.xlsx", engine="openpyxl") as writer:
                write_quality_sheets(writer, evaluation["quality"])
        with pd.ExcelWriter(directory / "report.xlsx", engine="openpyxl") as writer:
            # 전체 알람 원문은 alarms.csv에 두고 Excel에는 행 수 제한이 없는 집계만 저장한다.
            alarm_summary.to_excel(writer, sheet_name="alarm_summary", index=False)
            if evaluation is not None:
                per_variable.to_excel(writer, sheet_name="metrics", index=False)
                write_quality_sheets(writer, evaluation["quality"])
    if settings["plots"]:
        plot_report(result, directory, settings)
    return directory


def save_quality_text(quality_result, directory, *, model_name):
    """계산된 복원 품질을 기존 전체·변수별 요약·순위·지표 설명 형식으로 저장한다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / "reconstruction_quality.txt"
    is_tranad = model_name.startswith("tranad")
    title = "TranAD v2" if model_name == "tranad_v2" else ("TranAD" if is_tranad else model_name.upper())
    prediction = "Predicted Phase 2 Output" if is_tranad else "Predicted Output"

    overall = quality_result["overall"]
    per_var = quality_result["per_variable"]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  {title} Early Warning System - Reconstruction Quality Report\n")
        f.write(f"  ({prediction} vs Input Ground Truth)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"  Total Samples: {quality_result['n_samples']}\n")
        f.write(f"  Total Variables: {quality_result['n_feats']}\n\n")

        # 전체 종합
        f.write("-" * 80 + "\n")
        f.write("  OVERALL SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"  {'Metric':<25} {'Value':>15}  Description\n")
        f.write(f"  {'-' * 70}\n")
        f.write(f"  {'Reconstruction MAE':<25} {overall['MAE']:>15.6f}  Original process scale (Lower=Better)\n")
        f.write(f"  {'RMSE':<25} {overall['RMSE']:>15.6f}  Root Mean Squared Error (Lower=Better)\n")
        f.write(f"  {'R2 Score':<25} {overall['R2']:>15.6f}  Coefficient of Determination (Higher=Better, max=1)\n")
        f.write(f"  {'MAPE (%)':<25} {overall['MAPE']:>15.4f}  Mean Absolute Percentage Error (Lower=Better)\n\n")
        f.write(f"  {'Mean Anomaly Score':<25} {overall['mean_anomaly_score']:>15.6f}  Final postprocessed score\n")
        f.write(f"  {'Mean Threshold':<25} {overall['mean_threshold']:>15.6f}  Mean of per-variable thresholds\n")
        f.write(f"  {'Alarm Rate (%)':<25} {overall['alarm_rate_pct']:>15.4f}  Overall alarm time ratio\n\n")

        f.write(f"  {'Recon. MAE (var mean)':<25} {overall['MAE_mean']:>15.6f}\n")
        f.write(f"  {'RMSE (per-var mean)':<25} {overall['RMSE_mean']:>15.6f}\n")
        f.write(f"  {'R2 (per-var mean)':<25} {overall['R2_mean']:>15.6f}\n")
        f.write(f"  {'MAPE (per-var mean)':<25} {overall['MAPE_mean']:>15.4f}\n\n")

        # 변수별 상세
        f.write("-" * 80 + "\n")
        f.write("  PER-VARIABLE DETAIL\n")
        f.write("-" * 80 + "\n")
        f.write("  Reconstruction metrics: original process scale | Operational metrics: final anomaly score\n")
        f.write(
            f"  {'Variable':<25} {'Recon.MAE':>12} {'RMSE':>12} {'R2':>12} {'MAPE(%)':>12} "
            f"{'Mean Score':>12} {'Threshold':>12} {'Alarm Rate(%)':>14}\n"
        )
        f.write(f"  {'-' * 115}\n")
        for m in per_var:
            f.write(
                f"  {m['name']:<25} {m['MAE']:>12.6f} {m['RMSE']:>12.6f} {m['R2']:>12.6f} "
                f"{m['MAPE']:>12.4f} {m['mean_anomaly_score']:>12.6f} {m['threshold']:>12.6f} "
                f"{m['alarm_rate_pct']:>14.4f}\n"
            )

        # 가장 잘 복원한 변수와 오차가 큰 변수
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("  BEST / WORST VARIABLES\n")
        f.write("-" * 80 + "\n")
        best_mae = min(per_var, key=lambda x: x["MAE"])
        worst_mae = max(per_var, key=lambda x: x["MAE"])
        best_r2 = max(per_var, key=lambda x: x["R2"])
        worst_r2 = min(per_var, key=lambda x: x["R2"])

        f.write(f"  Best  MAE : {best_mae['name']} = {best_mae['MAE']:.6f}\n")
        f.write(f"  Worst MAE : {worst_mae['name']} = {worst_mae['MAE']:.6f}\n")
        f.write(f"  Best  R2  : {best_r2['name']} = {best_r2['R2']:.6f}\n")
        f.write(f"  Worst R2  : {worst_r2['name']} = {worst_r2['R2']:.6f}\n")
        f.write("=" * 80 + "\n")

        # 지표 설명
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("  METRIC DESCRIPTIONS\n")
        f.write("-" * 80 + "\n")
        f.write("  Reconstruction MAE\n")
        f.write("    - 원본 공정 단위의 |예측값-실제값|을 분석 기간 전체에서 평균한 복원 오차입니다.\n")
        f.write("    - 값이 클수록 해당 태그를 모델이 전반적으로 잘 복원하지 못했다는 의미입니다.\n\n")
        f.write("  Mean Anomaly Score\n")
        f.write("    - 그래프 하단과 predictions.csv에 저장되는 시점별 최종 Anomaly Score의 기간 평균입니다.\n")
        f.write("    - 순간 이상값은 이 평균이 아니라 그래프 또는 predictions.csv의 시점별 Score로 확인합니다.\n\n")
        f.write("  Threshold\n")
        f.write("    - 태그별 최종 Anomaly Score와 비교하는 운영 임계값입니다.\n")
        f.write("    - Score가 Threshold를 초과한 뒤 연속 횟수 등 Alarm 규칙을 만족해야 최종 알람이 됩니다.\n\n")
        f.write("  Alarm Rate (%)\n")
        f.write("    - 분석 기간 중 최종 알람이 활성화된 시점의 비율입니다. Inhibit이 적용된 알람은 제외됩니다.\n\n")
        f.write("  RMSE\n")
        f.write("    - 원본 공정 단위의 제곱 오차 평균에 제곱근을 적용한 복원 지표입니다.\n")
        f.write("    - MAE보다 큰 오차에 더 민감하며 낮을수록 좋습니다.\n\n")
        f.write("  R2\n")
        f.write("    - 예측값이 실제값의 변동을 얼마나 설명하는지 나타냅니다. 1에 가까울수록 좋습니다.\n")
        f.write("    - 0은 평균값 예측 수준이고 음수는 평균값 예측보다 복원 성능이 낮다는 의미입니다.\n\n")
        f.write("  MAPE (%)\n")
        f.write("    - 실제값 대비 절대 오차의 백분율 평균입니다. 낮을수록 좋습니다.\n")
        f.write("    - 실제값이 0에 가까운 태그에서는 값이 과도하게 커질 수 있으므로 보조 지표로 사용합니다.\n")
        f.write("=" * 80 + "\n")

    return filepath


def plot_report(result, directory, settings):
    """모든 변수의 분석·복원 오차와 점수 비교·품질·성능 표를 저장한다."""
    from .udf.visualization import (
        plot_all_scoring_metrics,
        plot_performance_table,
        plot_quality_per_variable,
        plot_quality_summary,
        plot_training_curves,
        plot_variable_analysis,
    )

    directory = Path(directory) / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    names = result["feature_names"]
    labels = result["label"] if result["labels_provided"] else None
    model_name = result["model_name"]
    prediction_label = "Predicted (Phase 2)" if model_name.startswith("tranad") else "Predicted"
    dpi = settings["plot_dpi"]
    plot_variable_analysis(
        result["target"],
        result["prediction"],
        result["score"],
        labels,
        result["thresholds"],
        result["alarms"]["per_variable"],
        names,
        timestamps=result["timestamp"],
        segments=result.get("segments"),
        save_dir=directory,
        inhibit_mask=result["alarms"]["inhibited"],
        feature_descriptions=result.get("feature_descriptions"),
        crv_limits=result.get("crv_limits"),
        score_name=result["score_name"],
        prediction_label=prediction_label,
        dpi=dpi,
    )
    evaluation = result.get("evaluation")
    if evaluation is not None:
        quality = evaluation["quality"]
        plot_quality_per_variable(quality, directory, prediction_label=prediction_label, dpi=dpi)
        plot_quality_summary(quality, directory, dpi=dpi)
    scores = result["all_scores"]
    if model_name.startswith("tranad"):
        # 기존 비교 그림의 TRANAD와 MSE는 같은 Phase 2 오차를 서로 다른 범례로 표시한다.
        scores = {
            "tranad": scores["phase2_mse"],
            "mae": scores["mae"],
            "mse": scores["phase2_mse"],
            "hybrid": scores["hybrid"],
        }
    plot_all_scoring_metrics(scores, labels, directory, dpi=dpi)
    if evaluation is not None:
        metrics = result["evaluation"]["metrics"]
        overall = dict.fromkeys(("precision", "recall", "f1", "TP", "TN", "FP", "FN", "ROC_AUC"))
        overall.update(metrics.get("detection", {}))
        overall.update(metrics.get("ranking", {}))
        overall["threshold"] = float(np.mean(result["thresholds"]))
        per_variable = result["evaluation"]["per_variable"].to_dict("records")
        for row, threshold in zip(per_variable, result["thresholds"], strict=False):
            row["threshold"] = float(threshold)
        plot_performance_table(
            per_variable,
            overall,
            names,
            directory,
            model_name=model_name.upper() if model_name == "tft" else "TranAD",
            dpi=dpi,
        )
    if result.get("training_history"):
        plot_training_curves(result["training_history"], directory, dpi=dpi)
    # 새 그림을 모두 저장한 뒤 이 저장기가 만들던 이전 번호 전용 파일을 제거한다.
    for old_plot in directory.glob("variable_[0-9][0-9][0-9].png"):
        old_plot.unlink()


def save_calibration(calibration, directory):
    """학습 점수에서 보정한 통계와 임계값을 YAML로 저장한다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "calibration.yaml"
    _save_yaml(calibration, path, header=CALIBRATION_YAML_HEADER)
    return path


def save_evaluation_snapshot(config, execution, calibration):
    """독립 평가에 실제 적용한 설정·Calibration·요약을 함께 저장한다."""
    directory = Path(execution["output_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("config.yaml", config),
        ("summary.yaml", execution),
        ("calibration.yaml", calibration),
    ):
        _save_yaml(values, directory / name)


def alarm_events(result, overall_score):
    """모델·계기 이상 이벤트를 동일한 알람 파일 형식으로 결합한다."""
    events = pd.DataFrame(result["alarms"]["events"])
    if not events.empty:
        rows = pd.Index(result["timestamp"]).get_indexer(pd.to_datetime(events["timestamp"]))
        if (rows < 0).any():
            raise ValueError("알람 이벤트의 시각이 예측 결과에 없습니다")
        event_types = events.get("event_type", pd.Series("MODEL_ANOMALY", index=events.index))
        events["event_type"] = event_types.fillna("MODEL_ANOMALY")
        events["score_overall"] = overall_score[rows]
        names = np.asarray(result["feature_names"])
        messages = []
        for event, row in zip(events.to_dict("records"), rows, strict=False):
            if event["event_type"].startswith("CRV_"):
                message = event["message"]
            elif event["variable"] == "OVERALL":
                active = names[result["alarms"]["per_variable"][row].astype(bool)]
                message = f"[{event['level']}] {len(active)}개 변수 이상 감지: {', '.join(active[:5])}"
                if len(active) > 5:
                    message += "..."
            else:
                message = f"[{event['level']}] {event['variable']} 이상 감지"
            messages.append(message)
        events["message"] = messages
    instrument = pd.DataFrame(result.get("instrument_events", []))
    return pd.concat(
        [events.reindex(columns=ALARM_COLUMNS), instrument.reindex(columns=ALARM_COLUMNS)], ignore_index=True
    )


def summarize_alarm_events(events):
    """Excel 행 제한을 피하도록 알람 원문을 유형·레벨·변수별 개수로 집계한다."""
    columns = ["event_type", "level", "variable", "count"]
    if events.empty:
        return pd.DataFrame([{"event_type": "TOTAL", "level": "ALL", "variable": "ALL", "count": 0}], columns=columns)
    summary = events.groupby(columns[:-1], dropna=False, sort=False).size().reset_index(name="count")
    total = pd.DataFrame(
        [{"event_type": "TOTAL", "level": "ALL", "variable": "ALL", "count": len(events)}],
        columns=columns,
    )
    return pd.concat([total, summary], ignore_index=True)


def save_overall_text(result, directory):
    """탐지·순위 지표와 복원 품질을 기존 종합 텍스트 보고서 형식으로 저장한다."""
    metrics = result["evaluation"]["metrics"]
    model_name = result["model_name"]
    title = "TranAD" if model_name == "tranad" else model_name.upper()
    detection = dict.fromkeys(("precision", "recall", "f1", "TP", "TN", "FP", "FN", "ROC_AUC"))
    detection.update(metrics.get("detection", {}))
    detection["threshold"] = float(np.mean(result["thresholds"]))
    detection.update(dict.fromkeys(("Hit@100%", "Hit@150%", "NDCG@100%", "NDCG@150%")))
    detection.update(metrics.get("ranking", {}))
    path = Path(directory) / "overall_metrics.txt"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"{title} Early Warning System - Overall Results\n" + "=" * 50 + "\n")
        stream.write("\n[Anomaly Detection Metrics]\n")
        if not result["labels_provided"]:
            stream.write("N/A: ground-truth labels unavailable\n")
        for key, value in detection.items():
            stream.write(f"{key}: {'N/A' if value is None else value}\n")
        stream.write("\n" + "=" * 50 + "\n\n[Reconstruction Quality Metrics]\n")
        quality = metrics["reconstruction"]
        stream.write(f"Reconstruction MAE (Original Scale): {quality['MAE']:.6f} (Lower=Better)\n")
        stream.write(f"RMSE: {quality['RMSE']:.6f} (Lower=Better)\n")
        stream.write(f"R2:   {quality['R2']:.6f} (Higher=Better, max=1)\n")
        stream.write(f"MAPE: {quality['MAPE']:.4f}% (Lower=Better)\n")
        stream.write(f"Mean Anomaly Score: {quality['mean_anomaly_score']:.6f}\n")
        stream.write(f"Mean Threshold: {quality['mean_threshold']:.6f}\n")
        stream.write(f"Alarm Rate: {quality['alarm_rate_pct']:.4f}%\n")
    return path


def write_quality_sheets(writer, quality_result):
    """기존 전체 요약·변수별 평균·지표 설명 시트를 같은 계산 결과로 기록한다."""
    overall = quality_result["overall"]
    per_var = quality_result["per_variable"]

    # ─── Sheet 1: Overall Summary ───
    overall_data = {
        "Metric": [
            "Reconstruction MAE (Original Scale)",
            "RMSE",
            "R2 Score",
            "MAPE (%)",
            "Mean Anomaly Score",
            "Mean Threshold",
            "Alarm Rate (%)",
            "",
            "Reconstruction MAE (per-variable mean)",
            "RMSE (per-var mean)",
            "R2 (per-var mean)",
            "MAPE (per-var mean)",
            "",
            "Total Samples",
            "Total Variables",
        ],
        "Value": [
            overall["MAE"],
            overall["RMSE"],
            overall["R2"],
            overall["MAPE"],
            overall["mean_anomaly_score"],
            overall["mean_threshold"],
            overall["alarm_rate_pct"],
            "",
            overall["MAE_mean"],
            overall["RMSE_mean"],
            overall["R2_mean"],
            overall["MAPE_mean"],
            "",
            quality_result["n_samples"],
            quality_result["n_feats"],
        ],
        "Direction": [
            "Lower=Better",
            "Lower=Better",
            "Higher=Better (max=1)",
            "Lower=Better",
            "Operational score",
            "Per-variable mean",
            "Lower=Better",
            "",
            "Lower=Better",
            "Lower=Better",
            "Higher=Better (max=1)",
            "Lower=Better",
            "",
            "",
            "",
        ],
    }
    df_overall = pd.DataFrame(overall_data)

    # ─── Sheet 2: Per Variable Detail ───
    per_var_data = []
    for m in per_var:
        per_var_data.append(
            {
                "Variable": m["name"],
                "Index": m["index"],
                "Reconstruction MAE (Original Scale)": m["MAE"],
                "RMSE": m["RMSE"],
                "R2 Score": m["R2"],
                "MAPE (%)": m["MAPE"],
                "Mean Anomaly Score": m["mean_anomaly_score"],
                "Threshold": m["threshold"],
                "Alarm Rate (%)": m["alarm_rate_pct"],
            }
        )
    df_per_var = pd.DataFrame(per_var_data)

    # 요약 행 추가
    summary_row = pd.DataFrame(
        [
            {
                "Variable": "=== OVERALL ===",
                "Index": "",
                "Reconstruction MAE (Original Scale)": overall["MAE_mean"],
                "RMSE": overall["RMSE_mean"],
                "R2 Score": overall["R2_mean"],
                "MAPE (%)": overall["MAPE_mean"],
                "Mean Anomaly Score": overall["mean_anomaly_score"],
                "Threshold": overall["mean_threshold"],
                "Alarm Rate (%)": overall["alarm_rate_pct"],
            }
        ]
    )
    df_per_var = pd.concat([df_per_var, summary_row], ignore_index=True)

    # ─── Sheet 3: Metric Guide ───
    guide_data = {
        "Metric": ["Reconstruction MAE (Original Scale)", "RMSE", "R2 Score", "MAPE (%)"],
        "Full Name": [
            "Mean Absolute Error",
            "Root Mean Squared Error",
            "R-Squared (Coefficient of Determination)",
            "Mean Absolute Percentage Error",
        ],
        "Formula": [
            "mean(|predicted - actual|)",
            "sqrt(mean((predicted - actual)^2))",
            "1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)",
            "mean(|actual - predicted| / |actual|) * 100",
        ],
        "Best Value": ["0", "0", "1.0", "0%"],
        "Direction": [
            "Lower = Better",
            "Lower = Better",
            "Higher = Better",
            "Lower = Better",
        ],
        "Description": [
            "Average absolute difference. Primary monitoring metric.",
            "Similar to MAE but penalizes large errors more heavily.",
            "Proportion of variance explained. 1.0 = perfect, 0.0 = mean baseline.",
            "Percentage-based error. Intuitive but sensitive to near-zero values.",
        ],
    }
    df_guide = pd.DataFrame(guide_data)

    df_overall.to_excel(writer, sheet_name="Overall Summary", index=False)
    df_per_var.to_excel(writer, sheet_name="Per Variable", index=False)
    # '='로 시작하는 기존 요약 라벨은 Excel 수식이 아닌 문자열이다.
    writer.sheets["Per Variable"].cell(row=len(df_per_var) + 1, column=1).data_type = "s"
    df_guide.to_excel(writer, sheet_name="Metric Guide", index=False)


def save_tft_interpretation(weights, scores, names, indices, alarm_mask, directory):
    """표본의 변수·지연 중요도와 실제 알람 원인 순위를 기존 Excel 시트로 기록한다."""
    from .udf.visualization import per_variable_contribution

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(
        {
            "Variable": names,
            "Index": np.arange(len(names)),
            "Avg_VSN_Weight": weights.mean(axis=(0, 1)),
            "Mean_AnomalyScore": scores.mean(axis=0),
            "Max_AnomalyScore": scores.max(axis=0),
        }
    ).sort_values("Avg_VSN_Weight", ascending=False)
    # 현재 DataLoader의 윈도우 마지막 입력은 예측 대상 시점이므로 lag_0이다.
    lags = pd.DataFrame(
        weights.mean(axis=0), columns=names, index=[f"lag_{i - weights.shape[1] + 1}" for i in range(weights.shape[1])]
    )
    contribution = per_variable_contribution(scores, weights)
    count = min(5, len(names))
    columns = ["alarm_time_idx"] + [
        f"top{rank}_{field}" for rank in range(1, count + 1) for field in ("var", "contrib", "score")
    ]
    rows = []
    for sample in np.flatnonzero(alarm_mask):
        row = {"alarm_time_idx": int(indices[sample])}
        for rank, variable in enumerate(np.argsort(-contribution[sample], kind="stable")[:count], 1):
            row.update(
                {
                    f"top{rank}_var": names[variable],
                    f"top{rank}_contrib": contribution[sample, variable],
                    f"top{rank}_score": scores[sample, variable],
                }
            )
        rows.append(row)
    path = directory / "tft_interpretability.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        importance.to_excel(writer, sheet_name="Variable Importance", index=False)
        lags.to_excel(writer, sheet_name="Lag Importance")
        pd.DataFrame(rows, columns=columns).to_excel(writer, sheet_name="Alarm Root Causes", index=False)
    return path
