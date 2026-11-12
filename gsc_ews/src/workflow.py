"""단일 진입점에서 선택한 학습·평가·추론 모드를 공통 파이프라인으로 실행한다."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import lightning.pytorch as pl
import pandas as pd

from .alarm import AlarmGenerator
from .dataloader import PredictionData, build_dataloaders, build_prediction_data, load_tag_descriptions
from .evaluation import evaluate_predictions
from .inference import infer
from .model import load_checkpoint
from .save import save_calibration, save_evaluation_snapshot, save_experiment, save_report
from .scoring import calibrate, score_predictions, validate_calibration
from .trainer import build_trainer, fit_model, training_history
from .udf.configuration import (
    build_ews_settings,
    load_yaml,
    validate_ews_settings,
    validate_execution_config,
    validate_feature_settings,
)

CONSOLE_WIDTH = 88


def _print_section(title, rows):
    """제목과 주요 실행 정보를 구분선이 있는 표준 출력 블록으로 표시한다."""
    print(f"\n{'=' * CONSOLE_WIDTH}", flush=True)  # noqa: T201
    print(title, flush=True)  # noqa: T201
    print("-" * CONSOLE_WIDTH, flush=True)  # noqa: T201
    for label, value in rows:
        if value is not None:
            print(f"{label:<24}: {value}", flush=True)  # noqa: T201
    print("=" * CONSOLE_WIDTH, flush=True)  # noqa: T201


def _yes_no(value):
    """설정의 불리언 값을 사용자가 읽기 쉬운 문자열로 바꾼다."""
    return "사용" if value else "미사용"


def print_run_configuration(config, settings=None):
    """전체 설정을 노출하지 않고 현재 실행에 필요한 핵심 설정만 출력한다."""
    run = config["run"]
    mode = run["mode"]
    name = config["active_model"]
    model = config["models"][name]
    data = config["data"]
    loader = config["dataloader"]
    normalization = data["normalization"]
    missing = config["preprocessing"]["missing"]
    rows = [
        ("실행 모드", mode),
        ("설비 ID", config["equipment_id"]),
        ("모델", name),
        ("모델 버전", config["model_version"]),
        ("실행 ID", run["run_id"]),
        ("장치", run["device"]),
    ]
    if mode in {"train", "full"}:
        rows.append(("Train CSV", data["train_path"]))
    if mode in {"full", "evaluate", "analyze"}:
        rows.append(("Input CSV", data["test_path"]))
    if mode == "online":
        rows.append(("Online Input", run["input_path"]))
    if mode in {"evaluate", "online", "analyze"}:
        rows.append(("Checkpoint", run["checkpoint_path"]))
    if mode in {"evaluate", "online"}:
        rows.append(("Calibration", run["calibration_path"]))
    rows.extend(
        [
            ("Output", run["output_directory"] or "설비/모델/버전/실행ID 자동 생성"),
            ("Timestamp Column", data["timestamp_column"]),
            ("Feature Count", "CSV 로딩 후 확정" if data["feature_columns"] is None else len(data["feature_columns"])),
            ("Window Size", model["window_size"]),
            ("Dtype", model["dtype"]),
            ("Train Batch Size", loader["batch_size"]),
            ("Eval Batch Size", loader["eval_batch_size"]),
            ("Normalization", normalization["method"]),
            ("MinMax Clip", normalization["clip"]),
            ("Missing Strategy", missing["strategy"]),
        ]
    )
    if mode in {"train", "full"}:
        validation = (
            f"TimeSeriesSplit ({loader['time_series_splits']} folds)"
            if loader["time_series_split_enabled"]
            else f"시간순 Holdout ({loader['validation_ratio']:.0%})"
        )
        rows.extend(
            [
                ("Validation", validation),
                ("Epochs", model["epochs"]),
                ("Learning Rate", model["learning_rate"]),
                ("Dropout", model["dropout"]),
            ]
        )
    structure_labels = {
        "d_model": "D Model",
        "hidden_size": "Hidden Size",
        "latent_size": "Latent Size",
        "num_heads": "Attention Heads",
        "encoder_layers": "Encoder Layers",
        "decoder_layers": "Decoder Layers",
        "dim_feedforward": "Feedforward Size",
        "lstm_layers": "LSTM Layers",
        "num_layers": "Layers",
    }
    rows.extend((label, model[key]) for key, label in structure_labels.items() if key in model)
    if settings is not None and mode in {"train", "full"}:
        threshold = settings["threshold"]
        rows.extend(
            [
                ("Threshold Method", threshold["method"]),
                ("Default Threshold", threshold["default"]),
            ]
        )
    elif mode in {"evaluate", "online"}:
        rows.append(("Threshold Source", "Calibration 파일"))
    if mode == "analyze":
        analysis = config["analysis"]
        rows.extend(
            [
                ("Input Start", analysis.get("input_start") or "전체 시작"),
                ("Input End", analysis.get("input_end") or "전체 끝"),
                ("SHAP", _yes_no(analysis["shap"])),
                ("SHAP Method", analysis["shap_method"] if analysis["shap"] else None),
                ("SHAP Background", run["background_path"] if analysis["shap"] else None),
            ]
        )
    _print_section("GSC EWS - 현재 실행 설정", rows)


def print_data_summary(bundle):
    """학습·검증·테스트 DataLoader 구성 결과를 요약해 출력한다."""
    metadata = bundle.metadata
    rows = [
        ("Train Shape", bundle.train_frame.shape),
        ("Validation Rows", metadata["validation_rows"]),
        ("Test Shape", "미사용" if bundle.test_loader is None else bundle.test_frame.shape),
        ("Feature Count", bundle.n_features),
        ("Train Windows", len(bundle.train_loader.dataset)),
        ("Validation Windows", 0 if bundle.val_loader is None else len(bundle.val_loader.dataset)),
        ("Test Windows", 0 if bundle.test_loader is None else len(bundle.test_loader.dataset)),
        ("Train Segments", metadata["train_segments"]),
        ("Test Segments", metadata["test_segments"]),
        ("Test Labels", "있음" if metadata["labels_provided"] else "없음"),
        ("Instrument Fault Rows", metadata["test_instrument_fault_rows"]),
    ]
    _print_section("데이터 로딩 결과", rows)


def print_prediction_data_summary(data, title, *, timestamp_column=None):
    """평가·분석 입력 CSV가 실제 모델 입력으로 변환된 결과를 출력한다."""
    rows = [
        ("Input Shape", data.frame.shape),
        ("Feature Count", len(data.feature_names)),
        ("Window Count", len(data.loader.dataset)),
        ("Labels", "있음" if data.labels_provided else "없음"),
        ("Instrument Fault Rows", len(data.instrument_fault_rows)),
    ]
    if timestamp_column and timestamp_column in data.frame and not data.frame.empty:
        rows.extend(
            [
                ("Start Time", data.frame[timestamp_column].iloc[0]),
                ("End Time", data.frame[timestamp_column].iloc[-1]),
            ]
        )
    _print_section(title, rows)


@contextmanager
def timed_step(name, *, enabled=True):
    """Pipeline 단계의 시작과 종료 시점에 소요시간을 즉시 출력한다."""
    if not enabled:
        yield
        return
    started_at = perf_counter()
    print(f"\n{'=' * CONSOLE_WIDTH}", flush=True)  # noqa: T201
    print(f"[{name} 단계] 시작", flush=True)  # noqa: T201
    print("=" * CONSOLE_WIDTH, flush=True)  # noqa: T201
    try:
        yield
    except Exception:
        elapsed = perf_counter() - started_at
        print(f"[{name} 단계] 실패 | 소요시간: {elapsed:.2f}초", flush=True)  # noqa: T201
        print("=" * CONSOLE_WIDTH, flush=True)  # noqa: T201
        raise
    else:
        elapsed = perf_counter() - started_at
        print(f"[{name} 단계] 완료 | 소요시간: {elapsed:.2f}초", flush=True)  # noqa: T201
        print("=" * CONSOLE_WIDTH, flush=True)  # noqa: T201


def output_directory(config, settings, model_name):
    """사용자 지정 경로 또는 모드별 기본 결과 폴더를 선택한다."""
    if config["run"]["output_directory"]:
        return Path(config["run"]["output_directory"])
    directory = (
        Path(settings["output"]["directory"])
        / config["equipment_id"]
        / model_name
        / config["model_version"]
        / config["run"]["run_id"]
    )
    return directory / "online" if config["run"]["mode"] == "online" else directory


def apply_calibration_alarm(settings, calibration, feature_names):
    """Calibration의 운영 Alarm과 별도 CRV 기준을 결합해 실제 실행 설정을 만든다."""
    settings = deepcopy(settings)
    crv = settings["alarm"]["crv"]
    settings["alarm"] = deepcopy(calibration["alarm"])
    settings["alarm"]["crv"] = crv
    validate_ews_settings(settings)
    validate_feature_settings(settings, feature_names, include_threshold=False)
    return settings


def run_prediction(
    model,
    data,
    calibration,
    settings,
    *,
    trainer,
    processor=None,
    alarm_generator=None,
    index_offset=0,
    segment=None,
    show_steps=True,
):
    """배치·온라인에서 같은 추론·점수·알람 경로와 선택적 평가를 실행한다."""
    with timed_step("Inference", enabled=show_steps):
        prediction = infer(model, data, trainer=trainer)
    prediction["index"] += index_offset
    if segment is not None:
        prediction["segments"][:] = segment
    with timed_step("Anomaly Score 계산", enabled=show_steps):
        result = score_predictions(prediction, calibration, processor=processor)
    with timed_step("Alarm 판정", enabled=show_steps):
        alarms = alarm_generator or AlarmGenerator(result["thresholds"], result["feature_names"], settings["alarm"])
        result["alarms"] = alarms.process(result["score"], result["target"], result["timestamp"], result["segments"])
        result["crv_limits"] = settings["alarm"]["crv"]["per_variable"]
        instrument_events = []
        instrument_fault_rows = []
        timestamp_col = model.data_state["metadata"]["pipeline"]["data"]["timestamp_column"]
        for _, row in data.instrument_fault_rows.iterrows():
            values = pd.to_numeric(row[data.feature_names], errors="coerce").to_numpy(dtype=float)
            timestamp = pd.Timestamp(row[timestamp_col])
            events = alarms.instrument_fault_events(values, timestamp, include_zero=False)
            instrument_events.extend(events)
            instrument_fault_rows.append(
                {
                    "timestamp": timestamp,
                    "raw_values": values,
                    "source_columns": row.drop(labels=[timestamp_col, *data.feature_names]).to_dict(),
                    "inhibited": alarms.inhibited(values),
                    "alarm_overall": bool(events),
                }
            )
        result["instrument_events"] = instrument_events
        result["instrument_fault_rows"] = instrument_fault_rows
    with timed_step("Evaluation", enabled=show_steps and settings["evaluation"]["enabled"]):
        result["evaluation"] = evaluate_predictions(result) if settings["evaluation"]["enabled"] else None
    return result


def evaluate_model(model, config, settings, checkpoint_path, *, trainer, bundle=None, tag_descriptions=None):
    """full은 정상 OOF로 보정하고 evaluate는 저장된 보정값으로 평가한다."""
    config = deepcopy(config)
    if not settings["evaluation"]["enabled"]:
        config["data"]["labels_path"] = None
    name = model.data_state["metadata"]["active_model"]
    directory = output_directory(config, settings, name)
    calibration_path = config["run"]["calibration_path"] if config["run"]["mode"] == "evaluate" else None
    if config["run"]["mode"] == "evaluate":
        with timed_step("Calibration 파일 로드·검증"):
            calibration = load_yaml(calibration_path)
            validate_calibration(calibration, model, checkpoint_path)
            settings = apply_calibration_alarm(settings, calibration, model.data_state["feature_names"])
        with timed_step("평가 데이터 준비"):
            data = build_prediction_data(
                config["data"]["test_path"],
                model.data_state,
                model.model_config,
                labels_path=config["data"]["labels_path"] if settings["evaluation"]["enabled"] else None,
                label_columns=config["data"]["label_columns"],
            )
        timestamp_column = model.data_state["metadata"]["pipeline"]["data"]["timestamp_column"]
        print_prediction_data_summary(data, "평가 데이터 준비 결과", timestamp_column=timestamp_column)
    else:
        if bundle is None:
            raise RuntimeError("full 모드의 Calibration에는 학습·검증 DataBundle이 필요합니다")
        with timed_step("Calibration 계산·저장"):
            calibration = calibrate(model, bundle, settings, checkpoint_path, trainer=trainer)
            calibration_path = save_calibration(calibration, directory)
        data = PredictionData(
            bundle.test_loader,
            bundle.test_frame,
            bundle.scaler,
            bundle.feature_names,
            bundle.metadata["labels_provided"],
            bundle.instrument_fault_rows,
        )
    history = training_history(trainer) if config["run"]["mode"] == "full" else None
    result = run_prediction(model, data, calibration, settings, trainer=trainer)
    result["training_history"] = history
    if tag_descriptions is None:
        tag_descriptions = load_tag_descriptions(
            config["data"]["tag_description_path"], config["equipment_id"], result["feature_names"]
        )
    result["feature_descriptions"] = tag_descriptions
    with timed_step("결과 파일 저장"):
        save_report(result, settings["output"], directory=directory)
    summary = {
        "checkpoint": str(checkpoint_path),
        "calibration": str(calibration_path),
        "output_directory": str(directory),
        "rows": len(result["index"]),
        "_metrics": None if result["evaluation"] is None else result["evaluation"]["metrics"],
    }
    if config["run"]["mode"] == "full":
        summary["calibration_rows"] = len(trainer.oof_raw_score)
    if config["run"]["mode"] == "evaluate":
        with timed_step("평가 실행 설정 저장"):
            execution = {key: value for key, value in summary.items() if key != "_metrics"}
            save_evaluation_snapshot(config, execution, calibration)
    return summary


def run_online(model, config, settings, *, trainer):
    """저장 상태로 온라인 추론을 시작하고 모델을 다시 로드하지 않는다."""
    from .online_runner import watch_csv

    run = config["run"]
    with timed_step("Calibration 파일 로드·검증"):
        calibration = load_yaml(run["calibration_path"])
        validate_calibration(calibration, model, run["checkpoint_path"])
        settings = apply_calibration_alarm(settings, calibration, model.data_state["feature_names"])
    directory = output_directory(config, settings, model.data_state["metadata"]["active_model"])
    with timed_step("Online 입력 감시"):
        watch_csv(
            model,
            run["input_path"],
            calibration,
            settings,
            directory,
            process_data=partial(run_prediction, trainer=trainer, show_steps=False),
        )
    return {
        "checkpoint": run["checkpoint_path"],
        "calibration": run["calibration_path"],
        "output_directory": str(directory),
    }


def analyze_model(model, config, checkpoint_path, *, trainer):
    """저장된 모델과 입력 CSV로 Attention·변수 중요도·SHAP만 생성한다."""
    from .explainer import analyze

    run = config["run"]
    name = model.data_state["metadata"]["active_model"]
    directory = output_directory(config, config, name)
    with timed_step("분석 데이터 준비"):
        analysis = config["analysis"]
        data = build_prediction_data(
            config["data"]["test_path"],
            model.data_state,
            model.model_config,
            input_start=analysis.get("input_start"),
            input_end=analysis.get("input_end"),
        )
        background = None
        background_data = None
        if analysis["shap"]:
            background_data = build_prediction_data(run["background_path"], model.data_state, model.model_config)
            background = background_data.loader.dataset
    timestamp_column = model.data_state["metadata"]["pipeline"]["data"]["timestamp_column"]
    print_prediction_data_summary(data, "분석 대상 데이터 준비 결과", timestamp_column=timestamp_column)
    if background_data is not None:
        print_prediction_data_summary(
            background_data,
            "SHAP Background 데이터 준비 결과",
            timestamp_column=timestamp_column,
        )
    with timed_step("모델 해석"):
        model.to(trainer.strategy.root_device)
        analyze(
            model,
            data.loader.dataset,
            analysis,
            directory / "analysis",
            background_dataset=background,
            plot_directory=directory / "plots" / "interpretability",
            dpi=config["output"]["plot_dpi"],
            report_directory=directory,
            excel=config["output"]["excel"],
        )
    return {
        "checkpoint": str(checkpoint_path),
        "calibration": None,
        "input": config["data"]["test_path"],
        "background": run["background_path"],
        "output_directory": str(directory),
        "analysis_period": {
            "input_start": analysis.get("input_start"),
            "input_end": analysis.get("input_end"),
        },
        "rows": len(data.loader.dataset),
    }


def run_pipeline(config):
    """모드에 따라 필요한 단계만 실행하며 full은 최적 체크포인트로 평가까지 수행한다."""
    config = deepcopy(config)
    if config["run"]["run_id"] is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        config["run"]["run_id"] = f"{timestamp}_{uuid4().hex[:8]}"
    with timed_step("실행 설정 검증"):
        validate_execution_config(config)
    run = config["run"]
    mode = run["mode"]
    if run["input_path"] and mode in {"full", "evaluate", "analyze"}:
        config["data"]["test_path"] = run["input_path"]
    pl.seed_everything(int(run["seed"]), workers=True)
    if mode in {"train", "full"}:
        name = config["active_model"]
        settings = None
        if mode == "full" or not run["fast_dev_run"]:
            with timed_step("EWS 설정 준비"):
                settings = build_ews_settings(config, model=name)
        if settings is not None and not settings["evaluation"]["enabled"]:
            config["data"]["labels_path"] = None
        print_run_configuration(config, settings)
        with timed_step("데이터 전처리·DataLoader 준비"):
            bundle = build_dataloaders(config)
            tag_descriptions = load_tag_descriptions(
                config["data"]["tag_description_path"], config["equipment_id"], bundle.feature_names
            )
        print_data_summary(bundle)
        if settings is not None:
            with timed_step("피처 참조 설정 검증"):
                validate_feature_settings(settings, bundle.feature_names)
        with timed_step("Fold 검증·운영 모델 학습"):
            model, trainer = fit_model(
                bundle, config, ckpt_path=run["checkpoint_path"], fast_dev_run=run["fast_dev_run"]
            )
        checkpoint = trainer.checkpoint_callback
        checkpoint_path = checkpoint.best_model_path if checkpoint else ""
        if mode == "train":
            calibration_path = None
            calibration_rows = None
            if not run["fast_dev_run"]:
                if not checkpoint_path:
                    raise RuntimeError("Calibration에 사용할 최적 체크포인트가 생성되지 않았습니다")
                with timed_step("Calibration 계산·저장"):
                    calibration = calibrate(model, bundle, settings, checkpoint_path, trainer=trainer)
                    calibration_path = save_calibration(calibration, output_directory(config, settings, name))
                    calibration_rows = len(trainer.oof_raw_score)
            summary = {
                "mode": mode,
                "model": name,
                "checkpoint": checkpoint_path or None,
                "calibration": None if calibration_path is None else str(calibration_path),
                "output_directory": str(output_directory(config, config, name)),
                "cross_validation": trainer.cross_validation,
                "calibration_rows": calibration_rows,
                "fast_dev_run": run["fast_dev_run"],
            }
            with timed_step("학습 결과 저장"):
                save_experiment(config, trainer.cross_validation, summary)
            return summary
        if not checkpoint_path:
            raise RuntimeError("전체 실행에 사용할 최적 체크포인트가 생성되지 않았습니다")
        # 전체 정상 데이터로 재학습한 운영 모델과 OOF 보정값으로 평가를 이어간다.
        summary = evaluate_model(
            model,
            config,
            settings,
            checkpoint_path,
            trainer=trainer,
            bundle=bundle,
            tag_descriptions=tag_descriptions,
        )
        metrics = summary.pop("_metrics")
        with timed_step("실험 이력 저장"):
            save_experiment(config, trainer.cross_validation, summary, metrics)
    else:
        with timed_step("체크포인트 로드"):
            model = load_checkpoint(run["checkpoint_path"])
        name = model.data_state["metadata"]["active_model"]
        # 추론 정밀도도 현재 YAML이 아닌 저장된 모델 설정을 따른다.
        config["active_model"] = name
        config["model_version"] = model.data_state["metadata"].get("model_version", config["model_version"])
        config["equipment_id"] = model.data_state["metadata"].get("equipment_id", config["equipment_id"])
        config["models"][name] = model.model_config
        with timed_step("모델 실행 환경 준비"):
            trainer = build_trainer(config, for_prediction=True)
        if mode == "analyze":
            print_run_configuration(config)
            summary = analyze_model(model, config, run["checkpoint_path"], trainer=trainer)
        else:
            with timed_step("EWS 설정 준비"):
                settings = build_ews_settings(config, model=name)
            print_run_configuration(config, settings)
        if mode == "evaluate":
            summary = evaluate_model(model, config, settings, run["checkpoint_path"], trainer=trainer)
            summary.pop("_metrics", None)
        elif mode == "online":
            summary = run_online(model, config, settings, trainer=trainer)
    return {"mode": mode, "model": name, **summary}
