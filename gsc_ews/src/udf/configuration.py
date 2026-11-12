"""학습·모델·EWS에서 사용하는 YAML 읽기와 공통 설정 해석."""

import csv
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
EWS_SECTIONS = ("evaluation", "output", "analysis", "online")
MODEL_NAMES = ("tranad", "tranad_v2", "tft", "lstm_ae")
MODEL_VERSION_PATTERN = re.compile(r"^v\d+_\d+_\d+$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
EQUIPMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
CRV_COLUMNS = (
    "Equipment",
    "RTDB Tag ID",
    "Pre Alarm Low",
    "Pre Alarm High",
    "Alarm Low",
    "Alarm High",
    "Trip Limit",
    "EWS Threshold",
)
CRV_BAND_COLUMNS = {
    "Alarm Low": "alarm_low",
    "Pre Alarm Low": "pre_alarm_low",
    "Pre Alarm High": "pre_alarm_high",
    "Alarm High": "alarm_high",
}


class ConfigError(ValueError):
    """설정 파일의 위치와 항목을 함께 표시하는 오류."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """중복 키로 앞의 설정이 조용히 덮어써지는 것을 막는 로더."""

    def construct_mapping(self, node, deep=False):
        """동일 매핑 안의 키 중복을 확인한 후 안전하게 변환한다."""
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in keys:
                    raise ConfigError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
                keys.add(key)
            except TypeError as exc:
                raise ConfigError("YAML keys must be scalar values") from exc
        return super().construct_mapping(node, deep=deep)


def load_yaml(path):
    """UTF-8 YAML을 읽고 빈 문서·중복 키·잘못된 문법을 명확히 보고한다."""
    path = Path(path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as handle:
            values = yaml.load(handle, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError, ConfigError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ConfigError(f"{path}: expected a YAML mapping")
    return values


def merge_settings(base, overrides):
    """중첩 매핑을 복사하여 병합하고 리스트·null 값은 지정값으로 교체한다."""
    result = deepcopy(base)
    if not isinstance(overrides, Mapping):
        raise ConfigError("Config overrides must be a mapping")
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_settings(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def require(config, dotted_path):
    """누락된 설정의 전체 경로를 표시하며 값을 가져온다."""
    value = config
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"Missing config key: {dotted_path}")
        value = value[key]
    return value


def resolve_path(value, base):
    """상대 경로를 해당 값을 정의한 YAML 디렉터리 기준으로 해석한다."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("A path must be a non-empty string or null")
    return str((Path(base) / Path(value).expanduser()).resolve())


def load_crv_settings(path, equipment_id):
    """설비별 CRV CSV를 읽어 기존 AlarmGenerator 입력 형식으로 변환한다."""
    if path is None:
        return {"enabled": False, "per_variable": {}}
    path = Path(path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(CRV_COLUMNS):
                raise ConfigError(f"{path}: CRV CSV columns must be {list(CRV_COLUMNS)}")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    limits_by_tag = {}
    seen = set()
    numeric_columns = (*CRV_BAND_COLUMNS, "Trip Limit", "EWS Threshold")
    for line, row in enumerate(rows, start=2):
        if None in row or any(row[column] is None for column in CRV_COLUMNS):
            raise ConfigError(f"{path}:{line}: CRV CSV row does not match the header")
        equipment = row["Equipment"].strip()
        tag = row["RTDB Tag ID"].strip()
        if not equipment or not tag:
            raise ConfigError(f"{path}:{line}: Equipment and RTDB Tag ID cannot be empty")
        key = (equipment, tag)
        if key in seen:
            raise ConfigError(f"{path}:{line}: duplicate Equipment and RTDB Tag ID: {key}")
        seen.add(key)
        parsed = {}
        for column in numeric_columns:
            text = row[column].strip()
            if not text:
                parsed[column] = None
                continue
            try:
                value = float(text)
            except ValueError as exc:
                raise ConfigError(f"{path}:{line}: {column} must be a number or empty") from exc
            if not math.isfinite(value):
                raise ConfigError(f"{path}:{line}: {column} must be finite")
            parsed[column] = value
        if equipment != equipment_id:
            continue
        limits = {key: parsed[column] for column, key in CRV_BAND_COLUMNS.items()}
        configured = [
            limits[key]
            for key in ("alarm_low", "pre_alarm_low", "pre_alarm_high", "alarm_high")
            if limits[key] is not None
        ]
        if not configured:
            continue
        if configured != sorted(set(configured)):
            raise ConfigError(f"{path}:{line}: CRV Band values must be strictly increasing")
        limits_by_tag[tag] = limits
    return {"enabled": bool(limits_by_tag), "per_variable": limits_by_tag}


def _number(value, key, *, minimum=0, maximum=None, integer=False, exclusive=False):
    """불리언·문자열·NaN과 허용 범위를 벗어난 수치를 거부한다."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ConfigError(f"{key} must be a finite number")
    if integer and not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    if (value <= minimum if exclusive else value < minimum) or (maximum is not None and value > maximum):
        raise ConfigError(f"{key} is out of range")


def validate_analysis_period(analysis):
    """분석 시작·종료 시각의 형식과 순서를 검증한다."""
    parsed = {}
    for key in ("input_start", "input_end"):
        value = analysis.get(key)
        if value is None:
            parsed[key] = None
            continue
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"analysis.{key}은 따옴표로 감싼 날짜·시간 문자열 또는 null이어야 합니다")
        try:
            parsed[key] = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConfigError(f"analysis.{key}의 날짜·시간 형식이 올바르지 않습니다: {value}") from exc
    start = parsed["input_start"]
    end = parsed["input_end"]
    if start is None or end is None:
        return
    try:
        invalid_order = start > end
    except TypeError as exc:
        raise ConfigError("analysis.input_start와 input_end는 같은 시간대 형식을 사용해야 합니다") from exc
    if invalid_order:
        raise ConfigError("analysis.input_start는 input_end보다 늦을 수 없습니다")


def normalize_model_name(value):
    """모델명의 공백·대소문자를 정리하고 비어 있거나 문자열이 아닌 값은 거부한다."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("모델명은 비어 있지 않은 문자열이어야 합니다")
    return value.strip().lower()


def validate_model_config(name, options):
    """모델 생성 전에 자주 발생하는 YAML 입력 오류를 검사한다."""
    if name not in MODEL_NAMES:
        raise ConfigError(f"Unsupported active_model: {name}")
    if not isinstance(options, Mapping):
        raise ConfigError(f"models.{name} must be a mapping")
    common = {
        "window_size",
        "epochs",
        "dtype",
        "learning_rate",
        "weight_decay",
        "gradient_clip_val",
        "dropout",
        "num_heads",
        "loss",
        "anomaly_score",
    }
    tranad_keys = {
        "d_model",
        "encoder_layers",
        "decoder_layers",
        "dim_feedforward",
        "phase_weighting",
        "hybrid_mae_weight",
    }
    specific = {
        "tranad": tranad_keys,
        "tranad_v2": {*tranad_keys, "head_dim"},
        "tft": {
            "hidden_size",
            "lstm_layers",
            "forecast_horizon",
            "n_static_vars",
            "quantiles",
            "use_vsn",
            "use_static",
            "qband_low",
            "qband_high",
            "quantile_loss_weight",
            "qband_tolerance",
        },
        "lstm_ae": {
            "hidden_size",
            "latent_size",
            "num_layers",
            "hybrid_mae_weight",
        },
    }
    unknown = set(options) - common - specific[name]
    if unknown:
        raise ConfigError(f"Unknown config keys in models.{name}: {sorted(unknown)}")
    if name == "tranad_v2":
        structure_keys = {"d_model", "num_heads", "head_dim"}
        present = structure_keys & set(options)
        if present and present != structure_keys:
            raise ConfigError("TranAD v2 구조값은 생략하거나 d_model, num_heads, head_dim을 모두 지정해야 합니다")
        if present:
            for key in structure_keys:
                _number(options[key], f"models.tranad_v2.{key}", minimum=1, integer=True)
            if tuple(options[key] for key in ("d_model", "num_heads", "head_dim")) != (64, 4, 16):
                raise ConfigError("TranAD v2 구조는 기존 체크포인트와 호환되는 64/4/16으로 고정됩니다")
    for key in ("window_size", "epochs"):
        _number(require(options, key), f"models.{name}.{key}", minimum=2 if key == "window_size" else 1, integer=True)
    for key in ("learning_rate", "weight_decay", "gradient_clip_val"):
        _number(require(options, key), f"models.{name}.{key}", exclusive=key == "learning_rate")
    _number(require(options, "dropout"), f"models.{name}.dropout", maximum=1)
    if require(options, "dtype") not in {"float32", "float64"}:
        raise ConfigError(f"models.{name}.dtype must be float32 or float64")
    keys = {
        "tranad": ("encoder_layers", "decoder_layers", "dim_feedforward"),
        "tranad_v2": ("encoder_layers", "decoder_layers", "dim_feedforward"),
        "tft": ("hidden_size", "lstm_layers", "forecast_horizon", "n_static_vars"),
        "lstm_ae": ("hidden_size", "latent_size", "num_layers"),
    }
    for key in keys[name]:
        _number(require(options, key), f"models.{name}.{key}", minimum=1, integer=True)
    heads = options.get("num_heads")
    if heads is not None or name == "tft":
        _number(heads, f"models.{name}.num_heads", minimum=1, integer=True)
    if name == "tft":
        _number(require(options, "qband_tolerance"), "models.tft.qband_tolerance", exclusive=True)
        _number(require(options, "quantile_loss_weight"), "models.tft.quantile_loss_weight")
        if options["hidden_size"] % heads:
            raise ConfigError("models.tft.num_heads must divide hidden_size")
        quantiles = require(options, "quantiles")
        if not isinstance(quantiles, list) or not quantiles:
            raise ConfigError("models.tft.quantiles must be a non-empty list")
        for value in quantiles:
            _number(value, "models.tft.quantiles", minimum=0, maximum=1, exclusive=True)
        if quantiles != sorted(set(quantiles)) or quantiles[-1] >= 1:
            raise ConfigError("models.tft.quantiles must be strictly increasing inside (0, 1)")
        if not 0 < options["qband_low"] < options["qband_high"] < 1:
            raise ConfigError("models.tft qband must satisfy 0 < low < high < 1")
        for key in ("use_vsn", "use_static"):
            if not isinstance(require(options, key), bool):
                raise ConfigError(f"models.tft.{key} must be true or false")
    else:
        _number(require(options, "hybrid_mae_weight"), f"models.{name}.hybrid_mae_weight", maximum=1)


def load_config(path=None, *, model=None, run_overrides=None):
    """실행 설정을 검증하고 경로를 YAML 기준의 절대 경로로 변환한다."""
    path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    config = load_yaml(path)
    allowed = {
        "active_model",
        "model_version",
        "equipment_id",
        "run",
        "data",
        "preprocessing",
        "dataloader",
        "training",
        "models",
        "calibration_rules",
        "alarm_rules",
        *EWS_SECTIONS,
    }
    unknown = set(config) - allowed
    if unknown:
        raise ConfigError(f"Unknown config keys: {sorted(unknown)}")
    config["active_model"] = normalize_model_name(model if model is not None else require(config, "active_model"))
    model_version = require(config, "model_version")
    if not isinstance(model_version, str) or not MODEL_VERSION_PATTERN.fullmatch(model_version):
        raise ConfigError("model_version must use vMAJOR_MINOR_PATCH format")
    equipment_id = require(config, "equipment_id")
    if not isinstance(equipment_id, str) or not EQUIPMENT_ID_PATTERN.fullmatch(equipment_id):
        raise ConfigError("equipment_id must contain only letters, numbers, underscores, and hyphens")
    for section in (
        "run",
        "data",
        "preprocessing",
        "dataloader",
        "training",
        "models",
        "calibration_rules",
        "alarm_rules",
        *EWS_SECTIONS,
    ):
        if not isinstance(require(config, section), dict):
            raise ConfigError(f"{section} must be a mapping")
    config["run"] = merge_settings(config["run"], run_overrides or {})
    validate_run_config(config["run"])
    name = config["active_model"]
    validate_model_config(name, require(config, f"models.{name}"))
    validate_training_config(config["training"])
    validate_preprocessing_config(config["preprocessing"])
    for key in ("batch_size", "eval_batch_size", "num_workers", "time_series_splits"):
        _number(
            require(config, f"dataloader.{key}"),
            f"dataloader.{key}",
            minimum=0 if key == "num_workers" else (2 if key == "time_series_splits" else 1),
            integer=True,
        )
    validation_ratio = require(config, "dataloader.validation_ratio")
    _number(validation_ratio, "dataloader.validation_ratio", exclusive=True)
    if validation_ratio >= 1:
        raise ConfigError("dataloader.validation_ratio must be less than 1")
    for key in ("time_series_split_enabled", "shuffle_train", "pin_memory", "drop_last", "pad_start"):
        if not isinstance(require(config, f"dataloader.{key}"), bool):
            raise ConfigError(f"dataloader.{key} must be true or false")
    for key in ("train_path", "train_labels_path", "test_path", "labels_path", "tag_description_path", "crv_path"):
        value = require(config, f"data.{key}")
        config["data"][key] = resolve_path(value, path.parent)
    for key in (
        "checkpoint_path",
        "calibration_path",
        "input_path",
        "background_path",
        "output_directory",
    ):
        config["run"][key] = resolve_path(require(config, f"run.{key}"), path.parent)
    for dotted in ("training.checkpoint.directory", "training.log_directory"):
        value = resolve_path(require(config, dotted), path.parent)
        if value is None:
            raise ConfigError(f"{dotted} cannot be null")
        target = config
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    # Calibration·Alarm 규칙은 통합 설정에서 읽고 결과 폴더의 Config 스냅샷에 보존한다.
    defaults = config if path == DEFAULT_CONFIG_PATH else load_yaml(DEFAULT_CONFIG_PATH)
    settings = {section: config[section] for section in EWS_SECTIONS}
    check_config_keys(settings, {section: defaults[section] for section in EWS_SECTIONS}, "config")
    check_config_keys(config["calibration_rules"], defaults["calibration_rules"], "calibration_rules")
    check_config_keys(config["alarm_rules"], defaults["alarm_rules"], "alarm_rules")
    validate_analysis_period(config["analysis"])
    output = config["output"]
    if "directory" in output:
        output["directory"] = resolve_path(output["directory"], path.parent)
        if output["directory"] is None:
            raise ConfigError("output.directory cannot be null")
    return config


def validate_run_config(run):
    """모드·장치·실행 스위치의 입력 형식을 검사한다."""
    allowed = {
        "mode",
        "seed",
        "device",
        "deterministic",
        "checkpoint_path",
        "calibration_path",
        "input_path",
        "background_path",
        "output_directory",
        "fast_dev_run",
        "run_id",
    }
    unknown = set(run) - allowed
    if unknown:
        raise ConfigError(f"Unknown run keys: {sorted(unknown)}")
    if require(run, "mode") not in {"full", "train", "evaluate", "online", "analyze"}:
        raise ConfigError("run.mode must be full, train, evaluate, online or analyze")
    if require(run, "device") not in {"auto", "cpu", "cuda", "gpu", "mps"}:
        raise ConfigError("run.device must be auto, cpu, cuda, gpu or mps")
    _number(require(run, "seed"), "run.seed", integer=True)
    for key in ("fast_dev_run", "deterministic"):
        if not isinstance(require(run, key), bool):
            raise ConfigError(f"run.{key} must be true or false")
    run_id = require(run, "run_id")
    if run_id is not None and (not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id)):
        raise ConfigError("run.run_id must contain only letters, numbers, underscores, and hyphens")


def validate_preprocessing_config(preprocessing):
    """결측 처리 전략과 전처리 목록의 중복·자료형을 검사한다."""
    if set(preprocessing) != {"steps", "missing"}:
        raise ConfigError("preprocessing requires only steps and missing")
    if not isinstance(preprocessing["steps"], list):
        raise ConfigError("preprocessing.steps must be a list")
    legacy = {
        step if isinstance(step, str) else step.get("name")
        for step in preprocessing["steps"]
        if isinstance(step, str | Mapping)
    }
    if legacy & {"drop_missing", "forward_fill", "interpolate"}:
        raise ConfigError("결측 처리는 preprocessing.missing.strategy에서 하나만 선택해야 합니다")
    missing = preprocessing["missing"]
    allowed = {"strategy", "max_consecutive", "interpolation_method", "interpolation_direction"}
    if not isinstance(missing, Mapping) or set(missing) != allowed:
        raise ConfigError(f"preprocessing.missing requires {sorted(allowed)}")
    if missing["strategy"] not in {"drop", "forward_fill", "interpolate"}:
        raise ConfigError("preprocessing.missing.strategy must be drop, forward_fill, or interpolate")
    _number(missing["max_consecutive"], "preprocessing.missing.max_consecutive", minimum=1, integer=True)
    if missing["interpolation_method"] not in {"time", "linear"}:
        raise ConfigError("preprocessing.missing.interpolation_method must be time or linear")
    if missing["interpolation_direction"] not in {"forward", "backward", "both"}:
        raise ConfigError("preprocessing.missing.interpolation_direction must be forward, backward, or both")


def validate_execution_config(config):
    """실행할 모드에 필요한 값만 확인해 다른 모드의 데이터 로딩을 막는다."""
    run = config["run"]
    validate_run_config(run)
    mode = run["mode"]
    if run["fast_dev_run"] and mode != "train":
        raise ConfigError("run.fast_dev_run is available only in train mode")
    if mode in {"train", "full"} and run["checkpoint_path"]:
        raise ConfigError("TimeSeriesSplit train/full mode does not resume from one checkpoint")
    required = []
    if mode in {"full", "train"}:
        required += ["data.train_path"]
        if mode != "train" and not run["input_path"]:
            required += ["data.test_path"]
    if mode == "evaluate":
        required += [
            "run.checkpoint_path",
            "run.calibration_path",
            "run.input_path" if run["input_path"] else "data.test_path",
        ]
    if mode == "online":
        required += ["run.checkpoint_path", "run.calibration_path", "run.input_path"]
    if mode == "analyze":
        required += [
            "run.checkpoint_path",
            "run.input_path" if run["input_path"] else "data.test_path",
        ]
        if config["analysis"]["shap"]:
            required += ["run.background_path"]
    for key in required:
        if not require(config, key):
            raise ConfigError(f"{mode} mode requires {key}")
    if mode == "full":
        for key in ("include_scaler", "include_feature_schema"):
            if not config["training"]["checkpoint"][key]:
                raise ConfigError(f"full mode requires training.checkpoint.{key}=true")


def load_model_config(name, overrides=None):
    """직접 모델 생성 시에도 YAML의 모델 기본값을 사용한다."""
    defaults = require(load_yaml(DEFAULT_CONFIG_PATH), f"models.{name}")
    options = merge_settings(defaults, overrides or {})
    validate_model_config(name, options)
    return options


def load_training_config(overrides=None):
    """직접 어댑터 생성에 필요한 최적화 기본값을 YAML에서 읽는다."""
    settings = merge_settings(require(load_yaml(DEFAULT_CONFIG_PATH), "training"), overrides or {})
    validate_training_config(settings)
    return settings


def validate_training_config(settings):
    """최적화·조기 종료·체크포인트 설정을 학습 전에 검증한다."""
    if require(settings, "optimizer") not in {"adam", "adamw"}:
        raise ConfigError("training.optimizer must be adam or adamw")
    if require(settings, "scheduler") not in {None, "none", "step_lr"}:
        raise ConfigError("training.scheduler must be step_lr or none")
    if require(settings, "validation_metric") != "loss":
        raise ConfigError("training.validation_metric must be loss")
    for key in ("scheduler_step_size", "log_every_n_steps"):
        _number(require(settings, key), f"training.{key}", minimum=1, integer=True)
    _number(require(settings, "scheduler_gamma"), "training.scheduler_gamma", exclusive=True)
    _number(require(settings, "early_stopping.patience"), "training.early_stopping.patience", integer=True)
    _number(require(settings, "early_stopping.min_delta"), "training.early_stopping.min_delta")
    for key in (
        "early_stopping.enabled",
        "checkpoint.save_best_only",
        "checkpoint.include_scaler",
        "checkpoint.include_feature_schema",
        "enable_progress_bar",
    ):
        if not isinstance(require(settings, key), bool):
            raise ConfigError(f"training.{key} must be true or false")


def check_config_keys(values, reference, prefix):
    """명시적 설정의 오타를 임의의 기본값으로 숨기지 않는다."""
    if not isinstance(values, Mapping):
        raise ConfigError(f"{prefix} must be a mapping")
    unknown = set(values) - set(reference)
    if unknown:
        raise ConfigError(f"Unknown config keys in {prefix}: {sorted(unknown)}")
    for key, value in values.items():
        if key in {"per_variable", "levels"}:
            continue
        if isinstance(reference[key], Mapping):
            check_config_keys(value, reference[key], f"{prefix}.{key}")


def build_ews_settings(config, *, model):
    """통합 설정의 Calibration·Alarm 규칙으로 EWS 실행 설정을 만든다."""
    if model not in MODEL_NAMES:
        raise ConfigError(f"EWS model must be one of {', '.join(MODEL_NAMES)}")
    settings = {section: deepcopy(config[section]) for section in EWS_SECTIONS}
    rules = deepcopy(require(config, "calibration_rules"))
    settings.update(postprocessing=require(rules, "postprocessing"), threshold=require(rules, "threshold"))
    settings["alarm"] = deepcopy(require(config, "alarm_rules"))
    settings["alarm"]["crv"] = load_crv_settings(config["data"]["crv_path"], config["equipment_id"])
    validate_ews_settings(settings)
    return settings


def validate_feature_settings(settings, feature_names, *, include_threshold=True):
    """학습·추론 전에 피처를 참조하는 EWS 설정의 이름과 인덱스를 확인한다."""
    names = list(feature_names)
    if not names or len(names) != len(set(names)):
        raise ConfigError("학습 피처명은 비어 있지 않고 중복되지 않아야 합니다")

    if include_threshold:
        for key in settings["threshold"]["per_variable"]:
            if isinstance(key, bool) or not isinstance(key, str | int):
                raise ConfigError("threshold.per_variable 키는 피처명 또는 0부터 시작하는 인덱스여야 합니다")
            if isinstance(key, int) and not 0 <= key < len(names):
                raise ConfigError(f"threshold.per_variable에 없는 피처 인덱스가 있습니다: {key}")
            if isinstance(key, str) and key not in names:
                raise ConfigError(f"threshold.per_variable에 없는 피처가 있습니다: {key}")

    for condition in settings["alarm"]["inhibit_conditions"]:
        if condition["variable"] not in names:
            raise ConfigError(f"alarm.inhibit_conditions에 없는 피처가 있습니다: {condition['variable']}")
    for name in settings["alarm"]["crv"]["per_variable"]:
        if name not in names:
            raise ConfigError(f"alarm.crv.per_variable에 없는 피처가 있습니다: {name}")
    resolve_shap_targets(settings["analysis"].get("shap_target"), names)


def resolve_shap_targets(target, feature_names=None):
    """SHAP 대상 설정을 검증하고 실행 순서대로 목록화한다."""
    if target is None:
        return [None]
    targets = [target] if isinstance(target, str) else target
    if not isinstance(targets, list) or not targets or any(not isinstance(name, str) or not name for name in targets):
        raise ConfigError("analysis.shap_target은 변수명, 비어 있지 않은 변수명 목록 또는 null이어야 합니다")
    if len(targets) != len(set(targets)):
        raise ConfigError("analysis.shap_target에는 같은 변수를 중복 지정할 수 없습니다")
    if feature_names is not None:
        names = set(feature_names)
        missing = [name for name in targets if name is not None and name not in names]
        if missing:
            raise ConfigError(f"analysis.shap_target에 없는 피처가 있습니다: {missing}")
    return targets


def validate_alarm_settings(alarm, *, include_crv=True):
    """Calibration에 저장하거나 실행에 적용할 Alarm 규칙을 검증한다."""
    required = {
        "levels",
        "consecutive_count",
        "min_variables",
        "cooldown_seconds",
        "inhibit_enabled",
        "inhibit_conditions",
        "inhibit_logic",
    }
    if include_crv:
        required.add("crv")
    if not isinstance(alarm, Mapping) or set(alarm) != required:
        raise ConfigError(f"alarm requires {sorted(required)}")
    if not isinstance(alarm["levels"], Mapping) or not alarm["levels"]:
        raise ConfigError("alarm.levels must be a non-empty mapping")
    for key, value in alarm["levels"].items():
        if not isinstance(key, str) or key == "NORMAL":
            raise ConfigError("alarm.levels keys must be non-NORMAL strings")
        _number(value, f"alarm.levels.{key}", minimum=1)
    if alarm["inhibit_logic"] not in {"any", "all"}:
        raise ConfigError("alarm.inhibit_logic must be any or all")
    if not isinstance(alarm["inhibit_conditions"], list):
        raise ConfigError("alarm.inhibit_conditions must be a list")
    for condition in alarm["inhibit_conditions"]:
        if not isinstance(condition, Mapping) or set(condition) != {"variable", "operator", "value"}:
            raise ConfigError("알람 억제 조건에는 variable, operator, value가 필요합니다")
        if condition["operator"] not in {"<", "<=", ">", ">=", "==", "!="}:
            raise ConfigError("Unsupported alarm inhibit operator")
        _number(condition["value"], "alarm.inhibit.value", minimum=-float("inf"))
    for key in ("consecutive_count", "min_variables"):
        _number(alarm[key], f"alarm.{key}", minimum=1, integer=True)
    _number(alarm["cooldown_seconds"], "alarm.cooldown_seconds")
    if not isinstance(alarm["inhibit_enabled"], bool):
        raise ConfigError("alarm.inhibit_enabled must be true or false")
    if not include_crv:
        return
    crv = alarm["crv"]
    if not isinstance(crv["enabled"], bool):
        raise ConfigError("alarm.crv.enabled must be true or false")
    if not isinstance(crv["per_variable"], Mapping):
        raise ConfigError("alarm.crv.per_variable must be a mapping")
    crv_keys = ("alarm_low", "pre_alarm_low", "pre_alarm_high", "alarm_high")
    for feature, limits in crv["per_variable"].items():
        if not isinstance(feature, str) or not feature:
            raise ConfigError("alarm.crv.per_variable keys must be feature names")
        if not isinstance(limits, Mapping) or set(limits) != set(crv_keys):
            raise ConfigError(f"alarm.crv.per_variable.{feature} requires {', '.join(crv_keys)}")
        configured = []
        for key in crv_keys:
            value = limits[key]
            if value is not None:
                _number(value, f"alarm.crv.per_variable.{feature}.{key}", minimum=-float("inf"))
                configured.append(value)
        if not configured:
            raise ConfigError(f"alarm.crv.per_variable.{feature} requires at least one limit")
        if configured != sorted(set(configured)):
            raise ConfigError(f"alarm.crv.per_variable.{feature} limits must be strictly increasing")


def validate_ews_settings(settings):
    """모델별 병합을 마친 점수 후처리·임계값·알람·결과 설정을 검증한다."""
    threshold = settings["threshold"]
    if threshold["method"] not in {"manual", "percentile", "sigma", "pot"}:
        raise ConfigError("Unsupported threshold.method")
    for key in ("default", "sigma_k"):
        _number(threshold[key], f"threshold.{key}")
    _number(threshold["percentile"], "threshold.percentile", maximum=100)
    for key in ("pot_risk", "pot_init_quantile"):
        _number(threshold[key], f"threshold.{key}", maximum=1, exclusive=True)
        if threshold[key] >= 1:
            raise ConfigError(f"threshold.{key} must be less than 1")
    _number(threshold["pot_multiplier"], "threshold.pot_multiplier", exclusive=True)
    if not isinstance(threshold["per_variable"], Mapping):
        raise ConfigError("threshold.per_variable must be a mapping")
    for value in threshold["per_variable"].values():
        _number(value, "threshold.per_variable")
    validate_alarm_settings(settings["alarm"])
    for section, keys in {
        "postprocessing": ("smooth_window",),
        "output": ("plot_dpi",),
        "analysis": ("max_samples", "top_k", "background_samples", "shap_nsamples"),
    }.items():
        for key in keys:
            _number(settings[section][key], f"{section}.{key}", minimum=1, integer=True)
    for section, keys in {
        "evaluation": ("enabled",),
        "output": ("excel", "plots"),
        "analysis": ("attention", "shap"),
    }.items():
        for key in keys:
            if not isinstance(settings[section][key], bool):
                raise ConfigError(f"{section}.{key} must be true or false")
    postprocessing = settings["postprocessing"]
    allowed = {"smooth_window", "normalization_method", "quantile_level", "mae_weight"}
    unknown = set(postprocessing) - allowed
    if unknown:
        raise ConfigError(f"Unknown postprocessing keys: {sorted(unknown)}")
    if postprocessing["normalization_method"] not in {"none", "robust", "quantile"}:
        raise ConfigError("postprocessing.normalization_method must be none, robust, or quantile")
    _number(postprocessing["quantile_level"], "postprocessing.quantile_level", exclusive=True, maximum=1)
    if postprocessing["quantile_level"] >= 1:
        raise ConfigError("postprocessing.quantile_level must be less than 1")
    _number(postprocessing["mae_weight"], "postprocessing.mae_weight", maximum=1)
    if settings["analysis"]["shap_method"] not in {"kernel", "gradient", "permutation"}:
        raise ConfigError("Unsupported analysis.shap_method")
    resolve_shap_targets(settings["analysis"].get("shap_target"))
    validate_analysis_period(settings["analysis"])
    _number(settings["online"]["poll_interval_seconds"], "online.poll_interval_seconds", exclusive=True)
    filename = settings["online"]["database_filename"]
    if not isinstance(filename, str) or filename in {"", ".", ".."} or Path(filename).name != filename:
        raise ConfigError("online.database_filename은 경로가 아닌 파일명이어야 합니다")
