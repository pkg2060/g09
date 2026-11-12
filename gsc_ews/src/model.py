"""명시적 레지스트리로 Lightning 모델을 선택하고 학습·추론 모델을 생성한다."""

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import torch
from lightning.pytorch import LightningModule

from models.lstm_ae import LSTMAutoEncoderLightningModule
from models.tft import TFTLightningModule
from models.tranad import TranADLightningModule
from models.tranad_v2 import TranADV2LightningModule

from .udf.configuration import (
    ConfigError,
    normalize_model_name,
    require,
    validate_model_config,
    validate_training_config,
)

# 학습 생성과 체크포인트 복원은 같은 레지스트리를 사용한다.
MODEL_REGISTRY: Mapping[str, type[LightningModule]] = MappingProxyType(
    {
        "tranad": TranADLightningModule,
        "tranad_v2": TranADV2LightningModule,
        "tft": TFTLightningModule,
        "lstm_ae": LSTMAutoEncoderLightningModule,
    }
)


def get_model_class(name: str) -> type[LightningModule]:
    """등록된 이름만 허용하며 모듈 경로나 임의의 클래스명은 해석하지 않는다."""
    name = normalize_model_name(name)
    if name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ConfigError(f"지원하지 않는 모델입니다: {name}. 지원 모델: {supported}")
    return MODEL_REGISTRY[name]


def build_model(config: dict[str, Any], n_features: int) -> LightningModule:
    """활성 모델의 설정과 특성 수를 검증한 후 해당 Lightning 어댑터를 생성한다.

    dtype은 모델 설정으로 전달하여 가중치·입력 dtype·체크포인트 설정을 일치시킨다.
    호출자가 전달한 설정은 변경하지 않는다.
    """
    name = normalize_model_name(require(config, "active_model"))
    model_class = get_model_class(name)
    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
        raise ConfigError("n_features는 양의 정수여야 합니다")
    model_config = require(config, f"models.{name}")
    training_config = require(config, "training")
    validate_model_config(name, model_config)
    validate_training_config(training_config)
    return model_class(
        n_feats=n_features, model_config=deepcopy(model_config), training_config=deepcopy(training_config)
    )


def load_checkpoint(path, *, device="cpu") -> LightningModule:
    """저장된 모델명으로 어댑터를 선택하고 가중치·학습 데이터 상태를 복원한다."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = require(checkpoint, "data_state")
    if not isinstance(state, Mapping) or not {"scaler", "feature_names", "metadata"} <= state.keys():
        raise ConfigError("학습 처리 상태가 포함된 통합 Lightning 체크포인트가 필요합니다")
    require(state, "metadata.pipeline")
    name = normalize_model_name(require(state, "metadata.active_model"))
    parameters = require(checkpoint, "hyper_parameters")
    model = build_model(
        {
            "active_model": name,
            "models": {name: require(parameters, "model_config")},
            "training": require(parameters, "training_config"),
        },
        n_features=require(parameters, "n_feats"),
    )
    model.load_state_dict(require(checkpoint, "state_dict"), strict=True)
    model.on_load_checkpoint(checkpoint)
    if model.n_feats != len(state["feature_names"]):
        raise ConfigError("모델과 체크포인트 특성 수가 다릅니다")
    return model.to(device).eval()
