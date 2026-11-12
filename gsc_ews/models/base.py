"""전처리와 독립적인 이상탐지 모델의 공통 Lightning 수명주기."""

from copy import deepcopy

import lightning.pytorch as pl
import numpy as np
import torch
from src.udf.configuration import load_training_config


def checkpoint_value(value):
    """스케일러 상태를 안전하게 복원할 수 있는 기본 자료형으로 변환한다."""
    if isinstance(value, dict):
        return {key: checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [checkpoint_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class AnomalyLightningModule(pl.LightningModule):
    """표준 배치와 모델별 손실·점수를 연결하는 Lightning 어댑터."""

    def __init__(self, n_feats, model_config, training_config=None):
        """체크포인트 복원에 필요한 모델·최적화 설정을 저장한다."""
        super().__init__()
        if n_feats < 1:
            raise ValueError("n_feats must be positive")
        self.n_feats = int(n_feats)
        self.model_config = deepcopy(model_config)
        self.training_config = load_training_config(training_config)
        self.n_window = int(self.model_config["window_size"])
        self.input_dtype = getattr(torch, self.model_config["dtype"])
        self.save_hyperparameters(
            {
                "n_feats": self.n_feats,
                "model_config": self.model_config,
                "training_config": self.training_config,
            }
        )
        self.data_state = {}

    def forward(self, context, target=None):
        """[배치, 윈도우, 특성] 입력을 기존 모델의 두 출력으로 변환한다."""
        if context.ndim != 3 or context.shape[1:] != (self.n_window, self.n_feats):
            raise ValueError("context must have shape [batch, window_size, n_feats]")
        context = context.to(dtype=self.dtype)
        if target is None:
            target = context[:, -1]
        if target.shape != (len(context), self.n_feats):
            raise ValueError("target must have shape [batch, n_feats]")
        first, second = self.network(context.permute(1, 0, 2), target.to(dtype=self.dtype).unsqueeze(0))
        return first.squeeze(0), second.squeeze(0)

    def batch_outputs(self, batch):
        """모델 어댑터에서 손실과 공통 추론 출력을 계산한다.

        하위 모델은 모델 종류와 관계없이 ``prediction``과 ``anomaly_score``를
        ``[batch, feature]`` 형태의 Tensor로 반환해야 한다. Forecasting 모델은
        표시할 예측 시점의 값을 ``prediction``에 맞춰 반환한다.
        """
        raise NotImplementedError

    def _validate_prediction_outputs(self, outputs, batch_size):
        """공통 그래프·점수 계산에 필요한 모델 출력 키와 형태를 검증한다."""
        if not isinstance(outputs, dict):
            raise TypeError("batch_outputs()는 dict를 반환해야 합니다")
        expected_shape = (batch_size, self.n_feats)
        for name in ("prediction", "anomaly_score"):
            if name not in outputs:
                raise ValueError(f"모델 출력에 필수 항목 '{name}'이 없습니다")
            value = outputs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"모델 출력 '{name}'은 Tensor여야 합니다")
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"모델 출력 '{name}'은 [batch, feature] 형태 {expected_shape}여야 합니다: {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"모델 출력 '{name}'에 NaN 또는 무한대가 있습니다")

    def _shared_step(self, batch, stage):
        """배치 크기로 가중 평균한 epoch 손실을 기록한다."""
        outputs = self.batch_outputs(batch)
        loss = outputs["loss" if stage == "train" else "validation_loss"]
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["context"]),
            sync_dist=True,
        )
        self.log(
            f"{stage}_anomaly_score",
            outputs["anomaly_score"].mean(),
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["context"]),
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        """자동 최적화에 사용할 학습 손실을 반환한다."""
        self.log(
            "learning_rate",
            self.optimizers().param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["context"]),
        )
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """검증 손실을 기록하여 체크포인트와 조기 종료에 사용한다."""
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """테스트 손실과 이상 점수를 기록한다."""
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """원래 시점 순서와 예측·특성별 이상 점수를 반환하며 CPU 수집은 Trainer가 맡는다."""
        outputs = self.batch_outputs(batch)
        self._validate_prediction_outputs(outputs, len(batch["context"]))
        result = {key: value for key, value in outputs.items() if key not in {"loss", "validation_loss"}}
        result.update({key: batch[key] for key in ("target", "index", "timestamp", "label") if key in batch})
        return result

    def configure_optimizers(self):
        """설정에 따라 Adam 또는 AdamW와 epoch 단위 StepLR을 생성한다."""
        name = self.training_config["optimizer"]
        optimizer_types = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        optimizer = optimizer_types[name](
            self.parameters(),
            lr=float(self.model_config["learning_rate"]),
            weight_decay=float(self.model_config["weight_decay"]),
        )
        scheduler_name = self.training_config["scheduler"]
        if scheduler_name in {None, "none"}:
            return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(self.training_config["scheduler_step_size"]),
            gamma=float(self.training_config["scheduler_gamma"]),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def set_data_state(self, bundle):
        """외부 DataLoader가 만든 스케일러와 특성 순서를 보관한다."""
        options = self.training_config["checkpoint"]
        state = {"metadata": bundle.metadata}
        if options["include_scaler"]:
            state["scaler"] = bundle.scaler.state_dict()
        if options["include_feature_schema"]:
            state["feature_names"] = bundle.feature_names
        self.data_state = checkpoint_value(state)

    def on_save_checkpoint(self, checkpoint):
        """Lightning 체크포인트에 데이터 처리 상태를 함께 저장한다."""
        checkpoint["data_state"] = deepcopy(self.data_state)

    def on_load_checkpoint(self, checkpoint):
        """체크포인트의 스케일러·특성 순서를 복원하고 재학습 불일치를 막는다."""
        restored = checkpoint.get("data_state", {})
        for key in ("scaler", "feature_names"):
            if key in self.data_state and self.data_state[key] != restored.get(key):
                raise ValueError(f"Checkpoint {key} differs from the current DataBundle")
        saved = checkpoint.get("hyper_parameters", {}).get("model_config", {})
        for key, value in self.model_config.items():
            if key != "epochs" and key in saved and saved[key] != value:
                raise ValueError(f"Checkpoint model_config.{key} differs from current config")
        self.data_state = deepcopy(restored)
