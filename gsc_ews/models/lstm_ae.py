"""LSTM AutoEncoder의 네트워크 구조·손실·이상 점수와 Lightning 학습을 정의한다.

데이터 전처리와 DataLoader 구성은 공통 src 패키지에서 담당한다.
"""

import torch
import torch.nn as nn
from src.udf.configuration import load_model_config

from .base import AnomalyLightningModule


# ==============================================================================
# LSTM AutoEncoder Main Model
# ==============================================================================
class LSTMAutoEncoder(nn.Module):
    """LSTM 기반 Multivariate Time-Series AutoEncoder.

    입력 Window를 Encoder LSTM으로 압축하여 Latent Representation을 생성하고,
    Latent Vector를 전체 Timestep에 반복 입력하여 Decoder LSTM으로 원래 시계열을 복원한다.

    입력과 출력은 동일한 `(batch, window_size, n_feats)` 형태를 유지한다.

    Args:
        n_feats (int): 입력 Feature 수.
        model_config (dict): 검증된 LSTM-AE 모델 설정.
    """

    def __init__(
        self,
        n_feats: int,
        model_config: dict,
    ) -> None:
        """검증된 LSTM-AE 구조 설정을 모델 인스턴스에 적용한다."""
        super().__init__()

        options = model_config

        self.name = "LSTM-AE"
        self.n_feats = n_feats
        self.n_window = options["window_size"]
        self.hidden_size = options["hidden_size"]
        self.latent_size = options["latent_size"]
        self.num_layers = options["num_layers"]
        self.dropout = options["dropout"]

        # encoder
        self.encoder = nn.LSTM(
            input_size=n_feats,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        # latent projection
        self.to_latent = nn.Linear(
            self.hidden_size,
            self.latent_size,
        )

        # decoder
        self.decoder = nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        # output projection
        self.output_projection = nn.Linear(
            self.hidden_size,
            n_feats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 Window를 동일한 길이의 Multivariate Time-Series로 복원한다.

        Args:
            x (torch.Tensor): 입력 시계열.
                Shape `(batch, window_size, n_feats)`.

        Returns:
            torch.Tensor: 복원된 시계열.
                Shape `(batch, window_size, n_feats)`.
        """
        # encoder
        _, (hidden, _) = self.encoder(x)

        # final hidden state of the last lstm layer
        hidden_last = hidden[-1]

        # latent representation
        latent = torch.tanh(self.to_latent(hidden_last))

        # repeat latent vector across the time dimension
        decoder_input = latent.unsqueeze(1).expand(
            -1,
            x.size(1),
            -1,
        )

        # decoder
        decoded, _ = self.decoder(decoder_input)

        # output projection
        reconstructed = self.output_projection(decoded)

        return reconstructed


# ==============================================================================
# LSTM AutoEncoder Lightning Module
# ==============================================================================
class LSTMAutoEncoderLightningModule(AnomalyLightningModule):
    """LSTM-AE를 공통 Lightning 학습 파이프라인에 연결한다."""

    def __init__(
        self,
        n_feats: int,
        model_config: dict | None = None,
        training_config: dict | None = None,
    ) -> None:
        """models.lstm_ae 설정을 적용하고 LSTM-AE 모델을 생성한다."""
        options = load_model_config(
            "lstm_ae",
            model_config,
        )

        if options["loss"] != "mse":
            raise ValueError("LSTM-AE loss must be mse")

        if options["anomaly_score"] not in {
            "mae",
            "mse",
            "hybrid",
        }:
            raise ValueError("Unsupported LSTM-AE anomaly_score")

        super().__init__(
            n_feats,
            options,
            training_config,
        )

        self.network = LSTMAutoEncoder(
            n_feats=n_feats,
            model_config=options,
        )

        self.to(dtype=self.input_dtype)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """입력 Window 전체를 LSTM-AE로 복원한다.

        Args:
            context (torch.Tensor): 입력 Window.
                Shape `(batch, window_size, n_feats)`.
            target (torch.Tensor | None): 공통 모델 Interface 호환을 위한 Target.
                LSTM-AE Forward 연산에서는 사용하지 않는다.

        Returns:
            torch.Tensor: 복원된 입력 Window.
                Shape `(batch, window_size, n_feats)`.

        Raises:
            ValueError: 입력 Shape이 모델 설정과 일치하지 않는 경우.
        """
        if context.ndim != 3 or context.shape[1:] != (self.n_window, self.n_feats):
            raise ValueError("context must have shape [batch, window_size, n_feats]")

        context = context.to(dtype=self.input_dtype)

        return self.network(context)

    def reconstruct(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """입력 Window의 Reconstruction을 반환한다.

        Args:
            context (torch.Tensor): 입력 Window.
                Shape `(batch, window_size, n_feats)`.

        Returns:
            torch.Tensor: 복원된 입력 Window.
                Shape `(batch, window_size, n_feats)`.
        """
        return self(context)

    def batch_outputs(
        self,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """전체 Window 복원 손실과 마지막 시점의 Feature별 이상 점수를 계산한다.

        LSTM-AE는 전체 Window를 복원하도록 MSE Loss로 학습하며,
        EWS Pipeline에 전달되는 Prediction과 Anomaly Score는
        Window의 마지막 Timestep을 기준으로 계산한다.

        Args:
            batch (dict): DataLoader에서 전달된 Batch.
                `context`: `(batch, window_size, n_feats)`.
                `target`: `(batch, n_feats)`.

        Returns:
            dict[str, torch.Tensor]: 학습 손실, 검증 손실,
            Prediction 및 Feature별 Anomaly Score.
        """
        context = batch["context"].to(dtype=self.input_dtype)
        target = batch["target"].to(dtype=self.input_dtype)

        # full-window reconstruction
        reconstructed = self.reconstruct(context)

        # training loss: full-window mse
        reconstruction_error = (reconstructed - context).square()
        loss = reconstruction_error.mean()

        # ews output: last timestep reconstruction
        prediction = reconstructed[:, -1, :]

        # feature-wise error at the last timestep
        squared_error = (prediction - target).square()
        absolute_error = (prediction - target).abs()

        # anomaly score
        alpha = self.model_config["hybrid_mae_weight"]

        scores = {
            "mae": absolute_error,
            "mse": squared_error,
            "hybrid": (alpha * absolute_error + (1.0 - alpha) * squared_error),
        }

        return {
            "loss": loss,
            "validation_loss": loss,
            "prediction": prediction,
            "anomaly_score": scores[self.model_config["anomaly_score"]],
            **{f"score_{name}": value for name, value in scores.items()},
        }
