"""TranAD v2의 네트워크 구조·손실·이상 점수와 Lightning 학습을 정의한다.

TranAD v2의 고정 기본 구조는 모델 내부에서 관리하며 Config에는 노출하지 않는다.
Transformer FFN에 GELU와 Pre-LayerNorm을 적용한다.

데이터 전처리와 DataLoader 구성은 공통 src 패키지에서 담당한다.
"""

import math

import torch
import torch.nn as nn
from src.udf.configuration import load_model_config

from .base import AnomalyLightningModule

# ==============================================================================
# TranAD Network Components
# ==============================================================================


class PositionalEncoding(nn.Module):
    """입력 Sequence에 위치 정보를 적용한다.

    기존 TranAD v2의 Positional Encoding 계산식을 유지한다.

    Args:
        d_model (int): Transformer Hidden 차원.
        dropout (float): Dropout 비율.
        max_len (int): 최대 Sequence 길이.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        positional_encoding = torch.zeros(max_len, d_model)

        positional_encoding += torch.sin(position * div_term)
        positional_encoding += torch.cos(position * div_term)

        positional_encoding = positional_encoding.unsqueeze(1)

        self.register_buffer(
            "positional_encoding",
            positional_encoding,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: int = 0,
    ) -> torch.Tensor:
        """입력 Sequence에 위치 정보를 추가한다.

        Args:
            x (torch.Tensor): Shape (T, B, d_model).
            pos (int): 위치 정보 시작 인덱스.

        Returns:
            torch.Tensor: 위치 정보가 적용된 입력 Tensor.
        """
        x = x + self.positional_encoding[pos : pos + x.size(0)]

        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm 기반 Transformer Encoder Layer.

    Self-Attention과 GELU FFN을 Residual Connection으로 연결한다.

    Args:
        d_model (int): Transformer Hidden 차원.
        nhead (int): Multi-Head Attention Head 수.
        dim_feedforward (int): FFN Hidden 차원.
        dropout (float): Dropout 비율.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )

        # feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

        # layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_feedforward = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Self-Attention과 FFN 연산을 수행한다.

        Args:
            src (torch.Tensor): Shape (T, B, d_model).
            src_mask (torch.Tensor | None): Attention Mask.
            src_key_padding_mask (torch.Tensor | None): Padding Mask.

        Returns:
            torch.Tensor: Encoder Layer 출력.
        """

        # pre-layernorm + self-attention
        normalized_src = self.norm1(src)

        attention_output = self.self_attn(
            normalized_src,
            normalized_src,
            normalized_src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]

        src = src + self.dropout_attention(attention_output)

        # pre-layernorm + gelu feed-forward
        normalized_src = self.norm2(src)

        feedforward_output = self.linear2(self.dropout(self.activation(self.linear1(normalized_src))))

        src = src + self.dropout_feedforward(feedforward_output)

        return src


class TransformerDecoderLayer(nn.Module):
    """Pre-LayerNorm 기반 Transformer Decoder Layer.

    Self-Attention, Cross-Attention, GELU FFN을 순차적으로 수행한다.

    Args:
        d_model (int): Transformer Hidden 차원.
        nhead (int): Multi-Head Attention Head 수.
        dim_feedforward (int): FFN Hidden 차원.
        dropout (float): Dropout 비율.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )

        # cross-attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )

        # feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

        # layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout_self_attention = nn.Dropout(dropout)
        self.dropout_cross_attention = nn.Dropout(dropout)
        self.dropout_feedforward = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Decoder의 Attention 및 FFN 연산을 수행한다.

        Args:
            tgt (torch.Tensor): Decoder 입력.
            memory (torch.Tensor): Encoder Memory.
            tgt_mask (torch.Tensor | None): Target Attention Mask.
            memory_mask (torch.Tensor | None): Memory Attention Mask.
            tgt_key_padding_mask (torch.Tensor | None): Target Padding Mask.
            memory_key_padding_mask (torch.Tensor | None): Memory Padding Mask.

        Returns:
            torch.Tensor: Decoder Layer 출력.
        """

        # pre-layernorm + self-attention
        normalized_tgt = self.norm1(tgt)

        self_attention_output = self.self_attn(
            normalized_tgt,
            normalized_tgt,
            normalized_tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]

        tgt = tgt + self.dropout_self_attention(self_attention_output)

        # pre-layernorm + cross-attention
        normalized_tgt = self.norm2(tgt)

        cross_attention_output = self.multihead_attn(
            normalized_tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]

        tgt = tgt + self.dropout_cross_attention(cross_attention_output)

        # pre-layernorm + gelu feed-forward
        normalized_tgt = self.norm3(tgt)

        feedforward_output = self.linear2(self.dropout(self.activation(self.linear1(normalized_tgt))))

        tgt = tgt + self.dropout_feedforward(feedforward_output)

        return tgt


# ==============================================================================
# TranAD Main Model
# ==============================================================================


class TranAD(nn.Module):
    """Transformer 기반 Multivariate Time-Series 이상탐지 모델.

    입력 Window와 Focus Score를 Encoder에 전달하고,
    독립적인 두 Decoder를 통해 Phase 1/2 Reconstruction을 계산한다.

    Args:
        n_feats (int): 입력 Feature 수.
        model_config (dict): 검증된 TranAD 모델 설정.
    """

    def __init__(
        self,
        n_feats: int,
        model_config: dict,
    ) -> None:
        super().__init__()

        options = model_config

        self.name = "TranAD"
        self.n_feats = n_feats
        self.n_window = options["window_size"]

        # 기존 체크포인트와 동일한 TranAD v2 기본 구조를 사용한다.
        self.d_model = options.get("d_model", 64)
        self.n_heads = options.get("num_heads", 4)
        self.head_dim = options.get("head_dim", 16)

        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal num_heads * head_dim")

        dim_feedforward = options["dim_feedforward"]
        dropout = options["dropout"]
        encoder_layers = options["encoder_layers"]
        decoder_layers = options["decoder_layers"]

        self.input_projection = nn.Linear(2 * n_feats, self.d_model)
        self.target_projection = nn.Linear(n_feats, self.d_model)

        # positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=dropout,
            max_len=self.n_window,
        )

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )

        # phase 1 decoder
        decoder_layer_1 = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.decoder_1 = nn.TransformerDecoder(
            decoder_layer=decoder_layer_1,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        # phase 2 decoder
        decoder_layer_2 = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.decoder_2 = nn.TransformerDecoder(
            decoder_layer=decoder_layer_2,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        # output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, n_feats),
            nn.Sigmoid(),
        )

    def _encode(
        self,
        src: torch.Tensor,
        focus: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """입력 Window 및 Focus Score로 Encoder Memory를 생성한다.

        Args:
            src (torch.Tensor): Shape (T, B, D).
            focus (torch.Tensor): Shape (T, B, D).
            target (torch.Tensor): Shape (1, B, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Decoder Target과 Encoder Memory.
        """

        # concatenate original input and focus score
        encoded_input = torch.cat(
            (src, focus),
            dim=2,
        )

        encoded_input = self.input_projection(encoded_input * math.sqrt(self.n_feats))

        # positional encoding
        encoded_input = self.positional_encoding(encoded_input)

        # transformer encoder
        memory = self.encoder(encoded_input)

        decoder_target = self.target_projection(target)

        return decoder_target, memory

    def forward(
        self,
        src: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase 1 및 Phase 2 Reconstruction을 생성한다.

        Args:
            src (torch.Tensor): Shape (T, B, D).
            target (torch.Tensor): Shape (1, B, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Phase 1, Phase 2 Reconstruction.
                각 Shape은 (1, B, D).
        """

        # phase 1: reconstruction without focus score
        focus = torch.zeros_like(src)

        decoder_target, memory = self._encode(
            src=src,
            focus=focus,
            target=target,
        )

        phase_1 = self.output_projection(
            self.decoder_1(
                decoder_target,
                memory,
            )
        )

        # phase 2: reconstruction with phase 1 error
        focus = (phase_1 - src).square()

        decoder_target, memory = self._encode(
            src=src,
            focus=focus,
            target=target,
        )

        phase_2 = self.output_projection(
            self.decoder_2(
                decoder_target,
                memory,
            )
        )

        return phase_1, phase_2


# ==============================================================================
# TranAD Lightning Module
# ==============================================================================


class TranADV2LightningModule(AnomalyLightningModule):
    """TranAD v2를 공통 Lightning 학습 파이프라인에 연결한다."""

    def __init__(
        self,
        n_feats: int,
        model_config: dict | None = None,
        training_config: dict | None = None,
    ) -> None:
        """설정을 검증하고 TranAD 네트워크를 생성한다."""

        options = load_model_config(
            "tranad_v2",
            model_config,
        )

        if options["loss"] != "phase1_phase2_mse":
            raise ValueError("TranAD loss must be phase1_phase2_mse")

        if options["phase_weighting"] != "epoch_dependent":
            raise ValueError("TranAD phase_weighting must be epoch_dependent")

        if options["anomaly_score"] not in {
            "phase2_mse",
            "mae",
            "hybrid",
        }:
            raise ValueError("Unsupported TranAD anomaly_score")

        super().__init__(
            n_feats,
            options,
            training_config,
        )

        self.network = TranAD(
            n_feats=n_feats,
            model_config=options,
        )

        self.to(dtype=self.input_dtype)

    def batch_outputs(
        self,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """Phase 1·2 손실과 Phase 2 기반 이상 점수를 계산한다.

        Args:
            batch (dict): 학습 또는 추론 Batch.
                context: (B, T, D).
                target: (B, D).

        Returns:
            dict[str, torch.Tensor]:
                Loss, Reconstruction, Feature Anomaly Score.
        """

        target = batch["target"].to(dtype=self.dtype)

        # phase 1 / phase 2 reconstruction
        phase_1, phase_2 = self(
            batch["context"],
            target,
        )

        # reconstruction error
        phase_1_error = (phase_1 - target).square()
        phase_2_error = (phase_2 - target).square()

        # epoch-dependent phase weighting
        weight = 1.0 / (self.current_epoch + 1)

        loss = (weight * phase_1_error + (1.0 - weight) * phase_2_error).mean()

        # anomaly score
        absolute_error = (phase_2 - target).abs()

        alpha = self.model_config["hybrid_mae_weight"]

        scores = {
            "phase2_mse": phase_2_error,
            "mae": absolute_error,
            "hybrid": (alpha * absolute_error + (1.0 - alpha) * phase_2_error),
        }

        return {
            "loss": loss,
            "validation_loss": phase_2_error.mean(),
            "prediction": phase_2,
            "anomaly_score": scores[self.model_config["anomaly_score"]],
            **{f"score_{name}": value for name, value in scores.items()},
        }
