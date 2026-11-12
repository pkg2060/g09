"""TranAD의 네트워크 구조·손실·이상 점수와 Lightning 학습을 정의한다.

데이터 전처리와 DataLoader 구성은 공통 src 패키지에서 담당한다.
"""

import math

import torch
import torch.nn as nn
from src.udf.configuration import load_model_config

from .base import AnomalyLightningModule


class PositionalEncoding(nn.Module):
    """
    Position Encoding (Vaswani et al., "Attention is All You Need", 2017)

    [FIX] 공식 TranAD 코드(dlutils.py)와 정확히 동일하게 구현
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x, pos=0):
        """
        Args:
            x: (seq_len, batch_size, d_model)
            pos: 시작 위치 오프셋
        """
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)


# ==============================================================================
# Transformer Encoder Layer (공식 코드 dlutils.py와 동일)
# ==============================================================================
class TransformerEncoderLayer(nn.Module):
    """
    [FIX] 공식 TranAD 코드(dlutils.py)의 TransformerEncoderLayer와 동일하게 구현

    공식 코드에서는 LayerNorm을 사용하지 않음 (실험적으로 더 나은 결과)
    LeakyReLU 활성화 함수 사용
    """

    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        # Multi-Head Self Attention (논문 Eq. 2-3)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed-Forward Network (2-layer MLP)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # LeakyReLU activation (공식 구현과 동일)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        """
        공식 코드와 동일:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        """
        # Self-Attention: Q=K=V=src
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)

        # Feed-Forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src


# ==============================================================================
# Transformer Decoder Layer (공식 코드 dlutils.py와 동일)
# ==============================================================================
class TransformerDecoderLayer(nn.Module):
    """
    [FIX] 공식 TranAD 코드(dlutils.py)의 TransformerDecoderLayer와 동일하게 구현
    """

    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        # Masked Self-Attention (Eq. 5, 첫 번째 줄)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Cross-Attention with Encoder output (Eq. 5, 세 번째 줄)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed-Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Dropouts
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        **kwargs,
    ):
        """
        공식 코드와 동일:
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        """
        # Self-Attention on window data
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)

        # Cross-Attention: Query=window, Key=Value=encoder output
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)

        # Feed-Forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


# ==============================================================================
# TranAD Main Model (공식 코드 models.py와 동일)
# ==============================================================================
class TranAD(nn.Module):
    """
    TranAD: Deep Transformer Networks for Anomaly Detection

    [FIX] 공식 TranAD 코드(imperial-qore/TranAD/src/models.py)의
    TranAD 클래스와 정확히 동일한 구조로 수정

    공식 코드:
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2*feats, nhead=feats, ...)
        decoder_layers1 = TransformerDecoderLayer(d_model=2*feats, nhead=feats, ...)
        decoder_layers2 = TransformerDecoderLayer(d_model=2*feats, nhead=feats, ...)
        self.fcn = nn.Sequential(nn.Linear(2*feats, feats), nn.Sigmoid())
    """

    def __init__(self, n_feats, model_config):
        """검증된 구조 설정을 명시적으로 주입한다."""
        super().__init__()
        options = model_config
        self.name = "TranAD"
        self.n_feats = n_feats
        self.n_window = options["window_size"]
        self.d_model = options["d_model"] or 2 * n_feats
        if self.d_model != 2 * n_feats:
            raise ValueError("TranAD d_model must equal 2 * n_feats")
        self.n_heads = options["num_heads"] or n_feats
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("num_heads must divide 2 * n_feats")
        dropout = options["dropout"]
        dim_feedforward = options["dim_feedforward"]
        encoder_count = options["encoder_layers"]
        decoder_count = options["decoder_layers"]

        # ─── Positional Encoding ───
        # [FIX] 공식 코드: PositionalEncoding(2 * feats, 0.1, self.n_window)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, self.n_window)

        # ─── Transformer Encoder (논문 Eq. 4) ───
        # [FIX] 공식 코드: TransformerEncoderLayer(d_model=2*feats, nhead=feats,
        #                                          dim_feedforward=16, dropout=0.1)
        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=encoder_count, enable_nested_tensor=False
        )

        # ─── Transformer Decoder 1 ───
        # [FIX] 공식 코드와 동일
        decoder_layers1 = TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layers1, num_layers=decoder_count)

        # ─── Transformer Decoder 2 ───
        # [FIX] 공식 코드와 동일
        decoder_layers2 = TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layers2, num_layers=decoder_count)

        # ─── Output Layer (논문 Eq. 6) ───
        # [FIX] 공식 코드: nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
        self.fcn = nn.Sequential(nn.Linear(self.d_model, n_feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        """
        [FIX] 공식 코드의 encode 함수와 동일:

        def encode(self, src, c, tgt):
            src = torch.cat((src, c), dim=2)
            src = src * math.sqrt(self.n_feats)
            src = self.pos_encoder(src)
            memory = self.transformer_encoder(src)
            tgt = tgt.repeat(1, 1, 2)
            return tgt, memory
        """
        # Focus Score와 입력 연결 → (window_size, batch, 2*n_feats)
        src = torch.cat((src, c), dim=2)

        # Scaling + Positional Encoding
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)

        # Transformer Encoder
        memory = self.transformer_encoder(src)

        # 디코더 입력: tgt를 d_model 차원으로 확장
        tgt = tgt.repeat(1, 1, 2)

        return tgt, memory

    def forward(self, src, tgt):
        """
        [FIX] 공식 코드의 forward와 동일:

        def forward(self, src, tgt):
            # Phase 1 - Without anomaly scores
            c = torch.zeros_like(src)
            x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
            # Phase 2 - With anomaly scores
            c = (x1 - src) ** 2
            x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
            return x1, x2
        """
        # ═══ Phase 1: Focus Score = 0 ═══
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))

        # ═══ Phase 2: Focus Score = ||O1 - src||^2 ═══
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))

        return x1, x2


# Lightning 어댑터: 모델별 학습 손실과 이상 점수
class TranADLightningModule(AnomalyLightningModule):
    """기존 TranAD 구조를 Lightning의 자동 최적화로 학습한다."""

    def __init__(self, n_feats, model_config=None, training_config=None):
        """models.tranad 설정을 적용하고 원본 모델 구조를 생성한다."""
        options = load_model_config("tranad", model_config)
        if options["loss"] != "phase1_phase2_mse":
            raise ValueError("TranAD loss must be phase1_phase2_mse")
        if options["phase_weighting"] != "epoch_dependent":
            raise ValueError("TranAD phase_weighting must be epoch_dependent")
        if options["anomaly_score"] not in {"phase2_mse", "mae", "hybrid"}:
            raise ValueError("Unsupported TranAD anomaly_score")
        super().__init__(n_feats, options, training_config)
        self.network = TranAD(n_feats, model_config=options)
        self.to(dtype=self.input_dtype)

    def batch_outputs(self, batch):
        """1/(epoch+1) 가중 손실과 설정에서 선택한 Phase 2 점수를 계산한다."""
        target = batch["target"].to(dtype=self.dtype)
        first, second = self(batch["context"], target)
        first_error = (first - target).square()
        second_error = (second - target).square()
        weight = 1.0 / (self.current_epoch + 1)
        loss = (weight * first_error + (1.0 - weight) * second_error).mean()
        mae = (second - target).abs()
        # 원래 hybrid의 TranAD 점수와 MSE는 같은 값이므로 하나의 비중으로 합친다.
        alpha = self.model_config["hybrid_mae_weight"]
        scores = {"phase2_mse": second_error, "mae": mae, "hybrid": alpha * mae + (1 - alpha) * second_error}
        return {
            "loss": loss,
            "validation_loss": second_error.mean(),
            "prediction": second,
            "anomaly_score": scores[self.model_config["anomaly_score"]],
            **{f"score_{name}": value for name, value in scores.items()},
        }
