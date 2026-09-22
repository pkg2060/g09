"""TranAD reconstruction model adapted from the authors' public implementation.

Reference paper: https://arxiv.org/abs/2201.07284
Reference repository: https://github.com/imperial-qore/TranAD
Pinned revision: 7ffb98d0c18189cc3d9ab732b4cb0278200a0af0
Reviewed sources: src/models.py (TranAD), src/dlutils.py, main.py.

The public implementation differs from the paper: its attention blocks have no
LayerNorm; positional encoding adds sin+cos to every channel; its training uses
an epoch-weighted MSE instead of separate adversarial optimizers/MAML. This module
faithfully retains those published CODE choices. It is not an exact reproduction
of every algorithm in the paper. This project changes tensor layout to [B,L,F],
supports minibatched GPU inference and float32, and aligns the reconstruction to
the LAST input timestamp rather than copying the repository's shifted plotting.
See SOURCE_INFO for machine-readable provenance and explicit differences.

Upstream BSD 3-Clause License (applies to adapted architecture components):
Copyright (c) 2022, Shreshth Tuli. All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import Tensor, nn


SOURCE_INFO: dict[str, Any] = {
    "paper": "https://arxiv.org/abs/2201.07284",
    "repository": "https://github.com/imperial-qore/TranAD",
    "commit": "7ffb98d0c18189cc3d9ab732b4cb0278200a0af0",
    "reviewed_files": ["src/models.py", "src/dlutils.py", "main.py"],
    "implementation_basis": "Official public TranAD code, not exact paper reproduction",
    "retained": [
        "d_model=2*n_features and default n_heads=n_features",
        "shared encoder; independent first and second decoders; shared sigmoid head by default",
        "un-detached squared reconstruction residual used as second-stage focus",
        "custom residual attention blocks without LayerNorm; upstream LeakyReLU(True) means slope=1",
        "official sin+cos positional encoding and sqrt(n_features) scaling",
        "training MSE_1/epoch+(1-1/epoch)*MSE_2; inference uses decoder 2",
    ],
    "project_changes": [
        "batch-first tensor interface; bounded GPU minibatches and float32 default",
        "current window-end reconstruction and timestamp-aligned targets",
        "chronological validation, validation-MAE early stopping and safe checkpoints",
        "MAE scoring is a user requirement; public code scores squared error",
        "no point adjustment of labels or anomaly predictions",
        "optional linear output head is an explicit project adaptation for un-clipped MinMax values",
        "independent bounded-output experiment: scaled_sigmoid(z)=(1+2*margin)*sigmoid(z)-margin; no input mixing or score rescaling",
        "optional L1/Huber/MSE+L1 objectives retain the official two-stage epoch weighting",
    ],
    "paper_code_differences": [
        "public code omits paper LayerNorm, separate adversarial objectives and MAML",
        "public code uses 1/epoch weights, not the paper epsilon exponential schedule",
        "public code uses feedforward width 16; paper reports a different width",
    ],
}


def canonical_model_config(model_cfg: dict) -> dict:
    """Validate output semantics, preserving old checkpoints with implicit margin=0."""
    cfg = copy.deepcopy(model_cfg)
    activation = str(cfg.get("output_activation", "sigmoid"))
    if activation not in {"sigmoid", "linear", "scaled_sigmoid"}:
        raise ValueError("output_activation must be sigmoid, linear or scaled_sigmoid")
    value = cfg.get("output_margin", 0.0)
    if isinstance(value, bool):
        raise ValueError("output_margin must be finite and nonnegative, not boolean")
    try:
        margin = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("output_margin must be finite and nonnegative") from error
    if not math.isfinite(margin) or margin < 0 or not math.isfinite(1 + 2 * margin):
        raise ValueError("output_margin must be finite and nonnegative")
    if activation != "scaled_sigmoid" and margin != 0:
        raise ValueError("output_margin must be 0 for sigmoid and linear outputs")
    cfg["output_activation"] = activation
    cfg["output_margin"] = margin
    return cfg


class ScaledSigmoid(nn.Module):
    """Fixed output interval; the current sensor input is never mixed into the head."""

    def __init__(self, margin: float):
        super().__init__()
        cfg = canonical_model_config({"output_activation": "scaled_sigmoid", "output_margin": margin})
        self.margin = cfg["output_margin"]  # saved in model_cfg, not a trainable parameter

    def forward(self, logits: Tensor) -> Tensor:
        return (1 + 2 * self.margin) * torch.sigmoid(logits) - self.margin


class OfficialPositionalEncoding(nn.Module):
    """Keep upstream sin+cos channels, deliberately not alternating sin/cos."""

    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(d_model, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        values = torch.sin(positions * frequencies) + torch.cos(positions * frequencies)
        self.register_buffer("pe", values.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, : x.shape[1]])


class OfficialEncoderLayer(nn.Module):
    """Upstream residual attention + feedforward, without added normalization."""

    def __init__(self, width: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(width, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, width)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # Upstream writes LeakyReLU(True): the positional True sets the NEGATIVE
        # SLOPE to 1, not inplace=True. Preserve that behavior explicitly.
        self.activation = nn.LeakyReLU(negative_slope=1.0, inplace=False)

    def forward(self, src: Tensor) -> Tensor:
        # need_weights=False avoids allocating attention maps during GPU training.
        src = src + self.dropout1(self.self_attn(src, src, src, need_weights=False)[0])
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return src + self.dropout2(feedforward)


class OfficialDecoderLayer(nn.Module):
    """Single-timestamp query attends to only the supplied historical window."""

    def __init__(self, width: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(width, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, width)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=1.0, inplace=False)

    def forward(self, target: Tensor, memory: Tensor) -> Tensor:
        target = target + self.dropout1(
            self.self_attn(target, target, target, need_weights=False)[0]
        )
        target = target + self.dropout2(
            self.multihead_attn(target, memory, memory, need_weights=False)[0]
        )
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(target))))
        return target + self.dropout3(feedforward)


class TranAD(nn.Module):
    """Two-stage reconstruction with differentiable self-conditioning.

    x must contain only sensor values, in chronological windows [batch, time,
    features]. Alarm labels and Inhibit must never enter this interface.
    The output corresponds exactly to x[:, -1, :].
    """

    def __init__(self, n_features: int, model_cfg: dict):
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        self.name = "TranAD"
        self.n_features = self.n_feats = int(n_features)
        self.config = canonical_model_config(model_cfg)
        self.window = self.n_window = int(model_cfg["window"])
        dropout = float(model_cfg["dropout"])
        ff_dim = int(model_cfg["ff_dim"])
        heads = model_cfg.get("n_heads")
        heads = n_features if heads is None else int(heads)
        encoder_count = int(model_cfg["encoder_layers"])
        decoder_count = int(model_cfg["decoder_layers"])
        dtype_name = str(model_cfg["dtype"])
        output_activation = self.config["output_activation"]
        output_margin = self.config["output_margin"]
        if self.window < 1 or ff_dim < 1 or encoder_count < 1 or decoder_count < 1:
            raise ValueError("window, ff_dim and layer counts must be positive")
        if heads < 1 or (2 * n_features) % heads:
            raise ValueError("n_heads must divide 2*n_features")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if dtype_name not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        self.config["n_heads"] = heads
        self.pos_encoder = OfficialPositionalEncoding(2 * n_features, dropout, self.window)
        encoder = OfficialEncoderLayer(2 * n_features, heads, ff_dim, dropout)
        first_decoder = OfficialDecoderLayer(2 * n_features, heads, ff_dim, dropout)
        second_decoder = OfficialDecoderLayer(2 * n_features, heads, ff_dim, dropout)
        # deepcopy matches the official TransformerEncoder/Decoder stack initialization.
        self.encoder = nn.ModuleList([copy.deepcopy(encoder) for _ in range(encoder_count)])
        self.decoder1 = nn.ModuleList([copy.deepcopy(first_decoder) for _ in range(decoder_count)])
        self.decoder2 = nn.ModuleList([copy.deepcopy(second_decoder) for _ in range(decoder_count)])
        activation = {"sigmoid": nn.Sigmoid, "linear": nn.Identity,
                      "scaled_sigmoid": lambda: ScaledSigmoid(output_margin)}[output_activation]()
        self.fcn = nn.Sequential(nn.Linear(2 * n_features, n_features), activation)
        self.source_info = copy.deepcopy(SOURCE_INFO)
        self.source_info["selected_output_activation"] = output_activation
        self.source_info["selected_output_margin"] = output_margin
        self.source_info["selected_output_interval"] = (
            None if output_activation == "linear" else [-output_margin, 1 + output_margin])
        self.source_info["output_head_version"] = "bounded-output-v1"
        self.to(dtype=getattr(torch, dtype_name))

    def _encode(self, x: Tensor, focus: Tensor) -> tuple[Tensor, Tensor]:
        memory = self.pos_encoder(torch.cat((x, focus), dim=-1) * math.sqrt(self.n_features))
        for layer in self.encoder:
            memory = layer(memory)
        query = x[:, -1:, :].repeat(1, 1, 2)
        return query, memory

    def _decode(self, x: Tensor, focus: Tensor, layers: nn.ModuleList) -> Tensor:
        query, memory = self._encode(x, focus)
        for layer in layers:
            query = layer(query, memory)
        return self.fcn(query)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.window, self.n_features):
            raise ValueError(f"Expected [B,{self.window},{self.n_features}], got {tuple(x.shape)}")
        first = self._decode(x, torch.zeros_like(x), self.decoder1)
        focus = (first - x).square()  # do not detach: gradients flow into decoder 1.
        second = self._decode(x, focus, self.decoder2)
        return first[:, 0, :], second[:, 0, :]


def build_model(n_features: int, model_cfg: dict) -> nn.Module:
    """Build on CPU; Trainer/reconstruct enforce the caller's CUDA execution policy."""
    return TranAD(n_features, model_cfg)
