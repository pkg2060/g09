"""TFT의 네트워크 구조·손실·이상 점수와 Lightning 학습을 정의한다.

데이터 전처리와 DataLoader 구성은 공통 src 패키지에서 담당한다.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.udf.configuration import load_model_config

from .base import AnomalyLightningModule


class TimeDistributed(nn.Module):
    """
    (B, T, *)  →  (B*T, *)  로 펴서 module 적용한 후 다시 (B, T, *) 로 복원.
    TFT 논문에서 GRN 등을 시간 축마다 동일 가중치로 적용하기 위함.
    """

    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if x.dim() <= 2:
            return self.module(x)
        # x : (B, T, F)
        b, t = x.size(0), x.size(1)
        x_reshape = x.contiguous().view(b * t, *x.shape[2:])
        y = self.module(x_reshape)
        y = y.view(b, t, *y.shape[1:])
        return y


class GLU(nn.Module):
    """
    Gated Linear Unit (논문 Eq. 1)
        GLU(x) = sigma(W1 x + b1) ⊙ (W2 x + b2)
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x)) * self.fc2(x)


class GatedResidualNetwork(nn.Module):
    """
    GRN (Gated Residual Network) - 논문 Eq. 2-4
       eta_2 = ELU(W2 a + W3 c + b2)             # context c는 optional
       eta_1 = W1 eta_2 + b1
       output = LayerNorm( a + GLU(eta_1) )      # skip connection 차원 다르면 projection
    GRN은 TFT 전체에서 가장 빈번하게 사용되는 빌딩 블록.
    """

    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1, context_size=None):
        super().__init__()
        if output_size is None:
            output_size = hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size

        # 입력 - hidden 변환
        self.lin_a = nn.Linear(input_size, hidden_size)
        # context (optional)
        if context_size is not None:
            self.lin_c = nn.Linear(context_size, hidden_size, bias=False)
        # hidden → hidden
        self.lin_eta = nn.Linear(hidden_size, hidden_size)
        # GLU + LayerNorm
        self.glu = GLU(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        # skip projection (input_size != output_size일 때)
        if input_size != output_size:
            self.skip_proj = nn.Linear(input_size, output_size)
        else:
            self.skip_proj = None

    def forward(self, a, c=None):
        # a : (..., input_size)
        # c : (..., context_size)  or None
        skip = a if self.skip_proj is None else self.skip_proj(a)

        x = self.lin_a(a)
        if c is not None and self.context_size is not None:
            # context는 broadcasting 가능하도록 차원 맞춰서 더함
            ctx = self.lin_c(c)
            # ctx 가 (B, hidden) 인데 a 가 (B, T, hidden) 이면 unsqueeze 처리
            while ctx.dim() < x.dim():
                ctx = ctx.unsqueeze(-2)
            x = x + ctx
        x = F.elu(x)
        x = self.lin_eta(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.layer_norm(skip + x)


# ==============================================================================
# 1. Variable Selection Network (논문 Section 4.2)
# ==============================================================================
class VariableSelectionNetwork(nn.Module):
    """
    각 변수에 대해 독립 GRN을 통과시켜 처리한 뒤,
    softmax 기반 변수 가중치를 곱해 결합.

    입력: x : (..., n_vars, input_size_per_var)
          c : (optional) static context (..., context_size)
    출력: combined : (..., output_size)
          weights  : (..., n_vars)   → 해석성용
    """

    def __init__(self, n_vars, input_size_per_var, hidden_size, dropout=0.1, context_size=None):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_size = hidden_size
        self.input_size_per_var = input_size_per_var

        # 변수 가중치 생성용 GRN (전체 변수 flatten → softmax)
        self.weight_grn = GatedResidualNetwork(
            input_size=n_vars * input_size_per_var,
            hidden_size=hidden_size,
            output_size=n_vars,
            dropout=dropout,
            context_size=context_size,
        )

        # 변수별 GRN (변수마다 독립적으로 적용)
        self.per_var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=input_size_per_var,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                    context_size=None,
                )
                for _ in range(n_vars)
            ]
        )

    def forward(self, x, c=None):
        # x : (..., n_vars, input_size_per_var)
        # 1) 변수 가중치 (softmax)
        flat = x.reshape(*x.shape[:-2], self.n_vars * self.input_size_per_var)
        w_logits = self.weight_grn(flat, c=c)  # (..., n_vars)
        weights = torch.softmax(w_logits, dim=-1)  # (..., n_vars)

        # 2) 변수별 GRN
        per_var_outputs = []
        for i, grn in enumerate(self.per_var_grns):
            xi = x[..., i, :]  # (..., input_size_per_var)
            per_var_outputs.append(grn(xi))  # (..., hidden_size)
        per_var_stack = torch.stack(per_var_outputs, dim=-2)  # (..., n_vars, hidden)

        # 3) 가중합
        # weights : (..., n_vars) → unsqueeze 마지막 dim
        combined = (per_var_stack * weights.unsqueeze(-1)).sum(dim=-2)
        return combined, weights


# ==============================================================================
# 2. Interpretable Multi-Head Attention (논문 Section 4.4)
# ==============================================================================
class InterpretableMultiHeadAttention(nn.Module):
    """
    표준 multi-head attention과 달리, V(value) 변환 가중치를 모든 head가 공유하여
    head별 attention 가중치를 평균해도 해석 가능하도록 구성.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_attn)) V
    Output = (1/H) * sum_h Attention_h
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_head = hidden_size // num_heads

        # Q, K는 head마다 독립
        self.q_proj = nn.ModuleList([nn.Linear(hidden_size, self.d_head) for _ in range(num_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_size, self.d_head) for _ in range(num_heads)])
        # V는 모든 head가 공유 (해석성 핵심)
        self.v_proj = nn.Linear(hidden_size, self.d_head)
        # 출력 변환
        self.out_proj = nn.Linear(self.d_head, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v : (B, T, hidden)
        v_shared = self.v_proj(v)  # (B, T, d_head)

        head_outputs = []
        head_attentions = []
        for h in range(self.num_heads):
            qh = self.q_proj[h](q)  # (B, Tq, d_head)
            kh = self.k_proj[h](k)  # (B, Tk, d_head)
            scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)  # (B, Tq, Tk)
            ctx = torch.matmul(attn, v_shared)  # (B, Tq, d_head)
            head_outputs.append(ctx)
            head_attentions.append(attn)

        # head 평균 (논문 Eq. 14)
        out = torch.stack(head_outputs, dim=0).mean(dim=0)  # (B, Tq, d_head)
        out = self.out_proj(out)  # (B, Tq, hidden)
        # 해석용: 모든 head 의 attention 평균
        attn_avg = torch.stack(head_attentions, dim=0).mean(dim=0)  # (B, Tq, Tk)
        return out, attn_avg


# ==============================================================================
# 3. Static Covariate Encoder (논문 Section 4.1)
# ==============================================================================
class StaticCovariateEncoder(nn.Module):
    """
    Static input → 4가지 context vector 생성:
      c_s : VSN 가중치 컨텍스트
      c_e : Static enrichment 컨텍스트
      c_c : LSTM cell state 초기값
      c_h : LSTM hidden state 초기값
    """

    def __init__(self, hidden_size, n_static_vars, dropout=0.1):
        super().__init__()
        # static 변수 자체를 처리할 VSN
        self.static_vsn = VariableSelectionNetwork(
            n_vars=n_static_vars,
            input_size_per_var=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.grn_cs = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.grn_ce = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.grn_cc = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.grn_ch = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)

    def forward(self, static_x):
        # static_x : (B, n_static_vars, hidden_size)
        s, _ = self.static_vsn(static_x)
        return self.grn_cs(s), self.grn_ce(s), self.grn_cc(s), self.grn_ch(s)


# ==============================================================================
# 4. TFT 본체
# ==============================================================================
class TFTAnomalyModel(nn.Module):
    """
    Temporal Fusion Transformer 기반의 회전기계 이상탐지/예측 모델.

    [입력 / 출력 인터페이스 - Lightning 어댑터]
    -------------------------------------------------------------------
        forward(src, tgt) -> (x1, x2)
          src : (window_size, batch, n_feats)
          tgt : (1, batch, n_feats)              # = src[-1]을 어댑터가 만들어줌
          x1  : (1, batch, n_feats)              # quantile-median 1-step 예측
          x2  : (1, batch, n_feats)              # quantile-band 평균 1-step 예측

    [내부 동작]
    -------------------------------------------------------------------
      1) src 를 permute 하여 (B, T, n_feats) 로 변환 후 VSN/LSTM 입력
      2) Encoder(과거 window) + Decoder(미래 horizon)에 LSTM 적용
      3) Static Enrichment + Interpretable Self-Attention + GRN
      4) Quantile Heads → (B, horizon, n_feats, n_quantiles)
      5) horizon=1 의 median, low, high 를 추출하여 (1, B, n_feats) 형태로 반환

    [해석성 정보]
      forward 직후 self.last_interpretation 에 다음 정보 캐싱:
        - encoder_var_weights : (B, T_enc, n_feats)
        - decoder_var_weights : (B, T_dec, n_feats)
        - attention           : (B, T_dec, T_enc + T_dec)
        - quantile_outputs    : (B, horizon, n_feats, n_quantiles)
    """

    def __init__(self, n_feats, model_config):
        """어댑터에서 검증한 TFT 구조 설정을 모델 인스턴스별로 적용한다."""
        super().__init__()
        options = model_config
        self.name = "TFT"
        self.n_feats = n_feats
        self.n_window = options["window_size"]
        self.hidden_size = options["hidden_size"]
        self.dropout = options["dropout"]
        self.num_heads = options["num_heads"]
        self.n_lstm_layers = options["lstm_layers"]
        self.horizon = options["forecast_horizon"]
        self.quantiles = list(options["quantiles"])
        self.n_quantiles = len(self.quantiles)
        self.use_vsn = options["use_vsn"]
        self.use_static = options["use_static"]
        self.n_static_vars = options["n_static_vars"]
        self.qband_low = options["qband_low"]
        self.qband_high = options["qband_high"]

        # ── 입력 임베딩(per-variable) ──
        # 각 연속형 변수를 hidden_size 차원으로 매핑
        self.encoder_var_embeds = nn.ModuleList([nn.Linear(1, self.hidden_size) for _ in range(n_feats)])
        self.decoder_var_embeds = nn.ModuleList([nn.Linear(1, self.hidden_size) for _ in range(n_feats)])

        # ── Variable Selection Networks ──
        if self.use_vsn:
            self.encoder_vsn = VariableSelectionNetwork(
                n_vars=n_feats,
                input_size_per_var=self.hidden_size,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                context_size=self.hidden_size if self.use_static else None,
            )
            self.decoder_vsn = VariableSelectionNetwork(
                n_vars=n_feats,
                input_size_per_var=self.hidden_size,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                context_size=self.hidden_size if self.use_static else None,
            )

        # ── Static Encoder ──
        # 회전기계는 보통 단일 기계라 static 변수가 없음 → 0으로 채워 컨텍스트 생성
        if self.use_static:
            self.static_encoder = StaticCovariateEncoder(
                hidden_size=self.hidden_size,
                n_static_vars=self.n_static_vars,
                dropout=self.dropout,
            )
        else:
            # static context를 0 벡터로 만들기 위한 학습 가능한 placeholder 없이
            # forward에서 zeros 사용
            self.static_encoder = None

        # ── LSTM Encoder/Decoder (Locality Enhancement) ──
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.n_lstm_layers > 1 else 0.0,
        )
        self.lstm_decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.n_lstm_layers > 1 else 0.0,
        )

        # LSTM 출력에 적용되는 GLU + skip + LayerNorm (논문 Fig. 2 "Add & Norm")
        self.lstm_glu = TimeDistributed(GLU(self.hidden_size, self.hidden_size))
        self.lstm_layer_norm = nn.LayerNorm(self.hidden_size)

        # ── Static Enrichment GRN ──
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size if self.use_static else None,
        )

        # ── Interpretable Multi-Head Self-Attention ──
        self.attention = InterpretableMultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        self.attn_glu = TimeDistributed(GLU(self.hidden_size, self.hidden_size))
        self.attn_layer_norm = nn.LayerNorm(self.hidden_size)

        # ── Position-wise Feed-Forward (GRN) ──
        self.position_ff = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )
        self.ff_glu = TimeDistributed(GLU(self.hidden_size, self.hidden_size))
        self.ff_layer_norm = nn.LayerNorm(self.hidden_size)

        # ── Quantile Output Heads (변수별, quantile별) ──
        # 출력 : (B, T_dec, n_feats * n_quantiles)
        self.quantile_proj = nn.Linear(self.hidden_size, n_feats * self.n_quantiles)

        # 해석성 캐시
        self.last_interpretation = None

    # ─────────────────────────────────────────
    @staticmethod
    def _fix_nheads(num_heads, hidden_size):
        """hidden_size를 나누어 떨어지지 않으면 가까운 약수로 보정."""
        if hidden_size % num_heads == 0:
            return num_heads
        for h in range(num_heads, 0, -1):
            if hidden_size % h == 0:
                return h
        return 1

    # ─────────────────────────────────────────
    def _embed_variables(self, x, embed_modules):
        """
        x : (B, T, n_feats)
        return : (B, T, n_feats, hidden_size)  --> per-variable embedding
        """
        B, T, F = x.shape
        per_var = []
        for i in range(F):
            xi = x[..., i : i + 1]  # (B, T, 1)
            ei = embed_modules[i](xi)  # (B, T, hidden)
            per_var.append(ei)
        return torch.stack(per_var, dim=-2)  # (B, T, n_feats, hidden)

    # ─────────────────────────────────────────
    def _build_inputs(self, src, tgt):
        """
        src : (window, B, n_feats)  (trainer 가 permute(1,0,2) 후 주는 형태)
        tgt : (1,      B, n_feats)
        return enc_x, dec_x : (B, T_enc/T_dec, n_feats)
        """
        # encoder 입력: 과거 window (그대로 사용)
        enc_x = src.permute(1, 0, 2).contiguous()  # (B, T_enc, n_feats)

        # decoder 입력: 미래 horizon. 학습 시점엔 미래값이 없으므로
        # encoder의 마지막 관측치를 horizon만큼 복제(known future가 없는 경우의 표준 처리).
        # tgt는 (1, B, n_feats)로 들어오므로, horizon=1이면 그대로,
        # horizon>1이면 마지막 값을 반복 복제하여 사용.
        last_obs = enc_x[:, -1:, :]  # (B, 1, n_feats)
        if self.horizon == 1:
            dec_x = last_obs  # (B, 1, n_feats)
        else:
            dec_x = last_obs.repeat(1, self.horizon, 1)  # (B, horizon, n_feats)

        return enc_x, dec_x

    # ─────────────────────────────────────────
    def forward(self, src, tgt):
        """
        Lightning 어댑터 forward.
            src : (window, B, n_feats)
            tgt : (1     , B, n_feats)
        return:
            x1, x2 : (1, B, n_feats)   <-- 분위수 복원 출력
        """
        device = src.device
        B = src.size(1)

        enc_x, dec_x = self._build_inputs(src, tgt)  # (B, T_enc/T_dec, F)
        T_enc = enc_x.size(1)
        T_dec = dec_x.size(1)

        # ── 1) 변수별 임베딩 ──
        enc_emb = self._embed_variables(enc_x, self.encoder_var_embeds)
        dec_emb = self._embed_variables(dec_x, self.decoder_var_embeds)
        # shape : (B, T_*, n_feats, hidden)

        # ── 2) Static context (회전기계는 보통 zero vector) ──
        if self.use_static and self.static_encoder is not None:
            # 단일 기계 모니터링 시 zero 텐서로 placeholder
            static_in = torch.zeros(B, self.n_static_vars, self.hidden_size, device=device, dtype=enc_emb.dtype)
            c_s, c_e, c_c, c_h = self.static_encoder(static_in)
            # LSTM 초기 상태를 [층, 배치, 은닉 차원] 순서로 맞춘다.
            c_c = c_c.unsqueeze(0).repeat(self.n_lstm_layers, 1, 1)
            c_h = c_h.unsqueeze(0).repeat(self.n_lstm_layers, 1, 1)
        else:
            c_s = None
            c_e = None
            c_c = torch.zeros(self.n_lstm_layers, B, self.hidden_size, device=device, dtype=enc_emb.dtype)
            c_h = torch.zeros(self.n_lstm_layers, B, self.hidden_size, device=device, dtype=enc_emb.dtype)

        # ── 3) Variable Selection ──
        if self.use_vsn:
            enc_sel, enc_w = self.encoder_vsn(enc_emb, c=c_s)  # (B, T_enc, hidden)
            dec_sel, dec_w = self.decoder_vsn(dec_emb, c=c_s)  # (B, T_dec, hidden)
        else:
            # VSN 미사용 시 단순 평균
            enc_sel = enc_emb.mean(dim=-2)
            dec_sel = dec_emb.mean(dim=-2)
            enc_w = torch.full((B, T_enc, self.n_feats), 1.0 / self.n_feats, device=device, dtype=enc_emb.dtype)
            dec_w = torch.full((B, T_dec, self.n_feats), 1.0 / self.n_feats, device=device, dtype=enc_emb.dtype)

        # ── 4) LSTM Encoder/Decoder (Locality Enhancement) ──
        if c_h is not None:
            enc_lstm_out, (h_n, c_n) = self.lstm_encoder(enc_sel, (c_h, c_c))
        else:
            enc_lstm_out, (h_n, c_n) = self.lstm_encoder(enc_sel)
        dec_lstm_out, _ = self.lstm_decoder(dec_sel, (h_n, c_n))

        # GLU + skip(VSN output) + LayerNorm  (Fig. 2의 Add&Norm)
        enc_temp = self.lstm_layer_norm(enc_sel + self.lstm_glu(enc_lstm_out))
        dec_temp = self.lstm_layer_norm(dec_sel + self.lstm_glu(dec_lstm_out))
        # encoder 와 decoder를 시간축으로 연결 (self-attention용)
        temporal = torch.cat([enc_temp, dec_temp], dim=1)  # (B, T_enc+T_dec, hidden)

        # ── 5) Static Enrichment (각 시간 step에 GRN + static context) ──
        if c_e is not None:
            enriched = self.static_enrichment(temporal, c=c_e)
        else:
            enriched = self.static_enrichment(temporal)

        # ── 6) Interpretable Multi-Head Self-Attention ──
        # decoder 부분만 query로 사용 (causal: decoder가 encoder 미래를 보지 못하게)
        T_total = enriched.size(1)
        # mask : (T_dec, T_total) — decoder t는 [0..T_enc + t] 까지 attend
        mask = torch.zeros(T_dec, T_total, device=device, dtype=torch.bool)
        for t in range(T_dec):
            mask[t, : T_enc + t + 1] = True
        mask = mask.unsqueeze(0)  # (1, T_dec, T_total) → broadcast B

        q = enriched[:, T_enc:, :]  # decoder 부분만 query
        k = enriched
        v = enriched
        attn_out, attn_weights = self.attention(q, k, v, mask=mask)

        # GLU + skip + LayerNorm
        attn_temp = self.attn_layer_norm(q + self.attn_glu(attn_out))

        # ── 7) Position-wise Feed-Forward (GRN) ──
        ff_out = self.position_ff(attn_temp)
        ff_temp = self.ff_layer_norm(
            dec_temp + self.ff_glu(ff_out)  # decoder LSTM 출력으로 skip
        )

        # ── 8) Quantile Output ──
        # (B, T_dec, n_feats * n_quantiles) → reshape
        q_out = self.quantile_proj(ff_temp)
        q_out = q_out.view(B, T_dec, self.n_feats, self.n_quantiles)

        # ── 해석성 정보 캐싱 ──
        # detach된 버전은 시각화/해석용
        self.last_interpretation = {
            "encoder_var_weights": enc_w.detach(),  # (B, T_enc, n_feats)
            "decoder_var_weights": dec_w.detach(),  # (B, T_dec, n_feats)
            "attention": attn_weights.detach(),  # (B, T_dec, T_total)
            "quantile_outputs": q_out.detach(),  # (B, T_dec, n_feats, n_quant)
            "quantiles": self.quantiles,
            "T_enc": T_enc,
            "T_dec": T_dec,
        }
        # [NEW] grad-aware quantile output을 어댑터가 quantile loss 계산에 사용할 수
        # 있도록 별도 속성으로 노출 (학습 그래프와 연결됨).
        self.last_quantile_outputs_grad = q_out

        # ── Lightning 어댑터 출력 (1, B, n_feats) ──
        # x1 : median quantile (point estimate)
        # x2 : quantile-band 평균 (low + high) / 2
        med_idx = self._closest_q_index(0.5)
        low_idx = self._closest_q_index(self.qband_low)
        high_idx = self._closest_q_index(self.qband_high)

        # horizon=1 의 첫 step만 사용 (anomaly detection)
        median_pred = q_out[:, 0, :, med_idx]  # (B, n_feats)
        band_pred = 0.5 * (q_out[:, 0, :, low_idx] + q_out[:, 0, :, high_idx])  # (B, n_feats)

        x1 = median_pred.unsqueeze(0)  # (1, B, n_feats)
        x2 = band_pred.unsqueeze(0)  # (1, B, n_feats)
        return x1, x2

    # ─────────────────────────────────────────
    def _closest_q_index(self, q):
        """주어진 quantile 값과 가장 가까운 인덱스 반환."""
        diffs = [abs(qi - q) for qi in self.quantiles]
        return int(np.argmin(diffs))


# Lightning 어댑터: 모델별 학습 손실과 이상 점수
class TFTLightningModule(AnomalyLightningModule):
    """TFT 복원 모델을 분위수 손실 또는 기존 혼합 손실로 학습한다."""

    def __init__(self, n_feats, model_config=None, training_config=None):
        """models.tft 설정으로 네트워크와 손실 계산 방식을 초기화한다."""
        options = load_model_config("tft", model_config)
        if options["loss"] not in {"mse_quantile", "quantile"}:
            raise ValueError("TFT loss must be mse_quantile or quantile")
        if options["anomaly_score"] not in {"quantile_loss", "band_mse", "median_mse", "band_violation"}:
            raise ValueError("Unsupported TFT anomaly_score")
        if options["forecast_horizon"] != 1:
            raise ValueError("The reconstruction DataLoader supports forecast_horizon=1 only")
        super().__init__(n_feats, options, training_config)
        self.network = TFTAnomalyModel(n_feats, model_config=options)
        self.to(dtype=self.input_dtype)

    def batch_outputs(self, batch):
        """미분 가능한 분위수 출력으로 손실과 특성별 점수를 계산한다."""
        target = batch["target"].to(dtype=self.dtype)
        median, band = self(batch["context"], target)
        quantile_predictions = self.network.last_quantile_outputs_grad[:, 0]
        quantiles = quantile_predictions.new_tensor(self.network.quantiles)
        error = target.unsqueeze(-1) - quantile_predictions
        quantile_score = torch.maximum(quantiles * error, (quantiles - 1) * error).mean(-1)
        median_error = (median - target).square()
        band_error = (band - target).square()
        loss = quantile_score.mean()
        validation_loss = loss
        if self.model_config["loss"] == "mse_quantile":
            weight = 1.0 / (self.current_epoch + 1)
            loss = (weight * median_error + (1 - weight) * band_error).mean() + self.model_config[
                "quantile_loss_weight"
            ] * loss
            validation_loss = band_error.mean()
        low = quantile_predictions[..., self.network._closest_q_index(self.model_config["qband_low"])]
        high = quantile_predictions[..., self.network._closest_q_index(self.model_config["qband_high"])]
        # 예측 분위 구간 밖이면 거리만큼 점수, 안이면 0. 분위수 역전도 폭에 반영한다.
        half_width = (high - low).abs() * 0.5 * self.model_config["qband_tolerance"]
        violation = ((target - (low + high) * 0.5).abs() - half_width).clamp_min(0)
        scores = {
            "quantile_loss": quantile_score,
            "band_mse": band_error,
            "median_mse": median_error,
            "band_violation": violation,
        }
        prediction = median if self.model_config["anomaly_score"] != "band_mse" else band
        # 손실이 그래프를 소유하므로 모듈 속성에는 계산 그래프를 남기지 않는다.
        self.network.last_quantile_outputs_grad = None
        return {
            "loss": loss,
            "validation_loss": validation_loss,
            "prediction": prediction,
            "anomaly_score": scores[self.model_config["anomaly_score"]],
            **{f"score_{name}": value for name, value in scores.items()},
            "quantile_predictions": quantile_predictions,
        }
