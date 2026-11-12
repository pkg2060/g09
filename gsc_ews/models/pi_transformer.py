"""Pi-Transformer의 네트워크 구조·손실·이상 점수와 Lightning 학습을 정의한다.

데이터 전처리와 DataLoader 구성은 공통 src 패키지에서 담당한다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.udf.configuration import load_model_config

from .base import AnomalyLightningModule


# ==============================================================================
# Pi-Transformer Network Components
# ==============================================================================
################################## positional encoding ##################################
class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding을 입력 Embedding에 적용한다.

    Args:
        d_model (int): Embedding Dimension.
        max_len (int): Maximum Sequence Length.
        dropout (float): Dropout Rate.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # 시점별 위치 인덱스 생성
        position = torch.arange(max_len).unsqueeze(1)

        # embedding 차원별 주파수 조절 계수
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # positional encoding 행렬 초기화
        pe = torch.zeros(max_len, d_model)

        # 짝수 차원에는 sin, 홀수 차원에는 cos 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[-1]])

        # batch 차원 추가 (buffer 적용)
        self.register_buffer("pe", pe.unsqueeze(0))

        # dropout 적용
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 Embedding에 위치 정보를 더한다.

        Args:
            x (torch.Tensor): Shape (B, T, d_model).

        Returns:
            torch.Tensor: Shape (B, T, d_model).
        """
        # forward
        x = x + self.pe[:, : x.size(1)]

        return self.dropout(x)


################################## series attention ##################################
class SeriesAttention(nn.Module):
    """Multi-Head Self-Attention으로 시점 간 관계와 Context를 계산한다.

    Args:
        d_model (int): Embedding Dimension.
        num_heads (int): Attention Head 수.
        dropout (float): Dropout 비율.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        # q, k, v projection layer 정의
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)

        # multi-head 결합 후 output projection
        self.output_projection = nn.Linear(d_model, d_model)

        # attention context 계산 시 dropout 적용
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Series Attention Weight와 Context Representation을 생성한다.

        Args:
            x (torch.Tensor): Shape (B, T, d_model).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Context: Shape (B, T, d_model).
                Series Attention Weight: Shape (B, Heads, T, T).
        """
        # shape 정의
        batch_size, seq_len, _ = x.shape

        # q, k, v linear projection
        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        # head별 차원 분리: [batch, time step, 512] → [batch, 8, time step, 64]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention score 계산
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        # causal masking: 미래 시점의 Attention 제한
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()

        # apply masking
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # softmax로 series attention weight 생성: [batch, 8, t, t]
        series_attention = torch.softmax(scores, dim=-1)

        # attention weight에 value를 적용하여 context 생성
        context = torch.matmul(self.dropout(series_attention), v)

        # multi-head 결합: [batch, 8, t, 64] → [batch, t, 512]
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, self.d_model)

        # output projection
        context = self.output_projection(context)

        return context, series_attention


################################## prior attention ##################################
class PriorAttention(nn.Module):
    """시간적 자기유사성과 Phase 정보를 이용하여 Prior Attention을 생성한다.

    Args:
        d_model (int): Embedding Dimension.
        num_heads (int): Attention Head 수.
        gamma (float): Hurst-like Time Warp 민감도.
        time_sigma (float): Temporal Gaussian Kernel 폭.
        tau_min (float): Phase-Stiffness 최소값.
        tau_scale (float): Phase-Stiffness 크기 조절 계수.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        gamma: float = 1.0,
        time_sigma: float = 10.0,
        tau_min: float = 0.01,
        tau_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.time_sigma = time_sigma
        self.tau_min = tau_min
        self.tau_scale = tau_scale

        # Head별 Prior Parameter Prediction을 위한 Projection
        self.hurst_projection = nn.Linear(d_model, num_heads)
        self.phase_projection = nn.Linear(d_model, num_heads)
        self.stiffness_projection = nn.Linear(d_model, num_heads)

    @staticmethod
    def _analytic_signal(x: torch.Tensor) -> torch.Tensor:
        """FFT 기반 Hilbert Transform으로 Analytic Signal을 계산한다.

        Args:
            x (torch.Tensor): Shape (B, Heads, T).

        Returns:
            torch.Tensor: Complex Tensor, Shape (B, Heads, T).
        """
        seq_len = x.size(-1)

        # hilbert transform 위한 주파수 필터 생성
        h = torch.zeros(seq_len, device=x.device, dtype=x.dtype)
        h[0] = 1.0

        if seq_len % 2 == 0:
            h[1 : seq_len // 2] = 2.0
            h[seq_len // 2] = 1.0
        else:
            h[1 : (seq_len + 1) // 2] = 2.0

        # fft → hilbert 주파수 필터 적용 → ifft
        analytic = torch.fft.ifft(
            torch.fft.fft(x, dim=-1) * h,
            dim=-1,
        )

        return analytic

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prior Attention Weight와 Regularization에 필요한 파라미터를 생성한다.

        Args:
            x (torch.Tensor): Shape (B, T, d_model).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Prior Attention: Shape (B, Heads, T, T).
                Hurst-like Parameter: Shape (B, Heads, T).
                Phase-Stiffness: Shape (B, Heads, T).
                Prior Log Score: Shape (B, Heads, T, T).
        """
        batch_size, seq_len, _ = x.shape

        ################ temporal self-similarity ################
        # Hurst-like Parameter 생성: [B, T, 512] → [B, 8, T]
        hurst = torch.sigmoid(self.hurst_projection(x))
        hurst = hurst.transpose(1, 2)

        # Hurst-like Parameter 기반 시간 좌표 생성
        warped_step = torch.exp(self.gamma * hurst)
        psi = torch.cumsum(warped_step, dim=-1)

        # 시점 간 Warped-Time 거리 계산
        time_difference = psi.unsqueeze(-1) - psi.unsqueeze(-2)

        # Temporal Gaussian Kernel의 Log Weight 계산
        time_log_weight = -time_difference.square() / (2.0 * self.time_sigma**2)

        ################ phase synchrony ################
        # Head별 Scalar Projection 및 Hilbert Transform
        signal = self.phase_projection(x).transpose(1, 2)
        analytic = self._analytic_signal(signal)  # hilbert 적용

        # Analytic Signal에서 Phase 추출: [B, 8, T]
        phase = torch.atan2(analytic.imag, analytic.real + 1e-8)

        # Phase-Stiffness Parameter 생성 및 양수 변환
        stiffness_raw = self.stiffness_projection(x).transpose(1, 2)
        tau = self.tau_scale * F.softplus(stiffness_raw) + self.tau_min

        # 시점 간 Phase 거리 계산
        phase_difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        phase_distance = torch.sin(phase_difference / 2.0).square()

        # Phase Consistency Kernel의 Log Weight 계산
        phase_log_weight = -phase_distance / (2.0 * tau.unsqueeze(-1).square())

        ################ prior attention matrix ################
        # 8. Temporal / Phase Weight 결합
        prior_log_score = time_log_weight + phase_log_weight

        # 9. Causal Masking: 미래 시점에 대한 Attention 제한
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()

        masked_score = prior_log_score.masked_fill(~mask, float("-inf"))

        # 10. Row Normalization으로 Prior Attention Weight 생성
        prior_attention = torch.softmax(masked_score, dim=-1)

        return prior_attention, hurst, tau, prior_log_score


################################## dual-attention encoder layer (attention + ffn) ##################################
class DualAttentionEncoderLayer(nn.Module):
    """Series / Prior Attention을 계산하고 Feature Representation을 업데이트한다.

    Args:
        d_model (int): Embedding Dimension.
        num_heads (int): Attention Head 수.
        dim_feedforward (int): FFN Hidden Dimension.
        dropout (float): Dropout 비율.
        gamma (float): Hurst-like Time Warp 민감도.
        time_sigma (float): Temporal Gaussian Kernel 폭.
        tau_min (float): Phase-Stiffness 최소값.
        tau_scale (float): Phase-Stiffness 크기 조절 계수.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        gamma: float = 1.0,
        time_sigma: float = 10.0,
        tau_min: float = 0.01,
        tau_scale: float = 1.0,
    ) -> None:
        super().__init__()

        # Series Attention 정의
        self.series_attention = SeriesAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Prior Attention 정의
        self.prior_attention = PriorAttention(
            d_model=d_model,
            num_heads=num_heads,
            gamma=gamma,
            time_sigma=time_sigma,
            tau_min=tau_min,
            tau_scale=tau_scale,
        )

        # Attention / FFN 입력 정규화
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network 정의
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Residual Connection에 적용할 Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Attention을 계산하고 다음 Encoder Block에 전달할 Feature를 생성한다.

        Args:
            x (torch.Tensor): Shape (B, T, d_model).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                Feature Representation: Shape (B, T, d_model).
                Series / Prior Attention: Shape (B, Heads, T, T).
                Hurst / Phase-Stiffness: Shape (B, Heads, T).
                Prior Log Score: Shape (B, Heads, T, T).
        """

        # 1. Attention 입력 정규화
        normalized = self.norm1(x)

        # 2. Series Attention 및 Context Representation 계산
        context, series = self.series_attention(normalized)

        # 3. Prior Attention 및 Prior Parameter 계산
        prior, hurst, tau, prior_log_score = self.prior_attention(normalized)

        # 4. Series Context에 Residual Connection 적용
        x = x + self.dropout(context)

        # 5. Feed-Forward Network 및 Residual Connection 적용
        x = x + self.dropout(self.ffn(self.norm2(x)))

        # 6. Feature Representation 및 Attention 정보 반환
        return x, {
            "series": series,
            "prior": prior,
            "hurst": hurst,
            "stiffness": tau,
            "prior_log_score": prior_log_score,
        }


# ==============================================================================
# Pi-Transformer Main Model
# ==============================================================================
class PiTransformer(nn.Module):
    """Pi-Transformer의 전체 Encoder 구조와 Reconstruction을 정의한다.

    Args:
        n_feats (int): 입력 Feature 수.
        model_config (dict): 모델 Hyperparameter 설정.
    """

    def __init__(
        self,
        n_feats: int,
        model_config: dict,
    ) -> None:
        super().__init__()

        ############# model config #############
        self.n_feats = n_feats
        self.n_window = model_config["window_size"]

        self.d_model = model_config["d_model"]
        self.num_heads = model_config["num_heads"]
        self.head_dim = model_config["head_dim"]
        self.encoder_layers_count = model_config["encoder_layers"]

        if self.d_model != self.num_heads * self.head_dim:
            raise ValueError("d_model must equal num_heads * head_dim")

        ############# 2. Input Embedding & Positional Encoding #############
        # 입력 Feature를 Embedding Dimension으로 변환
        self.input_projection = nn.Linear(n_feats, self.d_model)

        # 시점별 Positional Encoding 적용
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.n_window,
            dropout=model_config["dropout"],
        )

        ############# 3. Encoder Layer Stacking #############
        # 설정된 Encoder Layer 수만큼 독립적인 Block 생성
        self.encoder_layers = nn.ModuleList(
            [
                DualAttentionEncoderLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    dim_feedforward=model_config["dim_feedforward"],
                    dropout=model_config["dropout"],
                    gamma=model_config["gamma"],
                    time_sigma=model_config["time_sigma"],
                    tau_min=model_config["tau_min"],
                    tau_scale=model_config["tau_scale"],
                )
                for _ in range(self.encoder_layers_count)
            ]
        )

        ############# 4. Reconstruction Head #############
        # 최종 Feature 정규화 및 원본 Feature 차원으로 복원
        self.final_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Linear(self.d_model, n_feats)

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encoder Block을 순차적으로 통과시켜 Reconstruction과 Attention을 반환한다.

        Args:
            x (torch.Tensor): Shape (B, T, n_feats).

        Returns:
            dict[str, torch.Tensor]:
                reconstruction: Shape (B, T, n_feats).
                series: Shape (Layers, B, Heads, T, T).
                prior: Shape (Layers, B, Heads, T, T).
                hurst: Shape (Layers, B, Heads, T).
                stiffness: Shape (Layers, B, Heads, T).
                prior_log_score: Shape (Layers, B, Heads, T, T).
        """
        ############# 1. Input Validation #############
        # validation
        if x.ndim != 3 or x.shape[1:] != (self.n_window, self.n_feats):
            raise ValueError("x must have shape [batch, window_size, n_feats]")

        ############# 2. Input Embedding & Positional Encoding #############
        # Linear Embedding: [B, T, 51] → [B, T, 512]
        x = self.input_projection(x)

        # Positional Encoding: [B, T, 512]
        x = self.positional_encoding(x)

        ############# 3. Encoder Layer Stacking #############
        # 각 Encoder Block의 Attention 및 Prior Parameter 보관
        attention_maps = []

        for layer in self.encoder_layers:
            # Feature는 다음 Block으로 전달하고 Attention 정보는 별도 수집
            x, attention = layer(x)
            attention_maps.append(attention)

        # Layer별 Attention 및 Prior Parameter를 하나의 Tensor로 결합
        attention_outputs = {
            key: torch.stack(
                [attention[key] for attention in attention_maps],
                dim=0,
            )
            for key in (
                "series",
                "prior",
                "hurst",
                "stiffness",
                "prior_log_score",
            )
        }

        ############# 4. Reconstruction Output #############
        # 마지막 Encoder Feature를 원본 Feature 차원으로 복원
        x = self.final_norm(x)
        reconstruction = self.output_projection(x)

        return {
            "reconstruction": reconstruction,
            **attention_outputs,
        }


# ==============================================================================
# Pi-Transformer Lightning Module
# ==============================================================================
class PiTransformerLightningModule(AnomalyLightningModule):
    """Pi-Transformer의 네트워크·손실함수·이상 점수를 통합하는 Lightning Wrapper.

    Main Model의 Forward 결과를 기반으로 손실함수와 교대 최적화를 수행하고,
    학습·검증·추론 출력을 기존 GSC 파이프라인에 연결한다.
    """

    ################ init ################
    # load_model_config("pi_transformer") + 설정 검증 + network 생성 + manual optimization 설정
    def __init__(
        self,
        n_features: int,
        model_config: dict | None = None,
        training_config: dict | None = None,
    ) -> None:
        """모델 설정을 로드하고 Pi-Transformer 네트워크와 수동 최적화를 초기화한다."""
        # 1. yaml 기반 모델 설정 로드 및 검증
        options = load_model_config("pi_transformer", model_config)

        # 2. Base Lightning Module 초기화
        super().__init__(n_features, options, training_config)

        # 3. Feature 수 저장
        self.n_features = n_features

        # 4. pi-transformer main model 생성
        self.network = PiTransformer(
            n_feats=n_features,
            model_config=options,
        )

        # 5. manual optimization 활성화
        self.automatic_optimization = False

        # 6. 수동 graident clipping 설정
        self.manual_gradient_clip_val = float(options["gradient_clip_val"])

        # 7. 모델 연산 자료형 설정
        self.to(dtype=self.input_dtype)

    ################ forward ################
    # 입력 Context를 Pi-Transformer Main Model에 전달하고 Reconstruction 및 Attention 결과 반환
    def forward(
        self,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """입력 시계열을 검증하고 Reconstruction 및 Attention 정보를 반환한다.

        Args:
            context (torch.Tensor): Shape (B, T, n_feats).

        Returns:
            dict[str, torch.Tensor]:
                reconstruction: Shape (B, T, n_feats).
                series: Shape (Layers, B, Heads, T, T).
                prior: Shape (Layers, B, Heads, T, T).
                hurst: Shape (Layers, B, Heads, T).
                stiffness: Shape (Layers, B, Heads, T).
                prior_log_score: Shape (Layers, B, Heads, T, T).
        """
        # 1. 입력 context shape 검증
        if context.ndim != 3 or context.shape[1:] != (self.n_window, self.n_features):
            raise ValueError("PiTransformerLightningModule - context must have shape [batch, window_size, n_features]")

        # 2. 모델 연산 자료형 변환
        context = context.to(dtype=self.input_dtype)

        # 3. pi-transformer main model 실행 및 결과 반환
        return self.network(context)

    ################ loss functions ################
    # reconstruction loss: 원본과 복원 Sequence의 MSE 계산
    def reconstruction_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """원본 시계열과 복원 시계열의 평균제곱오차를 계산한다.

        Args:
            original (torch.Tensor): 원본 Sequence, Shape (B, T, n_features).
            reconstructed (torch.Tensor): 복원 Sequence, Shape (B, T, n_features).

        Returns:
            torch.Tensor: Reconstruction Loss, Scalar.
        """
        # 1. 원본 및 복원 Sequence Shape 검증
        if original.ndim != 3 or original.shape != reconstructed.shape:
            raise ValueError(
                "reconstruction_loss - original and reconstructed must have the same [B, T, n_features] shape"
            )

        # 2. 전체 batch, time, feature에 대한 mse 계산
        return F.mse_loss(reconstructed, original)

    # attention divergence loss: Series / Prior의 Symmetric KL Divergence 계산
    def attention_divergence_loss(
        self,
        series: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Series / Prior Attention 간 Symmetric KL Divergence를 계산한다.

        Args:
            series (torch.Tensor): Series Attention, Shape (Layers, B, Heads, T, T).
            prior (torch.Tensor): Prior Attention, Shape (Layers, B, Heads, T, T).

        Returns:
            torch.Tensor: Attention Divergence Loss, Scalar.
        """
        # 1. series / prior attention shape 검증
        if series.ndim != 5 or series.shape != prior.shape:
            raise ValueError(
                "attention_divergence_loss - series and prior must have the same [Layers, B, Heads, T, T] shape"
            )

        # 2. casual mask 생성: 현재 및 과거 시점만 사용
        seq_len = series.size(-1)

        mask = torch.ones(seq_len, seq_len, device=series.device, dtype=torch.bool).tril()

        # 3. 미래 시점의 attention weight 제거
        series = series.masked_fill(~mask, 0.0)
        prior = prior.masked_fill(~mask, 0.0)

        # 4. log 계산 시, 0으로 인한 nan 방지
        eps = torch.finfo(series.dtype).eps

        log_series = torch.log(series.clamp_min(eps))
        log_prior = torch.log(prior.clamp_min(eps))

        # 5. 양방향 kl divergence 계산
        kl_sp = (series * (log_series - log_prior)).sum(dim=-1)
        kl_ps = (prior * (log_prior - log_series)).sum(dim=-1)

        # 6. symmetric kl divergence 계산
        divergence = kl_sp + kl_ps

        # 7. layer, batch, head, time 평균하여 scaler 생성
        return divergence.mean()

    # prior regularization loss: Smoothness + Hurst Distillation + Prior Score
    def prior_regularization_loss(
        self,
        hurst: torch.Tensor,
        stiffness: torch.Tensor,
        prior_log_score: torch.Tensor,
        hurst_reference: torch.Tensor | float,
    ) -> torch.Tensor:
        """Prior Parameter의 시간적 안정성과 기준 통계량에 대한 정규화 손실을 계산한다.

        Args:
            hurst (torch.Tensor): Hurst-like Parameter, Shape (Layers, B, Heads, T).
            stiffness (torch.Tensor): Phase-Stiffness, Shape (Layers, B, Heads, T).
            prior_log_score (torch.Tensor): Prior Log Score, Shape (Layers, B, Heads, T, T).
            hurst_reference (torch.Tensor | float): 정상 데이터에서 추정한 Hurst 기준값.

        Returns:
            torch.Tensor: Prior Regularization Loss, Scalar.
        """
        # 1. prior parameter shape 검증
        if hurst.ndim != 4 or hurst.shape != stiffness.shape:
            raise ValueError(
                "prior_regularization_loss - hurst and stiffness must have the same [Layers, B, Heads, T] shape"
            )

        # 2. smoothness regularization: 인접 시점의 stiffness 변화 제약
        stiffness_diff = stiffness[..., 1:] - stiffness[..., :-1]

        r_smooth = stiffness_diff.square().mean()

        # 3. hurt distillation: 정상 데이터의 기준 통계량과 비교
        reference = torch.as_tensor(  # reference 나중에 연결할 것 정상 train dataset에서 필요
            hurst_reference,
            device=hurst.device,
            dtype=hurst.dtype,
        ).detach()

        r_distill = (hurst - reference).square().mean()

        # 4. prior score regularization: 과도하게 작은 log score 제약
        seq_len = prior_log_score.size(-1)

        mask = torch.ones(seq_len, seq_len, device=prior_log_score.device, dtype=torch.bool).tril()

        limit = self.model_config["prior_log_score_limit"]
        score_penalty = F.relu(-prior_log_score - limit).square()

        # 5. casual mask 적용 후 유효한 prior score만 평균
        r_prior = score_penalty.masked_select(mask).mean()

        # 6. 세 정규화 항목 결합
        return self.model_config["lambda_smooth"] * r_smooth + r_distill + r_prior

    # compute loss: Reconstruction + Divergence + Regularization 결합 (L1 / L2)
    def compute_loss(
        self,
        original: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        pass_name: str,
    ) -> dict[str, torch.Tensor]:
        """Pass별 Stop-Gradient를 적용하고 최종 Loss를 계산한다.

        Args:
            original (torch.Tensor): 원본 Sequence, Shape (B, T, n_features).
            outputs (dict[str, torch.Tensor]): Pi-Transformer Main Model의 Forward 결과.
            pass_name (str): 학습 경로 구분 ("series" 또는 "prior").

        Returns:
            dict[str, torch.Tensor]:
                loss: 최종 Loss, Scalar.
                reconstruction: Reconstruction Loss, Scalar.
                divergence: Attention Divergence Loss, Scalar.
                regularization: Prior Regularization Loss, Scalar.
        """
        # 1. Reconstruction Loss 계산
        r_loss = self.reconstruction_loss(
            original,
            outputs["reconstruction"],
        )

        # 2. pass별 attention divergence gradient 경로 설정
        if pass_name == "series":
            series = outputs["series"]
            prior = outputs["prior"].detach()

        elif pass_name == "prior":
            series = outputs["series"].detach()
            prior = outputs["prior"]

        else:
            raise ValueError("compute_loss - pass_name must be 'series' or 'prior'")

        # 3. Attention Divergence Loss 계산
        d_loss = self.attention_divergence_loss(
            series,
            prior,
        )

        # 4. prior regularization loss 계산
        reg_loss = self.prior_regularization_loss(
            hurst=outputs["hurst"],
            stiffness=outputs["stiffness"],
            prior_log_score=outputs["prior_log_score"],
            hurst_reference=self.nominal_hurst_reference,
        )

        # 5. reconstruction + divergence + regularization 결합
        loss = r_loss + self.model_config["kappa"] * d_loss + self.model_config["lambda_reg"] * reg_loss

        # 6. 최종 loss 및 세부 loss 반환
        return {
            "loss": loss,
            "reconstruction": r_loss,
            "divergence": d_loss,
            "regularization": reg_loss,
        }

    ################ training & optimization ################
    # training_step: Stop-Gradient 기반 Pass 1 / Pass 2 순차 최적화
    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ) -> torch.Tensor:
        """동일 Mini-batch에서 Series와 Prior 경로를 순차적으로 최적화한다.

        Args:
            batch (dict): 학습 Batch.
            batch_idx (int): 현재 Batch Index.

        Returns:
            torch.Tensor: 두 Pass의 평균 Training Loss, Scalar.
        """
        # 1. Optimizer 및 입력 Context 준비
        optimizer = self.optimizers()
        context = batch["context"].to(dtype=self.input_dtype)

        ################ pass 1: series attention ################
        # 2. 기존 gradient 초기화
        optimizer.zero_grad()

        # 3. forward 및 loss 1 계산 (prior attention stop-gradient)
        outputs_1 = self(context)
        loss_1 = self.compute_loss(context, outputs_1, pass_name="series")

        # 4. loss 1 역전파
        self.manual_backward(loss_1["loss"])

        # 5. gradient clipping 및 파라미터 업데이트
        if self.manual_gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.manual_gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

        optimizer.step()

        ################ pass 2: prior attention ################
        # 6. 기존 gradient 초기화
        optimizer.zero_grad()

        # 7. 업데이트된 모델로 forward 재계산 및 loss 2 계산
        outputs_2 = self(context)
        loss_2 = self.compute_loss(context, outputs_2, pass_name="prior")

        # 8. loss 2 역전파
        self.manual_backward(loss_2["loss"])

        # 9. gradient clipping 및 파라미터 업데이트
        if self.manual_gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.manual_gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

        optimizer.step()

        ################ training metrics ################
        # 10. 두 pass 평균 후 training loss 생성
        train_loss = (loss_1["loss"].detach() + loss_2["loss"].detach()) / 2.0

        # 11. learning rate 기록
        self.log(
            "learning_rate",
            optimizer.optimizer.param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            batch_size=len(context),
        )

        # 12. training loss 기록
        self.log(
            "train_loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(context),
            sync_dist=True,
        )

        return train_loss

    # on_train_epoch_end: Learning Rate Scheduler 업데이트
    def on_train_epoch_end(self) -> None:
        """Epoch 종료 시 Learning Rate Scheduler를 업데이트한다."""

        # 1. Base Module에서 생성한 Scheduler 가져오기
        scheduler = self.lr_schedulers()

        # 2. Scheduler가 설정된 경우 Learning Rate 업데이트
        if scheduler is not None:
            scheduler.step()

    ################ anomaly scoring ################
    # mismatch stream: Encoder Layer별 Attention Divergence 집계
    def mismatch_stream(
        self,
        series: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Series / Prior Attention의 시점별 Mismatch Score를 계산한다.

        Args:
            series (torch.Tensor): Series Attention, Shape (Layers, B, Heads, T, T).
            prior (torch.Tensor): Prior Attention, Shape (Layers, B, Heads, T, T).

        Returns:
            torch.Tensor: Mismatch Score, Shape (B, T).
        """
        # 1. Series / Prior Attention Shape 검증
        if series.ndim != 5 or series.shape != prior.shape:
            raise ValueError("mismatch_stream - series and prior must have the same [Layers, B, Heads, T, T] shape")

        # 2. Causal Mask 생성
        seq_len = series.size(-1)
        mask = torch.ones(seq_len, seq_len, device=series.device, dtype=torch.bool).tril()

        # 3. 미래 시점 Attention 제거
        series = series.masked_fill(~mask, 0.0)
        prior = prior.masked_fill(~mask, 0.0)

        # 4. 수치 안정성을 위한 Log Probability 계산
        eps = torch.finfo(series.dtype).eps
        log_series = series.clamp_min(eps).log()
        log_prior = prior.clamp_min(eps).log()

        # 5. Symmetric KL Divergence 계산
        divergence = (series * (log_series - log_prior) + prior * (log_prior - log_series)).sum(dim=-1)

        # 6. Layer 및 Head 평균
        mismatch = divergence.mean(dim=(0, 2))

        # 7. Inverse Temperature 적용
        mismatch = self.model_config["inverse_temperature"] * mismatch

        return mismatch

    # energy stream: Reconstruction Error + Prior-Alignment Weight 계산
    def energy_stream(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        mismatch: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruction Error와 Prior-Alignment Weight를 결합하여 Energy Score를 계산한다.

        Args:
            original (torch.Tensor): 원본 Sequence, Shape (B, T, n_features).
            reconstructed (torch.Tensor): 복원 Sequence, Shape (B, T, n_features).
            mismatch (torch.Tensor): Mismatch Score, Shape (B, T).

        Returns:
            torch.Tensor: Energy Score, Shape (B, T).
        """
        # 1. 입력 Sequence Shape 검증
        if original.ndim != 3 or original.shape != reconstructed.shape:
            raise ValueError("energy_stream - original and reconstructed must have the same [B, T, n_features] shape")

        # 2. Feature별 Reconstruction Error 계산
        reconstruction_error = (original - reconstructed).square()

        # 3. Feature 평균을 통한 시점별 Reconstruction Error 계산
        row_error = reconstruction_error.mean(dim=-1)

        # 4. Mismatch Score를 Prior-Alignment Weight로 변환
        alignment_weight = torch.softmax(-mismatch, dim=-1)

        # 5. Reconstruction Error와 Alignment Weight 결합
        energy = row_error * alignment_weight

        # 6. 시점별 Energy Score 반환
        return energy

    # score fusion: Mismatch / Energy Score 정규화 및 결합
    def score_fusion(
        self,
        mismatch: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """Mismatch와 Energy를 정규화하고 최종 Fused Score를 계산한다.

        Args:
            mismatch (torch.Tensor): Mismatch Score, Shape (B, T).
            energy (torch.Tensor): Energy Score, Shape (B, T).

        Returns:
            torch.Tensor: Fused Anomaly Score, Shape (B, T).
        """
        # 1. Mismatch / Energy Score Shape 검증
        if mismatch.ndim != 2 or mismatch.shape != energy.shape:
            raise ValueError("score_fusion - mismatch and energy must have the same [B, T] shape")

        # 2. IQR가 0인 경우를 방지하기 위한 최소값 설정
        eps = torch.finfo(energy.dtype).eps

        energy_iqr = self.energy_iqr.clamp_min(eps)
        mismatch_iqr = self.mismatch_iqr.clamp_min(eps)

        # 3. 학습 데이터 기준 Energy Score 정규화
        energy_normalized = F.relu((energy - self.energy_median) / energy_iqr)

        # 4. 학습 데이터 기준 Mismatch Score 정규화
        mismatch_normalized = F.relu((mismatch - self.mismatch_median) / mismatch_iqr)

        # 5. 두 점수의 최댓값으로 Fused Score 계산
        fused_score = torch.maximum(
            energy_normalized,
            mismatch_normalized,
        )

        # 6. 시점별 최종 Anomaly Score 반환
        return fused_score

    ################ batch outputs ################
    # batch_outputs: loss, validation_loss, prediction, anomaly_score, score_*
    def batch_outputs(
        self,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """모델의 복원 결과와 이상 점수를 기존 파이프라인 형식으로 반환한다.

        Args:
            batch (dict): 입력 Batch.

        Returns:
            dict[str, torch.Tensor]:
                loss: Reconstruction Loss, Scalar.
                validation_loss: Reconstruction Loss, Scalar.
                prediction: 마지막 시점 복원값, Shape (B, n_features).
                anomaly_score: 최종 Fused Score, Shape (B, n_features).
                score_energy: Energy Score, Shape (B, n_features).
                score_mismatch: Mismatch Score, Shape (B, n_features).
                score_fused: Fused Score, Shape (B, n_features).
        """
        # 1. 입력 context 및 main model 실행
        context = batch["context"].to(dtype=self.input_dtype)
        outputs = self(context)

        # 2. reconstruction loss 계산
        reconstruction_loss = self.recondsturction_loss(
            context,
            outputs["recondstruction"],
        )

        # 3. mismatch stream 계산
        mismatch = self.mismatch_stream(
            outputs["series"],
            outputs["prior"],
        )

        # 4. energy stream 계산
        energy = self.energy_stream(
            context,
            outputs["reconstruction"],
            mismatch,
        )

        # 5. 최종 fused anomaly score 계산
        fused_score = self.score_fusion(mismatch, energy)

        # 6. 마지막 시점의 reconstruction & anomaly score 추출
        prediction = outputs["reconstruction"][:, -1, :]
        energy_last = energy[:, -1:]
        mismatch_last = mismatch[:, -1:]
        fused_last = fused_score[:, -1:]

        # 7. GSC 파이프라인 호환을 위해 행별 점수를 Feature 차원으로 확장
        energy_features = energy_last.expand_as(prediction)
        mismatch_features = mismatch_last.expand_as(prediction)
        fused_features = fused_last.expand_as(prediction)

        return {
            "loss": reconstruction_loss,
            "validation_loss": reconstruction_loss,
            "prediction": prediction,
            "anomaly_score": fused_features,
            "score_energy": energy_features,
            "score_mismatch": mismatch_features,
            "score_fused": fused_features,
        }
