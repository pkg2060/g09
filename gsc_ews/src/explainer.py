"""공통 모델 점수의 SHAP과 Attention, TFT 변수 선택 가중치를 분석한다."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .udf.configuration import resolve_shap_targets
from .udf.visualization import safe_filename_component


class AnomalyScoreModel(nn.Module):
    """SHAP이 사용하는 평탄한 입력을 표준 윈도우와 이상 점수로 연결한다.

    점수는 어댑터에서만 계산하므로 모델을 바꾸어도 별도의 오차 공식을 만들지 않는다.
    """

    def __init__(self, model, target_feature=None):
        """현재 모델의 윈도우 크기와 특성 수를 재사용한다."""
        super().__init__()
        self.model = model
        names = model.data_state.get("feature_names", [])
        if target_feature is not None and target_feature not in names:
            raise ValueError(f"SHAP 대상 변수가 모델 특성에 없습니다: {target_feature}")
        self.target_index = None if target_feature is None else names.index(target_feature)

    def forward(self, flat):
        """입력 dtype을 모델에 맞추면서 미분 그래프를 유지한다."""
        context = flat.to(device=self.model.device, dtype=self.model.dtype).reshape(
            -1, self.model.model_config["window_size"], self.model.n_feats
        )
        output = self.model.batch_outputs({"context": context, "target": context[:, -1]})
        scores = output["anomaly_score"]
        return (
            scores.mean(-1, keepdim=True)
            if self.target_index is None
            else scores[:, self.target_index : self.target_index + 1]
        )


def sample_windows(dataset, count, *, include=()):
    """시간순으로 고르게 고른 소수 윈도우만 실제 텐서로 구성한다."""
    required = np.asarray(include, dtype=int)[:count]
    uniform = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
    selected = np.sort(np.concatenate([required, np.setdiff1d(uniform, required)[: count - len(required)]]))
    return torch.stack([dataset[int(index)]["context"] for index in selected]), selected


def analyze(
    model,
    dataset,
    settings,
    directory,
    *,
    background_dataset=None,
    result=None,
    plot_directory=None,
    dpi=150,
    report_directory=None,
    excel=True,
):
    """Attention·TFT 변수 중요도와 선택적인 SHAP을 저장한다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    alarm_samples = []
    if result is not None and result["model_name"] == "tft":
        alarm_rows = result["index"][result["alarms"]["overall"].astype(bool)][:10]
        alarm_samples = np.flatnonzero(np.isin(dataset.end_indices, alarm_rows))
    windows, selected = sample_windows(dataset, settings["max_samples"], include=alarm_samples)
    windows = windows.to(model.device, dtype=model.dtype)
    pd.DataFrame(
        {
            "sample": selected,
            "index": dataset.end_indices[selected],
            "timestamp_ns": dataset.timestamps_ns[dataset.end_indices[selected]],
        }
    ).to_csv(directory / "samples.csv", index=False)
    model.eval()
    captured = {}
    hooks = []

    def request_heads(module, args, kwargs):
        """헤드별 가중치는 해석 실행 중에만 요청한다."""
        kwargs = dict(kwargs, need_weights=True, average_attn_weights=False)
        return args, kwargs

    def capture(name):
        """같은 Encoder가 두 번 호출되면 Phase 2의 가중치를 보관한다."""

        def hook(module, args, output):
            """Attention 텐서를 계산 그래프에서 분리한다."""
            if output[1] is not None:
                captured[name] = output[1].detach().cpu().numpy()

        return hook

    try:
        if settings["attention"]:
            for name, module in model.network.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    hooks.append(module.register_forward_pre_hook(request_heads, with_kwargs=True))
                    hooks.append(module.register_forward_hook(capture(name)))
        with torch.no_grad():
            outputs = model.batch_outputs({"context": windows, "target": windows[:, -1]})
        if hasattr(model.network, "last_interpretation"):
            interpretation = model.network.last_interpretation
            # VSN 가중치를 배치·시간축으로 평균하여 전역 변수 중요도를 계산한다.
            weights = interpretation["encoder_var_weights"].cpu().numpy()
            importance = weights.mean(axis=(0, 1))
            indices = dataset.end_indices[selected]
            scores = outputs["anomaly_score"].cpu().numpy()
            alarm_mask = np.zeros(len(selected), dtype=bool)
            if result is not None:
                rows = np.searchsorted(result["index"], indices)
                scores = result["score"][rows]
                alarm_mask = result["alarms"]["overall"][rows]
            contribution = weights.mean(axis=1) * scores
            names = model.data_state["feature_names"]
            pd.DataFrame(
                {"feature": names, "weight": importance, "score_contribution": contribution.mean(axis=0)}
            ).sort_values("weight", ascending=False).to_csv(directory / "variable_importance.csv", index=False)
            # 표본별 분위수·변수 가중치는 시점별 원인 분석에도 사용할 수 있다.
            np.savez_compressed(
                directory / "tft_interpretation.npz",
                variable_weights=weights,
                score_contribution=contribution,
                quantile_outputs=interpretation["quantile_outputs"].cpu().numpy(),
                quantiles=model.network.quantiles,
                scores=scores,
                indices=indices,
            )
            if settings["attention"]:
                captured["tft_attention"] = interpretation["attention"].cpu().numpy()
            if excel:
                from .save import save_tft_interpretation

                save_tft_interpretation(
                    weights, scores, names, indices, alarm_mask, report_directory or directory.parent
                )
            _tft_plots(
                model, weights, interpretation, indices, settings, Path(plot_directory or directory), result, dpi
            )
        if captured:
            np.savez_compressed(directory / "attention.npz", **captured)
            _attention_plots(
                {key: value for key, value in captured.items() if key != "tft_attention"}, directory, dpi=dpi
            )
    finally:
        for hook in hooks:
            hook.remove()
    if settings["shap"]:
        if background_dataset is None:
            raise ValueError("SHAP 배경은 학습 구간의 DataLoader에서 전달해야 합니다")
        background, _ = sample_windows(background_dataset, settings["background_samples"])
        names = model.data_state["feature_names"]
        targets = resolve_shap_targets(settings.get("shap_target"), names)
        raw_scores = outputs["anomaly_score"].detach().cpu().numpy()
        for index, target in enumerate(targets, start=1):
            values = shap_values(model, windows, background, settings, target)
            suffix = "" if len(targets) == 1 else f"_{index:02d}_{safe_filename_component(target)}"
            np.savez_compressed(
                directory / f"shap{suffix}.npz",
                values=values,
                windows=windows.cpu().numpy(),
                target_feature="" if target is None else target,
            )
            importance = np.abs(values).mean(axis=(0, 1))
            pd.DataFrame({"feature": names, "mean_abs_shap": importance}).sort_values(
                "mean_abs_shap", ascending=False
            ).to_csv(directory / f"shap_importance{suffix}.csv", index=False)
            _bar_plot(
                importance,
                names,
                directory / f"shap_importance{suffix}.png",
                settings["top_k"],
                "Mean |SHAP|",
                target_feature=target,
                method=settings["shap_method"],
                dpi=dpi,
            )
            target_scores = raw_scores.mean(axis=1) if target is None else raw_scores[:, names.index(target)]
            _waterfall_plot(
                values,
                windows.detach().cpu().numpy(),
                target_scores,
                names,
                directory / f"shap_waterfall{suffix}.png",
                settings["top_k"],
                target_feature=target,
                dpi=dpi,
            )


def shap_values(model, windows, background, settings, target_feature=None):
    """학습 배경으로 Kernel·Gradient·Permutation SHAP을 계산한다."""
    import shap

    wrapper = AnomalyScoreModel(model, target_feature).eval()
    flat = windows.flatten(1).to(model.device, dtype=model.dtype)
    reference = background.flatten(1).to(model.device, dtype=model.dtype)
    method = settings["shap_method"]
    samples = settings["shap_nsamples"]
    if method == "gradient":
        # cuDNN LSTM은 eval 모드에서 역전파를 제한하므로 분석 중에만 기본 연산을 쓴다.
        with torch.backends.cudnn.flags(enabled=False):
            values = shap.GradientExplainer(wrapper, reference).shap_values(flat, nsamples=samples)
    else:

        def predict(values):
            """SHAP의 대량 교란 입력을 작은 배치로 나누어 추론한다."""
            with torch.no_grad():
                result = [
                    wrapper(torch.as_tensor(chunk, device=model.device, dtype=model.dtype)).cpu().numpy()
                    for chunk in np.array_split(values, max(1, (len(values) + 127) // 128))
                ]
            return np.concatenate(result)

        if method == "kernel":
            values = shap.KernelExplainer(predict, reference.cpu().numpy()).shap_values(
                flat.cpu().numpy(), nsamples=samples, silent=True
            )
        elif method == "permutation":
            # 한 번의 permutation은 특성 수의 두 배 + 1번 평가가 필요하다.
            explainer = shap.Explainer(predict, reference.cpu().numpy(), algorithm="permutation")
            values = explainer(flat.cpu().numpy(), max_evals=max(samples, 2 * flat.shape[1] + 1)).values
        else:
            raise ValueError(f"Unsupported SHAP method: {method}")
    values = np.asarray(values[0] if isinstance(values, list) else values)
    return values.reshape(len(windows), model.model_config["window_size"], model.n_feats)


def _bar_plot(values, names, path, top_k, label, *, target_feature=None, method=None, dpi=150):
    """상위 변수 중요도를 가로 막대그래프로 저장한다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = np.argsort(values)[-top_k:]
    fig, axis = plt.subplots(figsize=(14, 8))
    colors = ["#E74C3C" if names[index] == target_feature else "#3498DB" for index in selected]
    axis.barh(np.asarray(names)[selected], np.asarray(values)[selected], color=colors)
    axis.set_title(
        f"Variable Importance for Alarm\n(Target: {target_feature or 'Overall'}, Method: {method})",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(True, alpha=0.3, axis="x")
    if target_feature is not None:
        from matplotlib.patches import Patch

        axis.legend(
            handles=[
                Patch(color="#E74C3C", label=f"{target_feature} (self)"),
                Patch(color="#3498DB", label="Other variables (cross-variable)"),
            ]
        )
    axis.set_xlabel(label)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _waterfall_plot(values, windows, scores, names, path, top_k, *, target_feature=None, dpi=150):
    """가장 이상 점수가 큰 표본의 변수별 SHAP 기여도를 Waterfall로 저장한다."""
    import matplotlib.pyplot as plt
    import shap

    sample = int(np.argmax(scores))
    contributions = values[sample].sum(axis=0)
    prediction = float(scores[sample])
    baseline = prediction - float(contributions.sum())
    explanation = shap.Explanation(
        values=contributions,
        base_values=baseline,
        data=windows[sample, -1],
        feature_names=names,
    )
    shap.plots.waterfall(explanation, max_display=min(top_k, len(names)), show=False)
    figure = plt.gcf()
    figure.set_size_inches(12, max(5, min(10, len(names) * 0.45 + 2)))
    figure.suptitle(
        f"SHAP Waterfall (Target: {target_feature or 'Overall'}, Highest-score sample)",
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _attention_plots(captured, directory, *, dpi=150):
    """표본별 원본 헤드에서 평균·헤드별 Attention과 시간 중요도 그림을 만든다."""
    from .udf.visualization import plot_attention_details

    for name, weights in captured.items():
        plot_attention_details(weights, name, directory, dpi=dpi)


def _tft_plots(model, weights, interpretation, indices, settings, directory, result, dpi):
    """기존 TFT 해석 그림을 표본의 실제 시점과 전체 분위수 예측에 연결한다."""
    from .udf.visualization import (
        plot_attention_heatmap,
        plot_attention_temporal_pattern,
        plot_quantile_forecast,
        plot_root_cause_at_alarms,
        plot_var_importance_heatmap,
        plot_variable_importance,
    )

    names = model.data_state["feature_names"]
    plot_variable_importance(weights, names, directory, dpi=dpi, top_k=settings["top_k"])
    plot_var_importance_heatmap(weights, names, directory, dpi=dpi, sample_indices=indices)
    if settings["attention"]:
        attention = interpretation["attention"].cpu().numpy()
        plot_attention_heatmap(attention, indices, directory, dpi=dpi)
        plot_attention_temporal_pattern(attention, interpretation["T_enc"], directory, dpi=dpi)
    if result is not None:
        rows = np.searchsorted(result["index"], indices)
        alarm_indices = np.flatnonzero(result["alarms"]["overall"][rows]).tolist()
        plot_root_cause_at_alarms(
            result["score"][rows], weights, names, alarm_indices, directory, dpi=dpi, sample_indices=indices
        )
        from .dataloader import FeatureScaler

        scaler = FeatureScaler.from_state_dict(model.data_state["scaler"])
        qs = np.asarray(model.network.quantiles)
        positions = [
            int(np.argmin(np.abs(qs - q)))
            for q in (model.model_config["qband_low"], 0.5, model.model_config["qband_high"])
        ]
        low, median, high = [scaler.inverse_transform(result["quantiles"][..., i]) for i in positions]
        plot_quantile_forecast(result["target"], median, low, high, names, directory, dpi=dpi, quantiles=qs[positions])
