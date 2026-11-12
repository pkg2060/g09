"""기존 실험의 그림 구성·범례·파일명을 유지하는 공통 시각화 함수."""

import os
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 프로젝트에 동봉할 수 있는 폰트 폴더. 서버에 시스템 폰트를 설치할 수 없을 때 .ttf/.otf를 넣는다.
PROJECT_FONT_DIRECTORY = Path(__file__).resolve().parents[2] / "fonts"
# 앞에서부터 설치된 첫 번째 한글 폰트를 사용한다. Linux·Windows·macOS 기본 폰트를 모두 포함한다.
KOREAN_FONT_CANDIDATES = (
    "NanumGothic",
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Malgun Gothic",
    "AppleGothic",
    "UnDotum",
)


def configure_korean_font():
    """Tag Description 등 한글이 Plot에서 네모로 깨지지 않도록 한글 폰트를 등록한다.

    프로젝트 ``fonts/`` 폴더의 폰트를 먼저 등록한 뒤 설치된 후보 폰트를 순서대로 찾는다.
    한글 폰트가 없으면 경고만 출력하고 기본 폰트로 계속 그린다.
    """
    if PROJECT_FONT_DIRECTORY.is_dir():
        for path in sorted(PROJECT_FONT_DIRECTORY.iterdir()):
            if path.suffix.lower() in {".ttf", ".otf", ".ttc"}:
                font_manager.fontManager.addfont(str(path))
    installed = {font.name for font in font_manager.fontManager.ttflist}
    project_fonts = sorted(
        font.name for font in font_manager.fontManager.ttflist if Path(font.fname).parent == PROJECT_FONT_DIRECTORY
    )
    selected = next((name for name in (*project_fonts, *KOREAN_FONT_CANDIDATES) if name in installed), None)
    # 한글 폰트에 따라 유니코드 마이너스 기호가 없을 수 있으므로 ASCII 하이픈을 사용한다.
    plt.rcParams["axes.unicode_minus"] = False
    if selected is None:
        warnings.warn(
            "한글 폰트를 찾지 못해 Plot의 한글이 네모로 표시될 수 있습니다. "
            f"NanumGothic 등을 설치하거나 .ttf 파일을 {PROJECT_FONT_DIRECTORY}에 넣으세요",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected, *plt.rcParams["font.sans-serif"]]
    return selected


KOREAN_FONT = configure_korean_font()


def safe_filename_component(value, *, maximum_length=80):
    """외부 특성명을 출력 폴더 내부의 안전한 파일명 조각으로 바꾼다."""
    cleaned = re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("._-")
    return (cleaned or "feature")[:maximum_length]


def format_metric(value, digits=4):
    """라벨이 없거나 정의되지 않는 지표를 숫자로 꾸미지 않고 표시한다."""
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def plot_variable_analysis(
    test_target,
    test_pred,
    scores,
    labels,
    thresholds,
    alarm_per_var,
    feature_names,
    timestamps=None,
    segments=None,
    save_dir=None,
    inhibit_mask=None,
    *,
    feature_descriptions=None,
    crv_limits=None,
    score_name,
    prediction_label,
    dpi,
):
    """원본 단위 입력·예측과 최종 후처리 Anomaly Score를 변수별로 그린다.

    하단 점수는 원본 단위 MAE가 아니다. 자체적으로 scores > threshold 비교를 하지 않고 저장된 알람과 같은 판정을
    표시하며, 현재 파이프라인에서 정렬한 타깃을 다시 이동시키지 않는다.
    """
    import matplotlib.dates as mdates

    os.makedirs(save_dir, exist_ok=True)

    n_feats = test_target.shape[1]
    T = test_target.shape[0]

    # X축: timestamps가 있으면 DateTime, 없으면 정수 인덱스
    use_datetime = timestamps is not None and len(timestamps) == T
    if use_datetime:
        import pandas as pd

        x_axis = pd.to_datetime(timestamps).values if not hasattr(timestamps, "values") else timestamps.values
    else:
        x_axis = np.arange(T)

    # 추론 단계에서 이미 원본 공정 단위로 복원한 값을 사용한다.
    plot_x, plot_target, plot_pred, plot_scores = x_axis, test_target, test_pred, scores
    if segments is not None:
        segments = np.asarray(segments)
        if segments.shape != (T,):
            raise ValueError("그래프 구간 ID와 추론 결과의 길이가 다릅니다")
        boundaries = np.flatnonzero(segments[1:] != segments[:-1]) + 1
        if len(boundaries):
            # 정지 구간에는 추론값이 없으므로 그래프 배열에만 NaN을 넣는다.
            # Matplotlib은 NaN에서 선을 끊어 정지 전후 값이 연속 운전처럼 붙어 보이지 않는다.
            plot_x = np.insert(x_axis, boundaries, x_axis[boundaries])
            plot_target = np.insert(np.asarray(test_target, dtype=float), boundaries, np.nan, axis=0)
            plot_pred = np.insert(np.asarray(test_pred, dtype=float), boundaries, np.nan, axis=0)
            plot_scores = np.insert(np.asarray(scores, dtype=float), boundaries, np.nan, axis=0)
    ylabel = "Value (Original Scale)"

    import gc

    for i in range(n_feats):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        var_name = feature_names[i] if i < len(feature_names) else f"Var_{i}"
        description = (feature_descriptions or {}).get(var_name)
        title = f"Variable Analysis: {var_name}"
        title += f"  [{description}]" if description else ""
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # ─── 상단: Input vs Predicted ───
        ax1.plot(plot_x, plot_target[:, i], color="blue", alpha=0.7, linewidth=0.8, label="Input (Ground Truth)")
        ax1.plot(plot_x, plot_pred[:, i], color="orange", alpha=0.7, linewidth=0.8, label=prediction_label)

        # CRV는 원본 공정값 기준이므로 정규화 점수가 아닌 상단 원본 단위 축에 표시한다.
        limits = (crv_limits or {}).get(var_name, {})
        for key, label, color, linestyle in (
            ("alarm_low", "CRV Alarm Low", "#C0392B", "--"),
            ("pre_alarm_low", "CRV Pre Alarm Low", "#E67E22", ":"),
            ("pre_alarm_high", "CRV Pre Alarm High", "#E67E22", ":"),
            ("alarm_high", "CRV Alarm High", "#C0392B", "--"),
        ):
            value = limits.get(key)
            if value is not None:
                ax1.axhline(value, color=color, linestyle=linestyle, linewidth=1.2, label=f"{label} ({value:g})")

        # 실제 Anomaly 구간 표시 (있는 경우)
        if labels is not None:
            anomaly_mask = labels[:, i] > 0.5
            if anomaly_mask.any():
                ax1.fill_between(
                    x_axis,
                    ax1.get_ylim()[0],
                    ax1.get_ylim()[1],
                    where=anomaly_mask,
                    alpha=0.15,
                    color="red",
                    label="True Anomaly Region",
                )

        # 억제 구간 회색 음영 (Inhibit 구간 — Anomaly Score가 있어도 Alarm 억제)
        if inhibit_mask is not None and np.any(inhibit_mask):
            y_min_1, y_max_1 = ax1.get_ylim()
            ax1.fill_between(
                x_axis,
                y_min_1,
                y_max_1,
                where=inhibit_mask,
                alpha=0.15,
                color="gray",
                label="Inhibit (No Alarm)",
                zorder=1,
            )
            ax1.set_ylim(y_min_1, y_max_1)  # fill_between으로 ylim 변경 방지

        # 감지된 Anomaly 표시 — AlarmGenerator의 판정 결과 사용
        alarm_mask = alarm_per_var[:, i] > 0
        if alarm_mask.any():
            alarm_indices = np.where(alarm_mask)[0]
            if use_datetime:
                ax1.scatter(
                    x_axis[alarm_indices],
                    plot_target[alarm_indices, i],
                    color="red",
                    s=10,
                    zorder=5,
                    alpha=0.6,
                    label=f"Alarm (n={len(alarm_indices)})",
                )
            else:
                ax1.scatter(
                    alarm_indices,
                    plot_target[alarm_indices, i],
                    color="red",
                    s=10,
                    zorder=5,
                    alpha=0.6,
                    label=f"Alarm (n={len(alarm_indices)})",
                )

        ax1.set_ylabel(ylabel, fontsize=10)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ─── 하단: Anomaly Scores + Threshold ───
        score_label = f"Final Anomaly Score (source: {score_name.upper()})"
        ax2.plot(plot_x, plot_scores[:, i], color="purple", linewidth=0.8, label=score_label)
        ax2.axhline(
            y=thresholds[i], color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({thresholds[i]:.4f})"
        )

        # 억제 구간 회색 음영 (Anomaly Score 그래프에도 표시)
        if inhibit_mask is not None and np.any(inhibit_mask):
            ax2.fill_between(
                x_axis,
                0,
                ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1,
                where=inhibit_mask,
                alpha=0.15,
                color="gray",
                label="Inhibit (No Alarm)",
                zorder=1,
            )

        # Alarm 구간 강조 (AlarmGenerator 판정 기준 — 실제 알람만 빨간 음영)
        ax2.fill_between(x_axis, 0, scores[:, i], where=alarm_mask, alpha=0.3, color="red", label="Alarm Region")

        if labels is not None:
            anomaly_mask = labels[:, i] > 0.5
            if anomaly_mask.any():
                ax2.fill_between(
                    x_axis,
                    ax2.get_ylim()[0],
                    ax2.get_ylim()[1],
                    where=anomaly_mask,
                    alpha=0.1,
                    color="blue",
                    label="True Anomaly",
                )

        # X축 설정
        if use_datetime:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate(rotation=30, ha="right")
            ax2.set_xlabel("DateTime", fontsize=10)
        else:
            ax2.set_xlabel("Time Step", fontsize=10)

        ax2.set_ylabel(score_label, fontsize=10)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(save_dir, f"variable_{i:02d}_{safe_filename_component(var_name)}.png")
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if (i + 1) % 10 == 0:
            gc.collect()


def plot_training_curves(history, save_dir, *, dpi):
    """
    요구사항 6-2: Scoring 지표 변화 그래프
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    epochs = history["epoch"]

    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    if history.get("val_loss"):
        ax1.plot(epochs, history["val_loss"], "r-s", markersize=4, label="Validation Loss")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title("Loss Curves", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(epochs, history["lr"], "g-^", markersize=4, label="Learning Rate")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Learning Rate", fontsize=10)
    ax2.set_title("Learning Rate Schedule", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, "training_curves.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_all_scoring_metrics(test_scores_all, labels, save_dir, *, dpi):
    """
    요구사항 6-2 확장: 모든 Scoring 지표를 한 그래프에 표현
    """
    os.makedirs(save_dir, exist_ok=True)

    methods = list(test_scores_all.keys())
    colors = {
        "tranad": "blue",
        "mae": "green",
        "mse": "red",
        "hybrid": "purple",
        "phase2_mse": "blue",
        "quantile_loss": "blue",
        "band_mse": "green",
        "median_mse": "red",
        "band_violation": "purple",
    }

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle("All Scoring Metrics Comparison (Mean across variables)", fontsize=14, fontweight="bold")

    T = list(test_scores_all.values())[0].shape[0]
    x_axis = np.arange(T)

    for method in methods:
        mean_score = np.mean(test_scores_all[method], axis=1)
        color = colors.get(method, "gray")
        ax.plot(x_axis, mean_score, color=color, alpha=0.7, linewidth=0.8, label=f"{method.upper()} Score")

    if labels is not None:
        overall_label = (np.sum(labels, axis=1) >= 1).astype(float)
        anomaly_mask = overall_label > 0.5
        if anomaly_mask.any():
            ymin, ymax = ax.get_ylim()
            ax.fill_between(
                x_axis, ymin, ymax, where=anomaly_mask, alpha=0.15, color="red", label="True Anomaly Region"
            )

    ax.set_xlabel("Time Step", fontsize=10)
    ax.set_ylabel("Anomaly Score (Mean)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, "all_scoring_metrics.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_performance_table(metrics_per_var, overall_metrics, feature_names, save_dir, *, model_name, dpi):
    """
    요구사항 6-3: 최종 성능지표 표 (논문 Table 2와 동일한 형식)
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")
    ax.set_title(f"{model_name} Performance Summary", fontsize=14, fontweight="bold", pad=20)

    if overall_metrics.get("precision") is None:
        ax.text(0.5, 0.04, "N/A: ground-truth labels unavailable", ha="center", transform=ax.transAxes, fontsize=9)
    headers = ["Metric", "Value"]
    table_data = []

    for key in ["precision", "recall", "f1", "TP", "TN", "FP", "FN", "ROC_AUC", "threshold"]:
        if key in overall_metrics:
            val = overall_metrics[key]
            if val is None:
                table_data.append([key.upper(), "N/A"])
            elif isinstance(val, float):
                if key in ["precision", "recall", "f1", "ROC_AUC"]:
                    table_data.append([key.upper(), f"{val:.4f}"])
                else:
                    table_data.append([key.upper(), f"{val:.6f}"])
            else:
                table_data.append([key.upper(), str(val)])

    for key in ["Hit@100%", "Hit@150%", "NDCG@100%", "NDCG@150%"]:
        if key in overall_metrics:
            table_data.append([key, format_metric(overall_metrics[key])])

    table = ax.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center", colWidths=[0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#D6E4F0")
            else:
                table[i, j].set_facecolor("#FFFFFF")

    plt.tight_layout()
    filepath = os.path.join(save_dir, "performance_table.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if metrics_per_var:
        n_vars = len(metrics_per_var)
        fig, ax = plt.subplots(figsize=(14, max(4, n_vars * 0.4 + 2)))
        ax.axis("tight")
        ax.axis("off")
        ax.set_title("Per-Variable Performance", fontsize=14, fontweight="bold", pad=20)

        headers_var = ["Variable", "Precision", "Recall", "F1", "Threshold"]
        table_data_var = []

        for i, m in enumerate(metrics_per_var):
            name = feature_names[i] if i < len(feature_names) else f"Var_{i}"
            table_data_var.append(
                [
                    name,
                    format_metric(m.get("precision"), 4),
                    format_metric(m.get("recall"), 4),
                    format_metric(m.get("f1"), 4),
                    format_metric(m.get("threshold"), 6),
                ]
            )

        table_var = ax.table(
            cellText=table_data_var,
            colLabels=headers_var,
            cellLoc="center",
            loc="center",
            colWidths=[0.25, 0.15, 0.15, 0.15, 0.2],
        )
        table_var.auto_set_font_size(False)
        table_var.set_fontsize(9)
        table_var.scale(1.1, 1.3)

        for j in range(len(headers_var)):
            table_var[0, j].set_facecolor("#4472C4")
            table_var[0, j].set_text_props(color="white", fontweight="bold")

        for i in range(1, len(table_data_var) + 1):
            for j in range(len(headers_var)):
                if i % 2 == 0:
                    table_var[i, j].set_facecolor("#D6E4F0")

        plt.tight_layout()
        filepath_var = os.path.join(save_dir, "performance_per_variable.png")
        fig.savefig(filepath_var, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_quality_per_variable(quality_result, save_dir, *, prediction_label, dpi):
    """변수별 입력·예측 오버레이와 시점별 절대 오차를 원본 단위로 그린다."""
    import gc

    os.makedirs(save_dir, exist_ok=True)

    predictions = quality_result["predictions"]
    targets = quality_result["targets"]
    per_var = quality_result["per_variable"]
    feature_names = quality_result["feature_names"]
    mae_per_var = quality_result["timeseries"]["mae_per_var"]
    n_feats = quality_result["n_feats"]
    T = quality_result["n_samples"]
    x_axis = np.arange(T)

    # 절대 오차도 원본 스케일로 표시한다.
    plot_targets, plot_preds, plot_mae = targets, predictions, mae_per_var
    ylabel, err_ylabel = "Value (Original Scale)", "Absolute Error (Original Scale)"
    scale_note = "[원본 공정 단위]"

    for i in range(n_feats):
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        var_name = feature_names[i] if i < len(feature_names) else f"Var_{i}"
        m = per_var[i]

        fig.suptitle(
            f"Reconstruction Quality: {var_name}  {scale_note}\n"
            f"Reconstruction MAE={m['MAE']:.6f}  Mean Anomaly Score={m['mean_anomaly_score']:.6f}  "
            f"Threshold={m['threshold']:.6f}  Alarm Rate={m['alarm_rate_pct']:.2f}%\n"
            f"RMSE={m['RMSE']:.6f}  "
            f"R\u00b2={m['R2']:.4f}  MAPE={m['MAPE']:.2f}%",
            fontsize=12,
            fontweight="bold",
        )

        # ─── 상단: Input vs Predicted ───
        ax1 = axes[0]
        ax1.plot(x_axis, plot_targets[:, i], color="blue", alpha=0.7, linewidth=0.8, label="Input (Ground Truth)")
        ax1.plot(x_axis, plot_preds[:, i], color="orange", alpha=0.7, linewidth=0.8, label=prediction_label)
        ax1.set_ylabel(ylabel, fontsize=10)
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Input vs Predicted", fontsize=11)

        # ─── 하단: 시점별 MAE ───
        ax2 = axes[1]
        mean_ae_i = np.mean(plot_mae[:, i])
        ax2.fill_between(x_axis, 0, plot_mae[:, i], alpha=0.5, color="coral", label="|Predicted - Input|")
        ax2.plot(x_axis, plot_mae[:, i], color="red", linewidth=0.5, alpha=0.7)
        ax2.axhline(y=mean_ae_i, color="darkred", linestyle="--", linewidth=1.5, label=f"Mean AE = {mean_ae_i:.6f}")
        ax2.set_xlabel("Time Step", fontsize=10)
        ax2.set_ylabel(err_ylabel, fontsize=10)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Point-wise Absolute Error", fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(save_dir, f"recon_quality_{i:02d}_{safe_filename_component(var_name)}.png")
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if (i + 1) % 10 == 0:
            gc.collect()


def plot_quality_summary(quality_result, save_dir, *, dpi):
    """복원 품질 요약과 전체·변수별 지표 표를 기존 형식으로 그린다."""

    os.makedirs(save_dir, exist_ok=True)

    per_var = quality_result["per_variable"]
    overall = quality_result["overall"]
    n_feats = quality_result["n_feats"]
    mae_per_step = quality_result["timeseries"]["mae_per_step"]

    var_names = [m["name"] for m in per_var]
    mae_values = [m["MAE"] for m in per_var]
    rmse_values = [m["RMSE"] for m in per_var]
    r2_values = [m["R2"] for m in per_var]

    # ─── Figure 1: 변수별 MAE/RMSE 비교 + R² + 시계열 MAE ───
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Reconstruction Quality Summary", fontsize=16, fontweight="bold")

    # (1) 변수별 MAE 막대 그래프
    ax1 = axes[0, 0]
    x_pos = np.arange(n_feats)
    ax1.bar(x_pos, mae_values, color="steelblue", alpha=0.8, label="MAE")
    ax1.axhline(
        y=overall["MAE_mean"], color="red", linestyle="--", linewidth=1.5, label=f"Mean MAE = {overall['MAE_mean']:.6f}"
    )
    ax1.set_xlabel("Variable", fontsize=10)
    ax1.set_ylabel("Reconstruction MAE (Original Scale)", fontsize=10)
    ax1.set_title("Reconstruction MAE per Variable (Original Scale, Lower = Better)", fontsize=12)
    if n_feats <= 30:
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(var_names, rotation=45, ha="right", fontsize=7)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # (2) 변수별 RMSE 막대 그래프
    ax2 = axes[0, 1]
    ax2.bar(x_pos, rmse_values, color="darkorange", alpha=0.8, label="RMSE")
    ax2.axhline(
        y=overall["RMSE_mean"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean RMSE = {overall['RMSE_mean']:.6f}",
    )
    ax2.set_xlabel("Variable", fontsize=10)
    ax2.set_ylabel("RMSE", fontsize=10)
    ax2.set_title("RMSE per Variable (Lower = Better)", fontsize=12)
    if n_feats <= 30:
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(var_names, rotation=45, ha="right", fontsize=7)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # (3) 변수별 R² Score 막대 그래프
    ax3 = axes[1, 0]
    colors_r2 = [
        "green" if v >= 0.9 else "steelblue" if v >= 0.7 else "orange" if v >= 0.5 else "red" for v in r2_values
    ]
    ax3.bar(x_pos, r2_values, color=colors_r2, alpha=0.8)
    ax3.axhline(
        y=overall["R2_mean"],
        color="darkgreen",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean R\u00b2 = {overall['R2_mean']:.4f}",
    )
    ax3.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("Variable", fontsize=10)
    ax3.set_ylabel("R\u00b2 Score", fontsize=10)
    ax3.set_title("R\u00b2 Score per Variable (Higher = Better, max=1)", fontsize=12)
    if n_feats <= 30:
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(var_names, rotation=45, ha="right", fontsize=7)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    # 범례: 색상 의미
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", alpha=0.8, label="R\u00b2 >= 0.9 (Excellent)"),
        Patch(facecolor="steelblue", alpha=0.8, label="R\u00b2 >= 0.7 (Good)"),
        Patch(facecolor="orange", alpha=0.8, label="R\u00b2 >= 0.5 (Fair)"),
        Patch(facecolor="red", alpha=0.8, label="R\u00b2 < 0.5 (Poor)"),
    ]
    ax3.legend(handles=legend_elements, fontsize=8, loc="lower right")

    # (4) 전체 시점별 MAE 추이 (역정규화 적용 가능)
    ax4 = axes[1, 1]

    plot_mae_ts, overall_mae_val = mae_per_step, overall["MAE"]
    mae_ylabel = "Mean Absolute Error (Original Scale)"
    scale_suffix = " [Original Scale]"

    T = len(plot_mae_ts)
    x_time = np.arange(T)
    ax4.plot(x_time, plot_mae_ts, color="coral", linewidth=0.6, alpha=0.7)
    ax4.fill_between(x_time, 0, plot_mae_ts, alpha=0.3, color="coral")
    ax4.axhline(
        y=overall_mae_val, color="red", linestyle="--", linewidth=1.5, label=f"Overall MAE = {overall_mae_val:.6f}"
    )
    ax4.set_xlabel("Time Step", fontsize=10)
    ax4.set_ylabel(mae_ylabel, fontsize=10)
    ax4.set_title(f"MAE over Time (All Variables Mean){scale_suffix}", fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, "reconstruction_quality_summary.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ─── Figure 2: 성능 테이블 이미지 ───
    fig2, ax_table = plt.subplots(figsize=(12, 6))
    ax_table.axis("tight")
    ax_table.axis("off")
    ax_table.set_title("Reconstruction Quality and Operational Score Summary", fontsize=14, fontweight="bold", pad=20)

    # 전체 요약 테이블
    table_data = [
        ["Reconstruction MAE (Original Scale)", f"{overall['MAE']:.6f}", "Lower = Better"],
        ["RMSE", f"{overall['RMSE']:.6f}", "Lower = Better"],
        ["R\u00b2 Score", f"{overall['R2']:.6f}", "Higher = Better (max=1)"],
        ["MAPE (%)", f"{overall['MAPE']:.4f}", "Lower = Better"],
        ["Mean Anomaly Score", f"{overall['mean_anomaly_score']:.6f}", "Final postprocessed score"],
        ["Mean Threshold", f"{overall['mean_threshold']:.6f}", "Per-variable mean"],
        ["Alarm Rate (%)", f"{overall['alarm_rate_pct']:.4f}", "Lower = Better"],
        ["", "", ""],
        ["Reconstruction MAE (per-variable mean)", f"{overall['MAE_mean']:.6f}", ""],
        ["RMSE (per-var mean)", f"{overall['RMSE_mean']:.6f}", ""],
        ["R\u00b2 (per-var mean)", f"{overall['R2_mean']:.6f}", ""],
        ["MAPE (per-var mean)", f"{overall['MAPE_mean']:.4f}", ""],
    ]

    headers = ["Metric", "Value", "Direction"]
    table = ax_table.table(
        cellText=table_data, colLabels=headers, cellLoc="center", loc="center", colWidths=[0.3, 0.25, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#2E7D32")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#E8F5E9")

    plt.tight_layout()
    filepath2 = os.path.join(save_dir, "reconstruction_quality_table.png")
    fig2.savefig(filepath2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)

    # ─── Figure 3: 변수별 성능 테이블 이미지 ───
    n_rows = min(n_feats + 1, 50)  # 최대 50행까지
    fig3, ax_var_table = plt.subplots(figsize=(20, max(4, n_rows * 0.4 + 2)))
    ax_var_table.axis("tight")
    ax_var_table.axis("off")
    ax_var_table.set_title(
        "Per-Variable Reconstruction Quality and Operational Score", fontsize=14, fontweight="bold", pad=20
    )

    var_table_data = []
    for m in per_var[:49]:  # 최대 49개 변수
        var_table_data.append(
            [
                m["name"],
                f"{m['MAE']:.6f}",
                f"{m['mean_anomaly_score']:.6f}",
                f"{m['threshold']:.6f}",
                f"{m['alarm_rate_pct']:.2f}%",
                f"{m['RMSE']:.6f}",
                f"{m['R2']:.4f}",
                f"{m['MAPE']:.2f}%",
            ]
        )
    # 전체 평균 행
    var_table_data.append(
        [
            "=== MEAN ===",
            f"{overall['MAE_mean']:.6f}",
            f"{overall['mean_anomaly_score']:.6f}",
            f"{overall['mean_threshold']:.6f}",
            f"{overall['alarm_rate_pct']:.2f}%",
            f"{overall['RMSE_mean']:.6f}",
            f"{overall['R2_mean']:.4f}",
            f"{overall['MAPE_mean']:.2f}%",
        ]
    )

    var_headers = [
        "Variable",
        "Reconstruction MAE",
        "Mean Anomaly Score",
        "Threshold",
        "Alarm Rate",
        "RMSE",
        "R\u00b2",
        "MAPE",
    ]
    var_table = ax_var_table.table(
        cellText=var_table_data,
        colLabels=var_headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.18, 0.14, 0.14, 0.11, 0.11, 0.11, 0.09, 0.10],
    )
    var_table.auto_set_font_size(False)
    var_table.set_fontsize(9)
    var_table.scale(1.1, 1.3)

    for j in range(len(var_headers)):
        var_table[0, j].set_facecolor("#2E7D32")
        var_table[0, j].set_text_props(color="white", fontweight="bold")

    # 마지막 행 (MEAN) 강조
    last_row = len(var_table_data)
    for j in range(len(var_headers)):
        var_table[last_row, j].set_facecolor("#C8E6C9")
        var_table[last_row, j].set_text_props(fontweight="bold")

    for i in range(1, len(var_table_data)):
        for j in range(len(var_headers)):
            if i % 2 == 0:
                var_table[i, j].set_facecolor("#E8F5E9")

    plt.tight_layout()
    filepath3 = os.path.join(save_dir, "reconstruction_quality_per_variable_table.png")
    fig3.savefig(filepath3, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)


def aggregate_var_selection(var_weights):
    """
    var_weights : (T, T_enc, n_feats)  → 각 시점에서 encoder의 시간 step별 변수 가중치

    반환:
        avg_per_var      : (n_feats,)         전체 평균 중요도
        avg_over_time    : (T, n_feats)       시점 평균 (시간축으로 본 중요도 추이)
        avg_per_time_lag : (T_enc, n_feats)   window 내 시점(lag)별 중요도
    """
    avg_per_var = var_weights.mean(axis=(0, 1))  # (n_feats,)
    avg_over_time = var_weights.mean(axis=1)  # (T, n_feats)
    avg_per_time_lag = var_weights.mean(axis=0)  # (T_enc, n_feats)
    return avg_per_var, avg_over_time, avg_per_time_lag


def plot_variable_importance(var_weights, feature_names, save_dir, *, dpi, top_k=20, prefix="encoder"):
    """
    전체 데이터셋에서의 변수 중요도 막대그래프 (Top-K).
    """
    os.makedirs(save_dir, exist_ok=True)

    avg_per_var, _, _ = aggregate_var_selection(var_weights)
    n_feats = len(avg_per_var)
    feature_names = feature_names if feature_names else [f"Var_{i}" for i in range(n_feats)]

    # 정렬
    idx_sorted = np.argsort(avg_per_var)[::-1]
    top_k = min(top_k, n_feats)
    idx_top = idx_sorted[:top_k]

    fig, ax = plt.subplots(figsize=(12, max(4, top_k * 0.3 + 2)))
    bars = ax.barh(range(top_k), avg_per_var[idx_top][::-1], color="steelblue", alpha=0.85)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in idx_top[::-1]], fontsize=9)
    ax.set_xlabel("Average Variable Selection Weight", fontsize=10)
    ax.set_title(
        f"TFT Variable Importance (Top {top_k}, {prefix})\n= Average softmax weight from Variable Selection Network",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")
    # 값 표시
    for b in bars:
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2, f"{b.get_width():.4f}", va="center", fontsize=8)

    plt.tight_layout()
    fp = os.path.join(save_dir, f"interp_var_importance_{prefix}.png")
    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_var_importance_heatmap(
    var_weights, feature_names, save_dir, *, dpi, sample_indices=None, prefix="encoder", max_time_points=2000
):
    """
    시간(t) × 변수 의 importance heatmap.
    데이터가 너무 길면 down-sample.
    """
    os.makedirs(save_dir, exist_ok=True)

    _, avg_over_time, _ = aggregate_var_selection(var_weights)  # (T, n_feats)
    T, F = avg_over_time.shape
    x = np.arange(T) if sample_indices is None else np.asarray(sample_indices)
    if T > max_time_points:
        step = T // max_time_points
        avg_over_time = avg_over_time[::step]
        x = x[::step]
        T = avg_over_time.shape[0]

    fig, ax = plt.subplots(figsize=(14, max(4, F * 0.25 + 2)))
    im = ax.pcolormesh(x, np.arange(F), avg_over_time.T, cmap="YlOrRd", shading="nearest")
    ax.set_xlabel("Time Step", fontsize=10)
    ax.set_ylabel("Variable", fontsize=10)
    ax.set_yticks(range(F))
    ax.set_yticklabels(feature_names[:F], fontsize=8)
    ax.set_title(
        f"TFT Variable Importance over Time ({prefix})\nBrighter = More important at that time",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="VSN Weight")
    plt.tight_layout()
    fp = os.path.join(save_dir, f"interp_var_importance_heatmap_{prefix}.png")
    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_attention_heatmap(attention, sample_indices, save_dir, *, dpi):
    """
    Attention 가중치(decoder query → encoder/decoder key) heatmap.
    표본 추론에서 얻은 가중치를 재사용하며 제목에는 원래 행 번호를 표시한다.
    """
    os.makedirs(save_dir, exist_ok=True)
    indices = np.linspace(0, len(attention) - 1, min(4, len(attention)), dtype=int)
    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4))
    for ax, idx in zip(np.atleast_1d(axes), indices, strict=False):
        im = ax.imshow(attention[idx], aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_title(f"t={sample_indices[idx]}", fontsize=10)
        ax.set_xlabel("Key Time Step")
        ax.set_ylabel("Query (Decoder) Step")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("TFT Multi-Head Self-Attention (averaged over heads)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fp = os.path.join(save_dir, "interp_attention_heatmap.png")
    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_attention_temporal_pattern(attention, T_enc, save_dir, *, dpi):
    """
    여러 샘플의 attention을 평균하여 "어느 lag가 가장 중요한가"를 시각화.
    """
    from matplotlib.patches import Patch

    os.makedirs(save_dir, exist_ok=True)
    avg_attn = attention.mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(avg_attn))
    ax.bar(x, avg_attn, color=["steelblue"] * T_enc + ["orange"] * (len(avg_attn) - T_enc))
    ax.set_xlabel("Time Step Position (0..T_enc-1: past window | T_enc..: forecast)", fontsize=10)
    ax.set_ylabel("Average Attention Weight", fontsize=10)
    ax.set_title("Average Attention Pattern (Temporal Importance for 1-step Forecast)", fontsize=12, fontweight="bold")
    ax.axvline(x=T_enc - 0.5, color="red", linestyle="--", alpha=0.6, label="Encoder/Decoder boundary (t=-1)")
    ax.legend(
        [Patch(color="steelblue", label="Past (encoder)"), Patch(color="orange", label="Decoder query (current)")],
        ["Past (encoder)", "Decoder query (current)"],
        fontsize=9,
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fp = os.path.join(save_dir, "interp_attention_temporal.png")
    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fp


def per_variable_contribution(scores, var_weights):
    """
    score    : (T, n_feats)        - 변수별 anomaly score
    var_weights : (T, T_enc, n_feats) - VSN encoder weight (시간 평균)

    반환:
        contribution : (T, n_feats) = score * mean_VSN_weight_over_lag
                       (정규화: 각 시점에서 합이 1이 되도록)
    """
    weight_t = var_weights.mean(axis=1)  # (T, n_feats)
    raw = scores * weight_t
    sums = raw.sum(axis=1, keepdims=True) + 1e-12
    return raw / sums


def plot_root_cause_at_alarms(
    scores, var_weights, feature_names, alarm_indices, save_dir, *, dpi, sample_indices, top_k=5, max_alarms=10
):
    """
    알람 시점들에 대해 root-cause Top-K 변수 막대그래프를 생성합니다.

    Args:
        scores         : (T, n_feats)
        var_weights    : (T, T_enc, n_feats)
        feature_names  : list[str]
        alarm_indices  : list[int]
        max_alarms     : 너무 많은 알람을 모두 그리지 않도록 제한
    """
    os.makedirs(save_dir, exist_ok=True)

    contribution = per_variable_contribution(scores, var_weights)  # (T, F)
    F = contribution.shape[1]
    top_k = min(top_k, F)
    feature_names = feature_names if feature_names else [f"Var_{i}" for i in range(F)]

    if not alarm_indices:
        return None

    sample_alarms = alarm_indices[:max_alarms]
    n = len(sample_alarms)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, t_idx in zip(axes, sample_alarms, strict=False):
        contrib_t = contribution[t_idx]
        idx_sorted = np.argsort(contrib_t)[::-1][:top_k]
        ax.barh(range(top_k), contrib_t[idx_sorted][::-1], color="crimson", alpha=0.8)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([feature_names[i] for i in idx_sorted[::-1]], fontsize=8)
        ax.set_xlabel("Contribution (normalized)")
        ax.set_title(f"Alarm @ t={sample_indices[t_idx]} - Top {top_k} Root Causes", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    # 사용 안 한 axes 숨기기
    for ax in axes[len(sample_alarms) :]:
        ax.axis("off")

    fig.suptitle("TFT Alarm Root Cause Analysis (Score x Variable Weight)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fp = os.path.join(save_dir, "interp_alarm_root_cause.png")
    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_quantile_forecast(target, median, low_q, high_q, feature_names, save_dir, *, dpi, quantiles, max_points=2000):
    """원본 단위의 입력·중앙값·분위수 구간을 기존 변수별 파일명으로 그린다."""
    os.makedirs(save_dir, exist_ok=True)
    y_unit_label = "Value (original unit)"
    x = np.arange(len(target))

    T, F = target.shape
    if T > max_points:
        step = int(np.ceil(T / max_points))
        x = x[::step]
        target = target[::step]
        median = median[::step]
        low_q = low_q[::step]
        high_q = high_q[::step]
        T = target.shape[0]

    for i in range(F):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.fill_between(
            x, low_q[:, i], high_q[:, i], alpha=0.3, color="orange", label=f"[{quantiles[0]}, {quantiles[2]}] band"
        )
        ax.plot(x, median[:, i], color="red", linewidth=0.8, label=f"Median (q={quantiles[1]})")
        ax.plot(x, target[:, i], color="blue", linewidth=0.6, alpha=0.8, label="Actual")
        ax.set_xlabel("Time Step")
        ax.set_ylabel(y_unit_label)
        var_name = feature_names[i] if i < len(feature_names) else f"Var_{i}"
        ax.set_title(f"TFT Quantile Forecast: {var_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fp = os.path.join(save_dir, f"interp_qforecast_{i:02d}_{safe_filename_component(var_name)}.png")
        fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_attention_details(weights, name, directory, *, dpi):
    """표본·헤드·쿼리·키 배열에서 기존 TranAD Attention 해석 그림을 저장한다."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    per_head = weights.mean(axis=0)
    average = per_head.mean(axis=0)
    heads, queries, keys = per_head.shape
    key_labels = [f"t-{keys - 1 - i}" if i < keys - 1 else "t" for i in range(keys)]
    query_labels = [f"t-{queries - 1 - i}" if i < queries - 1 else "t" for i in range(queries)]

    # 쿼리 길이가 1인 디코더에서도 키 길이를 별도로 사용한다.
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(average, cmap="viridis", aspect="auto", interpolation="nearest", vmin=0, vmax=average.max())
    ax.set_xticks(range(keys), key_labels, rotation=45, ha="right")
    ax.set_yticks(range(queries), query_labels)
    ax.set(
        xlabel="Source Time Step (Key)",
        ylabel="Target Time Step (Query)",
        title=f"Average Attention Map\n({name}, {heads} heads, {len(weights)} samples)",
    )
    if max(queries, keys) <= 20:
        for query in range(queries):
            for key in range(keys):
                value = average[query, key]
                ax.text(
                    key,
                    query,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > average.max() * 0.6 else "black",
                )
    fig.colorbar(im, ax=ax, label="Attention Weight")
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 헤드별 개별 Attention Map: 동일한 색상 범위로 비교한다.
    columns = min(4, heads)
    rows = int(np.ceil(heads / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False)
    fig.suptitle(f"Per-Head Attention Maps\n({name}, {len(weights)} samples)", fontsize=14, fontweight="bold")
    for head, ax in enumerate(axes.flat):
        if head >= heads:
            ax.axis("off")
            continue
        im = ax.imshow(
            per_head[head], cmap="viridis", aspect="auto", interpolation="nearest", vmin=0, vmax=per_head.max()
        )
        ax.set_title(f"Head {head}", fontsize=10, fontweight="bold")
        if max(queries, keys) <= 15:
            ax.set_xticks(range(keys), key_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(queries), query_labels, fontsize=7)
        ax.set(xlabel="Source (Key)", ylabel="Target (Query)")
    fig.tight_layout(rect=[0, 0, 0.92, 0.93])
    colorbar = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=colorbar, label="Attention Weight")
    fig.savefig(directory / f"{name}_per_head.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 열 방향 평균은 각 키 시점이 쿼리들로부터 받은 Attention을 나타낸다.
    temporal = average.mean(axis=0)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(
        range(keys), temporal, color=plt.cm.YlOrRd(temporal / (temporal.max() + 1e-10)), edgecolor="gray", linewidth=0.5
    )
    ax.set_xticks(range(keys), key_labels, rotation=45, ha="right")
    ax.set(
        xlabel="Time Step",
        ylabel="Mean Attention Weight (Column-wise)",
        title=f"Temporal Importance from Attention\n({name}, {len(weights)} samples)",
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(directory / f"{name}_temporal_importance.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
