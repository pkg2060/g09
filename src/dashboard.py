"""한국어 EWS 결과 대시보드. 모든 시계열 원본을 태그별 오프라인 JS로 보존한다.

이 모듈은 결과를 시각화할 뿐, 모델 추론/임계값/Alarm을 변경하지 않는다.
정답 Label과 Inhibit는 비교 및 채점에만 사용한다.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _native(value: Any) -> Any:
    """numpy/pandas 값과 날짜를 JSON으로 안전하게 변환한다."""
    if isinstance(value, Mapping):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return _native(value.tolist())
    if hasattr(value, "item"):
        return _native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(_native(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _sequence(value: Any) -> list:
    if value is None:
        return []
    native = _native(value)
    if not isinstance(native, list):
        raise ValueError("시계열 입력은 list 또는 numpy 배열이어야 합니다.")
    return native


def _matrix(value: Any, name: str, n: int, f: int, optional: bool = False) -> Any:
    if optional and (value is None or len(value) == 0):
        return None
    if value is None:
        raise ValueError(f"{name}: 필수 시계열이 없습니다.")
    # Preserve ndarray backing memory; convert only one tag at a time for export.
    shape = getattr(value, "shape", None)
    valid = tuple(shape) == (n, f) if shape is not None else (
        len(value) == n and all(isinstance(row, (list, tuple)) and len(row) == f for row in value)
    )
    if not valid:
        raise ValueError(f"{name}: 예상 shape은 ({n}, {f})이며 모든 시점·태그의 정렬이 같아야 합니다.")
    return value


def _column(matrix: Any, j: int, n: int) -> list:
    if matrix is None:
        return [None] * n
    if hasattr(matrix, "shape"):
        return _sequence(matrix[:, j])
    return _sequence([row[j] for row in matrix])


def _binary(value: Any) -> int | None:
    if value is None:
        return None
    if value in (0, False):
        return 0
    if value in (1, True):
        return 1
    raise ValueError(f"Alarm/Label/Inhibit는 0, 1 또는 null이어야 합니다: {value!r}")


def _comparison(prediction: list, truth: list) -> dict:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for p, t in zip(prediction, truth):
        p, t = _binary(p), _binary(t)
        if p is None or t is None:
            continue
        counts["tp" if p and t else "tn" if not p and not t else "fp" if p else "fn"] += 1
    total = sum(counts.values())
    def ratio(a: int, b: int) -> float | None:
        return a / b if b else None
    return dict(counts, evaluated=total, omitted=len(truth) - total,
                accuracy=ratio(counts["tp"] + counts["tn"], total),
                precision=ratio(counts["tp"], counts["tp"] + counts["fp"]),
                recall=ratio(counts["tp"], counts["tp"] + counts["fn"]),
                f1=ratio(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]),
                positive_rate=ratio(counts["tp"] + counts["fn"], total))


def render_dashboard(output_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    """dashboard.html과 dashboard_data/*.js를 저장하고 HTML Path를 반환한다.

    필수: tags, timestamps, actual[N,F], reconstructed[N,F], scores[N,F],
    thresholds[F], final_alarm[N,F]. 선택: labels[N,F], predicted_inhibit[N],
    inhibit_labels[N], metadata, metrics, history, audit_summary.

    한국어 metadata 권장 필드: title, description, value_unit, normalization,
    stop_rule, alarm_rule, selected_checkpoint, evaluation_note.
    실제/복원값은 입력 단위를 유지하며 MAE는 metrics의 정규화 MAE만 사용한다.
    원본 full-resolution 배열을 저장하며 화면 해상도만 동적으로 조정한다.
    """
    output_dir = Path(output_dir)
    tags = [str(tag) for tag in _sequence(payload.get("tags"))]
    times = _sequence(payload.get("timestamps"))
    n, f = len(times), len(tags)
    if not n or not f:
        raise ValueError("대시보드 생성에는 하나 이상의 시점과 태그가 필요합니다.")
    if len(set(tags)) != f:
        raise ValueError("동일한 Tag 이름이 중복되어 있습니다.")
    actual = _matrix(payload.get("actual"), "actual", n, f)
    recon = _matrix(payload.get("reconstructed"), "reconstructed", n, f)
    scores = _matrix(payload.get("scores"), "scores", n, f)
    alarms = _matrix(payload.get("final_alarm"), "final_alarm", n, f)
    labels = _matrix(payload.get("labels"), "labels", n, f, optional=True)
    thresholds = _sequence(payload.get("thresholds"))
    if len(thresholds) != f:
        raise ValueError(f"thresholds: 태그별 임계값 {f}개가 필요합니다.")
    predicted_stop = _sequence(payload.get("predicted_inhibit")) or [None] * n
    true_stop = _sequence(payload.get("inhibit_labels")) or [None] * n
    if len(predicted_stop) != n or len(true_stop) != n:
        raise ValueError("predicted_inhibit/inhibit_labels의 길이는 timestamps와 같아야 합니다.")
    predicted_stop = [_binary(v) for v in predicted_stop]
    true_stop = [_binary(v) for v in true_stop]
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "dashboard_data"
    data_dir.mkdir(exist_ok=True)
    tag_metrics = []
    # No downsampling occurs here: every source row is exported for every tag.
    for j, tag in enumerate(tags):
        alarm = [_binary(value) for value in _column(alarms, j, n)]
        truth = [_binary(value) for value in _column(labels, j, n)]
        comparison = _comparison(alarm, truth)
        comparison.update(tag=tag, threshold=thresholds[j], alarm_count=sum(v == 1 for v in alarm))
        tag_metrics.append(comparison)
        data = {"tag": tag, "actual": _column(actual, j, n),
                "reconstructed": _column(recon, j, n), "scores": _column(scores, j, n),
                "threshold": thresholds[j], "final_alarm": alarm, "labels": truth}
        (data_dir / f"tag_{j:04d}.js").write_text(f"window.EWS_TAG_READY({j},{_json(data)});\n", encoding="utf-8")
    common = {"timestamps": times, "predicted_inhibit": predicted_stop, "inhibit_labels": true_stop}
    (data_dir / "common.js").write_text(f"window.EWS_COMMON={_json(common)};\n", encoding="utf-8")
    metrics = _native(payload.get("metrics", {}))
    stop_applicable = metrics.get("stop_rule_status", {}).get("applicable", True)
    mae = next((metrics[key] for key in ("normalized_test_mae", "test_mae", "mae") if metrics.get(key) is not None), None)
    all_tag_labels_complete = all(m["evaluated"] == n for m in tag_metrics)
    # Match pipeline policy: compare known labels, expose missing coverage separately.
    tag_gate = all(m["evaluated"] > 0 and m["accuracy"] is not None and m["accuracy"] >= .95 for m in tag_metrics)
    mae_gate = isinstance(mae, (int, float)) and not isinstance(mae, bool) and math.isfinite(mae) and 0 <= mae < .02
    summary = {"metadata": _native(payload.get("metadata", {})), "metrics": metrics,
               "tags": tags, "n": n, "tag_metrics": tag_metrics, "normalized_mae": mae,
               "all_labels_complete": all_tag_labels_complete, "mae_pass": mae_gate,
               "tag_pass": tag_gate, "phase2_pass": mae_gate and tag_gate,
               "stop_metrics": _comparison(predicted_stop if stop_applicable else [None] * n, true_stop),
               "stop_rule_applicable": stop_applicable,
               "stop_truth_known_minutes": sum(value is not None for value in true_stop),
               "stop_truth_stop_minutes": sum(value == 1 for value in true_stop),
               "history": _native(payload.get("history", [])),
               "experiments": _native(payload.get("experiments", [])),
               "experiment_sources": _native(payload.get("experiment_sources", [])),
               "audit_summary": _native(payload.get("audit_summary", {}))}
    assets = Path(__file__).with_name("dashboard_assets")
    template = (assets / "dashboard.html").read_text(encoding="utf-8")
    css = (assets / "dashboard.css").read_text(encoding="utf-8")
    js = (assets / "dashboard.js").read_text(encoding="utf-8")
    shap_js = (assets / "shap.js").read_text(encoding="utf-8")
    title = summary["metadata"].get("title", "회전기계 EWS · 운전 이상징후 검증")
    page = template.replace("__TITLE__", html.escape(str(title))).replace("__STYLE__", css).replace("__SUMMARY__", _json(summary)).replace("__SCRIPT__", js).replace("__SHAP_SCRIPT__", shap_js)
    path = output_dir / "dashboard.html"
    path.write_text(page, encoding="utf-8")
    return path
