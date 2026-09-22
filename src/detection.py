"""Tag-aligned anomaly scores, process-stop rules, thresholds and honest metrics.

Arrays are [time, feature] in the caller's saved feature order. This module never
uses labels to change scores, thresholds, predictions or process-stop decisions.
Metric dictionaries contain Python scalars/None and can be serialized as JSON.

POT reference: https://github.com/imperial-qore/TranAD/blob/main/src/spot.py
and https://github.com/imperial-qore/TranAD/blob/main/src/pot.py (read 2026-09-21).
``spot`` uses the configured official SPOT implementation's Grimshaw
initialization and returns its fixed extreme quantile. ``pot`` independently
fits a GPD using SciPy MLE. Neither mode performs streaming updates, dataset
multipliers or label-assisted point adjustment. These differences are explicit.
"""
from __future__ import annotations

import operator
import importlib.util
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.stats import genpareto


def _matrix(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must have shape [N, F], with F > 0")
    if np.isinf(result).any():
        raise ValueError(f"{name} contains infinity")
    return result


def _names(feature_names, count):
    names = list(feature_names)
    if len(names) != count or len(set(names)) != count:
        raise ValueError("feature_names must be unique and match the feature count")
    return names


def _mask(mask, shape):
    if mask is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(mask)
    if result.dtype != np.bool_:
        raise ValueError("validity mask must be boolean; NaN is not a valid mask")
    if result.shape == (shape[0],):
        return np.broadcast_to(result[:, None], shape)
    if result.shape != shape:
        raise ValueError("validity mask must have shape [N] or [N, F]")
    return result


def point_scores(actual_scaled, recon_scaled):
    """Return absolute error per timestamp/Tag, in the supplied scale.

Do not average over time here: one Tag at one timestamp must trigger an alarm.
Unavailable reconstructions remain NaN, never zero. Overall point MAE can be
computed separately as ``scores.mean(axis=1)`` (or with an explicit valid mask).
"""
    actual = _matrix(actual_scaled, "actual_scaled")
    recon = _matrix(recon_scaled, "recon_scaled")
    if actual.shape != recon.shape:
        raise ValueError("actual_scaled and recon_scaled shapes differ")
    return np.abs(actual - recon)


def apply_release_hold(raw_stop, timestamps, hold_minutes):
    """현재/과거 센서 정지만 사용해 재기동 해제를 지연한다. gap에서 초기화.

    hold=1이면 stop[t] = raw[t] OR (정확히1분전 raw[t-1]). 미래 시점은
    사용하지 않는다. hold>1도 연속 1분 시각 구간 안에서만 전달한다.
    """
    stopped = np.asarray(raw_stop)
    if stopped.ndim != 1 or stopped.dtype != np.bool_:
        raise ValueError('raw_stop must be a one-dimensional boolean array')
    if isinstance(hold_minutes, (bool, np.bool_)):
        raise ValueError('release_hold_minutes must be a nonnegative integer')
    try:
        hold = int(hold_minutes)
        valid = float(hold_minutes) == hold and hold >= 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError('release_hold_minutes must be a nonnegative integer')
    if hold == 0:
        return stopped.copy()
    if timestamps is None:
        raise ValueError('release_hold_minutes > 0 requires actual timestamps; row shifts are forbidden')
    times = np.asarray(timestamps, dtype='datetime64[ns]')
    if times.shape != stopped.shape or np.isnat(times).any():
        raise ValueError('Release hold needs one valid timestamp per sensor row')
    if len(times) > 1 and (np.diff(times) <= np.timedelta64(0, 'ns')).any():
        raise ValueError('Release hold timestamps must be strictly increasing')
    if not len(times):
        return stopped.copy()
    # A 30-second interval, 2-minute interval or larger gap all start a fresh
    # segment: row adjacency alone never establishes a one-minute predecessor.
    rows = np.arange(len(times))
    segment_start = np.maximum.accumulate(np.where(
        np.r_[True, np.diff(times) != np.timedelta64(1, 'm')], rows, 0))
    last_raw_stop = np.maximum.accumulate(np.where(stopped, rows, -1))
    return stopped | ((last_raw_stop >= segment_start) & (rows-last_raw_stop <= hold))


def process_stop(df, alarm_cfg, timestamps=None):
    """Evaluate raw-unit sensor conditions, never Inhibit or Alarm_ labels.

Config keys: enabled (True), mode ('or' or 'and'), conditions (list of
{tag, op, value}), release_hold_minutes (0 if omitted). Positive release hold
requires real timestamps and only spans uninterrupted one-minute sensor rows.
Missing/non-numeric/non-finite condition sensors raise clearly
rather than silently disabling alarms or guessing an operating state.
"""
    cfg = dict(alarm_cfg or {})
    if not cfg.get("enabled", True):
        return np.zeros(len(df), dtype=bool)
    conditions = cfg.get("conditions", [
        {"tag": "PDC4303_7.PV", "op": "<=", "value": 0.2},
        {"tag": "PPVI43032A.PV", "op": "<=", "value": 2.5},
    ])
    mode = str(cfg.get("mode", "or")).lower()
    if mode not in {"and", "or"}:
        raise ValueError("process-stop mode must be 'and' or 'or'")
    if not conditions:
        raise ValueError("enabled process-stop logic requires at least one condition")
    operators = {"<": operator.lt, "<=": operator.le, ">": operator.gt,
                 ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
                 "≤": operator.le, "≥": operator.ge}
    masks = []
    for condition in conditions:
        tag = str(condition["tag"])
        if tag.casefold().startswith(("alarm_", "inhibit")):
            raise ValueError(f"Evaluation truth column {tag!r} cannot gate alarms")
        if tag not in df:
            raise ValueError(f"Missing process-stop sensor: {tag}")
        op = condition["op"]
        if op not in operators:
            raise ValueError(f"Unsupported process-stop operator: {op}")
        try:
            values = np.asarray(df[tag], dtype=float)
            limit = float(condition["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Process-stop sensor/limit must be numeric: {tag}") from exc
        if values.shape != (len(df),) or not np.isfinite(values).all() or not np.isfinite(limit):
            raise ValueError(f"Process-stop sensor/limit is missing or non-finite: {tag}")
        masks.append(operators[op](values, limit))
    reduce = np.logical_or.reduce if mode == "or" else np.logical_and.reduce
    raw_stop = np.asarray(reduce(masks), dtype=bool)
    return apply_release_hold(raw_stop, timestamps, cfg.get('release_hold_minutes', 0))


def generate_alarms(scores, thresholds, stop):
    """Return (Tag alarms [N,F], Overall alarm [N]) using strict ``>``.

NaN scores represent unavailable inference and produce no alarm; the caller
must exclude these points from metrics using a validity mask. There is no
debouncing, minimum duration, label-based point adjustment or hidden smoothing.
"""
    scores = _matrix(scores, "scores")
    limits = np.asarray(thresholds, dtype=float)
    stopped = np.asarray(stop)
    if limits.shape != (scores.shape[1],) or not np.isfinite(limits).all() or (limits < 0).any():
        raise ValueError("thresholds must be finite non-negative values of shape [F]")
    if stopped.shape != (scores.shape[0],) or stopped.dtype != np.bool_:
        raise ValueError("stop must be a boolean array of shape [N]")
    if (scores[np.isfinite(scores)] < 0).any():
        raise ValueError("MAE anomaly scores cannot be negative")
    alarms = (scores > limits[None, :]) & np.isfinite(scores) & ~stopped[:, None]
    return alarms, alarms.any(axis=1)


def _pot_threshold(values, cfg):
    """Fit fixed upper-tail POT quantile from reference scores only."""
    level = float(cfg.get("level", cfg.get("spot_level", 0.98)))
    risk = float(cfg.get("q", cfg.get("spot_q", 1e-4)))
    minimum = int(cfg.get("min_exceedances", 10))
    if not (0 < level < 1) or not (0 < risk < 1 - level):
        raise ValueError("POT requires 0 < level < 1 and 0 < q < 1-level")
    if minimum < 3:
        raise ValueError("POT min_exceedances must be at least 3")
    initial = float(np.quantile(values, level))
    excess = values[values > initial] - initial
    if len(excess) < minimum:
        raise ValueError(f"POT has {len(excess)} tail exceedances; needs {minimum}. "
                         "Supply more reference data or explicitly lower level.")
    fraction = len(excess) / len(values)
    if risk >= fraction:
        raise ValueError("POT q must be below the empirical exceedance fraction")
    shape, _, scale = genpareto.fit(excess, floc=0)
    value = initial + genpareto.isf(risk / fraction, shape, loc=0, scale=scale)
    if not np.isfinite(value) or value < initial or scale <= 0:
        raise ValueError("POT tail fit did not produce a valid upper threshold")
    return float(value)


def _spot_threshold(values, cfg):
    """Official TranAD SPOT/Grimshaw initialization, frozen before Test.

Config ``spot_source_path`` must point to the official TranAD src/spot.py,
downloaded with its license and source revision retained by the caller. We
intentionally do not call run(): its dynamic=False behavior in the reference
may replace the calibrated threshold using the subsequent stream. A fixed
threshold vector cannot represent full dynamic streaming SPOT.
"""
    source_path = cfg.get("spot_source_path")
    if not source_path:
        raise ValueError("SPOT requires spot_source_path pointing to official TranAD src/spot.py; "
                         "use method='pot' explicitly for independent SciPy GPD MLE")
    source = Path(source_path)
    if not source.is_file():
        raise ValueError(f"Official SPOT source file does not exist: {source}")
    level = float(cfg.get("level", cfg.get("spot_level", 0.98)))
    risk = float(cfg.get("q", cfg.get("spot_q", 1e-4)))
    minimum = int(cfg.get("min_exceedances", 10))
    if not (0 < level < 1) or not (0 < risk < 1 - level) or minimum < 3:
        raise ValueError("SPOT needs 0 < level < 1, 0 < q < 1-level and min_exceedances >= 3")
    initial = np.sort(values)[int(level * len(values))]
    count = int(np.sum(values > initial))
    if count < minimum:
        raise ValueError(f"SPOT has {count} tail exceedances; needs {minimum}. "
                         "Supply more reference data or explicitly lower level.")
    spec = importlib.util.spec_from_file_location("ews_official_tranad_spot", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load official SPOT source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    detector = module.SPOT(risk)
    detector.fit(np.array(values, copy=True), np.empty(0, dtype=float))
    detector.initialize(level=level, min_extrema=False, verbose=False)
    value = float(detector.extreme_quantile)
    if not np.isfinite(value) or value < initial:
        raise ValueError("Official SPOT initialization did not produce a valid upper threshold")
    return value


def calculate_thresholds(reference_scores, threshold_cfg, feature_names):
    """Calculate a threshold vector in feature_names order, without Test input.

Manual: manual/default/manual_default (0.2) plus per_tag overrides. Manual
values always come from the current Config; saved threshold files are ignored.
Automatic: percentile (99.5), sigma (3 population standard deviations), or
    pot (SciPy GPD MLE), or spot (official Grimshaw initialization loaded from
    spot_source_path). Both are fixed calibration, level=.98, q=1e-4, with no
    dynamic Test updates. per_tag is manual-only.
Pass reference_split='train'/'validation'. Explicit Test provenance is rejected
unless allow_test_reference=True, in which case a development-evaluation warning
is emitted. The caller remains responsible for actual split provenance.
"""
    cfg = dict(threshold_cfg or {})
    names = list(feature_names)
    _names(names, len(names))
    if not names:
        raise ValueError("At least one feature is required")
    method = str(cfg.get("method", "manual")).lower()
    if method == "manual":
        default = cfg.get("manual", cfg.get("manual_default", cfg.get("default", 0.2)))
        overrides = cfg.get("per_tag", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("per_tag must map feature names to manual thresholds")
        unknown = set(overrides) - set(names)
        if unknown:
            raise ValueError(f"Manual thresholds reference unknown Tags: {sorted(unknown)}")
        values = np.array([float(overrides.get(name, default)) for name in names])
    else:
        reference = _matrix(reference_scores, "reference_scores")
        _names(names, reference.shape[1])
        if (reference[np.isfinite(reference)] < 0).any():
            raise ValueError("Reference MAE scores cannot be negative")
        split = str(cfg.get("reference_split", "unspecified")).lower()
        if split in {"test", "test.csv", "evaluation", "holdout"}:
            if not cfg.get("allow_test_reference", False):
                raise ValueError("Automatic threshold calibration cannot use Test/holdout scores")
            warnings.warn("Thresholds use Test/holdout: report development evaluation, "
                          "not independent final performance", RuntimeWarning, stacklevel=2)
        values = []
        for j, name in enumerate(names):
            column = reference[:, j]
            column = column[np.isfinite(column)]
            if not len(column):
                raise ValueError(f"No finite reference scores for {name}")
            if method == "percentile":
                percentile = float(cfg.get("percentile", 99.5))
                if not (0 < percentile <= 100):
                    raise ValueError("percentile must be in (0,100]")
                value = float(np.percentile(column, percentile))
            elif method in {"sigma", "3-sigma", "3sigma"}:
                multiplier = float(cfg.get("sigma", 3.0))
                if not np.isfinite(multiplier) or multiplier < 0:
                    raise ValueError("sigma must be finite and non-negative")
                value = float(column.mean() + multiplier * column.std(ddof=0))
            elif method == "pot":
                value = _pot_threshold(column, cfg)
            elif method == "spot":
                value = _spot_threshold(column, cfg)
            else:
                raise ValueError(f"Unknown threshold method: {method}")
            values.append(value)
        values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Thresholds must be finite and non-negative")
    return values


def _safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _counts_metrics(labels, alarms):
    true = np.asarray(labels, dtype=bool)
    predicted = np.asarray(alarms, dtype=bool)
    tp = int(np.count_nonzero(true & predicted))
    fp = int(np.count_nonzero(~true & predicted))
    fn = int(np.count_nonzero(true & ~predicted))
    tn = int(np.count_nonzero(~true & ~predicted))
    return {"n": len(true), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": _safe_div(tp + tn, len(true)),
            "precision": _safe_div(tp, tp + fp), "recall": _safe_div(tp, tp + fn),
            "f1": _safe_div(2 * tp, 2 * tp + fp + fn), "fpr": _safe_div(fp, fp + tn)}


def _event_metrics(labels, alarms, valid, times):
    """Events are contiguous valid label=1 runs, split by gaps/missing labels.

At a run's first valid positive observation, measure the delay to its first
actual alarm within that run. An excluded/unknown interval breaks the event;
we do not assume its hidden onset. No detection means delay=None, not zero.
"""
    positive = valid & (labels == 1)
    # Data are sampled each minute by contract. A gap > one minute splits an
    # event; no interpolation across deleted input rows or missing Labels.
    gap = np.zeros(len(times), dtype=bool)
    if len(times) > 1:
        gap[1:] = np.diff(times) > np.timedelta64(60, "s")
    events, delays = [], []
    i = 0
    while i < len(labels):
        if not positive[i]:
            i += 1
            continue
        start = i
        i += 1
        while i < len(labels) and positive[i] and not gap[i]:
            i += 1
        end = i
        detections = np.flatnonzero(alarms[start:end])
        detected = bool(len(detections))
        first = start + int(detections[0]) if detected else None
        delay = float((times[first] - times[start]) / np.timedelta64(1, "s")) if detected else None
        if detected:
            delays.append(delay)
        events.append({"start": np.datetime_as_string(times[start], unit="s"),
                       "end": np.datetime_as_string(times[end - 1], unit="s"),
                       "detected": detected, "delay_seconds": delay})
    return {"event_count": len(events), "detected_event_count": len(delays),
            "missed_event_count": len(events) - len(delays),
            "event_recall": _safe_div(len(delays), len(events)),
            "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
            "events": events}


def classification_metrics(labels, alarms, timestamps, feature_names, valid_mask=None):
    """Unadjusted point metrics plus event onset delay, with missing Labels excluded.

Overall metrics are MICRO metrics across valid timestamp/Tag pairs, not the
accuracy of Overall Alarm. Missing Labels (NaN) never become normal labels.
Undefined ratios are None. valid_mask may be [N] or [N,F], and should exclude
unavailable initial-window reconstructions. Process-stop inclusion is a caller
decision and must be reported; it is not silently removed here.
"""
    labels = _matrix(labels, "labels")
    predicted = np.asarray(alarms)
    if predicted.shape != labels.shape:
        raise ValueError("labels and alarms shapes differ")
    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("alarms must contain only booleans or 0/1")
    predicted = predicted.astype(bool)
    names = _names(feature_names, labels.shape[1])
    known = np.isfinite(labels)
    if not np.isin(labels[known], [0, 1]).all():
        raise ValueError("Known labels must be 0 or 1; missing labels must be NaN")
    try:
        times = np.asarray(timestamps, dtype="datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps must be parseable datetimes") from exc
    if times.shape != (len(labels),) or np.isnat(times).any():
        raise ValueError("timestamps must contain N valid datetimes")
    if len(times) > 1 and (np.diff(times) <= np.timedelta64(0, "ns")).any():
        raise ValueError("timestamps must be strictly increasing without duplicates")
    valid = known & _mask(valid_mask, labels.shape)
    by_tag = {}
    for j, name in enumerate(names):
        selected = valid[:, j]
        result = _counts_metrics(labels[selected, j], predicted[selected, j])
        result["missing_label_count"] = int((~known[:, j]).sum())
        result["excluded_count"] = int((~selected).sum())
        result.update(_event_metrics(labels[:, j], predicted[:, j], selected, times))
        by_tag[name] = result
    overall = _counts_metrics(labels[valid], predicted[valid])
    overall["missing_label_count"] = int((~known).sum())
    overall["excluded_count"] = int((~valid).sum())
    return {"per_tag": by_tag, "overall": overall,
            "semantics": {"overall": "micro across valid timestamp/Tag pairs",
                          "undefined_ratios": "null (zero denominator)",
                          "point_adjustment": False,
                          "missing_labels": "excluded, never imputed as normal",
                          "events": "valid positive runs; missing/excluded points and >60s gaps split runs",
                          "delay": "seconds to first alarm within run; missed event=null"}}


def _regression_row(actual, recon):
    n = len(actual)
    if not n:
        return {"n": 0, "mae": None, "mse": None, "rmse": None, "r2": None,
                "mape": None, "mape_n": 0, "mape_zero_excluded": 0}
    error = actual - recon
    mse = float(np.mean(error ** 2))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    nonzero = actual != 0
    return {"n": n, "mae": float(np.mean(np.abs(error))), "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(1 - np.sum(error ** 2) / denominator) if n >= 2 and denominator > 0 else None,
            "mape": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
            "mape_n": int(nonzero.sum()), "mape_zero_excluded": int((~nonzero).sum())}


def regression_metrics(actual, recon, feature_names, mask=None):
    """Finite paired-point metrics in caller's scale; MAPE is a percentage.

Overall MAE/MSE/RMSE/MAPE are MICRO across valid timestamp/Tag pairs. Overall
R2 is the arithmetic mean of defined per-Tag R2 values, avoiding the misleading
R2 obtained by pooling differently scaled sensors. Zero actuals are excluded
only from MAPE; they remain in every other metric. Constant-target/<2-point R2
is None, not a fabricated perfect score. NaN pairs and explicit masks exclude
points from all metrics, with counts reported.
"""
    actual = _matrix(actual, "actual")
    recon = _matrix(recon, "recon")
    if actual.shape != recon.shape:
        raise ValueError("actual and recon shapes differ")
    names = _names(feature_names, actual.shape[1])
    valid = np.isfinite(actual) & np.isfinite(recon) & _mask(mask, actual.shape)
    by_tag = {}
    for j, name in enumerate(names):
        row = _regression_row(actual[valid[:, j], j], recon[valid[:, j], j])
        row["excluded_count"] = int((~valid[:, j]).sum())
        by_tag[name] = row
    overall = _regression_row(actual[valid], recon[valid])
    defined = [row["r2"] for row in by_tag.values() if row["r2"] is not None]
    overall["r2"] = float(np.mean(defined)) if defined else None
    overall["r2_defined_tag_count"] = len(defined)
    overall["excluded_count"] = int((~valid).sum())
    return {"per_tag": by_tag, "overall": overall,
            "semantics": {"scale": "same as supplied arrays; caller must label raw/scaled",
                          "overall": "micro paired-point MAE/MSE/RMSE/MAPE; macro defined-tag R2",
                          "mape": "percent; exact-zero actuals excluded only from MAPE",
                          "undefined_r2": "null for constant targets or fewer than 2 points"}}
