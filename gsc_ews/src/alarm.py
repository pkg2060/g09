"""이상 점수와 임계값에 Debouncing·쿨다운·알람 억제 조건을 적용한다."""

import operator

import numpy as np
import pandas as pd

CRV_BANDS = (
    ("ALARM_LOW", "alarm_low", "CRV_ALARM_LOW", "CRV - Alarm Low"),
    ("PRE_ALARM_LOW", "pre_alarm_low", "CRV_PRE_ALARM_LOW", "CRV - Pre Alarm Low"),
    ("PRE_ALARM_HIGH", "pre_alarm_high", "CRV_PRE_ALARM_HIGH", "CRV - Pre Alarm High"),
    ("ALARM_HIGH", "alarm_high", "CRV_ALARM_HIGH", "CRV - Alarm High"),
)


class AlarmGenerator:
    """알람 상태와 이벤트를 하나의 판정에서 생성하여 CSV와 로그의 차이를 막는다."""

    def __init__(self, thresholds, feature_names, settings):
        """명시적으로 전달한 임계값과 설정으로 상태를 초기화한다."""
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.feature_names = list(feature_names)
        self.settings = settings
        self.consecutive_counts = np.zeros(len(feature_names), dtype=int)
        self.last_alarm_time = {}
        self.crv_counts = np.zeros((len(feature_names), len(CRV_BANDS)), dtype=int)
        self.last_crv_alarm_time = {}
        self.previous_segment = None
        self.crv_limits = {}
        self.operations = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }
        if self.thresholds.shape != (len(feature_names),):
            raise ValueError("임계값과 특성 수가 다릅니다")
        if settings["min_variables"] > len(feature_names):
            raise ValueError("alarm.min_variables exceeds the feature count")
        for condition in settings["inhibit_conditions"]:
            if condition["variable"] not in feature_names:
                raise ValueError(f"억제 조건에 없는 특성이 있습니다: {condition['variable']}")
        if settings["crv"]["enabled"]:
            for variable, limits in settings["crv"]["per_variable"].items():
                if variable not in feature_names:
                    raise ValueError(f"CRV 설정에 없는 특성이 있습니다: {variable}")
                self.crv_limits[self.feature_names.index(variable)] = limits

    def inhibited(self, raw_values):
        """원본 스케일의 변수값으로 EWS·계기 이상 알람의 억제 조건을 평가한다."""
        if not self.settings["inhibit_enabled"]:
            return False
        results = [
            self.operations[item["operator"]](raw_values[self.feature_names.index(item["variable"])], item["value"])
            for item in self.settings["inhibit_conditions"]
        ]
        return bool(results) and (all(results) if self.settings["inhibit_logic"] == "all" else any(results))

    def reset(self):
        """공정 정지나 시계열 단절 뒤 연속 알람 판정을 새로 시작한다."""
        self.consecutive_counts[:] = 0
        self.crv_counts[:] = 0
        self.previous_segment = None

    def instrument_fault_events(self, raw_values, timestamp, *, include_zero):
        """운전 중 원본 결측값과 선택적 0값을 계기 이상 이벤트로 변환한다."""
        values = np.asarray(raw_values, dtype=float)
        if values.shape != (len(self.feature_names),):
            raise ValueError("계기 이상 판정값과 특성 수가 다릅니다")
        timestamp = pd.Timestamp(timestamp)
        if pd.isna(timestamp):
            raise ValueError("계기 이상 알람에는 유효한 타임스탬프가 필요합니다")
        if self.inhibited(values):
            return []
        level = (
            "WARNING"
            if "WARNING" in self.settings["levels"]
            else min(self.settings["levels"], key=self.settings["levels"].get)
        )
        events = []
        for name, value in zip(self.feature_names, values, strict=True):
            if np.isfinite(value) and (value != 0 or not include_zero):
                continue
            reason = "0값" if np.isfinite(value) else "NA 또는 비수치"
            events.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "variable": name,
                    "level": level,
                    "event_type": "INSTRUMENT_FAULT",
                    "message": f"[{level}] {name} 계기 이상: {reason}",
                }
            )
        return events

    def process(self, scores, raw_values, timestamps, segments):
        """배치마다 이어지는 상태를 유지하며 실제 시각으로 쿨다운을 계산한다.

        억제 시점과 시간 단절에서 연속 횟수를 리셋한다. 알람 상태는 쿨다운 중에도
        유지하고, 이벤트만 반복 기록하지 않는다. 최소 변수 조건은 전체 알람에 적용한다.
        """
        scores, raw_values = np.asarray(scores), np.asarray(raw_values)
        if scores.shape != raw_values.shape or scores.shape[1] != len(self.feature_names):
            raise ValueError("점수와 원본값의 shape가 일치해야 합니다")
        if len(timestamps) != len(scores) or len(segments) != len(scores):
            raise ValueError("타임스탬프·구간·점수의 길이가 다릅니다")
        active = np.zeros_like(scores, dtype=bool)
        levels = np.full(scores.shape, "NORMAL", dtype=object)
        overall = np.zeros(len(scores), dtype=bool)
        inhibited = np.zeros(len(scores), dtype=bool)
        events = []
        crv_active = np.zeros_like(scores, dtype=bool)
        crv_overall = np.zeros(len(scores), dtype=bool)
        crv_directions = np.full(scores.shape, "NORMAL", dtype=object)
        ordered_levels = sorted(self.settings["levels"].items(), key=lambda item: item[1])
        for row, (score, raw, timestamp, segment) in enumerate(
            zip(scores, raw_values, timestamps, segments, strict=False)
        ):
            timestamp = pd.Timestamp(timestamp)
            if pd.isna(timestamp):
                raise ValueError("알람에는 유효한 타임스탬프가 필요합니다")
            if segment != self.previous_segment:
                self.consecutive_counts[:] = 0
                self.crv_counts[:] = 0
            self.previous_segment = segment
            inhibited[row] = self.inhibited(raw)
            new_events = []
            if inhibited[row]:
                # 억제 해제 후 Debouncing이 새로 시작되도록 연속 횟수도 리셋한다.
                self.consecutive_counts[:] = 0
            else:
                self.consecutive_counts = np.where(score > self.thresholds, self.consecutive_counts + 1, 0)
                active[row] = self.consecutive_counts >= self.settings["consecutive_count"]
                overall[row] = active[row].sum() >= self.settings["min_variables"]
                ratios = score / np.maximum(self.thresholds, 1e-12)
                for level, multiplier in ordered_levels:
                    # 낮은 레벨부터 갱신하여 높은 레벨이 마지막에 남도록 한다.
                    levels[row, active[row] & (ratios >= multiplier)] = level
                for index in np.flatnonzero(active[row]):
                    key = (int(index), levels[row, index])
                    previous = self.last_alarm_time.get(key)
                    if (
                        previous is not None
                        and (timestamp - previous).total_seconds() < self.settings["cooldown_seconds"]
                    ):
                        continue
                    new_events.append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "variable": self.feature_names[index],
                            "level": levels[row, index],
                            "event_type": "MODEL_ANOMALY",
                            "score": float(score[index]),
                            "threshold": float(self.thresholds[index]),
                            "ratio": float(ratios[index]),
                            "consecutive_count": int(self.consecutive_counts[index]),
                        }
                    )
                    self.last_alarm_time[key] = timestamp
                if overall[row] and new_events:
                    highest = max(levels[row, active[row]], key=self.settings["levels"].__getitem__)
                    new_events.insert(
                        0,
                        {
                            "timestamp": timestamp.isoformat(),
                            "variable": "OVERALL",
                            "level": highest,
                            "event_type": "MODEL_ANOMALY",
                            "anomalous_count": int(active[row].sum()),
                        },
                    )
            for index, limits in self.crv_limits.items():
                selected = None
                if limits["alarm_low"] is not None and raw[index] < limits["alarm_low"]:
                    selected = 0
                elif limits["pre_alarm_low"] is not None and raw[index] < limits["pre_alarm_low"]:
                    selected = 1
                elif limits["alarm_high"] is not None and raw[index] > limits["alarm_high"]:
                    selected = 3
                elif limits["pre_alarm_high"] is not None and raw[index] > limits["pre_alarm_high"]:
                    selected = 2
                next_count = 0 if selected is None else self.crv_counts[index, selected] + 1
                self.crv_counts[index] = 0
                if selected is not None:
                    self.crv_counts[index, selected] = next_count
                if selected is None or self.crv_counts[index, selected] < self.settings["consecutive_count"]:
                    continue
                band, limit_key, event_type, message = CRV_BANDS[selected]
                crv_active[row, index] = True
                crv_directions[row, index] = band
                key = (int(index), band, event_type)
                previous = self.last_crv_alarm_time.get(key)
                if previous is not None and (timestamp - previous).total_seconds() < self.settings["cooldown_seconds"]:
                    continue
                new_events.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "variable": self.feature_names[index],
                        "level": band,
                        "event_type": event_type,
                        "raw_value": float(raw[index]),
                        "crv_limit": limits[limit_key],
                        "consecutive_count": int(self.crv_counts[index, selected]),
                        "message": message,
                    }
                )
                self.last_crv_alarm_time[key] = timestamp
            crv_overall[row] = crv_active[row].any()
            events.extend(new_events)
        return {
            "per_variable": active,
            "overall": overall,
            "levels": levels,
            "inhibited": inhibited,
            "crv_per_variable": crv_active,
            "crv_overall": crv_overall,
            "crv_directions": crv_directions,
            "events": events,
        }

    def state_dict(self):
        """온라인 재시작을 위한 연속 감지·쿨다운 상태를 반환한다."""
        return {
            "consecutive_counts": self.consecutive_counts.tolist(),
            "crv_counts": self.crv_counts.tolist(),
            "previous_segment": self.previous_segment,
            "last_alarm_time": [
                [index, level, timestamp.isoformat()] for (index, level), timestamp in self.last_alarm_time.items()
            ],
            "last_crv_alarm_time": [
                [index, direction, level, timestamp.isoformat()]
                for (index, direction, level), timestamp in self.last_crv_alarm_time.items()
            ],
        }

    def load_state_dict(self, state):
        """현재 특성·레벨 설정과 일치하는 알람 상태만 복원한다."""
        counts = np.asarray(state["consecutive_counts"], dtype=int)
        if counts.shape != self.consecutive_counts.shape or (counts < 0).any():
            raise ValueError("저장된 알람 상태가 특성과 맞지 않습니다")
        times = {}
        crv_times = {}
        for index, level, timestamp in state["last_alarm_time"]:
            if not 0 <= index < len(counts) or level not in self.settings["levels"]:
                raise ValueError("저장된 알람 변수·레벨이 잘못되었습니다")
            times[(index, level)] = pd.Timestamp(timestamp)
        crv_counts = np.asarray(state.get("crv_counts", np.zeros_like(self.crv_counts)), dtype=int)
        if crv_counts.shape != self.crv_counts.shape or (crv_counts < 0).any():
            raise ValueError("저장된 CRV 알람 상태가 특성과 맞지 않습니다")
        valid_bands = {band: event_type for band, _, event_type, _ in CRV_BANDS}
        for index, band, event_type, timestamp in state.get("last_crv_alarm_time", []):
            if not 0 <= index < len(counts) or valid_bands.get(band) != event_type:
                raise ValueError("저장된 CRV 알람 변수·단계가 잘못되었습니다")
            crv_times[(index, band, event_type)] = pd.Timestamp(timestamp)
        self.consecutive_counts = counts
        self.crv_counts = crv_counts
        self.previous_segment = state["previous_segment"]
        self.last_alarm_time = times
        self.last_crv_alarm_time = crv_times
