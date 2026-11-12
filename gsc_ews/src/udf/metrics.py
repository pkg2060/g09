"""복원 점수와 라벨을 분리하여 탐지·변수 순위 지표를 계산한다."""

import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score


def detection_metrics(scores, labels, thresholds):
    """이진 판정으로 Precision/Recall/F1을, 연속 점수로 ROC-AUC를 계산한다."""
    scores, labels = np.asarray(scores), np.asarray(labels) > 0
    predicted = scores > thresholds
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "ROC_AUC": float(roc_auc_score(labels.ravel(), scores.ravel())) if np.unique(labels).size == 2 else None,
    }


def ranking_metrics(scores, labels, percentages=(100, 150)):
    """이상 변수의 Hit@k%와 NDCG@k%를 계산한다.

    이상 라벨이 있는 시점에서 실제 이상 변수 수의 k%만큼 상위 점수를 선택한다.
    """
    result = {}
    for percentage in percentages:
        hits, discounted = [], []
        for score, label in zip(scores, labels, strict=False):
            actual = set(np.flatnonzero(label > 0))
            if not actual:
                continue
            count = min(len(score), max(1, round(percentage * len(actual) / 100)))
            selected = set(np.argsort(score)[-count:])
            hits.append(len(actual & selected) / len(actual))
            discounted.append(
                float(ndcg_score((label > 0)[None, :], score[None, :], k=count)) if len(score) > 1 else 1.0
            )
        result[f"Hit@{percentage}%"] = float(np.mean(hits)) if hits else None
        result[f"NDCG@{percentage}%"] = float(np.mean(discounted)) if discounted else None
    return result


def point_adjust(scores, labels, threshold, segments):
    """라벨로 이상 구간 전체를 보정하는 별도 평가용 결과를 반환한다.

    이 보정은 실제 운전 알람에는 사용하지 않는다. 시간 단절에서 이상 구간을 끊는다.
    """
    predicted, actual = np.asarray(scores) > threshold, np.asarray(labels) > 0
    start = 0
    while start < len(actual):
        if not actual[start]:
            start += 1
            continue
        end = start + 1
        while end < len(actual) and actual[end] and segments[end] == segments[start]:
            end += 1
        predicted[start:end] = predicted[start:end].any()
        start = end
    return predicted
