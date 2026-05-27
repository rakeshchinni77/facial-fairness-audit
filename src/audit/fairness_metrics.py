"""Fairness metric utilities for initial audit reporting.

Overall accuracy is insufficient in biometric verification because a single
accuracy number can hide asymmetric error behavior. FAR and FRR are the core
operating-point metrics here because they expose the tradeoff between false
accepts and false rejects, which is exactly what downstream audit analysis
needs before mitigation work begins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfusionSummary:
    """Confusion-matrix and verification-rate summary."""

    tp: int
    tn: int
    fp: int
    fn: int
    far: float
    frr: float
    support: int
    positive_support: int
    negative_support: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": int(self.tp),
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "far": float(self.far),
            "frr": float(self.frr),
            "support": int(self.support),
            "positive_support": int(self.positive_support),
            "negative_support": int(self.negative_support),
        }


def _to_numpy(values: Any, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array


def compute_confusion_summary(true_labels: Any, predicted_labels: Any) -> ConfusionSummary:
    """Compute stable FAR/FRR and confusion counts from labels and predictions."""

    labels = _to_numpy(true_labels, dtype=np.int32)
    predictions = _to_numpy(predicted_labels, dtype=np.int32)
    if labels.shape[0] != predictions.shape[0]:
        raise ValueError("True labels and predicted labels must have the same length")

    positives = labels == 1
    negatives = labels == 0
    predicted_positive = predictions == 1
    predicted_negative = ~predicted_positive

    tp = int(np.sum(predicted_positive & positives))
    tn = int(np.sum(predicted_negative & negatives))
    fp = int(np.sum(predicted_positive & negatives))
    fn = int(np.sum(predicted_negative & positives))

    positive_support = int(np.sum(positives))
    negative_support = int(np.sum(negatives))
    support = int(labels.shape[0])

    # FAR and FRR are the verification metrics that reveal bias at the operating
    # threshold; a subgroup can look “accurate” while still being over-accepted or
    # over-rejected relative to other groups.
    far = fp / max(negative_support, 1)
    frr = fn / max(positive_support, 1)

    logger.info(
        "Computed verification summary | tp=%s tn=%s fp=%s fn=%s far=%.6f frr=%.6f",
        tp,
        tn,
        fp,
        fn,
        far,
        frr,
    )
    return ConfusionSummary(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        far=float(far),
        frr=float(frr),
        support=support,
        positive_support=positive_support,
        negative_support=negative_support,
    )


def summarize_metric_rows(true_labels: Any, predicted_labels: Any) -> dict[str, Any]:
    """Return a JSON-serializable metric summary."""

    return compute_confusion_summary(true_labels, predicted_labels).as_dict()
