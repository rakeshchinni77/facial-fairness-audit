"""Cross-group audit utilities.

Cross-group evaluation matters because pairwise demographic interactions can
show a different failure pattern than single-subgroup reporting. A system can be
acceptable within each subgroup yet still over-accept or over-reject specific
cross-demographic pairings, which is why these comparisons are tracked
separately from the main subgroup summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.audit.fairness_metrics import compute_confusion_summary
from src.audit.subgroup_evaluator import extract_pair_subgroups


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossGroupResult:
    pair_key: str
    far: float
    frr: float
    support: int
    tp: int
    tn: int
    fp: int
    fn: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "far": float(self.far),
            "frr": float(self.frr),
            "support": int(self.support),
            "tp": int(self.tp),
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),
        }


def _cross_group_key(left: str, right: str) -> str:
    return f"{left}__vs__{right}"


def analyze_cross_groups(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute FAR/FRR for cross-demographic pairings.

    Rows are grouped by the two demographic values participating in the pair.
    Same-subgroup rows are skipped because they belong to the primary subgroup
    summary, not the cross-group comparison layer.
    """

    if frame.empty:
        return {}

    pair_frame = extract_pair_subgroups(frame)
    results: dict[str, dict[str, Any]] = {}
    cross_group_frame = pair_frame[pair_frame["subgroup_left"] != pair_frame["subgroup_right"]]
    if cross_group_frame.empty:
        return results

    cross_group_frame = cross_group_frame.copy()
    cross_group_frame["cross_group_key"] = [
        _cross_group_key(str(left), str(right)) for left, right in zip(cross_group_frame["subgroup_left"], cross_group_frame["subgroup_right"])
    ]

    for pair_key, pair_rows in cross_group_frame.groupby("cross_group_key", dropna=False):
        summary = compute_confusion_summary(pair_rows["true_label"], pair_rows["predicted_label"])
        result = CrossGroupResult(
            pair_key=str(pair_key),
            far=summary.far,
            frr=summary.frr,
            support=summary.support,
            tp=summary.tp,
            tn=summary.tn,
            fp=summary.fp,
            fn=summary.fn,
        )
        results[str(pair_key)] = result.as_dict()
        logger.info(
            "Cross-group evaluated | pair=%s | support=%s | far=%.6f | frr=%.6f",
            pair_key,
            result.support,
            result.far,
            result.frr,
        )
    return results
