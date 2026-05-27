"""Subgroup audit evaluation utilities.

Subgroup auditing matters because an aggregate verification metric can bury a
large disparity between demographic slices. The point of this layer is to make
those slices visible before any mitigation or recalibration is attempted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.audit.fairness_metrics import compute_confusion_summary


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubgroupEvaluationResult:
    subgroup: str
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


def _subgroup_column(frame: pd.DataFrame) -> str:
    if "subgroup" in frame.columns:
        return "subgroup"
    if {"subgroup_a", "subgroup_b"}.issubset(frame.columns):
        return "subgroup_a"
    if {"group_a", "group_b"}.issubset(frame.columns):
        return "group_a"
    raise ValueError("No subgroup column found in audit dataframe")


def _pair_groups_from_row(row: pd.Series) -> tuple[str, str]:
    candidates = [
        ("subgroup_a", "subgroup_b"),
        ("group_a", "group_b"),
        ("demographic_a", "demographic_b"),
    ]
    for left_key, right_key in candidates:
        if left_key in row.index and right_key in row.index:
            left = str(row[left_key])
            right = str(row[right_key])
            return left, right

    subgroup_value = str(row.get("subgroup", "unknown"))
    if "__vs__" in subgroup_value:
        left, right = subgroup_value.split("__vs__", 1)
        return left.strip(), right.strip()
    if " vs " in subgroup_value:
        left, right = subgroup_value.split(" vs ", 1)
        return left.strip(), right.strip()
    return subgroup_value, subgroup_value


def _expand_to_single_subgroup_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pair rows into one row per participating subgroup.

    This keeps subgroup reporting meaningful for mixed-demographic pairs by
    counting each participating demographic slice, not just the left-hand side.
    """

    if frame.empty:
        return frame.copy()

    expanded_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        left_group, right_group = _pair_groups_from_row(row)
        row_dict = row.to_dict()
        expanded_rows.append({**row_dict, "audited_subgroup": left_group})
        if right_group != left_group:
            expanded_rows.append({**row_dict, "audited_subgroup": right_group})
    return pd.DataFrame(expanded_rows)


def evaluate_subgroups(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute FAR/FRR and support for each subgroup present in the audit set."""

    if frame.empty:
        return {}

    if {"subgroup_a", "subgroup_b"}.issubset(frame.columns) or {"group_a", "group_b"}.issubset(frame.columns):
        expanded_frame = _expand_to_single_subgroup_rows(frame)
        subgroup_key = "audited_subgroup"
    else:
        expanded_frame = frame.copy()
        subgroup_key = _subgroup_column(frame)

    results: dict[str, dict[str, Any]] = {}
    for subgroup_name, subgroup_frame in expanded_frame.groupby(subgroup_key, dropna=False):
        summary = compute_confusion_summary(subgroup_frame["true_label"], subgroup_frame["predicted_label"])
        result = SubgroupEvaluationResult(
            subgroup=str(subgroup_name),
            far=summary.far,
            frr=summary.frr,
            support=summary.support,
            tp=summary.tp,
            tn=summary.tn,
            fp=summary.fp,
            fn=summary.fn,
        )
        results[str(subgroup_name)] = result.as_dict()
        logger.info(
            "Subgroup evaluated | subgroup=%s | support=%s | far=%.6f | frr=%.6f",
            subgroup_name,
            result.support,
            result.far,
            result.frr,
        )
    return results


def extract_pair_subgroups(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach pair-level subgroup labels for cross-group analysis."""

    if frame.empty:
        return frame.copy()

    pair_frame = frame.copy()
    pair_groups = pair_frame.apply(_pair_groups_from_row, axis=1)
    pair_frame["subgroup_left"] = [left for left, _ in pair_groups]
    pair_frame["subgroup_right"] = [right for _, right in pair_groups]
    return pair_frame
