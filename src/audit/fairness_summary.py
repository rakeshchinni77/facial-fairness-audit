"""Disaggregated fairness summary generation for audit reporting.

Overall accuracy is misleading in biometric verification because a single score
can hide large subgroup gaps in false acceptance and false rejection behavior.
This layer therefore summarizes the already-computed audit outputs at the
subgroup and cross-group level without changing thresholds, embeddings, or the
underlying audit metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FairnessSummaryConfig:
    """Configuration for fairness summary generation."""

    initial_audit_path: Path = Path("results/initial_audit.json")
    cross_group_metrics_path: Path = Path("results/cross_group_metrics.json")
    output_path: Path = Path("results/fairness_summary.json")
    low_support_threshold: int = 5


def configure_logging() -> None:
    """Configure summary logging for standalone execution."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {json_path}")
    logger.info("Audit loading complete | path=%s", json_path)
    return payload


def _is_metadata_key(key: str) -> bool:
    return key in {"overall", "metadata"}


def _is_cross_group_key(key: str) -> bool:
    return "__vs__" in key


def _is_subgroup_key(key: str) -> bool:
    return not _is_metadata_key(key) and not _is_cross_group_key(key)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _subgroup_entries(initial_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    subgroup_entries: dict[str, dict[str, Any]] = {}
    for key, value in initial_audit.items():
        if not _is_subgroup_key(key):
            continue
        if not isinstance(value, dict):
            continue
        subgroup_entries[str(key)] = value
    logger.info("Subgroup parsing complete | subgroup_count=%s", len(subgroup_entries))
    return subgroup_entries


def _rank_groups_by_metric(groups: dict[str, dict[str, Any]], metric_name: str) -> list[tuple[str, dict[str, Any]]]:
    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, int, str]:
        group_name, metrics = item
        metric_value = _coerce_float(metrics.get(metric_name, 0.0))
        support = _coerce_int(metrics.get("support", 0))
        return (-metric_value, support, group_name)

    return sorted(groups.items(), key=sort_key)


def _low_support_warnings(groups: dict[str, dict[str, Any]], threshold: int) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for group_name, metrics in sorted(groups.items(), key=lambda item: item[0]):
        support = _coerce_int(metrics.get("support", 0))
        if support < threshold:
            warnings.append(
                {
                    "group": group_name,
                    "support": support,
                    # Small sample sizes can produce volatile FAR/FRR values, so
                    # these groups are flagged as statistically unstable rather
                    # than treated as equally reliable estimates.
                    "statistically_unstable": True,
                }
            )
    return warnings


def _fairness_risk_level(far_gap: float, frr_gap: float) -> str:
    # Large FAR/FRR gaps indicate demographic bias because different groups are
    # being accepted or rejected at meaningfully different rates under the same
    # operating threshold.
    if far_gap > 0.30 or frr_gap > 0.30:
        return "HIGH"
    if far_gap > 0.15:
        return "MEDIUM"
    return "LOW"


def _extract_overall_metrics(initial_audit: dict[str, Any]) -> dict[str, Any]:
    overall = initial_audit.get("overall", {})
    return {
        "far": round(_coerce_float(overall.get("far", 0.0)), 6),
        "frr": round(_coerce_float(overall.get("frr", 0.0)), 6),
        "total_pairs": _coerce_int(overall.get("support", 0)),
    }


def _largest_gap_summary(groups: dict[str, dict[str, Any]], metric_name: str) -> dict[str, Any]:
    if not groups:
        return {f"best_group_{metric_name}": 0.0, f"worst_group_{metric_name}": 0.0, "gap": 0.0}

    ranked = _rank_groups_by_metric(groups, metric_name)
    worst_group, worst_metrics = ranked[0]
    best_group, best_metrics = ranked[-1]
    worst_value = _coerce_float(worst_metrics.get(metric_name, 0.0))
    best_value = _coerce_float(best_metrics.get(metric_name, 0.0))
    return {
        f"best_group_{metric_name}": round(best_value, 6),
        f"worst_group_{metric_name}": round(worst_value, 6),
        "gap": round(worst_value - best_value, 6),
    }


def _worst_group_summary(groups: dict[str, dict[str, Any]], metric_name: str) -> dict[str, Any]:
    if not groups:
        return {"group": None, metric_name: 0.0, "support": 0}
    ranked = _rank_groups_by_metric(groups, metric_name)
    worst_group, worst_metrics = ranked[0]
    return {
        "group": worst_group,
        metric_name: round(_coerce_float(worst_metrics.get(metric_name, 0.0)), 6),
        "support": _coerce_int(worst_metrics.get("support", 0)),
    }


def _most_affected_demographics(groups: dict[str, dict[str, Any]]) -> list[str]:
    if not groups:
        return []

    def impact_score(item: tuple[str, dict[str, Any]]) -> tuple[float, int, str]:
        group_name, metrics = item
        far = _coerce_float(metrics.get("far", 0.0))
        frr = _coerce_float(metrics.get("frr", 0.0))
        support = _coerce_int(metrics.get("support", 0))
        return (-max(far, frr), support, group_name)

    ranked = sorted(groups.items(), key=impact_score)
    top_groups = [group_name for group_name, _ in ranked[:2]]
    logger.info("Most affected demographics selected | groups=%s", top_groups)
    return top_groups


def _cross_group_summary(cross_group_metrics: dict[str, Any]) -> dict[str, Any]:
    if not cross_group_metrics:
        return {"worst_cross_group": None, "worst_cross_group_far": 0.0, "worst_cross_group_frr": 0.0}

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int, str]:
        pair_name, metrics = item
        far = _coerce_float(metrics.get("far", 0.0))
        frr = _coerce_float(metrics.get("frr", 0.0))
        support = _coerce_int(metrics.get("support", 0))
        return (-max(far, frr), -far, support, pair_name)

    worst_pair, worst_metrics = sorted(
        ((key, value) for key, value in cross_group_metrics.items() if isinstance(value, dict)),
        key=sort_key,
    )[0]
    return {
        "worst_cross_group": worst_pair,
        "worst_cross_group_far": round(_coerce_float(worst_metrics.get("far", 0.0)), 6),
        "worst_cross_group_frr": round(_coerce_float(worst_metrics.get("frr", 0.0)), 6),
    }


def build_fairness_summary(initial_audit: dict[str, Any], cross_group_metrics: dict[str, Any], config: FairnessSummaryConfig) -> dict[str, Any]:
    """Build the fairness summary from the already computed audit artifacts."""

    subgroup_metrics = _subgroup_entries(initial_audit)
    overall_metrics = _extract_overall_metrics(initial_audit)

    worst_far_group = _worst_group_summary(subgroup_metrics, "far")
    worst_frr_group = _worst_group_summary(subgroup_metrics, "frr")
    largest_far_gap = _largest_gap_summary(subgroup_metrics, "far")
    largest_frr_gap = _largest_gap_summary(subgroup_metrics, "frr")
    most_affected_demographics = _most_affected_demographics(subgroup_metrics)
    cross_group_summary = _cross_group_summary(cross_group_metrics)
    low_support_warnings = _low_support_warnings(subgroup_metrics, config.low_support_threshold)

    fairness_risk_level = _fairness_risk_level(
        _coerce_float(largest_far_gap.get("gap", 0.0)),
        _coerce_float(largest_frr_gap.get("gap", 0.0)),
    )

    summary: dict[str, Any] = {
        "overall_metrics": overall_metrics,
        "worst_far_group": worst_far_group,
        "worst_frr_group": worst_frr_group,
        "largest_far_gap": largest_far_gap,
        "largest_frr_gap": largest_frr_gap,
        "most_affected_demographics": most_affected_demographics,
        "cross_group_summary": cross_group_summary,
        "fairness_risk_level": fairness_risk_level,
        "low_support_warnings": low_support_warnings,
    }
    logger.info(
        "Fairness summary generated | far_gap=%.6f | frr_gap=%.6f | risk=%s",
        _coerce_float(largest_far_gap.get("gap", 0.0)),
        _coerce_float(largest_frr_gap.get("gap", 0.0)),
        fairness_risk_level,
    )
    return summary


def save_fairness_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Write the summary to disk with stable JSON ordering."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Fairness summary exported to %s", path)
    return path


def run_fairness_summary(config: FairnessSummaryConfig | None = None) -> dict[str, Any]:
    """Load audit artifacts, compute disparity analysis, and save the summary."""

    configure_logging()
    cfg = config or FairnessSummaryConfig()
    logger.info("Loading audit artifacts for fairness summary")
    initial_audit = _load_json(cfg.initial_audit_path)
    cross_group_metrics = _load_json(cfg.cross_group_metrics_path)
    logger.info("Cross-group analysis loaded | pair_count=%s", len(cross_group_metrics))

    summary = build_fairness_summary(initial_audit, cross_group_metrics, cfg)
    save_fairness_summary(summary, cfg.output_path)

    worst_far = summary["worst_far_group"]
    worst_frr = summary["worst_frr_group"]
    largest_far_gap = summary["largest_far_gap"]
    largest_frr_gap = summary["largest_frr_gap"]
    print(f"WORST_FAR_GROUP={worst_far['group']}")
    print(f"WORST_FRR_GROUP={worst_frr['group']}")
    print(f"FAR_GAP={largest_far_gap['gap']:.6f}")
    print(f"FRR_GAP={largest_frr_gap['gap']:.6f}")
    print(f"FAIRNESS_RISK_LEVEL={summary['fairness_risk_level']}")
    return summary


def main() -> None:
    """Executable entry point."""

    run_fairness_summary()


if __name__ == "__main__":
    main()
