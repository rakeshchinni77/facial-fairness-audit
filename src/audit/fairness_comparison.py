"""Fairness comparison utilities for before/after audit reports."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FairnessComparisonConfig:
	"""Configuration for fairness comparison export."""

	initial_audit_path: Path = Path("results/initial_audit.json")
	initial_cross_group_path: Path = Path("results/cross_group_metrics.json")
	mitigated_audit_path: Path = Path("results/mitigated_audit.json")
	output_path: Path = Path("results/fairness_comparison.json")


def _load_json(path: str | Path) -> dict[str, Any]:
	json_path = Path(path)
	if not json_path.exists():
		raise FileNotFoundError(f"JSON file not found: {json_path}")
	return json.loads(json_path.read_text(encoding="utf-8"))


def _is_metric_record(value: Any) -> bool:
	return isinstance(value, dict) and {"far", "frr", "support"}.issubset(value.keys())


def _subgroup_metrics(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
	return {
		key: value
		for key, value in report.items()
		if key not in {"overall", "metadata", "cross_group_metrics"} and _is_metric_record(value) and "__vs__" not in key
	}


def _cross_group_metrics(report: dict[str, Any], fallback_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
	cross_group = report.get("cross_group_metrics")
	if isinstance(cross_group, dict):
		return {str(key): value for key, value in cross_group.items() if _is_metric_record(value)}
	if fallback_path is not None:
		fallback = Path(fallback_path)
		if fallback.exists():
			payload = _load_json(fallback)
			if isinstance(payload, dict):
				return {str(key): value for key, value in payload.items() if _is_metric_record(value)}
	return {}


def _pair_gap(metrics: dict[str, Any]) -> float:
	return float(max(float(metrics.get("far", 0.0)), float(metrics.get("frr", 0.0))))


def _metric_improvement(before: float, after: float) -> float:
	return float(before - after)


def build_fairness_comparison(
	initial_report: dict[str, Any],
	mitigated_report: dict[str, Any],
	initial_cross_group: dict[str, dict[str, Any]],
	mitigated_cross_group: dict[str, dict[str, Any]],
) -> dict[str, Any]:
	"""Compare fairness before and after mitigation."""

	initial_subgroups = _subgroup_metrics(initial_report)
	mitigated_subgroups = _subgroup_metrics(mitigated_report)
	all_groups = sorted(set(initial_subgroups) | set(mitigated_subgroups))

	subgroup_deltas: dict[str, dict[str, Any]] = {}
	metric_rows: list[dict[str, Any]] = []
	for group in all_groups:
		initial_metrics = initial_subgroups.get(group)
		mitigated_metrics = mitigated_subgroups.get(group)
		if not initial_metrics or not mitigated_metrics:
			continue
		delta_far = float(mitigated_metrics["far"]) - float(initial_metrics["far"])
		delta_frr = float(mitigated_metrics["frr"]) - float(initial_metrics["frr"])
		subgroup_deltas[group] = {
			"delta_far": delta_far,
			"delta_frr": delta_frr,
			"initial_far": float(initial_metrics["far"]),
			"mitigated_far": float(mitigated_metrics["far"]),
			"initial_frr": float(initial_metrics["frr"]),
			"mitigated_frr": float(mitigated_metrics["frr"]),
			"support": int(mitigated_metrics.get("support", initial_metrics.get("support", 0))),
		}
		metric_rows.append({"group": group, "metric": "far", "improvement": _metric_improvement(float(initial_metrics["far"]), float(mitigated_metrics["far"])), "value": float(mitigated_metrics["far"]), "delta": delta_far})
		metric_rows.append({"group": group, "metric": "frr", "improvement": _metric_improvement(float(initial_metrics["frr"]), float(mitigated_metrics["frr"])), "value": float(mitigated_metrics["frr"]), "delta": delta_frr})

	positive_improvements = [row for row in metric_rows if row["improvement"] > 0]
	if positive_improvements:
		largest_improvement = max(positive_improvements, key=lambda row: row["improvement"])
	else:
		largest_improvement = max(metric_rows, key=lambda row: row["improvement"], default={"group": "n/a", "metric": "far", "improvement": 0.0, "value": 0.0})

	remaining_gap_rows = [
		{
			"group": group,
			"metric": "far" if float(metrics["far"]) >= float(metrics["frr"]) else "frr",
			"value": float(max(float(metrics["far"]), float(metrics["frr"]))),
		}
		for group, metrics in mitigated_subgroups.items()
	]
	largest_remaining_gap = max(remaining_gap_rows, key=lambda row: row["value"], default={"group": "n/a", "metric": "far", "value": 0.0})

	initial_subgroup_gap = sum(max(float(metrics["far"]), float(metrics["frr"])) for metrics in initial_subgroups.values()) / max(len(initial_subgroups), 1)
	mitigated_subgroup_gap = sum(max(float(metrics["far"]), float(metrics["frr"])) for metrics in mitigated_subgroups.values()) / max(len(mitigated_subgroups), 1)
	average_disparity_reduction = initial_subgroup_gap - mitigated_subgroup_gap

	initial_cross_keys = set(initial_cross_group) & set(mitigated_cross_group)
	cross_group_deltas: dict[str, dict[str, Any]] = {}
	cross_group_rows: list[dict[str, Any]] = []
	for pair_key in sorted(initial_cross_keys):
		initial_metrics = initial_cross_group[pair_key]
		mitigated_metrics = mitigated_cross_group[pair_key]
		delta_far = float(mitigated_metrics["far"]) - float(initial_metrics["far"])
		delta_frr = float(mitigated_metrics["frr"]) - float(initial_metrics["frr"])
		initial_gap = _pair_gap(initial_metrics)
		mitigated_gap = _pair_gap(mitigated_metrics)
		cross_group_deltas[pair_key] = {
			"delta_far": delta_far,
			"delta_frr": delta_frr,
			"initial_far": float(initial_metrics["far"]),
			"mitigated_far": float(mitigated_metrics["far"]),
			"initial_frr": float(initial_metrics["frr"]),
			"mitigated_frr": float(mitigated_metrics["frr"]),
			"initial_gap": initial_gap,
			"mitigated_gap": mitigated_gap,
		}
		cross_group_rows.append({"pair": pair_key, "improvement": initial_gap - mitigated_gap, "value": mitigated_gap})

	if cross_group_rows:
		most_improved_cross_group_pairing = max(cross_group_rows, key=lambda row: row["improvement"])
		cross_group_fairness_improvement = sum(row["improvement"] for row in cross_group_rows) / len(cross_group_rows)
	else:
		most_improved_cross_group_pairing = {"pair": "n/a", "improvement": 0.0, "value": 0.0}
		cross_group_fairness_improvement = 0.0

	overall_before = initial_report.get("overall", {})
	overall_after = mitigated_report.get("overall", {})
	fairness_improved = (
		float(overall_after.get("far", 0.0)) <= float(overall_before.get("far", 0.0))
		and float(overall_after.get("frr", 0.0)) <= float(overall_before.get("frr", 0.0))
		and average_disparity_reduction >= 0.0
		and cross_group_fairness_improvement >= 0.0
	)

	return {
		"overall_improvement": {
			"far_before": float(overall_before.get("far", 0.0)),
			"far_after": float(overall_after.get("far", 0.0)),
			"frr_before": float(overall_before.get("frr", 0.0)),
			"frr_after": float(overall_after.get("frr", 0.0)),
		},
		"largest_improvement": {
			"group": str(largest_improvement.get("group", "n/a")),
			"metric": str(largest_improvement.get("metric", "far")),
			"improvement": float(largest_improvement.get("improvement", 0.0)),
		},
		"largest_remaining_gap": {
			"group": str(largest_remaining_gap.get("group", "n/a")),
			"metric": str(largest_remaining_gap.get("metric", "far")),
			"value": float(largest_remaining_gap.get("value", 0.0)),
		},
		"fairness_improved": bool(fairness_improved),
		"average_disparity_reduction": float(average_disparity_reduction),
		"cross_group_fairness_improvement": float(cross_group_fairness_improvement),
		"best_improved_subgroup": largest_improvement.get("group", "n/a"),
		"worst_remaining_subgroup": largest_remaining_gap.get("group", "n/a"),
		"most_improved_cross_group_pairing": most_improved_cross_group_pairing,
		"subgroup_deltas": subgroup_deltas,
		"cross_group_deltas": cross_group_deltas,
	}


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
	logger.info("Fairness comparison exported to %s", output_path)
	return output_path


def run_fairness_comparison(config: FairnessComparisonConfig | None = None) -> dict[str, Any]:
	"""Load audit artifacts, compute the comparison, and export it."""

	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	cfg = config or FairnessComparisonConfig()
	initial_report = _load_json(cfg.initial_audit_path)
	mitigated_report = _load_json(cfg.mitigated_audit_path)
	initial_cross_group = _cross_group_metrics(initial_report, fallback_path=cfg.initial_cross_group_path)
	mitigated_cross_group = _cross_group_metrics(mitigated_report)
	comparison = build_fairness_comparison(initial_report, mitigated_report, initial_cross_group, mitigated_cross_group)
	save_json(comparison, cfg.output_path)
	logger.info("Fairness comparison complete")
	print(f"FAIRNESS_IMPROVED={comparison['fairness_improved']}")
	print(f"OVERALL_FAR_BEFORE={comparison['overall_improvement']['far_before']:.6f}")
	print(f"OVERALL_FAR_AFTER={comparison['overall_improvement']['far_after']:.6f}")
	print(f"OVERALL_FRR_BEFORE={comparison['overall_improvement']['frr_before']:.6f}")
	print(f"OVERALL_FRR_AFTER={comparison['overall_improvement']['frr_after']:.6f}")
	return comparison


def main() -> None:
	"""Executable entry point for fairness comparison."""

	run_fairness_comparison()


if __name__ == "__main__":
	main()