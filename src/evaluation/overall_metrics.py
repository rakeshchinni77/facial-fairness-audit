"""Trade-off evaluation for initial vs mitigated fairness performance."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverallMetricsConfig:
	"""Configuration for the overall trade-off evaluation."""

	initial_audit_path: Path = Path("results/initial_audit.json")
	mitigated_audit_path: Path = Path("results/mitigated_audit.json")
	threshold_path: Path = Path("results/threshold_analysis.json")
	fairness_comparison_path: Path = Path("results/fairness_comparison.json")
	output_path: Path = Path("results/overall_metrics.json")


def configure_logging() -> None:
	"""Configure structured logging for the trade-off report."""

	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _load_json(path: str | Path) -> dict[str, Any]:
	json_path = Path(path)
	if not json_path.exists():
		raise FileNotFoundError(f"JSON artifact not found: {json_path}")
	return json.loads(json_path.read_text(encoding="utf-8"))


def _overall_metrics(report: dict[str, Any]) -> dict[str, Any]:
	overall = report.get("overall") if isinstance(report, dict) else {}
	if not isinstance(overall, dict):
		overall = {}
	return overall


def _threshold_from_payload(payload: dict[str, Any]) -> float:
	threshold = payload.get("optimal_threshold")
	if threshold is None:
		threshold = payload.get("threshold", 0.0)
	return float(threshold)


def _accuracy_from_far_frr(far: float, frr: float) -> float:
	return float(1.0 - ((float(far) + float(frr)) / 2.0))


def _disparity_reduction(comparison: dict[str, Any]) -> float:
	for key in ("average_disparity_reduction", "cross_group_fairness_improvement"):
		value = comparison.get(key)
		if isinstance(value, (int, float)):
			return float(value)
	return 0.0


def _is_fairness_improved(initial_far: float, mitigated_far: float, initial_frr: float, mitigated_frr: float, disparity_reduction: float) -> bool:
	far_degradation = float(mitigated_far - initial_far)
	frr_improved = mitigated_frr < initial_frr
	disparity_improved = disparity_reduction > 0.0
	non_catastrophic_far = far_degradation <= 0.05
	return bool(frr_improved and disparity_improved and non_catastrophic_far)


def _scientific_summary(initial_far: float, mitigated_far: float, initial_frr: float, mitigated_frr: float, fairness_improved: bool, disparity_reduction: float) -> str:
	far_change = mitigated_far - initial_far
	frr_change = mitigated_frr - initial_frr
	if fairness_improved:
		return (
			"The mitigation strategy improved rejection fairness while preserving acceptable acceptance behavior, "
			f"reducing subgroup disparity by {disparity_reduction:.6f} and lowering FRR by {abs(frr_change):.6f}."
		)
	if far_change > 0 and frr_change < 0:
		return (
			"The mitigation strategy reduced FRR but increased FAR, indicating a deployment trade-off between "
			"accessibility and security that should be resolved before operational use."
		)
	if far_change > 0:
		return (
			"The mitigated model improved some fairness signals but increased FAR, suggesting the operating point "
			"became more permissive and may require post-mitigation recalibration."
		)
	if frr_change < 0:
		return (
			"The mitigated model improved rejection fairness and preserved acceptance behavior, but the disparity "
			"reduction was insufficient to justify declaring the system fairness-improved."
		)
	return (
		"The mitigation run did not produce a clear fairness improvement under the current operating threshold, "
		"so the verification trade-off remains unresolved."
	)


def build_overall_metrics(
	initial_report: dict[str, Any],
	mitigated_report: dict[str, Any],
	threshold_payload: dict[str, Any],
	fairness_comparison: dict[str, Any],
) -> dict[str, Any]:
	"""Aggregate pre- and post-mitigation verification metrics."""

	initial_overall = _overall_metrics(initial_report)
	mitigated_overall = _overall_metrics(mitigated_report)
	initial_far = float(initial_overall.get("far", 0.0))
	initial_frr = float(initial_overall.get("frr", 0.0))
	mitigated_far = float(mitigated_overall.get("far", 0.0))
	mitigated_frr = float(mitigated_overall.get("frr", 0.0))
	threshold = _threshold_from_payload(threshold_payload)

	initial_accuracy = _accuracy_from_far_frr(initial_far, initial_frr)
	mitigated_accuracy = _accuracy_from_far_frr(mitigated_far, mitigated_frr)
	accuracy_change = mitigated_accuracy - initial_accuracy
	far_change = mitigated_far - initial_far
	frr_change = mitigated_frr - initial_frr
	disparity_reduction = _disparity_reduction(fairness_comparison)
	fairness_improved = _is_fairness_improved(initial_far, mitigated_far, initial_frr, mitigated_frr, disparity_reduction)

	return {
		"initial_model": {
			"accuracy": float(initial_accuracy),
			"threshold": float(threshold),
			"far": float(initial_far),
			"frr": float(initial_frr),
		},
		"mitigated_model": {
			"accuracy": float(mitigated_accuracy),
			"threshold": float(threshold),
			"far": float(mitigated_far),
			"frr": float(mitigated_frr),
		},
		"tradeoff_analysis": {
			"accuracy_change": float(accuracy_change),
			"far_change": float(far_change),
			"frr_change": float(frr_change),
			"fairness_improved": bool(fairness_improved),
			"summary": _scientific_summary(initial_far, mitigated_far, initial_frr, mitigated_frr, fairness_improved, disparity_reduction),
		},
	}


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
	"""Persist the overall metrics report with stable formatting."""

	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
	logger.info("Overall metrics exported to %s", output_path)
	return output_path


def run_overall_metrics(config: OverallMetricsConfig | None = None) -> dict[str, Any]:
	"""Load existing artifacts and export the trade-off report."""

	configure_logging()
	cfg = config or OverallMetricsConfig()
	logger.info("Loading artifacts for overall trade-off evaluation")
	initial_report = _load_json(cfg.initial_audit_path)
	mitigated_report = _load_json(cfg.mitigated_audit_path)
	threshold_payload = _load_json(cfg.threshold_path)
	fairness_comparison = _load_json(cfg.fairness_comparison_path)
	logger.info("Computing overall metrics and fairness-performance trade-off")
	report = build_overall_metrics(initial_report, mitigated_report, threshold_payload, fairness_comparison)
	save_json(report, cfg.output_path)
	logger.info("Trade-off evaluation complete")
	print(f"INITIAL_ACCURACY={report['initial_model']['accuracy']:.6f}")
	print(f"MITIGATED_ACCURACY={report['mitigated_model']['accuracy']:.6f}")
	print(f"ACCURACY_CHANGE={report['tradeoff_analysis']['accuracy_change']:.6f}")
	print(f"FAIRNESS_IMPROVED={report['tradeoff_analysis']['fairness_improved']}")
	return report


def main() -> None:
	"""Executable entry point for the trade-off evaluation layer."""

	run_overall_metrics()


if __name__ == "__main__":
	main()