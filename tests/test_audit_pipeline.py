from __future__ import annotations

import math

from tests.helpers import load_json_strict


def test_initial_audit_and_cross_group_artifacts_are_valid(results_dir) -> None:
	audit = load_json_strict(results_dir / "initial_audit.json")
	cross_group = load_json_strict(results_dir / "cross_group_metrics.json")
	fairness_summary = load_json_strict(results_dir / "fairness_summary.json")

	assert "overall" in audit, "Initial audit report must include overall metrics"
	assert math.isfinite(float(audit["overall"]["far"])), "Initial audit FAR must be finite"
	assert math.isfinite(float(audit["overall"]["frr"])), "Initial audit FRR must be finite"
	assert "metadata" in audit and "threshold" in audit["metadata"], "Initial audit metadata must include threshold information"
	assert audit["metadata"]["threshold"] == load_json_strict(results_dir / "threshold_analysis.json")["optimal_threshold"], "Audit must use the validation-selected threshold"
	assert fairness_summary["fairness_risk_level"] == "HIGH", "The fairness risk level should remain explicitly documented"
	assert cross_group, "Cross-group metrics must be generated"
	assert any(key.endswith("__vs__Male_40-59_Dark") for key in cross_group), "Cross-group metrics must include intersectional pairings"
