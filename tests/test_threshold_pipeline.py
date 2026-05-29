from __future__ import annotations

import math

from tests.helpers import load_json_strict


def test_threshold_analysis_artifact_is_valid(results_dir) -> None:
	threshold = load_json_strict(results_dir / "threshold_analysis.json")
	assert 0.0 <= float(threshold["roc_auc"]) <= 1.0, "ROC AUC must be a valid probability-like value"
	assert math.isfinite(float(threshold["optimal_threshold"])), "Threshold must be finite"
	assert 0.0 <= float(threshold["estimated_far"]) <= 1.0, "Estimated FAR must be bounded between 0 and 1"
	assert 0.0 <= float(threshold["estimated_frr"]) <= 1.0, "Estimated FRR must be bounded between 0 and 1"
	assert threshold["det_summary"]["far_points"] > 0, "DET analysis must contain at least one FAR point"
