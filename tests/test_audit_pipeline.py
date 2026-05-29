from __future__ import annotations

import math

import pytest

from src.audit.audit_pipeline import AuditPipelineConfig, load_threshold
from tests.helpers import load_json_artifact


def test_initial_audit_json_contains_subgroup_and_overall_metrics(project_root):
    audit = load_json_artifact(project_root / "results" / "initial_audit.json")
    assert "overall" in audit, "Initial audit report must include overall metrics"
    overall = audit["overall"]
    for key in ("far", "frr", "support", "tp", "tn", "fp", "fn"):
        assert key in overall, f"Overall audit metrics missing key: {key}"
        value = overall[key]
        if isinstance(value, float):
            assert math.isfinite(value), f"Overall metric {key} must be finite"

    subgroup_keys = [key for key in audit if key not in {"overall", "metadata"}]
    assert subgroup_keys, "Initial audit must include subgroup metrics"
    first_subgroup = audit[subgroup_keys[0]]
    for key in ("far", "frr", "support"):
        assert key in first_subgroup, f"Subgroup metric missing key: {key}"
        assert math.isfinite(float(first_subgroup[key])), f"Subgroup metric {key} must be finite"


def test_cross_group_metrics_are_generated(project_root):
    cross_group = load_json_artifact(project_root / "results" / "cross_group_metrics.json")
    assert cross_group, "Cross-group metrics should not be empty"
    sample_pair = next(iter(cross_group.values()))
    for key in ("far", "frr", "support"):
        assert key in sample_pair, f"Cross-group metric missing key: {key}"


def test_analysis_json_reports_high_risk_and_non_deployment(project_root):
    analysis = load_json_artifact(project_root / "results" / "analysis.json")
    assert analysis["fairness_risk_level"] == "HIGH", "Analysis should classify the current audit as high risk"
    assert analysis["deployment_readiness"]["recommended_for_high_stakes_use"] is False, "Analysis should not recommend high-stakes deployment"


def test_audit_loader_rejects_missing_threshold(tmp_path):
    with pytest.raises(FileNotFoundError, match="Threshold file not found"):
        load_threshold(tmp_path / "missing_threshold.json")


def test_audit_pipeline_config_defaults_to_cpu():
    config = AuditPipelineConfig()
    assert config.device == "cpu", "Audit pipeline should default to CPU execution"
