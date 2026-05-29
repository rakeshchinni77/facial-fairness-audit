from __future__ import annotations

import math

import pytest

from src.audit.threshold_selector import ThresholdSelectorConfig, ValidationPairDataset
from tests.helpers import load_json_artifact


def test_threshold_analysis_json_is_valid(project_root):
    payload = load_json_artifact(project_root / "results" / "threshold_analysis.json")
    assert 0.0 <= payload["roc_auc"] <= 1.0, "ROC AUC must be between 0 and 1"
    assert math.isfinite(float(payload["optimal_threshold"])), "Threshold must be finite"
    assert 0.0 <= payload["estimated_far"] <= 1.0, "Estimated FAR must be between 0 and 1"
    assert 0.0 <= payload["estimated_frr"] <= 1.0, "Estimated FRR must be between 0 and 1"
    stats = payload["validation_statistics"]
    assert stats["positive_examples"] > 0 and stats["negative_examples"] > 0, "Validation statistics should report positive and negative examples"


def test_validation_pair_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Validation pair CSV not found"):
        ValidationPairDataset(tmp_path / "missing.csv")


def test_threshold_selector_config_defaults_are_cpu_compatible():
    config = ThresholdSelectorConfig()
    assert config.device == "cpu", "Threshold selection should default to CPU execution"
