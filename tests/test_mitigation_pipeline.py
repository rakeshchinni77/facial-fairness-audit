from __future__ import annotations

import pytest

from src.mitigation.mitigation_pipeline import MitigationConfig, load_mitigated_model
from src.models.triplet_network import TripletNetwork
from tests.helpers import load_json_artifact


def test_mitigated_artifacts_exist_and_are_consistent(project_root):
    mitigated_model_path = project_root / "artifacts" / "best_mitigated_model.pth"
    comparison_path = project_root / "results" / "fairness_comparison.json"
    summary_path = project_root / "artifacts" / "mitigation_training_summary.json"

    assert mitigated_model_path.exists(), f"Mitigated checkpoint missing: {mitigated_model_path}"
    assert mitigated_model_path.stat().st_size > 0, "Mitigated checkpoint should not be empty"
    assert comparison_path.exists(), f"Fairness comparison missing: {comparison_path}"
    assert summary_path.exists(), f"Mitigation summary missing: {summary_path}"

    comparison = load_json_artifact(comparison_path)
    assert "average_disparity_reduction" in comparison, "Fairness comparison missing average_disparity_reduction"
    assert "fairness_improved" in comparison, "Fairness comparison missing fairness_improved flag"


def test_mitigation_pipeline_loads_model(project_root):
    config = MitigationConfig(
        processed_dir=project_root / "data" / "processed",
        artifacts_dir=project_root / "artifacts",
        results_dir=project_root / "results",
        checkpoint_path=project_root / "artifacts" / "best_model.pth",
        mitigated_checkpoint_path=project_root / "artifacts" / "mitigated_model.pth",
        best_mitigated_checkpoint_path=project_root / "artifacts" / "best_mitigated_model.pth",
        pretrained=False,
    )
    model = load_mitigated_model(config, {"Male_0-19_Light": 1.0})
    assert isinstance(model, TripletNetwork), "Mitigation pipeline should return a TripletNetwork"


def test_mitigation_loader_rejects_missing_checkpoint(tmp_path):
    config = MitigationConfig(checkpoint_path=tmp_path / "missing_checkpoint.pth", pretrained=False)
    with pytest.raises(FileNotFoundError, match="Mitigation checkpoint not found"):
        load_mitigated_model(config, {"Male_0-19_Light": 1.0})
