from __future__ import annotations

from tests.helpers import load_json_artifact


def test_core_model_artifacts_exist_and_are_non_empty(project_root):
    for relative_path in (
        "artifacts/best_model.pth",
        "artifacts/best_mitigated_model.pth",
        "artifacts/mitigated_model.pth",
        "artifacts/model.pth",
    ):
        path = project_root / relative_path
        assert path.exists(), f"Missing model artifact: {path}"
        assert path.stat().st_size > 0, f"Empty model artifact: {path}"


def test_summary_artifacts_exist(project_root):
    for relative_path in (
        "artifacts/training_summary.json",
        "artifacts/mitigation_training_summary.json",
        "results/rebalancing_summary.json",
    ):
        path = project_root / relative_path
        payload = load_json_artifact(path)
        assert payload, f"Artifact should not be empty: {path}"
