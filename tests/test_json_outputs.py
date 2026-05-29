from __future__ import annotations

from tests.helpers import EXPECTED_RESULTS_JSONS, assert_json_finite, load_json_artifact


def test_all_results_json_files_are_valid_and_finite(project_root):
    for name in EXPECTED_RESULTS_JSONS:
        path = project_root / "results" / name
        assert path.exists(), f"Missing results JSON: {path}"
        payload = load_json_artifact(path)
        assert_json_finite(payload, path=name)


def test_results_directory_contains_expected_json_artifacts(project_root):
    result_files = sorted(path.name for path in (project_root / "results").glob("*.json"))
    assert "threshold_analysis.json" in result_files, "Threshold analysis JSON should be present"
    assert "initial_audit.json" in result_files, "Initial audit JSON should be present"
    assert "mitigated_audit.json" in result_files, "Mitigated audit JSON should be present"
