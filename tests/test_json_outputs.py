from __future__ import annotations

from tests.helpers import assert_json_finite, load_json_strict


def test_all_results_json_files_are_valid_and_finite(results_dir) -> None:
	json_files = sorted(results_dir.glob("*.json"))
	assert json_files, "Expected JSON result artifacts to exist"
	for json_path in json_files:
		payload = load_json_strict(json_path)
		assert_json_finite(payload, location=json_path.name)
