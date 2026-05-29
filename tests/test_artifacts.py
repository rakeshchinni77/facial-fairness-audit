from __future__ import annotations

from tests.helpers import load_checkpoint_strict, load_json_strict


def test_core_artifacts_exist_and_are_non_empty(artifacts_dir) -> None:
	for name in ("best_model.pth", "best_mitigated_model.pth", "model.pth", "mitigated_model.pth"):
		path = artifacts_dir / name
		assert path.exists(), f"Required artifact is missing: {path}"
		assert path.stat().st_size > 0, f"Required artifact is empty: {path}"


def test_json_and_checkpoint_error_handling(tmp_path) -> None:
	invalid_json = tmp_path / "broken.json"
	invalid_json.write_text("{not-valid-json", encoding="utf-8")
	try:
		load_json_strict(invalid_json)
		raise AssertionError("Expected invalid JSON to raise a ValueError")
	except ValueError as exc:
		assert "Invalid JSON" in str(exc), "Invalid JSON should raise a readable error"

	invalid_checkpoint = tmp_path / "broken.pth"
	invalid_checkpoint.write_text("not-a-checkpoint", encoding="utf-8")
	try:
		load_checkpoint_strict(invalid_checkpoint)
		raise AssertionError("Expected invalid checkpoint to raise a ValueError")
	except ValueError as exc:
		assert "Unable to load checkpoint" in str(exc) or "Unsupported checkpoint format" in str(exc), "Invalid checkpoints should raise a readable error"
