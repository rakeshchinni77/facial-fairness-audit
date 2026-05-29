from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def first_existing_path(candidates: Iterable[Path]) -> Path:
	for candidate in candidates:
		if Path(candidate).exists():
			return Path(candidate)
	raise FileNotFoundError(f"No candidate path exists: {[str(candidate) for candidate in candidates]}")


def load_json_strict(path: str | Path) -> Any:
	json_path = Path(path)
	if not json_path.exists():
		raise FileNotFoundError(f"JSON file not found: {json_path}")
	try:
		return json.loads(json_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		raise ValueError(f"Invalid JSON in {json_path}: {exc.msg}") from exc


def assert_json_finite(payload: Any, location: str = "root") -> None:
	if payload is None or isinstance(payload, (str, bool)):
		return
	if isinstance(payload, int):
		return
	if isinstance(payload, float):
		if not math.isfinite(payload):
			raise AssertionError(f"Non-finite numeric value at {location}: {payload!r}")
		return
	if isinstance(payload, dict):
		for key, value in payload.items():
			assert_json_finite(value, f"{location}.{key}")
		return
	if isinstance(payload, (list, tuple)):
		for index, value in enumerate(payload):
			assert_json_finite(value, f"{location}[{index}]")
		return
	raise AssertionError(f"Unsupported JSON value at {location}: {type(payload)!r}")


def load_checkpoint_strict(path: str | Path) -> dict[str, Any]:
	checkpoint_path = Path(path)
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
	try:
		checkpoint = torch.load(checkpoint_path, map_location="cpu")
	except Exception as exc:  # pragma: no cover - defensive wrapper for corrupted checkpoints
		raise ValueError(f"Unable to load checkpoint {checkpoint_path}: {exc}") from exc
	if not isinstance(checkpoint, dict):
		raise ValueError(f"Checkpoint must be a mapping: {checkpoint_path}")
	for key in ("model_state_dict", "state_dict", "embedding_model_state_dict"):
		value = checkpoint.get(key)
		if isinstance(value, dict):
			return value
	if all(isinstance(key, str) for key in checkpoint.keys()):
		return checkpoint
	raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def load_csv_strict(path: str | Path, required_columns: set[str] | None = None) -> pd.DataFrame:
	csv_path = Path(path)
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")
	try:
		frame = pd.read_csv(csv_path)
	except Exception as exc:  # pragma: no cover - defensive wrapper for malformed CSVs
		raise ValueError(f"Unable to read CSV {csv_path}: {exc}") from exc
	if required_columns is not None:
		missing_columns = set(required_columns) - set(frame.columns)
		if missing_columns:
			raise ValueError(f"CSV missing required columns: {sorted(missing_columns)}")
	return frame
