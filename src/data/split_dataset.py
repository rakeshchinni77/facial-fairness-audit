"""Dataset split planning utilities.

This module prepares stratified split plans for train, validation, and audit
sets while preserving demographic distributions. The actual fairness balancing
logic is intentionally deferred to a later phase.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def build_stratification_key(frame: pd.DataFrame) -> pd.Series:
	"""Create a demographic stratification key from age, gender, and race."""

	required_columns = {"age", "gender", "race"}
	missing_columns = required_columns - set(frame.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns for stratified split: {sorted(missing_columns)}")

	return frame["gender"].astype(str).str.strip() + "_" + frame["age"].astype(str).str.strip() + "_" + frame["race"].astype(str).str.strip()


def create_split_plan(
	metadata_path: str | Path,
	output_directory: str | Path,
	audit_fraction: float = 0.2,
	validation_fraction: float = 0.1,
	seed: int = 42,
) -> dict[str, Any]:
	"""Create a stratified split plan without performing destructive moves.

	TODO: expand this into a full balancing workflow after the audit pipeline
	defines its final demographic buckets.
	"""

	metadata_frame = pd.read_csv(Path(metadata_path))
	stratification_key = build_stratification_key(metadata_frame)
	output_path = Path(output_directory)
	output_path.mkdir(parents=True, exist_ok=True)

	logger.info("Planning train/validation/audit splits for %s rows", len(metadata_frame))

	try:
		train_frame, temp_frame = train_test_split(
			metadata_frame,
			test_size=validation_fraction + audit_fraction,
			random_state=seed,
			stratify=stratification_key,
		)
		temp_key = build_stratification_key(temp_frame)
		validation_ratio = validation_fraction / (validation_fraction + audit_fraction)
		validation_frame, audit_frame = train_test_split(
			temp_frame,
			test_size=1 - validation_ratio,
			random_state=seed,
			stratify=temp_key,
		)
	except ValueError:
		logger.warning("Stratified split fallback activated due to sparse demographic groups")
		train_frame, temp_frame = train_test_split(metadata_frame, test_size=validation_fraction + audit_fraction, random_state=seed)
		validation_frame, audit_frame = train_test_split(temp_frame, test_size=audit_fraction / (validation_fraction + audit_fraction), random_state=seed)

	split_plan = {
		"train": train_frame["sample_id"].tolist() if "sample_id" in train_frame.columns else train_frame.index.tolist(),
		"validation": validation_frame["sample_id"].tolist() if "sample_id" in validation_frame.columns else validation_frame.index.tolist(),
		"audit": audit_frame["sample_id"].tolist() if "sample_id" in audit_frame.columns else audit_frame.index.tolist(),
		"counts": {
			"train": int(len(train_frame)),
			"validation": int(len(validation_frame)),
			"audit": int(len(audit_frame)),
		},
		"notes": [
			"TODO: preserve demographic balancing more aggressively in a later phase.",
			"TODO: create dedicated fairness-balanced audit sampling rules.",
		],
	}

	split_path = output_path / "split_plan.json"
	split_path.write_text(json.dumps(split_plan, indent=2), encoding="utf-8")
	logger.info("Wrote split plan to %s", split_path)
	return split_plan