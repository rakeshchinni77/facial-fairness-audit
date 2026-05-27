"""Fairness-aware dataset splitting utilities.

This module performs a deterministic 70/15/15 split over the enriched FairFace
metadata using subgroup-based stratification. Audit isolation matters because
the held-out audit set must remain untouched by downstream model development.
Random splitting can distort subgroup coverage, so stratification preserves the
joint demographic structure needed for later fairness analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
	"""Configuration for reproducible dataset splitting."""

	metadata_path: Path = Path("data/processed/enriched_metadata.csv")
	train_ratio: float = 0.70
	validation_ratio: float = 0.15
	audit_ratio: float = 0.15
	random_seed: int = 42
	output_directory: Path = Path("data/processed")
	summary_output_path: Path = Path("results/split_summary.json")


def configure_logging() -> None:
	"""Configure module logging for standalone execution."""

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def load_enriched_metadata(metadata_path: str | Path) -> pd.DataFrame:
	"""Load enriched metadata and validate the required demographic columns."""

	frame = pd.read_csv(Path(metadata_path))
	required_columns = {"subgroup", "gender_mapped", "age_bin", "skin_tone"}
	missing_columns = required_columns - set(frame.columns)
	if missing_columns:
		raise ValueError(f"Missing required enriched metadata columns: {sorted(missing_columns)}")

	if "sample_id" in frame.columns and frame["sample_id"].duplicated().any():
		raise ValueError("Duplicate sample_id values detected in enriched metadata")

	if frame[list(required_columns)].isna().any().any():
		raise ValueError("Enriched metadata contains missing demographic values")

	logger.info("Metadata loaded: %s rows", len(frame))
	return frame


def build_stratification_key(frame: pd.DataFrame) -> pd.Series:
	"""Create a subgroup-based stratification key from the enriched metadata."""

	if "subgroup" not in frame.columns:
		raise ValueError("Missing required subgroup column for stratification")

	return frame["subgroup"].astype(str).str.strip()


def validate_ratios(train_ratio: float, validation_ratio: float, audit_ratio: float) -> None:
	"""Validate the requested split ratios."""

	total = train_ratio + validation_ratio + audit_ratio
	if abs(total - 1.0) > 1e-8:
		raise ValueError(f"Split ratios must sum to 1.0, received {total:.6f}")
	if min(train_ratio, validation_ratio, audit_ratio) <= 0:
		raise ValueError("Split ratios must all be greater than zero")


def ensure_all_groups_present(frame: pd.DataFrame, source_groups: set[str], split_name: str) -> None:
	"""Ensure every subgroup in the source appears in the given split.

	This protects against hidden fairness regressions caused by random splitting,
	which can silently erase rare demographic groups from a split.
	"""

	current_groups = set(frame["subgroup"].astype(str).unique())
	missing_groups = sorted(source_groups - current_groups)
	if missing_groups:
		raise ValueError(f"{split_name} split is missing subgroup(s): {missing_groups}")


def validate_split_integrity(
	metadata_frame: pd.DataFrame,
	train_frame: pd.DataFrame,
	validation_frame: pd.DataFrame,
	audit_frame: pd.DataFrame,
) -> None:
	"""Validate split overlap, duplicates, and subgroup coverage."""

	identifier_column = "sample_id" if "sample_id" in metadata_frame.columns else None
	if identifier_column is None:
		raise ValueError("sample_id column is required for split integrity validation")

	all_ids = pd.Index(metadata_frame[identifier_column].astype(str))
	if all_ids.duplicated().any():
		raise ValueError("Duplicate sample_id values detected before splitting")

	train_ids = set(train_frame[identifier_column].astype(str))
	validation_ids = set(validation_frame[identifier_column].astype(str))
	audit_ids = set(audit_frame[identifier_column].astype(str))

	if train_ids & validation_ids:
		raise ValueError("Train and validation splits overlap")
	if train_ids & audit_ids:
		raise ValueError("Train and audit splits overlap")
	if validation_ids & audit_ids:
		raise ValueError("Validation and audit splits overlap")

	union_size = len(train_ids | validation_ids | audit_ids)
	if union_size != len(metadata_frame):
		raise ValueError("Split union does not match source metadata size")

	source_groups = set(metadata_frame["subgroup"].astype(str).unique())
	for split_name, split_frame in (
		("train", train_frame),
		("validation", validation_frame),
		("audit", audit_frame),
	):
		if split_frame[identifier_column].duplicated().any():
			raise ValueError(f"Duplicate sample_id values detected in {split_name} split")
		ensure_all_groups_present(split_frame, source_groups, split_name)

	logger.info("Validation passed: splits are disjoint and preserve subgroup coverage")


def compute_distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
	"""Compute a normalized frequency table for a categorical column."""

	return {str(index): int(value) for index, value in frame[column].value_counts().sort_index().items()}


def build_split_summary(
	train_frame: pd.DataFrame,
	validation_frame: pd.DataFrame,
	audit_frame: pd.DataFrame,
	random_seed: int,
	train_ratio: float,
	validation_ratio: float,
	audit_ratio: float,
) -> dict[str, Any]:
	"""Build the JSON summary for the split outputs."""

	return {
		"random_seed": random_seed,
		"split_ratios": {
			"train": train_ratio,
			"validation": validation_ratio,
			"audit": audit_ratio,
		},
		"split_sizes": {
			"train": int(len(train_frame)),
			"validation": int(len(validation_frame)),
			"audit": int(len(audit_frame)),
		},
		"subgroup_distributions": {
			"train": compute_distribution(train_frame, "subgroup"),
			"validation": compute_distribution(validation_frame, "subgroup"),
			"audit": compute_distribution(audit_frame, "subgroup"),
		},
		"gender_distributions": {
			"train": compute_distribution(train_frame, "gender_mapped"),
			"validation": compute_distribution(validation_frame, "gender_mapped"),
			"audit": compute_distribution(audit_frame, "gender_mapped"),
		},
		"age_distributions": {
			"train": compute_distribution(train_frame, "age_bin"),
			"validation": compute_distribution(validation_frame, "age_bin"),
			"audit": compute_distribution(audit_frame, "age_bin"),
		},
		"skin_tone_distributions": {
			"train": compute_distribution(train_frame, "skin_tone"),
			"validation": compute_distribution(validation_frame, "skin_tone"),
			"audit": compute_distribution(audit_frame, "skin_tone"),
		},
	}


def create_split_plan(
	metadata_path: str | Path,
	output_directory: str | Path,
	train_ratio: float = 0.70,
	validation_ratio: float = 0.15,
	audit_ratio: float = 0.15,
	seed: int = 42,
) -> dict[str, Any]:
	"""Create a deterministic subgroup-stratified split plan.

	The audit set is isolated first, then train and validation are derived from
	the remaining data. This prevents audit leakage and preserves the held-out
	role of the audit split for future fairness evaluation.
	"""

	validate_ratios(train_ratio, validation_ratio, audit_ratio)
	metadata_frame = load_enriched_metadata(metadata_path)
	stratification_key = build_stratification_key(metadata_frame)
	output_path = Path(output_directory)
	output_path.mkdir(parents=True, exist_ok=True)

	logger.info("Split started")
	logger.info("Metadata loaded: %s rows", len(metadata_frame))
	logger.info("Stratification applied using subgroup column")

	remaining_ratio = validation_ratio + audit_ratio
	train_frame, remainder_frame = train_test_split(
		metadata_frame,
		test_size=remaining_ratio,
		random_state=seed,
		stratify=stratification_key,
	)
	if len(remainder_frame) == 0:
		raise ValueError("Remainder split is empty; unable to form validation and audit sets")

	remainder_key = build_stratification_key(remainder_frame)
	validation_share_of_remainder = validation_ratio / remaining_ratio
	validation_frame, audit_frame = train_test_split(
		remainder_frame,
		test_size=1 - validation_share_of_remainder,
		random_state=seed,
		stratify=remainder_key,
	)

	train_frame = train_frame.reset_index(drop=True)
	validation_frame = validation_frame.reset_index(drop=True)
	audit_frame = audit_frame.reset_index(drop=True)

	validate_split_integrity(metadata_frame, train_frame, validation_frame, audit_frame)

	train_path = output_path / "train_metadata.csv"
	validation_path = output_path / "validation_metadata.csv"
	audit_path = output_path / "audit_metadata.csv"

	train_frame.to_csv(train_path, index=False)
	validation_frame.to_csv(validation_path, index=False)
	audit_frame.to_csv(audit_path, index=False)

	logger.info("Split sizes -> train: %s validation: %s audit: %s", len(train_frame), len(validation_frame), len(audit_frame))
	logger.info("Validation passed")
	logger.info("Exports completed")

	split_summary = build_split_summary(
		train_frame=train_frame,
		validation_frame=validation_frame,
		audit_frame=audit_frame,
		random_seed=seed,
		train_ratio=train_ratio,
		validation_ratio=validation_ratio,
		audit_ratio=audit_ratio,
	)

	summary_path = Path("results") / "split_summary.json"
	summary_path.parent.mkdir(parents=True, exist_ok=True)
	summary_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
	return split_summary


def main() -> None:
	"""Execute the full fairness-aware splitting pipeline."""

	configure_logging()
	config = SplitConfig()
	split_summary = create_split_plan(
		metadata_path=config.metadata_path,
		output_directory=config.output_directory,
		train_ratio=config.train_ratio,
		validation_ratio=config.validation_ratio,
		audit_ratio=config.audit_ratio,
		seed=config.random_seed,
	)

	train_size = split_summary["split_sizes"]["train"]
	validation_size = split_summary["split_sizes"]["validation"]
	audit_size = split_summary["split_sizes"]["audit"]
	print(f"train size: {train_size}")
	print(f"validation size: {validation_size}")
	print(f"audit size: {audit_size}")
	print("subgroup balance summary:")
	print(json.dumps(split_summary["subgroup_distributions"], indent=2))


if __name__ == "__main__":
	main()