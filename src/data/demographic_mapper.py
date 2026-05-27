"""Demographic engineering for subgroup-aware fairness auditing.

This module loads FairFace metadata, normalizes demographic fields, derives
age bins and a skin-tone proxy, and writes evaluator-compatible artifacts.

Fairness note: the race-to-skin-tone mapping is only an approximate proxy used
for auditing. It does not claim to represent actual skin tone, and it should
not be treated as a biologically precise or socially complete label.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)

SUPPORTED_GENDERS = ["Male", "Female"]
AGE_BINS: dict[str, list[int]] = {
	"0-19": [0, 19],
	"20-39": [20, 39],
	"40-59": [40, 59],
	"60+": [60, 150],
}
SKIN_TONE_SCALE: dict[str, list[int]] = {
	"Light": [1, 2],
	"Medium": [3, 4],
	"Dark": [5, 6],
}

NUMERIC_GENDER_MAP: dict[str, str] = {
	"0": "Male",
	"1": "Female",
}

RACE_TO_SKIN_TONE_MAP: dict[str, str] = {
	"white": "Light",
	"east asian": "Light",
	"indian": "Medium",
	"southeast asian": "Medium",
	"black": "Dark",
	"middle eastern": "Medium",
	"latino hispanic": "Medium",
	"latino_hispanic": "Medium",
}

NUMERIC_RACE_MAP: dict[str, str] = {
	"0": "East Asian",
	"1": "Indian",
	"2": "Black",
	"3": "White",
	"4": "Middle Eastern",
	"5": "Latino_Hispanic",
	"6": "Southeast Asian",
}

FAIRFACE_AGE_CLASS_TO_BIN: dict[int, str] = {
	0: "0-19",
	1: "0-19",
	2: "0-19",
	3: "20-39",
	4: "20-39",
	5: "40-59",
	6: "60+",
	7: "60+",
}


@dataclass(frozen=True)
class DemographicConfig:
	"""Configuration for demographic engineering."""

	metadata_path: Path = Path("data/interim/metadata.csv")
	output_enriched_metadata_path: Path = Path("data/processed/enriched_metadata.csv")
	demographics_output_path: Path = Path("results/demographics.json")


def configure_logging() -> None:
	"""Configure logging for standalone execution."""

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def ensure_output_directories(config: DemographicConfig) -> None:
	"""Create required output directories."""

	config.output_enriched_metadata_path.parent.mkdir(parents=True, exist_ok=True)
	config.demographics_output_path.parent.mkdir(parents=True, exist_ok=True)


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
	"""Load the interim FairFace metadata and validate required columns."""

	frame = pd.read_csv(Path(metadata_path))
	required_columns = {"age", "gender", "race"}
	missing_columns = required_columns - set(frame.columns)
	if missing_columns:
		raise ValueError(f"Missing required metadata columns: {sorted(missing_columns)}")

	frame = frame.dropna(subset=["age", "gender", "race"]).reset_index(drop=True)
	logger.info("Loaded %s metadata rows", len(frame))
	return frame


def normalize_gender(value: Any) -> str:
	"""Normalize gender labels to the supported canonical values."""

	if pd.isna(value):
		logger.warning("Missing gender value; using safe default 'Unknown'")
		return "Unknown"

	text = str(value).strip().lower()
	if text in NUMERIC_GENDER_MAP:
		return NUMERIC_GENDER_MAP[text]
	if text in {"male", "m"}:
		return "Male"
	if text in {"female", "f"}:
		return "Female"
	logger.warning("Unsupported gender label '%s'; using safe default 'Unknown'", value)
	return "Unknown"


def parse_age_value(value: Any) -> int:
	"""Parse an age-like value into a bounded integer for binning."""

	if pd.isna(value):
		logger.warning("Missing age value; using safe default 0")
		return 0

	text = str(value).strip()
	if text.endswith("+"):
		text = text[:-1]
	if "-" in text:
		left, right = text.split("-", 1)
		return max(0, min(150, int(left)))
	try:
		return max(0, min(150, int(float(text))))
	except ValueError:
		logger.warning("Unsupported age value '%s'; using safe default 0", value)
		return 0


def assign_age_bin(age_value: Any) -> str:
	"""Assign a FairFace age value to the required bin labels."""

	age = parse_age_value(age_value)
	if age in FAIRFACE_AGE_CLASS_TO_BIN:
		return FAIRFACE_AGE_CLASS_TO_BIN[age]
	if 0 <= age <= 19:
		return "0-19"
	if 20 <= age <= 39:
		return "20-39"
	if 40 <= age <= 59:
		return "40-59"
	return "60+"


def normalize_race(value: Any) -> str:
	"""Normalize race labels and decode numeric FairFace encodings."""

	if pd.isna(value):
		logger.warning("Missing race value; using safe default 'unknown'")
		return "unknown"

	text = str(value).strip().lower().replace("_", " ")
	if text in NUMERIC_RACE_MAP:
		return NUMERIC_RACE_MAP[text]
	logger.warning("Unsupported race label '%s'; using safe default 'unknown'", value)
	return "unknown"


def map_race_to_skin_tone(race_value: Any) -> str:
	"""Map race labels to a coarse skin-tone proxy for fairness auditing.

	This is an approximate proxy only. It is useful for audit grouping, but it
	does not represent real skin tone with high fidelity and should not be used
	as a literal biological attribute.
	"""

	traced_race = normalize_race(race_value).lower().replace("_", " ")
	if traced_race not in RACE_TO_SKIN_TONE_MAP:
		logger.warning(
			"Race label '%s' could not be mapped to a skin-tone proxy; using safe default 'Unknown'",
			race_value,
		)
		return "Unknown"
	return RACE_TO_SKIN_TONE_MAP[traced_race]


def validate_demographics(frame: pd.DataFrame) -> None:
	"""Validate demographic columns for critical missing values."""

	critical_columns = ["age", "gender", "race"]
	missing_count = int(frame[critical_columns].isna().sum().sum())
	if missing_count:
		raise ValueError(f"Found {missing_count} critical missing demographic values")


def enrich_metadata(frame: pd.DataFrame) -> pd.DataFrame:
	"""Add normalized demographic columns and subgroup labels."""

	enriched = frame.copy()
	enriched["gender_mapped"] = enriched["gender"].apply(normalize_gender)
	enriched["age_bin"] = enriched["age"].apply(assign_age_bin)
	enriched["skin_tone"] = enriched["race"].apply(map_race_to_skin_tone)
	enriched["subgroup"] = (
		enriched["gender_mapped"].astype(str).str.strip()
		+ "_"
		+ enriched["age_bin"].astype(str).str.strip()
		+ "_"
		+ enriched["skin_tone"].astype(str).str.strip()
	)
	return enriched


def validate_subgroups(frame: pd.DataFrame) -> None:
	"""Ensure subgroup labels were created and are non-empty."""

	if "subgroup" not in frame.columns:
		raise ValueError("Missing subgroup column")
	if frame["subgroup"].isna().any() or (frame["subgroup"].astype(str).str.strip() == "").any():
		raise ValueError("Found invalid subgroup labels")


def build_subgroup_statistics(frame: pd.DataFrame) -> dict[str, Any]:
	"""Compute subgroup counts for reporting and downstream auditing."""

	counts = frame["subgroup"].value_counts(dropna=False).sort_index()
	return {
		"total_rows": int(len(frame)),
		"subgroup_counts": {str(index): int(value) for index, value in counts.items()},
		"gender_counts": {str(index): int(value) for index, value in frame["gender_mapped"].value_counts().items()},
		"age_bin_counts": {str(index): int(value) for index, value in frame["age_bin"].value_counts().items()},
		"skin_tone_counts": {str(index): int(value) for index, value in frame["skin_tone"].value_counts().items()},
	}


def save_demographics_json(output_path: str | Path) -> Path:
	"""Write the evaluator-compatible demographics definition file."""

	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"gender": SUPPORTED_GENDERS,
		"age_bins": AGE_BINS,
		"skin_tone_scale": SKIN_TONE_SCALE,
	}
	path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	return path


def save_enriched_metadata(frame: pd.DataFrame, output_path: str | Path) -> Path:
	"""Persist enriched metadata for later audit phases."""

	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(path, index=False)
	return path


def run_demographic_engineering(config: DemographicConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
	"""Execute the demographic engineering workflow."""

	ensure_output_directories(config)
	metadata_frame = load_metadata(config.metadata_path)
	validate_demographics(metadata_frame)
	enriched_frame = enrich_metadata(metadata_frame)
	validate_subgroups(enriched_frame)
	stats = build_subgroup_statistics(enriched_frame)

	save_demographics_json(config.demographics_output_path)
	save_enriched_metadata(enriched_frame, config.output_enriched_metadata_path)

	return enriched_frame, stats


def main() -> None:
	"""Run the demographic engineering pipeline and print subgroup statistics."""

	configure_logging()
	config = DemographicConfig()
	logger.info("Loading metadata from %s", config.metadata_path)
	enriched_frame, stats = run_demographic_engineering(config)
	logger.info("Saved demographics definition to %s", config.demographics_output_path)
	logger.info("Saved enriched metadata to %s", config.output_enriched_metadata_path)
	logger.info("Subgroup statistics: %s", stats)
	print(json.dumps(stats, indent=2))
	print(f"Generated {len(enriched_frame)} enriched demographic rows")


if __name__ == "__main__":
	main()