"""Dataset acquisition utilities for the FairFace subset.

This module downloads a reproducible subset of the FairFace dataset via the
Hugging Face `datasets` library, validates the records, and writes organized
metadata artifacts for downstream preprocessing and audit phases.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


logger = logging.getLogger(__name__)

SUPPORTED_DATASET_CONFIGS: tuple[str, ...] = ("0.25", "1.25")


@dataclass(frozen=True)
class DatasetDownloadConfig:
	"""Configuration for downloading and organizing a FairFace subset."""

	dataset_name: str = "HuggingFaceM4/FairFace"
	dataset_config: str = "0.25"
	split: str = "train"
	subset_size: int = 10_000
	seed: int = 42
	data_root: Path = Path("data")

	@property
	def raw_dir(self) -> Path:
		return self.data_root / "raw"

	@property
	def interim_dir(self) -> Path:
		return self.data_root / "interim"

	@property
	def processed_dir(self) -> Path:
		return self.data_root / "processed"


def configure_logging() -> None:
	"""Configure module-level logging for direct script execution."""

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def ensure_directories(config: DatasetDownloadConfig) -> None:
	"""Create the project data directories required by the pipeline."""

	for directory in (
		config.raw_dir,
		config.interim_dir,
		config.processed_dir,
		config.raw_dir / "images",
		config.raw_dir / "metadata",
	):
		directory.mkdir(parents=True, exist_ok=True)


def load_fairface_dataset(config: DatasetDownloadConfig) -> Any:
	"""Load the FairFace dataset through Hugging Face with a clear error path."""

	try:
		from datasets import load_dataset
	except ImportError as exc:  # pragma: no cover - runtime dependency guard.
		raise ImportError(
			"The Hugging Face 'datasets' package is required to download FairFace."
		) from exc

	if config.dataset_config not in SUPPORTED_DATASET_CONFIGS:
		raise ValueError(
			"Invalid dataset_config '%s'. Supported configs: %s"
			% (config.dataset_config, list(SUPPORTED_DATASET_CONFIGS))
		)

	logger.info(
		"Loading dataset='%s' config='%s' split='%s' subset_size=%s",
		config.dataset_name,
		config.dataset_config,
		config.split,
		config.subset_size,
	)
	return load_dataset(config.dataset_name, config.dataset_config, split=config.split)


def select_subset(dataset: Any, subset_size: int, seed: int) -> Any:
	"""Select a deterministic subset of the dataset."""

	total_rows = len(dataset)
	effective_size = min(subset_size, total_rows)
	if effective_size <= 0:
		raise ValueError("subset_size must be greater than zero.")

	if effective_size == total_rows:
		return dataset

	logger.info(
		"Selecting deterministic subset: %s rows from %s total rows",
		effective_size,
		total_rows,
	)
	shuffled = dataset.shuffle(seed=seed)
	return shuffled.select(range(effective_size))


def normalize_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
	"""Normalize metadata column names and drop invalid records."""

	normalized = frame.copy()
	normalized.columns = [column.strip().lower().replace(" ", "_") for column in normalized.columns]

	expected_columns = {"age", "gender", "race"}
	missing_columns = expected_columns - set(normalized.columns)
	if missing_columns:
		raise ValueError(f"Missing required FairFace metadata columns: {sorted(missing_columns)}")

	normalized = normalized.dropna(subset=["age", "gender", "race"]).reset_index(drop=True)
	normalized["age"] = normalized["age"].astype(str).str.strip()
	normalized["gender"] = normalized["gender"].astype(str).str.strip()
	normalized["race"] = normalized["race"].astype(str).str.strip()
	return normalized


def is_valid_image(image: Any) -> bool:
	"""Return True when an image-like object can be opened and verified."""

	try:
		if isinstance(image, Image.Image):
			candidate = image.copy()
			candidate.load()
			return True
		if isinstance(image, (str, Path)):
			with Image.open(image) as opened:
				opened.load()
			return True
	except (UnidentifiedImageError, OSError, ValueError):
		return False
	return False


def save_image(image: Any, destination: Path) -> bool:
	"""Persist a dataset image to disk when possible."""

	try:
		if isinstance(image, Image.Image):
			image.convert("RGB").save(destination, format="JPEG", quality=95)
			return True
		if isinstance(image, (str, Path)):
			with Image.open(image) as opened:
				opened.convert("RGB").save(destination, format="JPEG", quality=95)
			return True
	except (UnidentifiedImageError, OSError, ValueError):
		return False
	return False


def validate_record(record: dict[str, Any]) -> bool:
	"""Validate a FairFace record for image presence and metadata consistency."""

	image = record.get("image")
	if image is None or not is_valid_image(image):
		return False

	for field in ("age", "gender", "race"):
		value = record.get(field)
		if value is None or str(value).strip() == "":
			return False
	return True


def build_metadata_frame(dataset: Any, image_paths: list[str]) -> pd.DataFrame:
	"""Create a clean metadata frame from a dataset subset."""

	frame = dataset.to_pandas()
	frame = normalize_metadata_frame(frame)
	if len(frame) != len(image_paths):
		raise ValueError("Metadata rows and saved image paths are misaligned.")
	frame.insert(0, "image_path", image_paths)
	frame.insert(0, "sample_id", [f"fairface_{index:07d}" for index in range(len(frame))])
	return frame


def build_dataset_summary(frame: pd.DataFrame, subset_size: int, valid_records: int) -> dict[str, Any]:
	"""Build a lightweight summary of the downloaded dataset subset."""

	return {
		"dataset_name": "HuggingFaceM4/FairFace",
		"subset_size_requested": subset_size,
		"records_validated": valid_records,
		"records_saved": int(len(frame)),
		"columns": list(frame.columns),
		"gender_distribution": frame["gender"].value_counts().to_dict(),
		"race_distribution": frame["race"].value_counts().to_dict(),
		"age_distribution": frame["age"].value_counts().to_dict(),
	}


def download_fairface_subset(config: DatasetDownloadConfig) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
	"""Download, validate, and organize a deterministic FairFace subset."""

	ensure_directories(config)
	dataset = load_fairface_dataset(config)
	subset = select_subset(dataset, config.subset_size, config.seed)

	valid_records: list[dict[str, Any]] = []
	image_paths: list[str] = []
	total_records = len(subset)

	for index, record in enumerate(tqdm(subset, total=total_records, desc="Validating FairFace subset")):
		if not validate_record(record):
			logger.warning("Skipping invalid record at index %s", index)
			continue

		image_path = config.raw_dir / "images" / f"fairface_{index:07d}.jpg"
		if not save_image(record["image"], image_path):
			logger.warning("Skipping unreadable image at index %s", index)
			continue

		cleaned_record = {
			"age": str(record["age"]).strip(),
			"gender": str(record["gender"]).strip(),
			"race": str(record["race"]).strip(),
			"image_path": str(image_path),
		}
		valid_records.append(cleaned_record)
		image_paths.append(str(image_path))

	if not valid_records:
		raise RuntimeError("No valid FairFace records were found in the selected subset.")

	metadata_frame = pd.DataFrame(valid_records)
	metadata_frame.insert(0, "sample_id", [f"fairface_{index:07d}" for index in range(len(metadata_frame))])

	manifest = {
		"dataset_name": config.dataset_name,
		"dataset_config": config.dataset_config,
		"split": config.split,
		"subset_size_requested": config.subset_size,
		"subset_size_saved": int(len(metadata_frame)),
		"seed": config.seed,
		"raw_image_directory": str(config.raw_dir / "images"),
		"metadata_file": str(config.interim_dir / "metadata.csv"),
	}
	summary = build_dataset_summary(metadata_frame, config.subset_size, total_records)

	return metadata_frame, manifest, summary


def save_download_artifacts(
	config: DatasetDownloadConfig,
	metadata_frame: pd.DataFrame,
	manifest: dict[str, Any],
	summary: dict[str, Any],
) -> None:
	"""Persist the metadata and summary artifacts to disk."""

	metadata_path = config.interim_dir / "metadata.csv"
	manifest_path = config.interim_dir / "dataset_manifest.json"
	summary_path = config.interim_dir / "dataset_summary.json"

	metadata_frame.to_csv(metadata_path, index=False)
	manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for the dataset download pipeline."""

	parser = argparse.ArgumentParser(description="Download and organize a FairFace subset.")
	parser.add_argument("--dataset-name", default="HuggingFaceM4/FairFace")
	parser.add_argument("--dataset-config", default="0.25")
	parser.add_argument("--split", default="train")
	parser.add_argument("--subset-size", type=int, default=10_000)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--data-root", default="data")
	return parser.parse_args()


def main() -> None:
	"""Run the FairFace subset download and organization pipeline."""

	configure_logging()
	args = parse_args()
	config = DatasetDownloadConfig(
		dataset_name=args.dataset_name,
		dataset_config=args.dataset_config,
		split=args.split,
		subset_size=args.subset_size,
		seed=args.seed,
		data_root=Path(args.data_root),
	)

	logger.info("Starting FairFace dataset acquisition pipeline")
	metadata_frame, manifest, summary = download_fairface_subset(config)
	save_download_artifacts(config, metadata_frame, manifest, summary)

	logger.info("Downloaded and organized %s valid FairFace records", len(metadata_frame))
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()