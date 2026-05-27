"""Image preprocessing utilities for fairness-sensitive face verification.

The functions in this module intentionally keep preprocessing conservative and
deterministic. They support safe PIL loading, RGB conversion, resizing,
ImageNet normalization, tensor conversion, batch preprocessing, and saving
processed images for the downstream training pipeline.

Fairness note: excessive preprocessing can alter facial characteristics or
distributional signals in ways that may disproportionately affect subgroup
performance. This module therefore limits itself to standard resize and
normalization operations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm


logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class PreprocessingConfig:
	"""Configuration for the image preprocessing pipeline."""

	image_size: int = DEFAULT_IMAGE_SIZE
	mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
	std: tuple[float, float, float] = (0.229, 0.224, 0.225)
	data_root: Path = Path("data")

	@property
	def raw_image_dir(self) -> Path:
		return self.data_root / "raw" / "images"

	@property
	def processed_image_dir(self) -> Path:
		return self.data_root / "processed" / "images"


def configure_logging() -> None:
	"""Configure module logging for standalone execution."""

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def ensure_processed_directories(config: PreprocessingConfig) -> None:
	"""Create the processed image directory structure if needed."""

	config.raw_image_dir.mkdir(parents=True, exist_ok=True)
	config.processed_image_dir.mkdir(parents=True, exist_ok=True)


def discover_image_files(image_directory: str | Path) -> list[Path]:
	"""Discover image files under the raw image directory.

	The search is recursive so the pipeline remains compatible with future
	organizing schemes used in Docker, Windows, or Colab environments.
	"""

	directory = Path(image_directory)
	if not directory.exists():
		return []

	extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
	return sorted(
		[path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
	)


def build_preprocess_transform(config: PreprocessingConfig | None = None) -> transforms.Compose:
	"""Build the deterministic preprocessing transform for model input."""

	selected_config = config or PreprocessingConfig()
	return transforms.Compose(
		[
			transforms.Resize((selected_config.image_size, selected_config.image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=selected_config.mean, std=selected_config.std),
		]
	)


def load_image_safely(image_path: str | Path) -> Image.Image:
	"""Load an image safely and convert it to RGB.

	Raises:
		FileNotFoundError: if the image file does not exist.
		ValueError: if the image is corrupted or cannot be decoded.
	"""

	path = Path(image_path)
	if not path.exists():
		raise FileNotFoundError(f"Image not found: {path}")

	try:
		with Image.open(path) as image:
			return image.convert("RGB")
	except (UnidentifiedImageError, OSError) as exc:
		raise ValueError(f"Corrupted or unreadable image: {path}") from exc


def preprocess_image(image_path: str | Path, config: PreprocessingConfig | None = None) -> torch.Tensor:
	"""Load, resize, normalize, and convert a single image to a tensor."""

	selected_config = config or PreprocessingConfig()
	transform = build_preprocess_transform(selected_config)
	image = load_image_safely(image_path)
	return transform(image)


def resize_for_saving(image: Image.Image, image_size: int) -> Image.Image:
	"""Resize an RGB image before saving the processed copy.

	The saved artifact is intentionally limited to standard resizing so the
	process remains fairness-safe and does not introduce aggressive visual
	changes.
	"""

	return image.resize((image_size, image_size), Image.Resampling.BILINEAR)


def save_processed_image(image: Image.Image, destination: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> Path:
	"""Save a resized RGB image to the processed dataset directory."""

	destination_path = Path(destination)
	destination_path.parent.mkdir(parents=True, exist_ok=True)
	resized_image = resize_for_saving(image, image_size)
	resized_image.save(destination_path, format="JPEG", quality=95)
	return destination_path


def preprocess_batch(
	image_paths: Iterable[str | Path],
	output_directory: str | Path,
	config: PreprocessingConfig | None = None,
) -> dict[str, Any]:
	"""Preprocess a batch of images and save processed outputs.

	Returns a summary dictionary containing counts for processed, skipped, and
	failed files. The function is deterministic for a fixed file ordering.
	"""

	selected_config = config or PreprocessingConfig()
	ensure_processed_directories(selected_config)
	output_root = Path(output_directory)
	output_root.mkdir(parents=True, exist_ok=True)

	processed_count = 0
	skipped_files: list[str] = []
	failed_files: list[str] = []

	for image_path in image_paths:
		path = Path(image_path)
		try:
			image = load_image_safely(path)
			destination = output_root / path.name
			save_processed_image(image, destination, selected_config.image_size)
			processed_count += 1
		except FileNotFoundError:
			logger.warning("Skipping missing image: %s", path)
			skipped_files.append(str(path))
		except ValueError:
			logger.warning("Skipping corrupted image: %s", path)
			failed_files.append(str(path))

	summary = {
		"processed_count": processed_count,
		"skipped_count": len(skipped_files),
		"failed_count": len(failed_files),
		"processed_directory": str(output_root),
		"image_size": selected_config.image_size,
		"mean": list(selected_config.mean),
		"std": list(selected_config.std),
		"skipped_files": skipped_files,
		"failed_files": failed_files,
	}

	logger.info(
		"Processed %s images, skipped %s missing images, and flagged %s corrupt images",
		processed_count,
		len(skipped_files),
		len(failed_files),
	)
	return summary


def preprocess_image_directory(
	input_directory: str | Path,
	output_directory: str | Path,
	config: PreprocessingConfig | None = None,
) -> dict[str, Any]:
	"""Preprocess all discoverable images in a directory tree.

	Images are loaded safely, converted to RGB, resized, and saved under the
	processed output directory without overwriting the raw source files.
	Normalization is preserved in the tensor preprocessing path for downstream
	training compatibility, while saved images remain standard RGB JPEGs.
	"""

	selected_config = config or PreprocessingConfig()
	input_root = Path(input_directory)
	output_root = Path(output_directory)
	output_root.mkdir(parents=True, exist_ok=True)

	image_files = discover_image_files(input_root)
	logger.info("Preprocessing started")
	logger.info("Input directory: %s", input_root)
	logger.info("Output directory: %s", output_root)
	logger.info("Number of images found: %s", len(image_files))

	processed_count = 0
	skipped_count = 0
	failure_count = 0
	processed_paths: list[str] = []

	for image_path in tqdm(image_files, desc="Preprocessing images", unit="image"):
		relative_path = image_path.relative_to(input_root)
		destination_path = output_root / relative_path
		destination_path = destination_path.with_suffix(".jpg")
		try:
			image = load_image_safely(image_path)
			_ = preprocess_image(image_path, selected_config)
			save_processed_image(image, destination_path, selected_config.image_size)
			processed_count += 1
			processed_paths.append(str(destination_path))
		except FileNotFoundError:
			logger.warning("Skipping missing image: %s", image_path)
			skipped_count += 1
		except ValueError:
			logger.warning("Skipping corrupt image: %s", image_path)
			skipped_count += 1
		except OSError as exc:
			logger.warning("Skipping image due to write/load error %s: %s", image_path, exc)
			skipped_count += 1
			failure_count += 1

	summary = {
		"input_directory": str(input_root),
		"output_directory": str(output_root),
		"number_of_images_found": len(image_files),
		"processed_count": processed_count,
		"skipped_count": skipped_count,
		"failed_count": failure_count,
		"processed_files": processed_paths,
		"image_size": selected_config.image_size,
		"mean": list(selected_config.mean),
		"std": list(selected_config.std),
	}

	write_preprocessing_summary(summary, output_root)
	logger.info("Processed image count: %s", processed_count)
	logger.info("Skipped/corrupt image count: %s", skipped_count)
	logger.info("Preprocessing completed")
	return summary


def write_preprocessing_summary(summary: dict[str, Any], output_directory: str | Path) -> Path:
	"""Write a preprocessing summary JSON file."""

	output_path = Path(output_directory)
	output_path.mkdir(parents=True, exist_ok=True)
	summary_path = output_path / "preprocessing_summary.json"
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	return summary_path


def prepare_preprocessing_plan(metadata_path: str | Path) -> pd.DataFrame:
	"""Load metadata and log the intended preprocessing steps.

	The returned DataFrame is unchanged. This function exists so future phases
	have a clean, testable entrypoint for preprocessing orchestration.
	"""

	metadata_frame = pd.read_csv(Path(metadata_path))
	logger.info("Loaded %s records for preprocessing plan", len(metadata_frame))
	logger.info("TODO: validate resize, normalization, and augmentation policies")
	return metadata_frame


def main() -> None:
	"""Execute preprocessing over the raw FairFace image directory."""

	configure_logging()
	config = PreprocessingConfig()
	ensure_processed_directories(config)

	input_directory = config.raw_image_dir
	output_directory = config.processed_image_dir

	if not input_directory.exists():
		logger.info("Preprocessing started")
		logger.info("Input directory: %s", input_directory)
		logger.info("Output directory: %s", output_directory)
		logger.info("Number of images found: 0")
		summary = {
			"input_directory": str(input_directory),
			"output_directory": str(output_directory),
			"number_of_images_found": 0,
			"processed_count": 0,
			"skipped_count": 0,
			"failed_count": 0,
			"processed_files": [],
			"image_size": config.image_size,
			"mean": list(config.mean),
			"std": list(config.std),
		}
		write_preprocessing_summary(summary, output_directory)
		logger.info("Processed image count: 0")
		logger.info("Skipped/corrupt image count: 0")
		logger.info("Preprocessing completed")
		print(json.dumps(summary, indent=2))
		return

	summary = preprocess_image_directory(input_directory, output_directory, config)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()