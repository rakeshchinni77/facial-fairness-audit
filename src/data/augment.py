"""Fairness-safe augmentation utilities for future face verification training.

The augmentation policy is intentionally conservative to avoid distorting facial
attributes or introducing demographic shifts. Audit transforms avoid any
augmentation entirely so fairness evaluation remains faithful to the raw data
distribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from torchvision import transforms


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AugmentationConfig:
	"""Configuration for conservative augmentation policies."""

	image_size: int = 224
	horizontal_flip_probability: float = 0.5
	brightness_factor: float = 0.1
	rotation_degrees: float = 5.0


def build_training_transforms(config: AugmentationConfig | None = None) -> transforms.Compose:
	"""Build conservative training transforms.

	The allowed operations are intentionally limited to avoid aggressive visual
	changes that could alter facial characteristics or disproportionately affect
	fairness-sensitive features.
	"""

	selected_config = config or AugmentationConfig()
	logger.info(
		"Building training transforms with flip_p=%s brightness=±%s rotation=±%s image_size=%s",
		selected_config.horizontal_flip_probability,
		selected_config.brightness_factor,
		selected_config.rotation_degrees,
		selected_config.image_size,
	)
	return transforms.Compose(
		[
			transforms.RandomHorizontalFlip(p=selected_config.horizontal_flip_probability),
			transforms.RandomApply(
				[transforms.ColorJitter(brightness=selected_config.brightness_factor)],
				p=0.5,
			),
			transforms.RandomRotation(degrees=selected_config.rotation_degrees),
			transforms.RandomResizedCrop(size=(selected_config.image_size, selected_config.image_size), scale=(0.92, 1.0), ratio=(0.95, 1.05)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def build_validation_transforms(image_size: int = 224) -> transforms.Compose:
	"""Build validation transforms without augmentation."""

	return transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def build_audit_transforms(image_size: int = 224) -> transforms.Compose:
	"""Build audit transforms with no augmentation.

	Audit evaluation must reflect the original data distribution as closely as
	possible, so this transform intentionally avoids random crop, rotation,
	brightness changes, or flips.
	"""

	return transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)