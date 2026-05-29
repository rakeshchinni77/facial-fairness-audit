from __future__ import annotations

import torch

from PIL import Image

from src.data.augment import build_audit_transforms, build_training_transforms, build_validation_transforms
from src.data.preprocess import load_image_safely, preprocess_image


def test_preprocessing_outputs_rgb_images_and_finite_tensors(sample_image_path) -> None:
	image = load_image_safely(sample_image_path)
	assert image.mode == "RGB", "Loaded preprocessing input must be converted to RGB"
	processed = preprocess_image(sample_image_path)
	assert processed.shape == (3, 224, 224), f"Preprocessed tensor must be 3x224x224, got {tuple(processed.shape)}"
	assert torch.isfinite(processed).all(), "Normalized preprocessing tensor contains non-finite values"


def test_augmentation_policies_do_not_crash(sample_image_path) -> None:
	image = load_image_safely(sample_image_path)
	for transform in (build_training_transforms(), build_validation_transforms(), build_audit_transforms()):
		output = transform(image)
		assert output.shape == (3, 224, 224), "Transform must emit 224x224 tensors"
		assert torch.isfinite(output).all(), "Transform output contains non-finite values"
