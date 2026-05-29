from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.data.augment import build_training_transforms, build_validation_transforms
from src.data.preprocess import build_preprocess_transform, load_image_safely, preprocess_image
from tests.helpers import write_rgb_image


def test_preprocessing_outputs_rgb_tensor_with_expected_shape(tmp_path):
    image_path = write_rgb_image(tmp_path / "sample.png", size=(80, 120), mode="L")
    image = load_image_safely(image_path)
    assert image.mode == "RGB", "Images must be converted to RGB during safe loading"

    tensor = preprocess_image(image_path)
    assert tensor.shape == (3, 224, 224), f"Unexpected tensor shape: {tuple(tensor.shape)}"
    assert torch.isfinite(tensor).all(), "Preprocessed tensor contains non-finite values"


def test_validation_transform_is_finite(tmp_path):
    image_path = write_rgb_image(tmp_path / "validation.jpg")
    image = Image.open(image_path)
    tensor = build_validation_transforms()(image)
    assert tensor.shape == (3, 224, 224), f"Unexpected validation tensor shape: {tuple(tensor.shape)}"
    assert torch.isfinite(tensor).all(), "Validation transform produced non-finite values"


def test_training_augmentation_does_not_crash(tmp_path):
    image_path = write_rgb_image(tmp_path / "train.jpg")
    image = Image.open(image_path)
    tensor = build_training_transforms()(image)
    assert tensor.shape == (3, 224, 224), f"Unexpected augmented tensor shape: {tuple(tensor.shape)}"
    assert torch.isfinite(tensor).all(), "Augmentation transform produced non-finite values"


def test_preprocess_transform_configuration_is_consistent():
    transform = build_preprocess_transform()
    assert transform is not None, "Preprocess transform should be constructible"


def test_load_image_safely_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image_safely(tmp_path / "missing.jpg")
