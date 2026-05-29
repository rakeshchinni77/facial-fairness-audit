from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import ARTIFACTS_DIR, DATA_DIR, PLOTS_DIR, PROCESSED_DIR, RESULTS_DIR, load_csv_strict


@pytest.fixture(scope="session")
def project_root() -> Path:
	return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir() -> Path:
	return DATA_DIR


@pytest.fixture(scope="session")
def processed_dir() -> Path:
	return PROCESSED_DIR


@pytest.fixture(scope="session")
def results_dir() -> Path:
	return RESULTS_DIR


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
	return ARTIFACTS_DIR


@pytest.fixture(scope="session")
def plots_dir() -> Path:
	return PLOTS_DIR


@pytest.fixture(scope="session")
def sample_image_path(processed_dir: Path) -> Path:
	image_dir = processed_dir / "images"
	images = sorted(path for path in image_dir.glob("*.jpg") if path.is_file())
	if not images:
		raise FileNotFoundError(f"No processed images found in {image_dir}")
	return images[0]


@pytest.fixture(scope="session")
def enriched_metadata_frame(processed_dir: Path):
	return load_csv_strict(
		processed_dir / "enriched_metadata.csv",
		required_columns={"age", "gender", "race", "subgroup"},
	)


@pytest.fixture(scope="session")
def train_metadata_frame(processed_dir: Path):
	return load_csv_strict(
		processed_dir / "train_metadata.csv",
		required_columns={"sample_id", "subgroup", "gender_mapped", "age_bin", "skin_tone"},
	)


@pytest.fixture(scope="session")
def validation_metadata_frame(processed_dir: Path):
	return load_csv_strict(
		processed_dir / "validation_metadata.csv",
		required_columns={"sample_id", "subgroup", "gender_mapped", "age_bin", "skin_tone"},
	)


@pytest.fixture(scope="session")
def audit_metadata_frame(processed_dir: Path):
	return load_csv_strict(
		processed_dir / "audit_metadata.csv",
		required_columns={"sample_id", "subgroup", "gender_mapped", "age_bin", "skin_tone"},
	)


@pytest.fixture(scope="session")
def train_pairs_frame(processed_dir: Path):
	return load_csv_strict(
		processed_dir / "train_pairs.csv",
		required_columns={"image_a", "image_b", "label", "subgroup", "synthetic_identity_a", "synthetic_identity_b"},
	)


@pytest.fixture(scope="session")
def validation_pairs_frame(processed_dir: Path):
	return load_csv_strict(
		processed_dir / "validation_pairs.csv",
		required_columns={"image_a", "image_b", "label", "subgroup", "synthetic_identity_a", "synthetic_identity_b"},
	)


@pytest.fixture(scope="session")
def audit_pairs_frame(processed_dir: Path):
	return load_csv_strict(
		processed_dir / "audit_pairs.csv",
		required_columns={"image_a", "image_b", "label", "subgroup", "synthetic_identity_a", "synthetic_identity_b"},
	)


@pytest.fixture(scope="session")
def train_triplets_frame(processed_dir: Path):
	return load_csv_strict(
		processed_dir / "train_triplets.csv",
		required_columns={"anchor", "positive", "negative", "subgroup", "anchor_identity", "positive_identity", "negative_identity"},
	)
