from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.demographic_mapper import load_metadata
from tests.helpers import load_csv_strict


def test_metadata_fixture_has_required_columns_and_no_empty_rows(enriched_metadata_frame, tmp_path: Path) -> None:
	metadata_path = tmp_path / "metadata.csv"
	enriched_metadata_frame.loc[:, ["age", "gender", "race", "subgroup"]].head(12).to_csv(metadata_path, index=False)
	loaded = load_metadata(metadata_path)
	assert metadata_path.exists(), "Expected the temporary metadata.csv fixture to exist for validation"
	assert {"age", "gender", "race"}.issubset(loaded.columns), "metadata.csv must expose age, gender, and race columns"
	assert loaded[["age", "gender", "race"]].isna().sum().sum() == 0, "metadata.csv should not contain empty demographic rows"
	assert "subgroup" in enriched_metadata_frame.columns, "Processed metadata must include a subgroup column"


def test_processed_directories_exist(processed_dir: Path) -> None:
	assert processed_dir.exists() and processed_dir.is_dir(), "Processed directory is missing"
	assert (processed_dir / "images").exists(), "Processed image directory is missing"
	assert (processed_dir / "validation_pairs.csv").exists(), "Validation pairs CSV is missing"
	assert (processed_dir / "audit_pairs.csv").exists(), "Audit pairs CSV is missing"


def test_load_metadata_rejects_missing_columns(tmp_path: Path) -> None:
	broken_path = tmp_path / "metadata.csv"
	pd.DataFrame({"age": [30], "gender": ["Male"]}).to_csv(broken_path, index=False)
	with pytest.raises(ValueError, match="Missing required metadata columns"):
		load_metadata(broken_path)
