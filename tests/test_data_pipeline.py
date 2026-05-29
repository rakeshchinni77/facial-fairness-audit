from __future__ import annotations

import pandas as pd
import pytest

from src.data.demographic_mapper import enrich_metadata, load_metadata, validate_subgroups


def test_metadata_csv_exists_and_has_required_columns(project_root):
    metadata_path = project_root / "data" / "interim" / "metadata.csv"
    assert metadata_path.exists(), f"Expected metadata file is missing: {metadata_path}"

    frame = load_metadata(metadata_path)
    assert not frame.empty, "metadata.csv should not be empty"
    for column in ("age", "gender", "race"):
        assert column in frame.columns, f"Required column missing from metadata.csv: {column}"
        assert frame[column].notna().all(), f"metadata.csv contains empty values in column: {column}"


def test_processed_directories_exist(project_root):
    processed_dir = project_root / "data" / "processed"
    raw_dir = project_root / "data" / "raw"
    interim_dir = project_root / "data" / "interim"
    assert processed_dir.exists(), f"Processed directory missing: {processed_dir}"
    assert raw_dir.exists(), f"Raw directory missing: {raw_dir}"
    assert interim_dir.exists(), f"Interim directory missing: {interim_dir}"


def test_enriched_metadata_creates_subgroup_column(project_root):
    metadata_path = project_root / "data" / "interim" / "metadata.csv"
    frame = load_metadata(metadata_path)
    enriched = enrich_metadata(frame)
    validate_subgroups(enriched)
    assert "subgroup" in enriched.columns, "Enriched metadata must include subgroup labels"
    assert enriched["subgroup"].astype(str).str.strip().ne("").all(), "Subgroup labels must be non-empty"


def test_load_metadata_rejects_malformed_csv(tmp_path):
    malformed = tmp_path / "metadata.csv"
    pd.DataFrame({"age": [12], "gender": ["Male"]}).to_csv(malformed, index=False)

    with pytest.raises(ValueError, match="Missing required metadata columns"):
        load_metadata(malformed)


def test_load_metadata_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_metadata(tmp_path / "missing_metadata.csv")
