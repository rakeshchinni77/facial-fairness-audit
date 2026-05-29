from __future__ import annotations

import pandas as pd
import pytest

from src.data.split_dataset import create_split_plan, load_enriched_metadata, validate_ratios
from tests.helpers import build_balanced_metadata_frame


def test_split_pipeline_produces_701515_ratios(tmp_path):
    metadata = build_balanced_metadata_frame(rows_per_group=20)
    metadata_path = tmp_path / "enriched_metadata.csv"
    output_dir = tmp_path / "splits"
    metadata.to_csv(metadata_path, index=False)

    result = create_split_plan(metadata_path, output_dir, seed=7)

    train_path = output_dir / "train_metadata.csv"
    validation_path = output_dir / "validation_metadata.csv"
    audit_path = output_dir / "audit_metadata.csv"

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    audit = pd.read_csv(audit_path)

    total_rows = len(metadata)
    assert len(train) == 70, f"Expected 70 train rows, found {len(train)}"
    assert len(validation) == 15, f"Expected 15 validation rows, found {len(validation)}"
    assert len(audit) == 15, f"Expected 15 audit rows, found {len(audit)}"
    assert len(train) + len(validation) + len(audit) == total_rows, "Split sizes do not sum to the source size"

    assert set(train["sample_id"]).isdisjoint(validation["sample_id"]), "Train and validation sets overlap"
    assert set(train["sample_id"]).isdisjoint(audit["sample_id"]), "Train and audit sets overlap"
    assert set(validation["sample_id"]).isdisjoint(audit["sample_id"]), "Validation and audit sets overlap"

    source_groups = set(metadata["subgroup"])
    for split_name, frame in (("train", train), ("validation", validation), ("audit", audit)):
        split_groups = set(frame["subgroup"])
        assert source_groups.issubset(split_groups), f"{split_name} split is missing subgroup coverage"

    assert result["split_sizes"]["train"] == 70, "Split summary returned the wrong train size"


def test_split_ratio_validation_rejects_invalid_totals():
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_ratios(0.5, 0.25, 0.10)


def test_load_enriched_metadata_rejects_missing_columns(tmp_path):
    malformed = tmp_path / "enriched_metadata.csv"
    pd.DataFrame({"sample_id": ["s1"], "subgroup": ["Male_0-19_Light"]}).to_csv(malformed, index=False)

    with pytest.raises(ValueError, match="Missing required enriched metadata columns"):
        load_enriched_metadata(malformed)
