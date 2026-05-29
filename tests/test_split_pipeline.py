from __future__ import annotations

import pytest

from tests.helpers import load_json_strict


def test_split_summary_reports_expected_ratios_and_sizes(results_dir) -> None:
	summary = load_json_strict(results_dir / "split_summary.json")
	assert summary["split_ratios"] == {"train": 0.7, "validation": 0.15, "audit": 0.15}, "Split ratios must remain 70/15/15"
	assert summary["split_sizes"] == {"train": 70, "validation": 15, "audit": 15}, "Split sizes must match the published split summary"


def test_splits_are_disjoint_and_audit_isolated(train_metadata_frame, validation_metadata_frame, audit_metadata_frame) -> None:
	train_ids = set(train_metadata_frame["sample_id"].astype(str))
	validation_ids = set(validation_metadata_frame["sample_id"].astype(str))
	audit_ids = set(audit_metadata_frame["sample_id"].astype(str))

	assert not (train_ids & validation_ids), "Train and validation splits must be disjoint"
	assert not (train_ids & audit_ids), "Train and audit splits must be disjoint"
	assert not (validation_ids & audit_ids), "Validation and audit splits must be disjoint"
	assert len(train_ids | validation_ids | audit_ids) == len(train_metadata_frame) + len(validation_metadata_frame) + len(audit_metadata_frame), "Split union must equal the full metadata population"


def test_splits_preserve_subgroup_coverage(train_metadata_frame, validation_metadata_frame, audit_metadata_frame, results_dir) -> None:
	summary = load_json_strict(results_dir / "split_summary.json")
	expected_groups = set(summary["split_sizes"].keys())  # not used for groups; keep summary access exercised
	train_groups = set(train_metadata_frame["subgroup"].astype(str))
	validation_groups = set(validation_metadata_frame["subgroup"].astype(str))
	audit_groups = set(audit_metadata_frame["subgroup"].astype(str))
	assert train_groups == validation_groups == audit_groups, "Every split must preserve subgroup coverage"
	assert len(train_groups) >= 1, "At least one subgroup must be present in the splits"
