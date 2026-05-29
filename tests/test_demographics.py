from __future__ import annotations

import pandas as pd

from src.data.demographic_mapper import AGE_BINS, RACE_TO_SKIN_TONE_MAP, assign_age_bin, enrich_metadata, map_race_to_skin_tone


def test_age_bins_are_mapped_correctly() -> None:
	assert assign_age_bin(0) == "0-19"
	assert assign_age_bin(19) == "0-19"
	assert assign_age_bin(20) == "20-39"
	assert assign_age_bin(39) == "20-39"
	assert assign_age_bin(40) == "40-59"
	assert assign_age_bin(59) == "40-59"
	assert assign_age_bin(60) == "60+"
	assert set(AGE_BINS) == {"0-19", "20-39", "40-59", "60+"}, "Age bin definitions must stay aligned with the audit spec"


def test_skin_tone_mappings_are_stable() -> None:
	assert map_race_to_skin_tone(3) == "Light"
	assert map_race_to_skin_tone(0) == "Light"
	assert map_race_to_skin_tone(1) == "Medium"
	assert map_race_to_skin_tone(6) == "Medium"
	assert map_race_to_skin_tone(2) == "Dark"
	assert RACE_TO_SKIN_TONE_MAP["white"] == "Light"


def test_enrich_metadata_builds_expected_subgroup_strings() -> None:
	frame = pd.DataFrame(
		{
			"age": [12, 31, 48, 70],
			"gender": ["Male", "Female", "1", "0"],
			"race": ["3", "1", "2", "0"],
		}
	)
	enriched = enrich_metadata(frame)
	expected = ["Male_0-19_Light", "Female_20-39_Medium", "Female_40-59_Dark", "Male_60+_Light"]
	assert enriched["subgroup"].tolist() == expected, "Subgroup strings must combine mapped gender, age bin, and skin tone"
