from __future__ import annotations

import pandas as pd

from src.data.demographic_mapper import AGE_BINS, assign_age_bin, enrich_metadata, map_race_to_skin_tone


def test_age_bin_definitions_are_present():
    assert set(AGE_BINS) == {"0-19", "20-39", "40-59", "60+"}, "Expected age bins are missing"


def test_skin_tone_mappings_are_present():
    expected = {
        "3": "Light",
        "0": "Light",
        "1": "Medium",
        "6": "Medium",
        "2": "Dark",
    }
    for race_code, skin_tone in expected.items():
        assert map_race_to_skin_tone(race_code) == skin_tone, f"Unexpected skin-tone mapping for FairFace race code {race_code}"


def test_subgroup_strings_are_generated_correctly():
    frame = pd.DataFrame(
        {
            "age": [17, 28, 44, 63],
            "gender": ["Male", "Female", "Male", "Female"],
            "race": ["3", "1", "2", "0"],
        }
    )
    enriched = enrich_metadata(frame)
    assert list(enriched["age_bin"]) == ["0-19", "20-39", "40-59", "60+"], "Age bins were not assigned correctly"
    assert list(enriched["skin_tone"]) == ["Light", "Medium", "Dark", "Light"], "Skin-tone mappings were not assigned correctly"
    assert enriched.loc[0, "subgroup"] == "Male_0-19_Light", "Subgroup label format is incorrect"
    assert enriched.loc[1, "subgroup"] == "Female_20-39_Medium", "Subgroup label format is incorrect"


def test_assign_age_bin_handles_string_and_numeric_inputs():
    assert assign_age_bin(12) == "0-19"
    assert assign_age_bin("37") == "20-39"
    assert assign_age_bin("61") == "60+"
