from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

EXPECTED_PLOTS = [
    "roc_curve_publication.png",
    "det_curve_publication.png",
    "subgroup_far_chart.png",
    "subgroup_frr_chart.png",
    "fairness_heatmap.png",
    "mitigation_comparison.png",
    "disparity_gap_plot.png",
    "fairness_dashboard.png",
]

EXPECTED_RESULTS_JSONS = [
    "analysis.json",
    "cross_group_metrics.json",
    "demographics.json",
    "fairness_comparison.json",
    "fairness_summary.json",
    "initial_audit.json",
    "mitigated_audit.json",
    "overall_metrics.json",
    "pair_statistics.json",
    "rebalancing_summary.json",
    "split_summary.json",
    "threshold_analysis.json",
]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_json_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def assert_json_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            assert_json_finite(nested, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            assert_json_finite(nested, f"{path}[{index}]")
        return
    if isinstance(value, float):
        assert math.isfinite(value), f"Non-finite numeric value at {path}: {value!r}"


def write_rgb_image(path: Path, *, size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (96, 128, 160), mode: str = "RGB") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fill_color: int | tuple[int, int, int]
    if mode == "L" and isinstance(color, tuple):
        fill_color = int(sum(color) / len(color))
    else:
        fill_color = color
    image = Image.new(mode, size=size, color=fill_color)
    image.save(path)
    return path


def build_balanced_metadata_frame(rows_per_group: int = 20) -> pd.DataFrame:
    groups = [
        ("Male", "0-19", "Light", "Male_0-19_Light"),
        ("Male", "20-39", "Medium", "Male_20-39_Medium"),
        ("Male", "40-59", "Dark", "Male_40-59_Dark"),
        ("Female", "0-19", "Dark", "Female_0-19_Dark"),
        ("Female", "20-39", "Light", "Female_20-39_Light"),
    ]
    rows: list[dict[str, Any]] = []
    for group_index, (gender, age_bin, skin_tone, subgroup) in enumerate(groups):
        for row_index in range(rows_per_group):
            rows.append(
                {
                    "sample_id": f"sample_{group_index:02d}_{row_index:03d}",
                    "age": 20 + row_index,
                    "gender": gender,
                    "race": skin_tone,
                    "image_path": f"data/raw/images/{subgroup}_{row_index:03d}.jpg",
                    "gender_mapped": gender,
                    "age_bin": age_bin,
                    "skin_tone": skin_tone,
                    "subgroup": subgroup,
                }
            )
    return pd.DataFrame(rows)


def make_pair_triplet_frame(image_paths: list[Path], subgroup: str) -> pd.DataFrame:
    if len(image_paths) < 4:
        raise ValueError("At least four images are required to build a synthetic identity frame")
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(image_paths):
        rows.append(
            {
                "sample_id": f"{subgroup}_{index:03d}",
                "subgroup": subgroup,
                "image_path": str(path),
            }
        )
    return pd.DataFrame(rows)
