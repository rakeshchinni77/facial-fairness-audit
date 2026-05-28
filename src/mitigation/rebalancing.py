"""Fairness-aware subgroup rebalancing utilities.

Balanced sampling improves fairness because a model trained mostly on dominant
subgroups can learn a decision boundary that fits the majority well while
stabilizing poorly on smaller groups. In biometric verification, that often shows
up as elevated FRR or FAR for underrepresented demographics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RebalancingConfig:
	"""Configuration for subgroup rebalancing."""

	triplets_path: Path = Path("data/processed/train_triplets.csv")
	output_path: Path = Path("results/rebalancing_summary.json")
	balanced_seed: int = 42
	target_strategy: str = "max_count"


def configure_logging() -> None:
	"""Configure structured logging for standalone use."""

	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def load_triplets(csv_path: str | Path) -> pd.DataFrame:
	"""Load the triplet training data and preserve subgroup labels."""

	path = Path(csv_path)
	if not path.exists():
		raise FileNotFoundError(f"Triplet CSV not found: {path}")
	frame = pd.read_csv(path)
	required_columns = {"anchor", "positive", "negative", "subgroup"}
	missing_columns = required_columns - set(frame.columns)
	if missing_columns:
		raise ValueError(f"Triplet CSV missing columns: {sorted(missing_columns)}")
	logger.info("Triplets loaded | path=%s | rows=%s", path, len(frame))
	return frame


def subgroup_distribution_summary(frame: pd.DataFrame) -> dict[str, int]:
	"""Return deterministic subgroup counts sorted by subgroup name."""

	if "subgroup" not in frame.columns:
		raise ValueError("Triplet frame missing subgroup column")
	counts = frame["subgroup"].astype(str).value_counts(dropna=False).sort_index()
	summary = {str(group): int(count) for group, count in counts.items()}
	logger.info("Subgroup distribution summarized | subgroup_count=%s", len(summary))
	return summary


def compute_subgroup_weights(frame: pd.DataFrame, epsilon: float = 1e-8) -> dict[str, float]:
	"""Compute inverse-frequency weights for subgroup-aware fine-tuning.

	The smaller the subgroup support, the larger the weight. This does not change
	the architecture; it only changes how strongly each subgroup contributes to
	the loss so weak groups are not drowned out by majority groups.
	"""

	counts = subgroup_distribution_summary(frame)
	total = float(sum(counts.values()))
	num_groups = float(max(len(counts), 1))
	weights = {
		group: float(total / (num_groups * max(count, 1) + epsilon))
		for group, count in counts.items()
	}
	mean_weight = float(np.mean(list(weights.values()))) if weights else 1.0
	if mean_weight > 0:
		weights = {group: float(weight / mean_weight) for group, weight in weights.items()}
	logger.info("Subgroup weights computed | group_count=%s", len(weights))
	return weights


def balanced_triplet_sampling(
	frame: pd.DataFrame,
	weights: dict[str, float] | None = None,
	seed: int = 42,
	target_count: int | None = None,
) -> pd.DataFrame:
	"""Oversample low-support subgroups to create a balanced triplet frame.

	A deterministic oversampling step reduces subgroup imbalance so fairness-aware
	fine-tuning sees a more even distribution of demographic slices.
	"""

	if frame.empty:
		return frame.copy()
	if "subgroup" not in frame.columns:
		raise ValueError("Triplet frame missing subgroup column")

	rng = np.random.default_rng(seed)
	grouped = {str(group): group_frame.reset_index(drop=True) for group, group_frame in frame.groupby(frame["subgroup"].astype(str), sort=True)}
	if not grouped:
		return frame.copy()
	if target_count is None:
		target_count = max(len(group_frame) for group_frame in grouped.values())

	balanced_frames: list[pd.DataFrame] = []
	for subgroup_name, group_frame in sorted(grouped.items(), key=lambda item: item[0]):
		group_size = len(group_frame)
		if group_size == 0:
			continue
		replace = group_size < target_count
		chosen_indices = rng.choice(group_frame.index.to_numpy(), size=target_count, replace=replace)
		balanced_frames.append(group_frame.loc[chosen_indices].reset_index(drop=True))
		logger.info(
			"Balanced sampling | subgroup=%s | original=%s | balanced=%s | replace=%s",
			subgroup_name,
			group_size,
			target_count,
			replace,
		)

	balanced_frame = pd.concat(balanced_frames, ignore_index=True)
	balanced_frame = balanced_frame.sort_values(by=["subgroup", "anchor", "positive", "negative"], kind="mergesort").reset_index(drop=True)
	return balanced_frame


def build_rebalancing_summary(
	original_frame: pd.DataFrame,
	balanced_frame: pd.DataFrame,
	weights: dict[str, float],
	seed: int,
) -> dict[str, Any]:
	"""Create a JSON-serializable summary of the rebalancing process."""

	original_counts = subgroup_distribution_summary(original_frame)
	balanced_counts = subgroup_distribution_summary(balanced_frame)
	balanced_weights = {group: round(float(weight), 6) for group, weight in sorted(weights.items(), key=lambda item: item[0])}
	return {
		"seed": int(seed),
		"original_subgroup_counts": original_counts,
		"balanced_subgroup_counts": balanced_counts,
		"subgroup_weights": balanced_weights,
		"balanced_rows": int(len(balanced_frame)),
		"original_rows": int(len(original_frame)),
	}


def save_rebalancing_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
	"""Persist the rebalancing summary as deterministic JSON."""

	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
	logger.info("Rebalancing summary exported to %s", path)
	return path


def run_rebalancing(config: RebalancingConfig | None = None) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
	"""Load, rebalance, and summarize the triplet training distribution."""

	configure_logging()
	cfg = config or RebalancingConfig()
	original_frame = load_triplets(cfg.triplets_path)
	weights = compute_subgroup_weights(original_frame)
	balanced_frame = balanced_triplet_sampling(original_frame, weights=weights, seed=cfg.balanced_seed)
	summary = build_rebalancing_summary(original_frame, balanced_frame, weights, cfg.balanced_seed)
	save_rebalancing_summary(summary, cfg.output_path)
	return balanced_frame, weights, summary
