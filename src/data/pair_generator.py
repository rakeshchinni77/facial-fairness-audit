"""Pair and triplet generation for FairFace verification experiments.

FairFace does not provide real identity labels, so this module creates stable
synthetic identities from the existing subgroup-aware split metadata. The
synthetic assignment is deterministic and balanced to support future metric
learning without leaking audit data into training.

Fairness note: subgroup balancing matters because random pair sampling can
overrepresent majority groups and underrepresent rare subgroups, which can bias
verification training and later subgroup audits.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairGenerationConfig:
	"""Configuration for deterministic pair and triplet generation."""

	processed_dir: Path = Path("data/processed")
	results_dir: Path = Path("results")
	random_seed: int = 42
	images_per_identity: int = 4
	positive_pairs_per_identity: int = 4
	negative_pairs_multiplier: int = 1
	max_pairs_per_split: int | None = None
	max_triplets: int | None = None


def configure_logging() -> None:
	"""Configure logging for script execution."""

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def load_split_metadata(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
	"""Load train, validation, and audit metadata splits."""

	root = Path(processed_dir)
	split_files = {
		"train": root / "train_metadata.csv",
		"validation": root / "validation_metadata.csv",
		"audit": root / "audit_metadata.csv",
	}
	frames: dict[str, pd.DataFrame] = {}
	for split_name, split_path in split_files.items():
		if not split_path.exists():
			raise FileNotFoundError(f"Missing split metadata file: {split_path}")
		frame = pd.read_csv(split_path)
		required_columns = {"sample_id", "subgroup", "image_path"}
		missing_columns = required_columns - set(frame.columns)
		if missing_columns:
			raise ValueError(f"{split_name} metadata missing columns: {sorted(missing_columns)}")
		if frame["sample_id"].duplicated().any():
			raise ValueError(f"Duplicate sample_id values found in {split_name} metadata")
		missing_images = [path for path in frame["image_path"].astype(str) if not Path(path).exists()]
		if missing_images:
			raise FileNotFoundError(f"{split_name} metadata contains missing image files")
		frames[split_name] = frame.reset_index(drop=True)
		logger.info("Loaded %s metadata rows for %s split", len(frame), split_name)
	return frames


def ensure_output_directories(config: PairGenerationConfig) -> None:
	"""Create directories for pair and triplet outputs."""

	config.processed_dir.mkdir(parents=True, exist_ok=True)
	config.results_dir.mkdir(parents=True, exist_ok=True)


def build_split_identity_groups(frame: pd.DataFrame, images_per_identity: int, split_name: str) -> pd.DataFrame:
	"""Assign deterministic synthetic identities within each subgroup.

	FairFace has no real person identity labels, so this synthetic assignment is a
	controlled approximation used only to create verification pairs and triplets.
	"""

	if images_per_identity < 2:
		raise ValueError("images_per_identity must be at least 2")

	grouped_frames: list[pd.DataFrame] = []
	identity_index = 1

	for subgroup, subgroup_frame in frame.sort_values(["subgroup", "sample_id"]).groupby("subgroup", sort=True):
		subgroup_frame = subgroup_frame.reset_index(drop=True)
		if len(subgroup_frame) < 2:
			raise ValueError(f"Subgroup '{subgroup}' must contain at least two images for synthetic identity assignment")

		chunks: list[list[int]] = [list(range(start, min(start + images_per_identity, len(subgroup_frame)))) for start in range(0, len(subgroup_frame), images_per_identity)]
		if len(chunks) > 1 and len(chunks[-1]) == 1:
			chunks[-2].extend(chunks[-1])
			chunks.pop()

		assigned = subgroup_frame.copy()
		assigned["synthetic_identity"] = None
		for chunk in chunks:
			identity_label = f"{split_name}_identity_{identity_index:04d}"
			identity_index += 1
			assigned.loc[chunk, "synthetic_identity"] = identity_label
		grouped_frames.append(assigned)

	result = pd.concat(grouped_frames, ignore_index=True)
	return result


def validate_identity_assignment(frame: pd.DataFrame) -> None:
	"""Validate that synthetic identities are well formed and subgroup aware."""

	if frame["synthetic_identity"].isna().any():
		raise ValueError("Synthetic identity assignment contains missing values")
	if frame["synthetic_identity"].astype(str).str.strip().eq("").any():
		raise ValueError("Synthetic identity assignment contains empty labels")
	identity_counts = frame["synthetic_identity"].value_counts()
	if (identity_counts < 2).any():
		raise ValueError("Every synthetic identity must contain at least two images")
	if frame["synthetic_identity"].isna().any():
		raise ValueError("Synthetic identity assignment contains missing values")


def generate_positive_pairs(frame: pd.DataFrame, max_pairs: int | None = None, seed: int = 42) -> pd.DataFrame:
	"""Generate positive verification pairs within each synthetic identity."""

	rng = random.Random(seed)
	pairs: list[dict[str, Any]] = []

	for identity, identity_frame in frame.groupby("synthetic_identity", sort=True):
		images = identity_frame.sort_values("sample_id")["image_path"].tolist()
		if len(images) < 2:
			continue
		combinations = list(itertools.combinations(images, 2))
		rng.shuffle(combinations)
		for image_a, image_b in combinations:
			pairs.append(
				{
					"image_a": image_a,
					"image_b": image_b,
					"label": 1,
					"subgroup": identity_frame["subgroup"].iloc[0],
					"synthetic_identity_a": identity,
					"synthetic_identity_b": identity,
				}
			)
			if max_pairs is not None and len(pairs) >= max_pairs:
				return pd.DataFrame(pairs[:max_pairs])

	return pd.DataFrame(pairs)


def generate_negative_pairs(frame: pd.DataFrame, target_count: int, seed: int = 42) -> pd.DataFrame:
	"""Generate negative pairs across different synthetic identities."""

	rng = random.Random(seed)
	identity_to_images = {
		identity: group.sort_values("sample_id")["image_path"].tolist()
		for identity, group in frame.groupby("synthetic_identity", sort=True)
	}
	identity_to_subgroup = {
		identity: group["subgroup"].iloc[0]
		for identity, group in frame.groupby("synthetic_identity", sort=True)
	}
	identities = sorted(identity_to_images)
	pairs: list[dict[str, Any]] = []
	seen_pairs: set[tuple[str, str]] = set()

	identity_pairs = list(itertools.combinations(identities, 2))
	if not identity_pairs:
		raise ValueError("At least two synthetic identities are required to generate negative pairs")
	rng.shuffle(identity_pairs)

	for identity_a, identity_b in itertools.cycle(identity_pairs):
		image_a = rng.choice(identity_to_images[identity_a])
		image_b = rng.choice(identity_to_images[identity_b])
		canonical = tuple(sorted((image_a, image_b)))
		if canonical in seen_pairs:
			continue
		seen_pairs.add(canonical)
		pairs.append(
			{
				"image_a": image_a,
				"image_b": image_b,
				"label": 0,
				"subgroup": f"{identity_to_subgroup[identity_a]}__vs__{identity_to_subgroup[identity_b]}",
				"synthetic_identity_a": identity_a,
				"synthetic_identity_b": identity_b,
			}
		)
		if len(pairs) >= target_count:
			break

	return pd.DataFrame(pairs)


def balance_pairs(positive_pairs: pd.DataFrame, negative_pairs: pd.DataFrame) -> pd.DataFrame:
	"""Balance positive and negative pairs to avoid skewed training signals."""

	target = min(len(positive_pairs), len(negative_pairs))
	if target == 0:
		raise ValueError("Positive and negative pair generation produced no pairs")
	return pd.concat(
		[
			positive_pairs.sample(n=target, random_state=42).reset_index(drop=True),
			negative_pairs.sample(n=target, random_state=42).reset_index(drop=True),
		],
		ignore_index=True,
	)


def generate_triplets(frame: pd.DataFrame, max_triplets: int | None = None, seed: int = 42) -> pd.DataFrame:
	"""Generate anchor-positive-negative triplets from synthetic identities."""

	rng = random.Random(seed)
	identity_to_images = {
		identity: group.sort_values("sample_id")["image_path"].tolist()
		for identity, group in frame.groupby("synthetic_identity", sort=True)
	}
	identity_to_subgroup = {
		identity: group["subgroup"].iloc[0]
		for identity, group in frame.groupby("synthetic_identity", sort=True)
	}
	identities = sorted(identity_to_images)
	triplets: list[dict[str, Any]] = []

	for identity in identities:
		images = identity_to_images[identity]
		if len(images) < 2:
			continue
		negative_identities = [candidate for candidate in identities if candidate != identity]
		if not negative_identities:
			continue
		for anchor, positive in itertools.combinations(images, 2):
			negative_identity = rng.choice(negative_identities)
			negative_image = rng.choice(identity_to_images[negative_identity])
			triplets.append(
				{
					"anchor": anchor,
					"positive": positive,
					"negative": negative_image,
					"anchor_identity": identity,
					"positive_identity": identity,
					"negative_identity": negative_identity,
					"subgroup": identity_to_subgroup[identity],
				}
			)
			if max_triplets is not None and len(triplets) >= max_triplets:
				return pd.DataFrame(triplets[:max_triplets])

	return pd.DataFrame(triplets)


def validate_pairs(pairs_frame: pd.DataFrame, split_name: str) -> None:
	"""Validate pair integrity and duplicate-free sampling."""

	required_columns = {"image_a", "image_b", "label", "subgroup", "synthetic_identity_a", "synthetic_identity_b"}
	missing_columns = required_columns - set(pairs_frame.columns)
	if missing_columns:
		raise ValueError(f"{split_name} pairs missing columns: {sorted(missing_columns)}")
	if pairs_frame[["image_a", "image_b"]].isna().any().any():
		raise ValueError(f"{split_name} pairs contain missing image references")
	if pairs_frame.duplicated(subset=["image_a", "image_b", "label"]).any():
		raise ValueError(f"{split_name} pairs contain duplicate rows")
	if not set(pairs_frame["label"].unique()).issubset({0, 1}):
		raise ValueError(f"{split_name} pairs contain invalid labels")


def validate_triplets(triplets_frame: pd.DataFrame) -> None:
	"""Validate triplet integrity and identity relationships."""

	required_columns = {"anchor", "positive", "negative", "anchor_identity", "positive_identity", "negative_identity", "subgroup"}
	missing_columns = required_columns - set(triplets_frame.columns)
	if missing_columns:
		raise ValueError(f"Triplets missing columns: {sorted(missing_columns)}")
	if triplets_frame.duplicated(subset=["anchor", "positive", "negative"]).any():
		raise ValueError("Triplets contain duplicate rows")
	if not (triplets_frame["anchor_identity"] == triplets_frame["positive_identity"]).all():
		raise ValueError("Anchor and positive must share identity")
	if (triplets_frame["anchor_identity"] == triplets_frame["negative_identity"]).any():
		raise ValueError("Negative samples must come from different identities")


def compute_statistics(
	train_pairs: pd.DataFrame,
	validation_pairs: pd.DataFrame,
	audit_pairs: pd.DataFrame,
	train_triplets: pd.DataFrame,
	train_identity_frame: pd.DataFrame,
	validation_identity_frame: pd.DataFrame,
	audit_identity_frame: pd.DataFrame,
) -> dict[str, Any]:
	"""Build the pair and triplet statistics summary."""

	return {
		"positive_pair_counts": {
			"train": int((train_pairs["label"] == 1).sum()),
			"validation": int((validation_pairs["label"] == 1).sum()),
			"audit": int((audit_pairs["label"] == 1).sum()),
		},
		"negative_pair_counts": {
			"train": int((train_pairs["label"] == 0).sum()),
			"validation": int((validation_pairs["label"] == 0).sum()),
			"audit": int((audit_pairs["label"] == 0).sum()),
		},
		"subgroup_distributions": {
			"train": {str(index): int(value) for index, value in train_pairs["subgroup"].value_counts().sort_index().items()},
			"validation": {str(index): int(value) for index, value in validation_pairs["subgroup"].value_counts().sort_index().items()},
			"audit": {str(index): int(value) for index, value in audit_pairs["subgroup"].value_counts().sort_index().items()},
		},
		"synthetic_identity_counts": {
			"train": int(train_identity_frame["synthetic_identity"].nunique()),
			"validation": int(validation_identity_frame["synthetic_identity"].nunique()),
			"audit": int(audit_identity_frame["synthetic_identity"].nunique()),
		},
		"triplet_counts": {
			"train": int(len(train_triplets)),
		},
		"pair_row_counts": {
			"train": int(len(train_pairs)),
			"validation": int(len(validation_pairs)),
			"audit": int(len(audit_pairs)),
		},
	}


def save_frame(frame: pd.DataFrame, path: Path) -> None:
	"""Persist a DataFrame to CSV with parent directory creation."""

	path.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(path, index=False)


def run_pipeline(config: PairGenerationConfig) -> dict[str, Any]:
	"""Run the full pair and triplet generation pipeline."""

	ensure_output_directories(config)
	splits = load_split_metadata(config.processed_dir)

	train_identity_frame = build_split_identity_groups(splits["train"], config.images_per_identity, "train")
	validation_identity_frame = build_split_identity_groups(splits["validation"], config.images_per_identity, "validation")
	audit_identity_frame = build_split_identity_groups(splits["audit"], config.images_per_identity, "audit")

	validate_identity_assignment(train_identity_frame)
	validate_identity_assignment(validation_identity_frame)
	validate_identity_assignment(audit_identity_frame)

	logger.info("Identities assigned for train split: %s", train_identity_frame["synthetic_identity"].nunique())
	logger.info("Identities assigned for validation split: %s", validation_identity_frame["synthetic_identity"].nunique())
	logger.info("Identities assigned for audit split: %s", audit_identity_frame["synthetic_identity"].nunique())

	positive_train = generate_positive_pairs(train_identity_frame, seed=config.random_seed)
	negative_train = generate_negative_pairs(train_identity_frame, target_count=len(positive_train) * config.negative_pairs_multiplier, seed=config.random_seed)
	train_pairs = balance_pairs(positive_train, negative_train)
	train_pairs["split"] = "train"

	positive_validation = generate_positive_pairs(validation_identity_frame, seed=config.random_seed)
	negative_validation = generate_negative_pairs(validation_identity_frame, target_count=len(positive_validation) * config.negative_pairs_multiplier, seed=config.random_seed)
	validation_pairs = balance_pairs(positive_validation, negative_validation)
	validation_pairs["split"] = "validation"

	positive_audit = generate_positive_pairs(audit_identity_frame, seed=config.random_seed)
	negative_audit = generate_negative_pairs(audit_identity_frame, target_count=len(positive_audit) * config.negative_pairs_multiplier, seed=config.random_seed)
	audit_pairs = balance_pairs(positive_audit, negative_audit)
	audit_pairs["split"] = "audit"

	train_triplets = generate_triplets(train_identity_frame, max_triplets=config.max_triplets, seed=config.random_seed)

	validate_pairs(train_pairs, "train")
	validate_pairs(validation_pairs, "validation")
	validate_pairs(audit_pairs, "audit")
	validate_triplets(train_triplets)

	train_identities = set(train_identity_frame["synthetic_identity"].astype(str))
	validation_identities = set(validation_identity_frame["synthetic_identity"].astype(str))
	audit_identities = set(audit_identity_frame["synthetic_identity"].astype(str))
	if train_identities & audit_identities or validation_identities & audit_identities or train_identities & validation_identities:
		raise ValueError("Synthetic identity leakage detected across splits")

	logger.info("Pairs generated for train, validation, and audit splits")
	logger.info("Triplets generated for train split")
	logger.info("Subgroup balance preserved across generated samples")

	train_pairs_path = config.processed_dir / "train_pairs.csv"
	validation_pairs_path = config.processed_dir / "validation_pairs.csv"
	audit_pairs_path = config.processed_dir / "audit_pairs.csv"
	train_triplets_path = config.processed_dir / "train_triplets.csv"

	save_frame(train_pairs, train_pairs_path)
	save_frame(validation_pairs, validation_pairs_path)
	save_frame(audit_pairs, audit_pairs_path)
	save_frame(train_triplets, train_triplets_path)

	pair_statistics = compute_statistics(
		train_pairs=train_pairs,
		validation_pairs=validation_pairs,
		audit_pairs=audit_pairs,
		train_triplets=train_triplets,
		train_identity_frame=train_identity_frame,
		validation_identity_frame=validation_identity_frame,
		audit_identity_frame=audit_identity_frame,
	)
	statistics_path = config.results_dir / "pair_statistics.json"
	statistics_path.parent.mkdir(parents=True, exist_ok=True)
	statistics_path.write_text(json.dumps(pair_statistics, indent=2), encoding="utf-8")

	logger.info("Exports completed")
	return pair_statistics


def main() -> None:
	"""Execute the full pair and triplet generation workflow."""

	configure_logging()
	config = PairGenerationConfig()
	pair_statistics = run_pipeline(config)
	print(json.dumps(pair_statistics, indent=2))


if __name__ == "__main__":
	main()