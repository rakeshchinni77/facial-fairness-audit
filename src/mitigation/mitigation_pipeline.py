"""Fairness-aware mitigation pipeline for the facial verification model.

This pipeline fine-tunes the existing best checkpoint with subgroup-balanced
sampling and subgroup-weighted triplet loss. It intentionally avoids changing
architecture, embedding dimensionality, or the audit methodology. The goal is to
reduce demographic unfairness without collapsing overall verification quality.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.mitigation.rebalancing import RebalancingConfig, build_rebalancing_summary, balanced_triplet_sampling, compute_subgroup_weights, load_triplets, save_rebalancing_summary
from src.mitigation.weighted_loss import FairnessWeightedTripletLoss
from src.models.embedding_model import FaceEmbeddingModel
from src.models.triplet_network import TripletNetwork
from src.training.callbacks import EarlyStopping, build_checkpoint_state, save_checkpoint
from src.training.validator import run_validation_epoch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MitigationConfig:
	"""Configuration for fairness-aware fine-tuning."""

	processed_dir: Path = Path("data/processed")
	artifacts_dir: Path = Path("artifacts")
	results_dir: Path = Path("results")
	checkpoint_path: Path = Path("artifacts/best_model.pth")
	mitigated_checkpoint_path: Path = Path("artifacts/mitigated_model.pth")
	best_mitigated_checkpoint_path: Path = Path("artifacts/best_mitigated_model.pth")
	rebalancing_summary_path: Path = Path("results/rebalancing_summary.json")
	training_summary_path: Path = Path("artifacts/mitigation_training_summary.json")
	device: str = "cpu"
	batch_size: int = 16
	validation_batch_size: int = 16
	learning_rate: float = 1e-5
	epochs: int = 1
	patience: int = 2
	margin: float = 0.3
	random_seed: int = 42
	image_size: int = 224
	max_train_rows: int | None = 240
	max_validation_rows: int | None = 64
	freeze_backbone: bool = True
	train_backbone_block: bool = True
	embedding_dim: int = 128
	pretrained: bool = False


def configure_logging() -> None:
	"""Configure structured logging for mitigation runs."""

	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


class FrameTripletDataset(Dataset):
	"""Triplet dataset backed by a pandas frame."""

	def __init__(self, frame: pd.DataFrame, image_size: int = 224) -> None:
		if frame.empty:
			raise ValueError("Triplet frame cannot be empty")
		required_columns = {"anchor", "positive", "negative", "subgroup"}
		missing_columns = required_columns - set(frame.columns)
		if missing_columns:
			raise ValueError(f"Triplet frame missing columns: {sorted(missing_columns)}")
		self.frame = frame.reset_index(drop=True).copy()
		self.transform = transforms.Compose(
			[
				transforms.Resize((image_size, image_size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]
		)

	def __len__(self) -> int:
		return len(self.frame)

	def _load_image(self, image_path: str) -> torch.Tensor:
		try:
			with Image.open(image_path) as image:
				return self.transform(image.convert("RGB"))
		except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
			raise ValueError(f"Unable to load mitigation image: {image_path}") from exc

	def __getitem__(self, index: int) -> dict[str, Any]:
		row = self.frame.iloc[int(index)]
		return {
			"anchor": self._load_image(str(row["anchor"])),
			"positive": self._load_image(str(row["positive"])),
			"negative": self._load_image(str(row["negative"])),
			"subgroup": str(row["subgroup"]),
		}


class FramePairDataset(Dataset):
	"""Validation pair dataset backed by a pandas frame."""

	def __init__(self, frame: pd.DataFrame, image_size: int = 224) -> None:
		if frame.empty:
			raise ValueError("Validation frame cannot be empty")
		required_columns = {"image_a", "image_b", "label"}
		missing_columns = required_columns - set(frame.columns)
		if missing_columns:
			raise ValueError(f"Validation frame missing columns: {sorted(missing_columns)}")
		self.frame = frame.reset_index(drop=True).copy()
		self.transform = transforms.Compose(
			[
				transforms.Resize((image_size, image_size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]
		)

	def __len__(self) -> int:
		return len(self.frame)

	def _load_image(self, image_path: str) -> torch.Tensor:
		try:
			with Image.open(image_path) as image:
				return self.transform(image.convert("RGB"))
		except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
			raise ValueError(f"Unable to load mitigation validation image: {image_path}") from exc

	def __getitem__(self, index: int) -> dict[str, Any]:
		row = self.frame.iloc[int(index)]
		return {
			"image_a": self._load_image(str(row["image_a"])),
			"image_b": self._load_image(str(row["image_b"])),
			"label": torch.tensor(int(row["label"]), dtype=torch.float32),
			"subgroup": str(row.get("subgroup", "unknown")),
		}


def _seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.set_num_threads(1)


def _balanced_subset(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
	"""Select an equal-count subgroup subset for a fast mitigation smoke test."""

	if max_rows is None or frame.empty:
		return frame.reset_index(drop=True).copy()
	subgroups = sorted(frame["subgroup"].astype(str).unique())
	if not subgroups:
		return frame.reset_index(drop=True).copy()
	per_group = max(1, max_rows // len(subgroups))
	rng = np.random.default_rng(seed)
	selected_frames: list[pd.DataFrame] = []
	for subgroup in subgroups:
		group_frame = frame[frame["subgroup"].astype(str) == subgroup].reset_index(drop=True)
		if group_frame.empty:
			continue
		replace = len(group_frame) < per_group
		indices = rng.choice(group_frame.index.to_numpy(), size=per_group, replace=replace)
		selected_frames.append(group_frame.loc[indices].reset_index(drop=True))
	if not selected_frames:
		return frame.reset_index(drop=True).copy()
	return pd.concat(selected_frames, ignore_index=True).sort_values(by=["subgroup", "anchor", "positive", "negative"], kind="mergesort").reset_index(drop=True)


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
	if isinstance(checkpoint, dict):
		for key in ("model_state_dict", "state_dict", "embedding_model_state_dict"):
			value = checkpoint.get(key)
			if isinstance(value, dict):
				return value
		if all(isinstance(key, str) for key in checkpoint.keys()):
			return checkpoint
	raise ValueError("Unsupported checkpoint format")


def _normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
	normalized: dict[str, Any] = {}
	for key, value in state_dict.items():
		new_key = key
		if new_key.startswith("module."):
			new_key = new_key[len("module.") :]
		if new_key.startswith("embedding_model."):
			new_key = new_key[len("embedding_model.") :]
		normalized[new_key] = value
	return normalized


def load_mitigated_model(config: MitigationConfig, subgroup_weights: dict[str, float]) -> TripletNetwork:
	"""Load the existing best checkpoint and prepare it for fairness fine-tuning."""

	model = TripletNetwork(
		FaceEmbeddingModel(
			embedding_dim=config.embedding_dim,
			pretrained=config.pretrained,
			freeze_backbone=config.freeze_backbone,
		)
	).to(config.device)
	checkpoint_path = config.checkpoint_path
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Mitigation checkpoint not found: {checkpoint_path}")
	checkpoint = torch.load(checkpoint_path, map_location=config.device)
	state_dict = _normalize_state_dict(_extract_state_dict(checkpoint))
	missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
	if missing_keys:
		logger.warning("Mitigation checkpoint missing keys: %s", sorted(missing_keys))
	if unexpected_keys:
		logger.warning("Mitigation checkpoint unexpected keys: %s", sorted(unexpected_keys))

	if config.freeze_backbone and hasattr(model.embedding_model, "backbone"):
		for parameter in model.embedding_model.backbone.parameters():
			parameter.requires_grad = False
		if hasattr(model.embedding_model.backbone, "backbone") and hasattr(model.embedding_model.backbone.backbone, "layer4"):
			for parameter in model.embedding_model.backbone.backbone.layer4.parameters():
				parameter.requires_grad = config.train_backbone_block
		for parameter in model.embedding_model.projection.parameters():
			parameter.requires_grad = True
		for parameter in model.embedding_model.batch_norm.parameters():
			parameter.requires_grad = True
	logger.info("Mitigated model initialized | subgroup_count=%s", len(subgroup_weights))
	return model


def build_optimizer(model: nn.Module, learning_rate: float) -> AdamW:
	trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
	if not trainable_parameters:
		raise ValueError("No trainable parameters found for mitigation")
	return AdamW(trainable_parameters, lr=learning_rate, weight_decay=1e-4)


def build_scheduler(optimizer: torch.optim.Optimizer) -> ReduceLROnPlateau:
	return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)


def run_mitigation_epoch(
	model: TripletNetwork,
	loss_fn: FairnessWeightedTripletLoss,
	optimizer: torch.optim.Optimizer,
	loader: DataLoader,
	device: torch.device,
) -> float:
	"""Run one fairness-aware fine-tuning epoch."""

	model.train()
	total_loss = 0.0
	total_batches = 0
	for batch in loader:
		anchor = batch["anchor"].to(device)
		positive = batch["positive"].to(device)
		negative = batch["negative"].to(device)
		subgroup_labels = [str(label) for label in batch["subgroup"]]
		optimizer.zero_grad(set_to_none=True)
		anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
		loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding, subgroup_labels=subgroup_labels)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		total_loss += float(loss.item())
		total_batches += 1
	return total_loss / max(total_batches, 1)


def build_training_summary(
	epochs_completed: int,
	final_training_loss: float,
	fairness_weights: dict[str, float],
	validation_loss: float,
	checkpoint_paths: dict[str, str],
	rebalancing_summary: dict[str, Any],
	best_validation_loss: float,
) -> dict[str, Any]:
	return {
		"epochs": int(epochs_completed),
		"final_training_loss": round(float(final_training_loss), 6),
		"subgroup_rebalancing_enabled": True,
		"weighted_triplet_loss_enabled": True,
		"fairness_weights": {group: round(float(weight), 6) for group, weight in sorted(fairness_weights.items(), key=lambda item: item[0])},
		"validation_loss": round(float(validation_loss), 6),
		"best_validation_loss": round(float(best_validation_loss), 6),
		"checkpoint_paths": checkpoint_paths,
		"rebalancing_summary": rebalancing_summary,
	}


def save_training_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
	logger.info("Mitigation training summary exported to %s", path)
	return path


def run_mitigation(config: MitigationConfig | None = None) -> dict[str, Any]:
	"""Run fairness-aware fine-tuning and save mitigated checkpoints."""

	configure_logging()
	cfg = config or MitigationConfig()
	_seed_everything(cfg.random_seed)
	device = torch.device(cfg.device)
	logger.info("Mitigation started on device=%s", device)

	original_triplets = load_triplets(cfg.processed_dir / "train_triplets.csv")
	weights = compute_subgroup_weights(original_triplets)
	balanced_triplets = balanced_triplet_sampling(original_triplets, weights=weights, seed=cfg.random_seed)
	rebalancing_summary = build_rebalancing_summary(original_triplets, balanced_triplets, weights, cfg.random_seed)
	save_rebalancing_summary(rebalancing_summary, cfg.rebalancing_summary_path)
	print(f"REBALANCED_SUBGROUPS={len(rebalancing_summary['balanced_subgroup_counts'])}")

	validation_frame = pd.read_csv(cfg.processed_dir / "validation_pairs.csv")
	if cfg.max_validation_rows is not None:
		validation_frame = validation_frame.head(cfg.max_validation_rows).reset_index(drop=True)
	balanced_triplets = _balanced_subset(balanced_triplets, cfg.max_train_rows, cfg.random_seed)

	train_dataset = FrameTripletDataset(balanced_triplets, image_size=cfg.image_size)
	validation_dataset = FramePairDataset(validation_frame, image_size=cfg.image_size)
	train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
	validation_loader = DataLoader(validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=0)

	model = load_mitigated_model(cfg, weights).to(device)
	loss_fn = FairnessWeightedTripletLoss(margin=cfg.margin, subgroup_weight_map=weights).to(device)
	optimizer = build_optimizer(model, cfg.learning_rate)
	scheduler = build_scheduler(optimizer)
	early_stopping = EarlyStopping(patience=cfg.patience)

	best_validation_loss = float("inf")
	artifacts_dir = cfg.artifacts_dir
	artifacts_dir.mkdir(parents=True, exist_ok=True)
	mitigated_checkpoint_path = cfg.mitigated_checkpoint_path
	best_mitigated_checkpoint_path = cfg.best_mitigated_checkpoint_path

	last_validation_loss = float("inf")
	epochs_completed = 0
	last_training_loss = float("inf")
	for epoch in range(1, cfg.epochs + 1):
		logger.info("Mitigation epoch %s/%s started", epoch, cfg.epochs)
		train_loss = run_mitigation_epoch(model, loss_fn, optimizer, train_loader, device)
		last_training_loss = float(train_loss)
		validation_result = run_validation_epoch(model.embedding_model, validation_loader, device, margin=cfg.margin)
		last_validation_loss = float(validation_result.validation_loss)
		scheduler.step(last_validation_loss)
		epochs_completed = epoch

		checkpoint_state = build_checkpoint_state(
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			epoch=epoch,
			best_validation_loss=best_validation_loss,
			metadata={
				"train_loss": train_loss,
				"validation_loss": last_validation_loss,
				"fairness_weights": weights,
				"rebalancing_summary_path": str(cfg.rebalancing_summary_path),
			},
		)
		save_checkpoint(checkpoint_state, mitigated_checkpoint_path)
		logger.info("Mitigated checkpoint saved to %s", mitigated_checkpoint_path)

		if last_validation_loss < best_validation_loss:
			best_validation_loss = last_validation_loss
			best_checkpoint_state = build_checkpoint_state(
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				epoch=epoch,
				best_validation_loss=best_validation_loss,
				metadata={
					"train_loss": train_loss,
					"validation_loss": last_validation_loss,
					"best_model": True,
					"fairness_weights": weights,
				},
			)
			save_checkpoint(best_checkpoint_state, best_mitigated_checkpoint_path)
			logger.info("Best mitigated checkpoint updated at epoch %s", epoch)

		if early_stopping.step(last_validation_loss):
			logger.info("Early stopping triggered for mitigation run")
			break

	checkpoint_paths = {
		"mitigated_model": str(mitigated_checkpoint_path),
		"best_mitigated_model": str(best_mitigated_checkpoint_path),
	}
	summary = build_training_summary(
		epochs_completed=epochs_completed,
		final_training_loss=last_training_loss,
		fairness_weights=weights,
		validation_loss=last_validation_loss,
		checkpoint_paths=checkpoint_paths,
		rebalancing_summary=rebalancing_summary,
		best_validation_loss=best_validation_loss,
	)
	save_training_summary(summary, cfg.training_summary_path)

	print(f"FAIRNESS_WEIGHT_COUNT={len(weights)}")
	print(f"MITIGATION_TRAINING_COMPLETED={epochs_completed}")
	print(f"MITIGATED_CHECKPOINT={mitigated_checkpoint_path}")
	print(f"BEST_MITIGATED_CHECKPOINT={best_mitigated_checkpoint_path}")
	return summary


def main() -> None:
	"""Executable entry point for mitigation training."""

	run_mitigation()


if __name__ == "__main__":
	main()
