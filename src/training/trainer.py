"""Training orchestration for the facial verification triplet pipeline.

This trainer wires together the CSV-backed triplet dataset, subgroup-aware
balanced batching, triplet loss optimization, validation, checkpointing, and
artifact persistence.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models.embedding_model import FaceEmbeddingModel
from src.models.losses import TripletLossWrapper
from src.models.triplet_network import TripletNetwork
from src.training.balanced_sampler import SubgroupBalancedBatchSampler
from src.training.callbacks import EarlyStopping, build_checkpoint_state, save_checkpoint
from src.training.metrics import MetricsTracker
from src.training.train_loop import run_training_epoch
from src.training.validator import run_validation_epoch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
	"""Configuration for the triplet-learning trainer."""

	processed_dir: Path = Path("data/processed")
	artifacts_dir: Path = Path("artifacts")	
	device: str = "cpu"
	batch_size: int = 32
	validation_batch_size: int = 32
	learning_rate: float = 1e-4
	epochs: int = 2
	patience: int = 2
	margin: float = 0.3
	images_per_identity: int = 4
	random_seed: int = 42
	freeze_backbone: bool = True
	train_backbone_block: bool = True
	model_embedding_dim: int = 128
	pretrained: bool = True
	max_train_rows: int | None = 512
	max_validation_rows: int | None = 256


def configure_logging() -> None:
	"""Configure module logging for standalone execution."""

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


class TripletCSVSampleDataset(Dataset):
	"""Dataset for triplet rows stored in CSV format."""

	def __init__(self, csv_path: str | Path, transform: transforms.Compose | None = None, max_rows: int | None = None) -> None:
		self.csv_path = Path(csv_path)
		if not self.csv_path.exists():
			raise FileNotFoundError(f"Triplet CSV not found: {self.csv_path}")
		frame = pd.read_csv(self.csv_path)
		required_columns = {"anchor", "positive", "negative"}
		missing_columns = required_columns - set(frame.columns)
		if missing_columns:
			raise ValueError(f"Triplet CSV missing columns: {sorted(missing_columns)}")
		if max_rows is not None:
			frame = frame.head(max_rows).reset_index(drop=True)
		self.frame = frame
		self.transform = transform or build_training_transform()

	def __len__(self) -> int:
		return len(self.frame)

	def _load_image(self, image_path: str) -> torch.Tensor:
		path = Path(image_path)
		try:
			with Image.open(path) as image:
				image = image.convert("RGB")
				return self.transform(image)
		except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
			raise ValueError(f"Unable to load image: {path}") from exc

	def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
		row = self.frame.iloc[int(index)]
		return {
			"anchor": self._load_image(str(row["anchor"])),
			"positive": self._load_image(str(row["positive"])),
			"negative": self._load_image(str(row["negative"])),
			"subgroup": str(row.get("subgroup", "unknown")),
		}


class PairCSVSampleDataset(Dataset):
	"""Dataset for validation pair rows stored in CSV format."""

	def __init__(self, csv_path: str | Path, transform: transforms.Compose | None = None, max_rows: int | None = None) -> None:
		self.csv_path = Path(csv_path)
		if not self.csv_path.exists():
			raise FileNotFoundError(f"Validation CSV not found: {self.csv_path}")
		frame = pd.read_csv(self.csv_path)
		required_columns = {"image_a", "image_b", "label"}
		missing_columns = required_columns - set(frame.columns)
		if missing_columns:
			raise ValueError(f"Validation CSV missing columns: {sorted(missing_columns)}")
		if max_rows is not None:
			frame = frame.head(max_rows).reset_index(drop=True)
		self.frame = frame
		self.transform = transform or build_validation_transform()

	def __len__(self) -> int:
		return len(self.frame)

	def _load_image(self, image_path: str) -> torch.Tensor:
		path = Path(image_path)
		try:
			with Image.open(path) as image:
				image = image.convert("RGB")
				return self.transform(image)
		except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
			raise ValueError(f"Unable to load image: {path}") from exc

	def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
		row = self.frame.iloc[int(index)]
		return {
			"image_a": self._load_image(str(row["image_a"])),
			"image_b": self._load_image(str(row["image_b"])),
			"label": torch.tensor(int(row["label"]), dtype=torch.float32),
			"subgroup": str(row.get("subgroup", "unknown")),
		}


def build_training_transform(image_size: int = 224) -> transforms.Compose:
	"""Build the preprocessing transform used for triplet training."""

	return transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def build_validation_transform(image_size: int = 224) -> transforms.Compose:
	"""Build the preprocessing transform used for validation pairs."""

	return transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def build_triplet_model(config: TrainerConfig) -> TripletNetwork:
	"""Initialize the pretrained ResNet18 embedding model and triplet network."""

	embedding_model = FaceEmbeddingModel(
		embedding_dim=config.model_embedding_dim,
		pretrained=config.pretrained,
		freeze_backbone=config.freeze_backbone,
	)

	if config.freeze_backbone and hasattr(embedding_model.backbone, "backbone"):
		for parameter in embedding_model.backbone.backbone.parameters():
			parameter.requires_grad = False
		for parameter in embedding_model.backbone.backbone.layer4.parameters():
			parameter.requires_grad = config.train_backbone_block
		for parameter in embedding_model.projection.parameters():
			parameter.requires_grad = True
		for parameter in embedding_model.batch_norm.parameters():
			parameter.requires_grad = True

	logger.info("Model initialized | embedding_dim=%s | pretrained=%s", config.model_embedding_dim, config.pretrained)
	return TripletNetwork(embedding_model)


def build_optimizer(model: torch.nn.Module, learning_rate: float) -> AdamW:
	"""Build the AdamW optimizer over trainable parameters."""

	trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
	if not trainable_parameters:
		raise ValueError("No trainable parameters found for optimization")
	return AdamW(trainable_parameters, lr=learning_rate, weight_decay=1e-4)


def build_scheduler(optimizer: torch.optim.Optimizer) -> ReduceLROnPlateau:
	"""Build a plateau-based learning rate scheduler."""

	return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)


def build_train_loader(dataset: TripletCSVSampleDataset, config: TrainerConfig) -> DataLoader:
	"""Create a subgroup-balanced training dataloader."""

	indices = list(range(len(dataset)))
	subgroups = dataset.frame["subgroup"].astype(str).tolist()
	sampler = SubgroupBalancedBatchSampler(
		indices=indices,
		subgroups=subgroups,
		batch_size=config.batch_size,
		random_seed=config.random_seed,
		shuffle=True,
	)
	return DataLoader(dataset, batch_sampler=sampler, num_workers=0)


def build_validation_loader(dataset: PairCSVSampleDataset, config: TrainerConfig) -> DataLoader:
	"""Create a validation dataloader."""

	return DataLoader(dataset, batch_size=config.validation_batch_size, shuffle=False, num_workers=0)


def train(config: TrainerConfig) -> dict[str, Any]:
	"""Run triplet training and validation, saving checkpoints along the way."""

	configure_logging()
	random.seed(config.random_seed)
	torch.manual_seed(config.random_seed)

	device = torch.device(config.device)
	logger.info("Training started on device=%s", device)

	train_dataset = TripletCSVSampleDataset(
		config.processed_dir / "train_triplets.csv",
		transform=build_training_transform(),
		max_rows=config.max_train_rows,
	)
	validation_dataset = PairCSVSampleDataset(
		config.processed_dir / "validation_pairs.csv",
		transform=build_validation_transform(),
		max_rows=config.max_validation_rows,
	)

	train_loader = build_train_loader(train_dataset, config)
	validation_loader = build_validation_loader(validation_dataset, config)

	model = build_triplet_model(config).to(device)
	loss_fn = TripletLossWrapper(margin=config.margin).to(device)
	optimizer = build_optimizer(model, learning_rate=config.learning_rate)
	scheduler = build_scheduler(optimizer)
	early_stopping = EarlyStopping(patience=config.patience)
	metrics_tracker = MetricsTracker()

	best_validation_loss = float("inf")
	artifacts_dir = config.artifacts_dir
	artifacts_dir.mkdir(parents=True, exist_ok=True)
	model_path = artifacts_dir / "model.pth"
	best_model_path = artifacts_dir / "best_model.pth"

	for epoch in range(1, config.epochs + 1):
		logger.info("Epoch %s/%s started", epoch, config.epochs)
		train_result = run_training_epoch(
			model=model,
			loss_fn=loss_fn,
			optimizer=optimizer,
			train_loader=train_loader,
			device=device,
			gradient_clip_norm=1.0,
		)
		validation_result = run_validation_epoch(
			model=model.embedding_model,
			validation_loader=validation_loader,
			device=device,
			margin=config.margin,
		)

		scheduler.step(validation_result.validation_loss)
		current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
		train_result.metrics.epoch = epoch
		train_result.metrics.validation_loss = validation_result.validation_loss
		train_result.metrics.learning_rate = current_lr
		train_result.metrics.metadata.update(
			{
				"train_batches": len(train_loader),
				"validation_batches": len(validation_loader),
			}
		)
		metrics_tracker.add(train_result.metrics)

		logger.info(
			"Epoch %s completed | train_loss=%.6f | val_loss=%.6f | lr=%.2e",
			epoch,
			train_result.loss,
			validation_result.validation_loss,
			current_lr,
		)
		print(f"epoch={epoch} train_loss={train_result.loss:.6f} validation_loss={validation_result.validation_loss:.6f}")

		checkpoint_state = build_checkpoint_state(
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			epoch=epoch,
			best_validation_loss=best_validation_loss,
			metadata={
				"training_loss": train_result.loss,
				"validation_loss": validation_result.validation_loss,
				"train_batches": len(train_loader),
				"validation_batches": len(validation_loader),
			},
		)
		save_checkpoint(checkpoint_state, model_path)
		print(f"checkpoint_saved={model_path}")

		if validation_result.validation_loss < best_validation_loss:
			best_validation_loss = validation_result.validation_loss
			best_checkpoint_state = build_checkpoint_state(
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				epoch=epoch,
				best_validation_loss=best_validation_loss,
				metadata={
					"training_loss": train_result.loss,
					"validation_loss": validation_result.validation_loss,
					"best_model": True,
				},
			)
			save_checkpoint(best_checkpoint_state, best_model_path)
			logger.info("Best model updated at epoch %s", epoch)
			print(f"best_model_updated={best_model_path}")

		if early_stopping.step(validation_result.validation_loss):
			logger.info("Early stopping triggered; ending training loop")
			break

	final_summary = {
		"best_validation_loss": best_validation_loss,
		"epochs_completed": len(metrics_tracker.history),
		"model_path": str(model_path),
		"best_model_path": str(best_model_path),
		"metrics": [metric.__dict__ for metric in metrics_tracker.history],
	}
	(config.artifacts_dir / "training_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
	logger.info("Training completed")
	return final_summary


def main() -> None:
	"""Execute a short smoke-test training run."""

	config = TrainerConfig()
	train(config)


if __name__ == "__main__":
	main()
