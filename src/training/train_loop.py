"""Training loop utilities for triplet learning."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import EpochMetrics, compute_distance_statistics, measure_epoch_duration


logger = logging.getLogger(__name__)


@dataclass
class TrainEpochResult:
	"""Container for a single training epoch result."""

	loss: float
	metrics: EpochMetrics


def run_training_epoch(
	model: torch.nn.Module,
	loss_fn: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	train_loader: DataLoader,
	device: torch.device,
	gradient_clip_norm: float = 1.0,
	log_interval: int = 1,
) -> TrainEpochResult:
	"""Run a single training epoch with gradient clipping and progress bars."""

	model.train()
	total_loss = 0.0
	total_batches = 0
	start_time = time.time()
	last_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
	metric_snapshot: dict[str, float] = {}

	progress = tqdm(train_loader, desc="Training", leave=False)
	for batch_index, batch in enumerate(progress, start=1):
		anchor = batch["anchor"].to(device)
		positive = batch["positive"].to(device)
		negative = batch["negative"].to(device)

		optimizer.zero_grad(set_to_none=True)
		anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
		loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
		optimizer.step()

		total_loss += float(loss.item())
		total_batches += 1
		metric_snapshot = compute_distance_statistics(anchor_embedding.detach(), positive_embedding.detach(), negative_embedding.detach())
		progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{last_lr:.2e}"})

		if log_interval > 0 and batch_index % log_interval == 0:
			logger.info("Training batch %s | loss=%.6f", batch_index, float(loss.item()))

	mean_loss = total_loss / max(total_batches, 1)
	duration = measure_epoch_duration(start_time)
	metrics = EpochMetrics(
		epoch=0,
		train_loss=mean_loss,
		validation_loss=0.0,
		mean_anchor_positive_distance=metric_snapshot.get("mean_anchor_positive_distance", 0.0),
		mean_anchor_negative_distance=metric_snapshot.get("mean_anchor_negative_distance", 0.0),
		mean_positive_negative_distance=metric_snapshot.get("mean_positive_negative_distance", 0.0),
		mean_positive_similarity=metric_snapshot.get("mean_positive_similarity", 0.0),
		mean_negative_similarity=metric_snapshot.get("mean_negative_similarity", 0.0),
		learning_rate=last_lr,
		duration_seconds=duration,
		metadata={"batch_count": total_batches},
	)
	logger.info("Training epoch complete | loss=%.6f | duration=%.2fs", mean_loss, duration)
	return TrainEpochResult(loss=mean_loss, metrics=metrics)
