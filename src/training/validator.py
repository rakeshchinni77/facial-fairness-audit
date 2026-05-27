"""Validation utilities for triplet learning."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_cosine_similarity


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
	"""Container for validation metrics."""

	validation_loss: float
	embedding_statistics: dict[str, float]
	pair_statistics: dict[str, float]
	batch_count: int


@torch.no_grad()
def run_validation_epoch(
	model: torch.nn.Module,
	validation_loader: DataLoader,
	device: torch.device,
	margin: float = 0.3,
) -> ValidationResult:
	"""Run validation over paired images and return summary statistics.

	This does not compute FAR/FRR; it only measures embedding separation and
	cosine similarity statistics for monitoring during triplet training.
	"""

	model.eval()
	total_loss = 0.0
	total_batches = 0
	positive_similarities: list[torch.Tensor] = []
	negative_similarities: list[torch.Tensor] = []
	mean_distances: list[float] = []

	progress = tqdm(validation_loader, desc="Validation", leave=False)
	for batch in progress:
		image_a = batch["image_a"].to(device)
		image_b = batch["image_b"].to(device)
		labels = batch["label"].to(device)

		embedding_a = model.extract_embedding(image_a)
		embedding_b = model.extract_embedding(image_b)
		similarity = compute_cosine_similarity(embedding_a, embedding_b)

		positive_mask = labels > 0.5
		negative_mask = ~positive_mask
		positive_loss = (
			(1.0 - similarity[positive_mask]).clamp(min=0.0).mean()
			if positive_mask.any()
			else torch.tensor(0.0, device=device)
		)
		negative_loss = (
			(similarity[negative_mask] - margin).clamp(min=0.0).mean()
			if negative_mask.any()
			else torch.tensor(0.0, device=device)
		)
		batch_loss = torch.stack([positive_loss, negative_loss]).mean()

		total_loss += float(batch_loss.item())
		total_batches += 1
		mean_distances.append(float(torch.linalg.vector_norm(embedding_a - embedding_b, dim=1).mean().item()))
		if positive_mask.any():
			positive_similarities.append(similarity[positive_mask])
		if negative_mask.any():
			negative_similarities.append(similarity[negative_mask])

	validation_loss = total_loss / max(total_batches, 1)
	mean_embedding_distance = float(sum(mean_distances) / len(mean_distances)) if mean_distances else 0.0
	positive_similarity = torch.cat(positive_similarities).mean().item() if positive_similarities else 0.0
	negative_similarity = torch.cat(negative_similarities).mean().item() if negative_similarities else 0.0

	result = ValidationResult(
		validation_loss=validation_loss,
		embedding_statistics={"mean_distance": mean_embedding_distance},
		pair_statistics={
			"mean_positive_similarity": float(positive_similarity),
			"mean_negative_similarity": float(negative_similarity),
		},
		batch_count=total_batches,
	)
	logger.info(
		"Validation complete | loss=%.6f | mean_distance=%.6f | positive_similarity=%.6f | negative_similarity=%.6f",
		result.validation_loss,
		result.embedding_statistics["mean_distance"],
		result.pair_statistics["mean_positive_similarity"],
		result.pair_statistics["mean_negative_similarity"],
	)
	return result
