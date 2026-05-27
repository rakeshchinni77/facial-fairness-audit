"""Training metrics utilities for the triplet-learning pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class EpochMetrics:
	"""Track metrics for a single epoch."""

	epoch: int
	train_loss: float = 0.0
	validation_loss: float = 0.0
	mean_anchor_positive_distance: float = 0.0
	mean_anchor_negative_distance: float = 0.0
	mean_positive_negative_distance: float = 0.0
	mean_positive_similarity: float = 0.0
	mean_negative_similarity: float = 0.0
	learning_rate: float = 0.0
	duration_seconds: float = 0.0
	metadata: dict[str, Any] = field(default_factory=dict)


class MetricsTracker:
	"""Maintain epoch-level metrics for logging and checkpoint decisions."""

	def __init__(self) -> None:
		self.history: list[EpochMetrics] = []

	def add(self, metrics: EpochMetrics) -> None:
		self.history.append(metrics)

	def latest(self) -> EpochMetrics | None:
		return self.history[-1] if self.history else None


def compute_cosine_similarity(embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
	"""Compute cosine similarity for normalized embeddings."""

	if embeddings_a.shape != embeddings_b.shape:
		raise ValueError("Cosine similarity inputs must share identical shapes")
	return torch.sum(embeddings_a * embeddings_b, dim=1)


def compute_distance_statistics(
	anchor_embedding: torch.Tensor,
	positive_embedding: torch.Tensor,
	negative_embedding: torch.Tensor,
) -> dict[str, float]:
	"""Compute embedding separation statistics for monitoring."""

	anchor_positive_distance = torch.linalg.vector_norm(anchor_embedding - positive_embedding, dim=1)
	anchor_negative_distance = torch.linalg.vector_norm(anchor_embedding - negative_embedding, dim=1)
	positive_negative_distance = torch.linalg.vector_norm(positive_embedding - negative_embedding, dim=1)
	positive_similarity = compute_cosine_similarity(anchor_embedding, positive_embedding)
	negative_similarity = compute_cosine_similarity(anchor_embedding, negative_embedding)

	return {
		"mean_anchor_positive_distance": float(anchor_positive_distance.mean().item()),
		"mean_anchor_negative_distance": float(anchor_negative_distance.mean().item()),
		"mean_positive_negative_distance": float(positive_negative_distance.mean().item()),
		"mean_positive_similarity": float(positive_similarity.mean().item()),
		"mean_negative_similarity": float(negative_similarity.mean().item()),
	}


def measure_epoch_duration(start_time: float) -> float:
	"""Return elapsed seconds for an epoch."""

	return float(time.time() - start_time)
