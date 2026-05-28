"""Weighted triplet loss for fairness-aware fine-tuning.

Weighted loss helps weaker groups because each subgroup contributes more evenly
to gradient updates instead of letting dominant groups steer the embedding space
alone. That makes the mitigation step fairness-aware without changing the model
architecture or the underlying triplet objective.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import torch
from torch import nn


logger = logging.getLogger(__name__)


class FairnessWeightedTripletLoss(nn.Module):
	"""Triplet margin loss with subgroup-aware sample weighting."""

	def __init__(
		self,
		margin: float = 0.3,
		epsilon: float = 1e-8,
		subgroup_weight_map: Mapping[str, float] | None = None,
	) -> None:
		super().__init__()
		if margin <= 0:
			raise ValueError("margin must be greater than zero")
		if epsilon <= 0:
			raise ValueError("epsilon must be greater than zero")
		self.margin = float(margin)
		self.epsilon = float(epsilon)
		self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2, reduction="none")
		self.subgroup_weight_map = {str(key): float(value) for key, value in (subgroup_weight_map or {}).items()}
		logger.info(
			"Fairness weighted triplet loss initialized | margin=%s | subgroup_count=%s",
			self.margin,
			len(self.subgroup_weight_map),
		)

	def _weights_from_labels(self, subgroup_labels: Sequence[str] | None, device: torch.device, batch_size: int) -> torch.Tensor | None:
		if subgroup_labels is None:
			return None
		resolved_weights = []
		for label in subgroup_labels:
			resolved_weights.append(float(self.subgroup_weight_map.get(str(label), 1.0)))
		weights = torch.tensor(resolved_weights, dtype=torch.float32, device=device)
		if weights.numel() != batch_size:
			raise ValueError("Subgroup weights must match batch size")
		return weights

	def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
		weights = torch.clamp(weights.float(), min=self.epsilon)
		mean_weight = torch.clamp(weights.mean(), min=self.epsilon)
		return weights / mean_weight

	def forward(
		self,
		anchor_embedding: torch.Tensor,
		positive_embedding: torch.Tensor,
		negative_embedding: torch.Tensor,
		subgroup_weights: torch.Tensor | None = None,
		subgroup_labels: Sequence[str] | None = None,
	) -> torch.Tensor:
		"""Compute subgroup-weighted triplet loss in a numerically stable way."""

		if anchor_embedding.shape != positive_embedding.shape or anchor_embedding.shape != negative_embedding.shape:
			raise ValueError("Triplet loss inputs must share identical shapes")
		if anchor_embedding.ndim != 2:
			raise ValueError("Triplet loss expects 2D tensors: [batch, embedding_dim]")

		per_sample_loss = self.loss_fn(anchor_embedding, positive_embedding, negative_embedding)
		if per_sample_loss.ndim != 1:
			per_sample_loss = per_sample_loss.reshape(-1)

		if subgroup_weights is None:
			subgroup_weights = self._weights_from_labels(subgroup_labels, device=per_sample_loss.device, batch_size=per_sample_loss.shape[0])
		if subgroup_weights is None:
			normalized_weights = torch.ones_like(per_sample_loss)
		else:
			if subgroup_weights.shape[0] != per_sample_loss.shape[0]:
				raise ValueError("Subgroup weights must match batch size")
			normalized_weights = self._normalize_weights(subgroup_weights.to(per_sample_loss.device))

		weighted_loss = torch.sum(per_sample_loss * normalized_weights) / torch.clamp(normalized_weights.sum(), min=self.epsilon)
		logger.debug(
			"Weighted triplet loss computed | batch=%s | mean_weight=%.6f | loss=%.6f",
			per_sample_loss.shape[0],
			float(normalized_weights.mean().item()),
			float(weighted_loss.item()),
		)
		return weighted_loss
