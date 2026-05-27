"""Metric-learning loss definitions for facial verification."""

from __future__ import annotations

import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)


class TripletLossWrapper(nn.Module):
	"""Thin wrapper around ``torch.nn.TripletMarginLoss``.

	By default, this uses Euclidean distance in embedding space. Since embeddings
	are L2-normalized upstream, this remains consistent with cosine-based ranking.
	"""

	def __init__(self, margin: float = 0.3) -> None:
		super().__init__()
		if margin <= 0:
			raise ValueError("margin must be greater than zero")
		self.margin = float(margin)
		self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)
		logger.info("Triplet loss initialized | margin=%s", self.margin)

	def forward(
		self,
		anchor_embedding: torch.Tensor,
		positive_embedding: torch.Tensor,
		negative_embedding: torch.Tensor,
	) -> torch.Tensor:
		"""Compute triplet margin loss for a batch of embeddings."""

		if anchor_embedding.shape != positive_embedding.shape or anchor_embedding.shape != negative_embedding.shape:
			raise ValueError("Triplet loss inputs must share identical shapes")
		if anchor_embedding.ndim != 2:
			raise ValueError("Triplet loss expects 2D tensors: [batch, embedding_dim]")
		return self.loss_fn(anchor_embedding, positive_embedding, negative_embedding)


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	loss_module = TripletLossWrapper(margin=0.3)
	anchor = torch.randn(4, 128)
	positive = torch.randn(4, 128)
	negative = torch.randn(4, 128)
	loss = loss_module(anchor, positive, negative)
	assert loss.ndim == 0, "Triplet loss should return a scalar tensor"
	print(f"Triplet loss smoke test passed. Loss={loss.item():.6f}")