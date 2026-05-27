"""Triplet network with shared embedding weights.

Triplet learning relies on shared weights so anchor, positive, and negative
examples are projected into the same embedding space consistently.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from src.models.embedding_model import FaceEmbeddingModel


logger = logging.getLogger(__name__)


class TripletNetwork(nn.Module):
	"""Shared-weights triplet network for metric learning."""

	def __init__(self, embedding_model: FaceEmbeddingModel) -> None:
		super().__init__()
		self.embedding_model = embedding_model
		logger.info("Triplet network initialized with shared embedding model")

	def forward(
		self,
		anchor: torch.Tensor,
		positive: torch.Tensor,
		negative: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Project anchor, positive, and negative with shared model weights."""

		anchor_embedding = self.embedding_model(anchor)
		positive_embedding = self.embedding_model(positive)
		negative_embedding = self.embedding_model(negative)

		if anchor_embedding.shape != positive_embedding.shape or anchor_embedding.shape != negative_embedding.shape:
			raise ValueError("Triplet embeddings must share identical shapes")

		return anchor_embedding, positive_embedding, negative_embedding


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	embedding_model = FaceEmbeddingModel(embedding_dim=128, pretrained=False)
	triplet_model = TripletNetwork(embedding_model=embedding_model)

	anchor = torch.randn(2, 3, 224, 224)
	positive = torch.randn(2, 3, 224, 224)
	negative = torch.randn(2, 3, 224, 224)
	a, p, n = triplet_model(anchor, positive, negative)
	assert a.shape == p.shape == n.shape == (2, 128), "Unexpected triplet embedding shapes"
	print("Triplet network smoke test passed.")