"""Embedding model definitions for metric learning.

Embedding vectors are L2-normalized so cosine similarity is stable and
comparable across samples and batches.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbone import ResNet18Backbone


logger = logging.getLogger(__name__)


class FaceEmbeddingModel(nn.Module):
	"""ResNet18-based embedding model with a configurable projection head."""

	def __init__(
		self,
		embedding_dim: int = 128,
		pretrained: bool = True,
		freeze_backbone: bool = False,
		backbone: ResNet18Backbone | None = None,
	) -> None:
		super().__init__()
		if embedding_dim <= 0:
			raise ValueError("embedding_dim must be greater than zero")

		self.backbone = backbone or ResNet18Backbone(
			pretrained=pretrained,
			freeze_backbone=freeze_backbone,
		)
		self.embedding_dim = int(embedding_dim)
		self.projection = nn.Linear(self.backbone.feature_dim, self.embedding_dim)
		self.batch_norm = nn.BatchNorm1d(self.embedding_dim)
		self.activation = nn.ReLU(inplace=True)

		logger.info(
			"Embedding model initialized | embedding_dim=%s | pretrained=%s",
			self.embedding_dim,
			pretrained,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Return normalized embeddings for an input image batch."""

		features = self.backbone(x)
		embeddings = self.projection(features)
		embeddings = self.batch_norm(embeddings)
		embeddings = self.activation(embeddings)

		# L2 normalization makes cosine similarity directly meaningful.
		normalized = F.normalize(embeddings, p=2, dim=1)

		if normalized.ndim != 2 or normalized.shape[1] != self.embedding_dim:
			raise ValueError(
				f"Unexpected embedding shape {tuple(normalized.shape)}; "
				f"expected [batch, {self.embedding_dim}]"
			)
		return normalized

	@torch.no_grad()
	def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
		"""Inference helper that returns normalized embeddings."""

		was_training = self.training
		self.eval()
		embeddings = self.forward(x)
		if was_training:
			self.train()
		return embeddings


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	model = FaceEmbeddingModel(embedding_dim=128, pretrained=False)
	dummy = torch.randn(4, 3, 224, 224)
	out = model(dummy)
	assert out.shape == (4, 128), f"Unexpected embedding shape: {tuple(out.shape)}"
	norms = torch.linalg.vector_norm(out, dim=1)
	assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Embeddings are not L2-normalized"
	print("Embedding model smoke test passed.")