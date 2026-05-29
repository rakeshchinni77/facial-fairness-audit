from __future__ import annotations

import torch

from src.models.backbone import ResNet18Backbone
from src.models.embedding_model import FaceEmbeddingModel


def test_resnet18_backbone_initializes_and_runs_on_cpu() -> None:
	backbone = ResNet18Backbone(pretrained=False, freeze_backbone=True)
	assert backbone.feature_dim == 512, "ResNet18 backbone should expose 512 features"
	output = backbone(torch.randn(2, 3, 224, 224))
	assert output.shape == (2, 512), f"Unexpected backbone output shape: {tuple(output.shape)}"


def test_embedding_model_returns_l2_normalized_embeddings() -> None:
	model = FaceEmbeddingModel(embedding_dim=128, pretrained=False, freeze_backbone=True)
	model.eval()
	output = model(torch.randn(2, 3, 224, 224))
	assert output.shape == (2, 128), f"Embedding model must return [batch, 128], got {tuple(output.shape)}"
	norms = torch.linalg.vector_norm(output, dim=1)
	assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), f"Embeddings must be L2-normalized, got norms {norms.tolist()}"
