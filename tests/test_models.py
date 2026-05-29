from __future__ import annotations

import torch

from src.models.backbone import ResNet18Backbone
from src.models.embedding_model import FaceEmbeddingModel


def test_resnet18_backbone_initializes_and_forwards_on_cpu():
    model = ResNet18Backbone(pretrained=False, freeze_backbone=False)
    dummy = torch.randn(2, 3, 224, 224)
    features = model(dummy)
    assert model.feature_dim == 512, "ResNet18 backbone should expose 512 features"
    assert features.shape == (2, 512), f"Unexpected backbone output shape: {tuple(features.shape)}"


def test_face_embedding_model_produces_l2_normalized_embeddings():
    model = FaceEmbeddingModel(embedding_dim=128, pretrained=False, freeze_backbone=False)
    dummy = torch.randn(2, 3, 224, 224)
    embeddings = model(dummy)
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    assert embeddings.shape == (2, 128), f"Unexpected embedding shape: {tuple(embeddings.shape)}"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Embeddings are not L2-normalized"
