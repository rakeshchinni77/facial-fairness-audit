from __future__ import annotations

import torch

from src.models.embedding_model import FaceEmbeddingModel
from src.models.triplet_network import TripletNetwork


def test_triplet_network_returns_three_embeddings_on_cpu():
    embedding_model = FaceEmbeddingModel(embedding_dim=128, pretrained=False, freeze_backbone=False)
    network = TripletNetwork(embedding_model=embedding_model)

    anchor = torch.randn(2, 3, 224, 224)
    positive = torch.randn(2, 3, 224, 224)
    negative = torch.randn(2, 3, 224, 224)

    anchor_embedding, positive_embedding, negative_embedding = network(anchor, positive, negative)
    assert anchor_embedding.shape == positive_embedding.shape == negative_embedding.shape == (2, 128), "Triplet network must return three aligned 128-d embeddings"
    assert anchor_embedding.device.type == "cpu", "Triplet network should run on CPU during validation"
