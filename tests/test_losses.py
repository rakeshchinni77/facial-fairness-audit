from __future__ import annotations

import torch

from src.mitigation.weighted_loss import FairnessWeightedTripletLoss
from src.models.losses import TripletLossWrapper


def _embedding_batch(batch_size: int = 4, embedding_dim: int = 128) -> torch.Tensor:
    embeddings = torch.randn(batch_size, embedding_dim)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def test_triplet_loss_returns_scalar_tensor():
    loss_fn = TripletLossWrapper(margin=0.3)
    loss = loss_fn(_embedding_batch(), _embedding_batch(), _embedding_batch())
    assert loss.ndim == 0, "Triplet loss should return a scalar tensor"
    assert torch.isfinite(loss), "Triplet loss should be finite"


def test_weighted_triplet_loss_handles_subgroup_labels_and_weights():
    loss_fn = FairnessWeightedTripletLoss(margin=0.3, subgroup_weight_map={"Male_0-19_Light": 2.0, "Female_20-39_Dark": 0.5})
    anchor = _embedding_batch()
    positive = _embedding_batch()
    negative = _embedding_batch()
    loss = loss_fn(anchor, positive, negative, subgroup_labels=["Male_0-19_Light", "Female_20-39_Dark", "Male_0-19_Light", "Female_20-39_Dark"])
    assert loss.ndim == 0, "Weighted triplet loss should return a scalar tensor"
    assert torch.isfinite(loss), "Weighted triplet loss should be finite"


def test_weighted_triplet_loss_rejects_bad_batch_shapes():
    loss_fn = FairnessWeightedTripletLoss(margin=0.3)
    anchor = _embedding_batch(batch_size=2)
    positive = _embedding_batch(batch_size=2)
    negative = _embedding_batch(batch_size=3)
    try:
        loss_fn(anchor, positive, negative)
    except ValueError as exc:
        assert "identical shapes" in str(exc), "Expected a readable shape mismatch error"
    else:
        raise AssertionError("Expected ValueError for mismatched triplet shapes")
