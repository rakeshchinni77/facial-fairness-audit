from __future__ import annotations

import torch
import torch.nn.functional as F

from src.mitigation.weighted_loss import FairnessWeightedTripletLoss
from src.models.losses import TripletLossWrapper


def test_triplet_loss_returns_scalar_and_is_finite() -> None:
	loss_fn = TripletLossWrapper(margin=0.3)
	anchor = F.normalize(torch.randn(4, 128), p=2, dim=1)
	positive = F.normalize(torch.randn(4, 128), p=2, dim=1)
	negative = F.normalize(torch.randn(4, 128), p=2, dim=1)
	loss = loss_fn(anchor, positive, negative)
	assert loss.ndim == 0, "Triplet loss must return a scalar tensor"
	assert torch.isfinite(loss).item(), "Triplet loss must be finite"


def test_weighted_triplet_loss_handles_subgroup_labels() -> None:
	loss_fn = FairnessWeightedTripletLoss(margin=0.3, subgroup_weight_map={"A": 2.0, "B": 0.5})
	anchor = F.normalize(torch.randn(4, 128), p=2, dim=1)
	positive = F.normalize(torch.randn(4, 128), p=2, dim=1)
	negative = F.normalize(torch.randn(4, 128), p=2, dim=1)
	loss = loss_fn(anchor, positive, negative, subgroup_labels=["A", "B", "A", "B"])
	assert loss.ndim == 0, "Weighted triplet loss must return a scalar tensor"
	assert torch.isfinite(loss).item(), "Weighted triplet loss must remain finite when subgroup weights are applied"
