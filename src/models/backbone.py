"""Backbone model definitions for facial verification.

ResNet18 is intentionally used as a lightweight baseline that remains efficient
on CPU while still providing strong visual features for metric learning.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision import models


logger = logging.getLogger(__name__)


class ResNet18Backbone(nn.Module):
	"""Feature extractor based on torchvision ResNet18.

	The final classification layer is removed to expose 512-dimensional features
	for downstream embedding projection.
	"""

	def __init__(self, pretrained: bool = True, freeze_backbone: bool = False) -> None:
		super().__init__()
		weights = models.ResNet18_Weights.DEFAULT if pretrained else None
		resnet = models.resnet18(weights=weights)
		self.feature_dim = int(resnet.fc.in_features)
		resnet.fc = nn.Identity()
		self.backbone = resnet

		if freeze_backbone:
			for parameter in self.backbone.parameters():
				parameter.requires_grad = False

		logger.info(
			"ResNet18 backbone initialized | pretrained=%s | freeze_backbone=%s | feature_dim=%s",
			pretrained,
			freeze_backbone,
			self.feature_dim,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Extract backbone features for a batch of input images."""

		features = self.backbone(x)
		if features.ndim != 2 or features.shape[1] != self.feature_dim:
			raise ValueError(
				f"Unexpected backbone output shape {tuple(features.shape)}; "
				f"expected [batch, {self.feature_dim}]"
			)
		return features


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	model = ResNet18Backbone(pretrained=False)
	dummy = torch.randn(2, 3, 224, 224)
	out = model(dummy)
	assert out.shape == (2, 512), f"Unexpected backbone shape: {tuple(out.shape)}"
	print("Backbone smoke test passed.")