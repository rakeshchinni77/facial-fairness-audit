"""Embedding extraction utilities for audit workflows.

The validation threshold depends on stable normalized embeddings: cosine
similarity on L2-normalized vectors is simply a dot product, which keeps the
score scale consistent across batches and checkpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.embedding_model import FaceEmbeddingModel


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingExtractorConfig:
    """Configuration for checkpoint-backed inference."""

    checkpoint_path: Path = Path("artifacts/best_model.pth")
    device: str = "cpu"
    embedding_dim: int = 128
    pretrained: bool = False
    image_size: int = 224


class EmbeddingExtractor:
    """Load a trained embedding checkpoint and run inference only."""

    def __init__(self, config: EmbeddingExtractorConfig | None = None) -> None:
        self.config = config or EmbeddingExtractorConfig()
        self.device = torch.device(self.config.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.model = self._load_model()
        self.model.eval()
        logger.info("Embedding checkpoint loaded from %s", self.config.checkpoint_path)

    def _load_model(self) -> FaceEmbeddingModel:
        model = FaceEmbeddingModel(
            embedding_dim=self.config.embedding_dim,
            pretrained=self.config.pretrained,
            freeze_backbone=False,
        ).to(self.device)

        checkpoint_path = self.config.checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        normalized_state_dict = self._normalize_state_dict(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(normalized_state_dict, strict=False)
        if missing_keys:
            logger.warning("Missing checkpoint keys: %s", sorted(missing_keys))
        if unexpected_keys:
            logger.warning("Unexpected checkpoint keys: %s", sorted(unexpected_keys))
        return model

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "embedding_model_state_dict"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            if all(isinstance(key, str) for key in checkpoint.keys()):
                return checkpoint
        raise ValueError("Unsupported checkpoint format")

    @staticmethod
    def _normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("embedding_model."):
                new_key = new_key[len("embedding_model.") :]
            normalized[new_key] = value
        return normalized

    def _ensure_tensor(self, image: Image.Image | np.ndarray | torch.Tensor | str | Path) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.ndim == 3:
                return tensor.float()
            raise ValueError("Expected a 3D image tensor")
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return self.transform(opened.convert("RGB"))
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
            return self.transform(pil_image.convert("RGB"))
        if isinstance(image, Image.Image):
            return self.transform(image.convert("RGB"))
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    @torch.no_grad()
    def embed_batch(self, images: torch.Tensor | np.ndarray | list[Any]) -> torch.Tensor:
        """Generate normalized embeddings for a batch of images."""

        if isinstance(images, torch.Tensor):
            batch = images.to(self.device)
        else:
            batch = torch.stack([self._ensure_tensor(image) for image in images], dim=0).to(self.device)
        embeddings = self.model.extract_embedding(batch)
        logger.info("Generated embeddings for batch size %s", batch.shape[0])
        return embeddings.detach()

    @torch.no_grad()
    def extract_pair_embeddings(
        self,
        image_a: Image.Image | np.ndarray | torch.Tensor | str | Path,
        image_b: Image.Image | np.ndarray | torch.Tensor | str | Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract two normalized embeddings from a pair of face images."""

        tensor_a = self._ensure_tensor(image_a).unsqueeze(0).to(self.device)
        tensor_b = self._ensure_tensor(image_b).unsqueeze(0).to(self.device)
        embedding_a = self.model.extract_embedding(tensor_a).squeeze(0)
        embedding_b = self.model.extract_embedding(tensor_b).squeeze(0)
        return embedding_a.detach(), embedding_b.detach()


def load_embedding_extractor(config: EmbeddingExtractorConfig | None = None) -> EmbeddingExtractor:
    """Factory used by the threshold pipeline."""

    return EmbeddingExtractor(config=config)
