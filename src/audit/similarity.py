"""Similarity computation utilities.

Cosine similarity is the right score here because the embedding model already
outputs L2-normalized vectors; that makes the score scale interpretable and
stable across batches, checkpoints, and validation runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _as_tensor(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.float()
    else:
        tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError("Expected embeddings with shape [batch, dim]")
    return tensor


def cosine_similarity(
    embeddings_a: Any,
    embeddings_b: Any,
    eps: float = 1e-8,
) -> Any:
    """Compute batch-safe cosine similarity for numpy arrays or torch tensors."""

    if isinstance(embeddings_a, torch.Tensor) or isinstance(embeddings_b, torch.Tensor):
        tensor_a = _as_tensor(embeddings_a)
        tensor_b = _as_tensor(embeddings_b)
        if tensor_a.shape != tensor_b.shape:
            raise ValueError("Embedding batches must share the same shape")
        numerator = torch.sum(tensor_a * tensor_b, dim=1)
        denominator = torch.clamp(
            torch.linalg.vector_norm(tensor_a, dim=1) * torch.linalg.vector_norm(tensor_b, dim=1),
            min=eps,
        )
        similarity = torch.clamp(numerator / denominator, min=-1.0, max=1.0)
        return similarity

    array_a = np.asarray(embeddings_a, dtype=np.float32)
    array_b = np.asarray(embeddings_b, dtype=np.float32)
    if array_a.ndim == 1:
        array_a = array_a[np.newaxis, :]
    if array_b.ndim == 1:
        array_b = array_b[np.newaxis, :]
    if array_a.shape != array_b.shape:
        raise ValueError("Embedding batches must share the same shape")
    numerator = np.sum(array_a * array_b, axis=1)
    denominator = np.clip(np.linalg.norm(array_a, axis=1) * np.linalg.norm(array_b, axis=1), a_min=eps, a_max=None)
    similarity = np.clip(numerator / denominator, -1.0, 1.0)
    return similarity


def pair_similarity_scores(
    embedding_pairs: list[tuple[Any, Any]],
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute cosine similarity scores for a list of embedding pairs."""

    if not embedding_pairs:
        return np.empty((0,), dtype=np.float32)
    scores: list[float] = []
    for embedding_a, embedding_b in embedding_pairs:
        score = cosine_similarity(embedding_a, embedding_b, eps=eps)
        if isinstance(score, torch.Tensor):
            scores.extend(score.detach().cpu().numpy().astype(np.float32).tolist())
        else:
            scores.extend(np.asarray(score, dtype=np.float32).tolist())
    return np.asarray(scores, dtype=np.float32)


def pair_similarity(
    embedding_a: Any,
    embedding_b: Any,
    eps: float = 1e-8,
) -> float:
    """Return a single cosine similarity score for one pair."""

    score = cosine_similarity(embedding_a, embedding_b, eps=eps)
    if isinstance(score, torch.Tensor):
        return float(score.squeeze().detach().cpu().item())
    return float(np.asarray(score).squeeze())
