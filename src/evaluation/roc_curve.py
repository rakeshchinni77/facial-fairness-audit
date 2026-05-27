"""ROC curve utilities for validation threshold analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RocAnalysis:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    roc_auc: float


def compute_roc_analysis(true_labels: Any, similarity_scores: Any) -> RocAnalysis:
    """Compute ROC coordinates and AUC from labels and similarity scores."""

    labels = np.asarray(true_labels, dtype=np.int32)
    scores = np.asarray(similarity_scores, dtype=np.float32)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("Labels and scores must have the same length")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    logger.info("ROC computed | auc=%.6f | pairs=%s", roc_auc, labels.shape[0])
    return RocAnalysis(fpr=fpr, tpr=tpr, thresholds=thresholds, roc_auc=roc_auc)


def save_roc_plot(analysis: RocAnalysis, output_path: str | Path) -> Path:
    """Save an ROC curve plot for inspection."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot(analysis.fpr, analysis.tpr, label=f"ROC AUC = {analysis.roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("ROC plot saved to %s", path)
    return path
