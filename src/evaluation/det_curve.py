"""DET curve utilities for validation threshold analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import det_curve


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetAnalysis:
    far: np.ndarray
    frr: np.ndarray
    thresholds: np.ndarray


def compute_det_analysis(true_labels: Any, similarity_scores: Any) -> DetAnalysis:
    """Compute DET-compatible FAR and FRR curves from scores."""

    labels = np.asarray(true_labels, dtype=np.int32)
    scores = np.asarray(similarity_scores, dtype=np.float32)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("Labels and scores must have the same length")
    far, frr, thresholds = det_curve(labels, scores)
    logger.info("DET computed | points=%s", len(thresholds))
    return DetAnalysis(far=far, frr=frr, thresholds=thresholds)


def save_det_plot(analysis: DetAnalysis, output_path: str | Path) -> Path:
    """Save a DET curve plot for inspection."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot(analysis.far, analysis.frr, label="DET curve")
    plt.xlabel("False Acceptance Rate")
    plt.ylabel("False Rejection Rate")
    plt.title("DET Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("DET plot saved to %s", path)
    return path
