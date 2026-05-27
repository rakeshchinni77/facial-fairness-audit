"""Threshold selection pipeline for validation similarity analysis.

Threshold selection matters because a single operating point controls the tradeoff
between false accepts and false rejects. The validation set is used here rather
than the audit set so the audit set remains reserved for downstream fairness
analysis and is not leaked into threshold tuning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.audit.embedding_extractor import EmbeddingExtractor, EmbeddingExtractorConfig, load_embedding_extractor
from src.audit.similarity import cosine_similarity
from src.evaluation.det_curve import DetAnalysis, compute_det_analysis, save_det_plot
from src.evaluation.roc_curve import RocAnalysis, compute_roc_analysis, save_roc_plot


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdSelectorConfig:
    """Configuration for validation similarity analysis."""

    validation_pairs_path: Path = Path("data/processed/validation_pairs.csv")
    checkpoint_path: Path = Path("artifacts/best_model.pth")
    results_path: Path = Path("results/threshold_analysis.json")
    plots_dir: Path = Path("artifacts/plots")
    batch_size: int = 32
    device: str = "cpu"
    max_rows: int | None = None
    image_size: int = 224


class ValidationPairDataset(Dataset):
    """Load validation pairs for batch-safe embedding and scoring."""

    def __init__(self, csv_path: str | Path, image_size: int = 224, max_rows: int | None = None) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Validation pair CSV not found: {self.csv_path}")
        frame = pd.read_csv(self.csv_path)
        required_columns = {"image_a", "image_b", "label"}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(f"Validation CSV missing columns: {sorted(missing_columns)}")
        if max_rows is not None:
            frame = frame.head(max_rows).reset_index(drop=True)
        self.frame = frame
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, image_path: str) -> torch.Tensor:
        try:
            with Image.open(image_path) as image:
                return self.transform(image.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Unable to load validation image: {image_path}") from exc

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        return {
            "image_a": self._load_image(str(row["image_a"])),
            "image_b": self._load_image(str(row["image_b"])),
            "true_label": torch.tensor(int(row["label"]), dtype=torch.int64),
            "subgroup": str(row.get("subgroup", "unknown")),
        }


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def load_validation_pairs_dataframe(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load the validation pair CSV without mutating its structure."""

    frame = pd.read_csv(path)
    if max_rows is not None:
        frame = frame.head(max_rows).reset_index(drop=True)
    return frame


def build_validation_loader(config: ThresholdSelectorConfig) -> DataLoader:
    dataset = ValidationPairDataset(
        config.validation_pairs_path,
        image_size=config.image_size,
        max_rows=config.max_rows,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def generate_validation_similarity_scores(
    extractor: EmbeddingExtractor,
    validation_loader: DataLoader,
) -> pd.DataFrame:
    """Generate embeddings and similarity scores for validation pairs."""

    rows: list[dict[str, Any]] = []
    for batch in validation_loader:
        image_a = batch["image_a"]
        image_b = batch["image_b"]
        labels = batch["true_label"].detach().cpu().numpy()
        subgroup_values = batch.get("subgroup")
        if isinstance(subgroup_values, list):
            subgroups = subgroup_values
        else:
            subgroups = [str(item) for item in subgroup_values]

        embeddings_a = extractor.embed_batch(image_a)
        embeddings_b = extractor.embed_batch(image_b)
        logger.info("Embeddings generated for validation batch size %s", image_a.shape[0])
        scores = cosine_similarity(embeddings_a, embeddings_b)
        score_values = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)

        for index in range(len(score_values)):
            rows.append(
                {
                    "similarity_score": float(score_values[index]),
                    "true_label": int(labels[index]),
                    "subgroup": str(subgroups[index]),
                }
            )
    frame = pd.DataFrame(rows)
    return frame


def select_optimal_threshold(
    similarity_scores: np.ndarray,
    true_labels: np.ndarray,
    roc_analysis: RocAnalysis,
) -> tuple[float, dict[str, float], np.ndarray]:
    """Select the operating point that maximizes TPR - FPR.

    This chooses the threshold with the best Youden-style separation on the
    validation set, which is the right place to tune the decision boundary before
    moving to audit analysis.
    """

    candidate_scores = np.asarray(similarity_scores, dtype=np.float32)
    labels = np.asarray(true_labels, dtype=np.int32)
    if candidate_scores.shape[0] != labels.shape[0]:
        raise ValueError("Scores and labels must have the same length")

    if roc_analysis.thresholds.size == 0:
        raise ValueError("ROC thresholds are empty")

    finite_mask = np.isfinite(roc_analysis.thresholds)
    candidate_indices = np.where(finite_mask)[0]
    if candidate_indices.size == 0:
        raise ValueError("ROC analysis produced no finite thresholds")
    youden_local_index = np.argmax(roc_analysis.tpr[finite_mask] - roc_analysis.fpr[finite_mask])
    youden_index = candidate_indices[youden_local_index]
    optimal_threshold = float(roc_analysis.thresholds[youden_index])

    predictions = candidate_scores >= optimal_threshold
    positives = labels == 1
    negatives = labels == 0
    true_positive = int(np.sum(predictions & positives))
    false_positive = int(np.sum(predictions & negatives))
    true_negative = int(np.sum(~predictions & negatives))
    false_negative = int(np.sum(~predictions & positives))

    estimated_far = false_positive / max(false_positive + true_negative, 1)
    estimated_frr = false_negative / max(false_negative + true_positive, 1)
    validation_statistics = {
        "positive_examples": float(np.sum(positives)),
        "negative_examples": float(np.sum(negatives)),
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "true_negative": float(true_negative),
        "false_negative": float(false_negative),
    }
    logger.info(
        "Threshold selected | threshold=%.6f | FAR=%.6f | FRR=%.6f",
        optimal_threshold,
        estimated_far,
        estimated_frr,
    )
    return optimal_threshold, {"estimated_far": estimated_far, "estimated_frr": estimated_frr, **validation_statistics}, predictions


def build_threshold_summary(
    threshold: float,
    roc_analysis: RocAnalysis,
    det_analysis: DetAnalysis,
    validation_frame: pd.DataFrame,
    validation_statistics: dict[str, float],
) -> dict[str, Any]:
    """Assemble the JSON payload for threshold analysis."""

    return {
        "optimal_threshold": float(threshold),
        "roc_auc": float(roc_analysis.roc_auc),
        "estimated_far": float(validation_statistics["estimated_far"]),
        "estimated_frr": float(validation_statistics["estimated_frr"]),
        "validation_pairs": int(len(validation_frame)),
        "validation_statistics": {
            "positive_examples": int(validation_statistics["positive_examples"]),
            "negative_examples": int(validation_statistics["negative_examples"]),
            "true_positive": int(validation_statistics["true_positive"]),
            "false_positive": int(validation_statistics["false_positive"]),
            "true_negative": int(validation_statistics["true_negative"]),
            "false_negative": int(validation_statistics["false_negative"]),
        },
        "det_summary": {
            "far_points": int(det_analysis.far.shape[0]),
            "frr_points": int(det_analysis.frr.shape[0]),
        },
    }


def save_threshold_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Threshold analysis saved to %s", path)
    return path


def run_threshold_selection(config: ThresholdSelectorConfig | None = None) -> dict[str, Any]:
    """Run validation similarity scoring and threshold optimization."""

    configure_logging()
    cfg = config or ThresholdSelectorConfig()
    extractor = load_embedding_extractor(
        EmbeddingExtractorConfig(
            checkpoint_path=cfg.checkpoint_path,
            device=cfg.device,
            image_size=cfg.image_size,
        )
    )
    validation_loader = build_validation_loader(cfg)
    validation_frame = generate_validation_similarity_scores(extractor, validation_loader)
    logger.info("Validation pair scoring complete | pairs=%s", len(validation_frame))

    roc_analysis = compute_roc_analysis(validation_frame["true_label"].to_numpy(), validation_frame["similarity_score"].to_numpy())
    logger.info("ROC computed with AUC %.6f", roc_analysis.roc_auc)
    det_analysis = compute_det_analysis(validation_frame["true_label"].to_numpy(), validation_frame["similarity_score"].to_numpy())

    optimal_threshold, validation_statistics, predictions = select_optimal_threshold(
        validation_frame["similarity_score"].to_numpy(),
        validation_frame["true_label"].to_numpy(),
        roc_analysis,
    )
    logger.info("Threshold selected at %.6f", optimal_threshold)

    plots_dir = cfg.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_roc_plot(roc_analysis, plots_dir / "roc_curve.png")
    save_det_plot(det_analysis, plots_dir / "det_curve.png")

    summary = build_threshold_summary(
        threshold=optimal_threshold,
        roc_analysis=roc_analysis,
        det_analysis=det_analysis,
        validation_frame=validation_frame,
        validation_statistics=validation_statistics,
    )
    save_threshold_summary(summary, cfg.results_path)
    print(f"ROC_AUC={roc_analysis.roc_auc:.6f}")
    print(f"OPTIMAL_THRESHOLD={optimal_threshold:.6f}")
    print(f"FAR_ESTIMATE={validation_statistics['estimated_far']:.6f}")
    print(f"FRR_ESTIMATE={validation_statistics['estimated_frr']:.6f}")
    print(f"PROCESSED_PAIR_COUNT={len(validation_frame)}")
    return {
        **summary,
        "predictions": predictions.tolist(),
        "scores_preview": validation_frame.head(5).to_dict(orient="records"),
    }


def main() -> None:
    """Executable entry point for threshold analysis."""

    run_threshold_selection()


if __name__ == "__main__":
    main()
