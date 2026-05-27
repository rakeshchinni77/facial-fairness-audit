"""Initial fairness audit pipeline for facial verification.

This pipeline is intentionally separate from threshold tuning. The threshold is
loaded from the earlier validation phase, then the audit set is scored at that
fixed operating point so we can observe subgroup and cross-group disparities
without contaminating the calibration data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.audit.cross_group_analysis import analyze_cross_groups
from src.audit.embedding_extractor import EmbeddingExtractorConfig, load_embedding_extractor
from src.audit.fairness_metrics import compute_confusion_summary
from src.audit.similarity import cosine_similarity
from src.audit.subgroup_evaluator import evaluate_subgroups


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditPipelineConfig:
    audit_pairs_path: Path = Path("data/processed/audit_pairs.csv")
    threshold_path: Path = Path("results/threshold_analysis.json")
    demographics_path: Path = Path("results/demographics.json")
    checkpoint_path: Path = Path("artifacts/best_model.pth")
    output_path: Path = Path("results/initial_audit.json")
    cross_group_output_path: Path = Path("results/cross_group_metrics.json")
    plots_dir: Path = Path("artifacts/plots")
    batch_size: int = 32
    device: str = "cpu"
    image_size: int = 224
    max_rows: int | None = None


class AuditPairDataset(Dataset):
    """Batch-safe loader for audit verification pairs."""

    def __init__(self, csv_path: str | Path, image_size: int = 224, max_rows: int | None = None) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Audit pair CSV not found: {self.csv_path}")
        frame = pd.read_csv(self.csv_path)
        required_columns = {"image_a", "image_b", "label"}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(f"Audit CSV missing columns: {sorted(missing_columns)}")
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
            raise ValueError(f"Unable to load audit image: {image_path}") from exc

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        return {
            "image_a": self._load_image(str(row["image_a"])),
            "image_b": self._load_image(str(row["image_b"])),
            "true_label": torch.tensor(int(row["label"]), dtype=torch.int64),
            "subgroup": str(row.get("subgroup", "unknown")),
            "subgroup_left": str(row.get("subgroup_a", row.get("subgroup", "unknown"))),
            "subgroup_right": str(row.get("subgroup_b", row.get("subgroup", "unknown"))),
        }


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def load_threshold(path: str | Path) -> float:
    """Load the fixed operating threshold from validation analysis."""

    threshold_path = Path(path)
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    if "optimal_threshold" not in payload:
        raise ValueError("threshold_analysis.json missing optimal_threshold")
    threshold = float(payload["optimal_threshold"])
    logger.info("Threshold loaded from %s | threshold=%.6f", threshold_path, threshold)
    return threshold


def load_demographics(path: str | Path) -> list[str]:
    """Load known demographic groups for reporting and validation."""

    demographics_path = Path(path)
    if not demographics_path.exists():
        raise FileNotFoundError(f"Demographics file not found: {demographics_path}")
    payload = json.loads(demographics_path.read_text(encoding="utf-8"))
    groups: list[str] = []
    for gender in payload.get("gender", []):
        for age_bin in payload.get("age_bins", {}).keys():
            for skin_tone in payload.get("skin_tone_scale", {}).keys():
                groups.append(f"{gender}_{age_bin}_{skin_tone}")
    return groups


def build_audit_loader(config: AuditPipelineConfig) -> DataLoader:
    dataset = AuditPairDataset(
        config.audit_pairs_path,
        image_size=config.image_size,
        max_rows=config.max_rows,
    )
    logger.info("Audit pair loading complete | rows=%s", len(dataset))
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def score_audit_pairs(extractor, audit_loader: DataLoader) -> pd.DataFrame:
    """Generate embeddings and similarity scores for the isolated audit set."""

    rows: list[dict[str, Any]] = []
    for batch in audit_loader:
        embeddings_a = extractor.embed_batch(batch["image_a"])
        embeddings_b = extractor.embed_batch(batch["image_b"])
        logger.info("Embedding generation complete for audit batch size %s", batch["image_a"].shape[0])
        scores = cosine_similarity(embeddings_a, embeddings_b)
        score_values = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores
        labels = batch["true_label"].detach().cpu().numpy()
        subgroup_values = batch["subgroup"]
        left_values = batch["subgroup_left"]
        right_values = batch["subgroup_right"]
        if isinstance(subgroup_values, list):
            subgroups = subgroup_values
            left_groups = left_values
            right_groups = right_values
        else:
            subgroups = [str(item) for item in subgroup_values]
            left_groups = [str(item) for item in left_values]
            right_groups = [str(item) for item in right_values]

        for index in range(len(score_values)):
            rows.append(
                {
                    "similarity_score": float(score_values[index]),
                    "true_label": int(labels[index]),
                    "predicted_label": 0,
                    "subgroup": str(subgroups[index]),
                    "subgroup_left": str(left_groups[index]),
                    "subgroup_right": str(right_groups[index]),
                }
            )
    frame = pd.DataFrame(rows)
    logger.info("Similarity computation complete | processed_pairs=%s", len(frame))
    return frame


def apply_threshold(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Apply the fixed validation threshold to the audit scores."""

    scored_frame = frame.copy()
    scored_frame["predicted_label"] = (scored_frame["similarity_score"] >= threshold).astype(int)
    return scored_frame


def build_report(frame: pd.DataFrame, threshold: float, demographic_groups: list[str]) -> dict[str, Any]:
    """Build the nested JSON report for overall, subgroup, and cross-group views."""

    overall = compute_confusion_summary(frame["true_label"], frame["predicted_label"]).as_dict()
    subgroup_metrics = evaluate_subgroups(frame)
    cross_group_metrics = analyze_cross_groups(frame)

    report: dict[str, Any] = {
        "overall": overall,
        "metadata": {
            "threshold": float(threshold),
            "total_audit_pairs": int(len(frame)),
            "demographic_group_count": int(len(demographic_groups)),
            "subgroup_count": int(len(subgroup_metrics)),
            "cross_group_count": int(len(cross_group_metrics)),
        },
    }
    report.update(subgroup_metrics)
    return report, cross_group_metrics


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("JSON exported to %s", output_path)
    return output_path


def run_audit_pipeline(config: AuditPipelineConfig | None = None) -> dict[str, Any]:
    """Run the initial fairness audit and export the report artifacts."""

    configure_logging()
    cfg = config or AuditPipelineConfig()
    threshold = load_threshold(cfg.threshold_path)
    demographic_groups = load_demographics(cfg.demographics_path)
    extractor = load_embedding_extractor(
        EmbeddingExtractorConfig(
            checkpoint_path=cfg.checkpoint_path,
            device=cfg.device,
            image_size=cfg.image_size,
        )
    )
    audit_loader = build_audit_loader(cfg)
    scored_frame = score_audit_pairs(extractor, audit_loader)
    scored_frame = apply_threshold(scored_frame, threshold)
    report, cross_group_metrics = build_report(scored_frame, threshold, demographic_groups)
    save_json(report, cfg.output_path)
    save_json(cross_group_metrics, cfg.cross_group_output_path)

    overall = report["overall"]
    subgroup_items = {key: value for key, value in report.items() if key not in {"overall", "metadata"}}
    worst_subgroup = None
    worst_score = float("-inf")
    for subgroup_name, metrics in subgroup_items.items():
        score = float(metrics["far"]) + float(metrics["frr"])
        if score > worst_score:
            worst_score = score
            worst_subgroup = subgroup_name

    print(f"OVERALL_FAR={overall['far']:.6f}")
    print(f"OVERALL_FRR={overall['frr']:.6f}")
    print(f"TOTAL_AUDIT_PAIRS={len(scored_frame)}")
    print(f"SUBGROUP_COUNT={len(subgroup_items)}")
    print(f"WORST_PERFORMING_SUBGROUP={worst_subgroup or 'n/a'}")
    logger.info("Audit pipeline completed")
    return {
        "report": report,
        "cross_group_metrics": cross_group_metrics,
        "scored_frame": scored_frame,
    }


def main() -> None:
    """Executable entry point."""

    run_audit_pipeline()


if __name__ == "__main__":
    main()
