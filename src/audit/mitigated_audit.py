"""Mitigated fairness audit pipeline.

This mirrors the initial audit pipeline but swaps in the mitigated checkpoint
so we can compare subgroup and cross-group fairness after re-training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.audit.audit_pipeline import (
	AuditPipelineConfig,
	apply_threshold,
	build_audit_loader,
	load_demographics,
	load_threshold,
	save_json,
	score_audit_pairs,
)
from src.audit.cross_group_analysis import analyze_cross_groups
from src.audit.embedding_extractor import EmbeddingExtractorConfig, load_embedding_extractor
from src.audit.fairness_metrics import compute_confusion_summary
from src.audit.subgroup_evaluator import evaluate_subgroups


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MitigatedAuditConfig:
	"""Configuration for the mitigated audit run."""

	audit_pairs_path: Path = Path("data/processed/audit_pairs.csv")
	threshold_path: Path = Path("results/threshold_analysis.json")
	demographics_path: Path = Path("results/demographics.json")
	checkpoint_path: Path = Path("artifacts/best_mitigated_model.pth")
	output_path: Path = Path("results/mitigated_audit.json")
	batch_size: int = 32
	device: str = "cpu"
	image_size: int = 224
	max_rows: int | None = None
	embedding_dim: int = 128
	pretrained: bool = False


def configure_logging() -> None:
	"""Configure structured logging for mitigated audit runs."""

	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_mitigated_report(frame: pd.DataFrame, threshold: float, demographic_groups: list[str]) -> dict[str, Any]:
	"""Build the mitigated audit report with overall, subgroup, and cross-group metrics."""

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
			"checkpoint_path": None,
		},
		"cross_group_metrics": cross_group_metrics,
	}
	report.update(subgroup_metrics)
	return report


def run_mitigated_audit(config: MitigatedAuditConfig | None = None) -> dict[str, Any]:
	"""Run the mitigated fairness audit and export the report artifact."""

	configure_logging()
	cfg = config or MitigatedAuditConfig()
	logger.info("Mitigated audit started")
	threshold = load_threshold(cfg.threshold_path)
	demographic_groups = load_demographics(cfg.demographics_path)
	loader = build_audit_loader(
		AuditPipelineConfig(
			audit_pairs_path=cfg.audit_pairs_path,
			threshold_path=cfg.threshold_path,
			demographics_path=cfg.demographics_path,
			checkpoint_path=cfg.checkpoint_path,
			output_path=cfg.output_path,
			batch_size=cfg.batch_size,
			device=cfg.device,
			image_size=cfg.image_size,
			max_rows=cfg.max_rows,
		)
	)
	extractor = load_embedding_extractor(
		EmbeddingExtractorConfig(
			checkpoint_path=cfg.checkpoint_path,
			device=cfg.device,
			embedding_dim=cfg.embedding_dim,
			pretrained=cfg.pretrained,
			image_size=cfg.image_size,
		)
	)
	logger.info("Embeddings generated via mitigated checkpoint %s", cfg.checkpoint_path)
	scored_frame = score_audit_pairs(extractor, loader)
	logger.info("Similarity scoring complete | pairs=%s", len(scored_frame))
	scored_frame = apply_threshold(scored_frame, threshold)
	report = build_mitigated_report(scored_frame, threshold, demographic_groups)
	report["metadata"]["checkpoint_path"] = str(cfg.checkpoint_path)
	save_json(report, cfg.output_path)
	logger.info("Mitigated audit exported to %s", cfg.output_path)

	overall = report["overall"]
	print(f"MITIGATED_OVERALL_FAR={overall['far']:.6f}")
	print(f"MITIGATED_OVERALL_FRR={overall['frr']:.6f}")
	print(f"MITIGATED_AUDIT_PAIRS={len(scored_frame)}")
	print(f"MITIGATED_SUBGROUP_COUNT={len([k for k in report.keys() if k not in {'overall', 'metadata', 'cross_group_metrics'}])}")
	return {
		"report": report,
		"scored_frame": scored_frame,
	}


def main() -> None:
	"""Executable entry point for mitigated audit scoring."""

	run_mitigated_audit()


if __name__ == "__main__":
	main()