"""Publication-quality visualization utilities for fairness audit reporting."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.audit.embedding_extractor import EmbeddingExtractorConfig, load_embedding_extractor
from src.audit.threshold_selector import ThresholdSelectorConfig, build_validation_loader, generate_validation_similarity_scores
from src.evaluation.det_curve import DetAnalysis, compute_det_analysis
from src.evaluation.roc_curve import RocAnalysis, compute_roc_analysis


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualizationConfig:
	"""Configuration for the visualization export pipeline."""

	initial_audit_path: Path = Path("results/initial_audit.json")
	mitigated_audit_path: Path = Path("results/mitigated_audit.json")
	fairness_summary_path: Path = Path("results/fairness_summary.json")
	fairness_comparison_path: Path = Path("results/fairness_comparison.json")
	cross_group_metrics_path: Path = Path("results/cross_group_metrics.json")
	threshold_path: Path = Path("results/threshold_analysis.json")
	overall_metrics_path: Path = Path("results/overall_metrics.json")
	validation_pairs_path: Path = Path("data/processed/validation_pairs.csv")
	checkpoint_path: Path = Path("artifacts/best_model.pth")
	plots_dir: Path = Path("artifacts/plots")
	batch_size: int = 32
	device: str = "cpu"
	image_size: int = 224
	max_rows: int | None = None


def configure_logging() -> None:
	"""Configure logging for visualization runs."""

	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def configure_style() -> None:
	"""Apply a restrained publication-style theme."""

	sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
	plt.rcParams.update(
		{
			"figure.dpi": 300,
			"savefig.dpi": 300,
			"axes.titlesize": 13,
			"axes.labelsize": 11,
			"xtick.labelsize": 9,
			"ytick.labelsize": 9,
			"legend.fontsize": 9,
			"font.family": "DejaVu Sans",
			"axes.spines.top": False,
			"axes.spines.right": False,
		}
	)


def _load_json(path: str | Path) -> dict[str, Any]:
	json_path = Path(path)
	if not json_path.exists():
		raise FileNotFoundError(f"Artifact not found: {json_path}")
	return json.loads(json_path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
	try:
		if value is None:
			return float(default)
		return float(value)
	except (TypeError, ValueError):
		return float(default)


def _subgroup_metrics(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
	metrics: dict[str, dict[str, Any]] = {}
	for key, value in report.items():
		if key in {"overall", "metadata", "cross_group_metrics"}:
			continue
		if isinstance(value, dict) and {"far", "frr", "support"}.issubset(value.keys()):
			metrics[str(key)] = value
	return metrics


def _cross_group_metrics(report: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
	value = report.get("cross_group_metrics")
	if isinstance(value, dict):
		return {str(key): metric for key, metric in value.items() if isinstance(metric, dict)}
	if isinstance(fallback, dict):
		return {str(key): metric for key, metric in fallback.items() if isinstance(metric, dict)}
	return {}


def _overall_metrics(report: dict[str, Any]) -> dict[str, Any]:
	value = report.get("overall")
	return value if isinstance(value, dict) else {}


def _threshold_from_payload(payload: dict[str, Any]) -> float:
	return _safe_float(payload.get("optimal_threshold", payload.get("threshold", 0.0)))


def _best_threshold_index(thresholds: np.ndarray, threshold: float) -> int:
	if thresholds.size == 0:
		return 0
	finite_mask = np.isfinite(thresholds)
	finite_indices = np.where(finite_mask)[0]
	if finite_indices.size == 0:
		return int(np.argmax(thresholds))
	return int(finite_indices[np.argmin(np.abs(thresholds[finite_mask] - threshold))])


def _prepare_output_path(path: str | Path) -> Path:
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	return output_path


def _sort_subgroups_for_display(initial_report: dict[str, Any], mitigated_report: dict[str, Any]) -> list[str]:
	initial_metrics = _subgroup_metrics(initial_report)
	mitigated_metrics = _subgroup_metrics(mitigated_report)
	all_groups = sorted(set(initial_metrics) | set(mitigated_metrics))
	return sorted(
		all_groups,
		key=lambda group: (
			-_safe_float(mitigated_metrics.get(group, {}).get("frr", initial_metrics.get(group, {}).get("frr", 0.0))),
			-_safe_float(mitigated_metrics.get(group, {}).get("far", initial_metrics.get(group, {}).get("far", 0.0))),
			group,
		),
	)


def _build_validation_curve_data(config: VisualizationConfig) -> tuple[RocAnalysis, DetAnalysis, float]:
	logger.info("Loading validation artifacts for ROC/DET curves")
	threshold_payload = _load_json(config.threshold_path)
	threshold = _threshold_from_payload(threshold_payload)
	extractor = load_embedding_extractor(
		EmbeddingExtractorConfig(
			checkpoint_path=config.checkpoint_path,
			device=config.device,
			image_size=config.image_size,
		)
	)
	loader = build_validation_loader(
		ThresholdSelectorConfig(
			validation_pairs_path=config.validation_pairs_path,
			checkpoint_path=config.checkpoint_path,
			results_path=config.threshold_path,
			plots_dir=config.plots_dir,
			batch_size=config.batch_size,
			device=config.device,
			max_rows=config.max_rows,
			image_size=config.image_size,
		)
	)
	validation_frame = generate_validation_similarity_scores(extractor, loader)
	roc_analysis = compute_roc_analysis(validation_frame["true_label"].to_numpy(), validation_frame["similarity_score"].to_numpy())
	det_analysis = compute_det_analysis(validation_frame["true_label"].to_numpy(), validation_frame["similarity_score"].to_numpy())
	logger.info("Validation curves prepared | pairs=%s | roc_auc=%.6f", len(validation_frame), roc_analysis.roc_auc)
	return roc_analysis, det_analysis, threshold


def save_publication_roc_plot(analysis: RocAnalysis, threshold: float, output_path: str | Path) -> Path:
	"""Save a publication-quality ROC plot."""

	path = _prepare_output_path(output_path)
	best_index = _best_threshold_index(analysis.thresholds, threshold)
	fig, ax = plt.subplots(figsize=(6.5, 6.0))
	ax.plot(analysis.fpr, analysis.tpr, color="#1f77b4", linewidth=2.4, label=f"ROC curve (AUC = {analysis.roc_auc:.3f})")
	ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.2, label="Random baseline")
	ax.scatter(
		analysis.fpr[best_index],
		analysis.tpr[best_index],
		color="#d62728",
		s=38,
		zorder=5,
		label=f"Operating threshold = {threshold:.3f}",
	)
	ax.annotate(
		f"Threshold\n{threshold:.3f}",
		xy=(analysis.fpr[best_index], analysis.tpr[best_index]),
		xytext=(12, -20),
		textcoords="offset points",
		arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.9},
		fontsize=8,
		bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "lw": 0.8},
	)
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_title("ROC Curve for Validation Threshold Selection")
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.02)
	ax.grid(True, alpha=0.22)
	ax.legend(frameon=False, loc="lower right")
	fig.tight_layout()
	fig.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	logger.info("ROC publication plot saved to %s", path)
	return path


def save_publication_det_plot(analysis: DetAnalysis, threshold: float, output_path: str | Path) -> Path:
	"""Save a publication-quality DET plot with an operating-point marker."""

	path = _prepare_output_path(output_path)
	best_index = _best_threshold_index(analysis.thresholds, threshold)
	fig, ax = plt.subplots(figsize=(6.5, 6.0))
	ax.plot(analysis.far, analysis.frr, color="#2ca02c", linewidth=2.4, label="DET curve")
	ax.scatter(
		analysis.far[best_index],
		analysis.frr[best_index],
		color="#d62728",
		s=38,
		zorder=5,
		label=f"Operating threshold = {threshold:.3f}",
	)
	ax.annotate(
		f"Threshold\n{threshold:.3f}",
		xy=(analysis.far[best_index], analysis.frr[best_index]),
		xytext=(12, -20),
		textcoords="offset points",
		arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.9},
		fontsize=8,
		bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "lw": 0.8},
	)
	ax.set_xlabel("False Acceptance Rate")
	ax.set_ylabel("False Rejection Rate")
	ax.set_title("DET Curve for Validation Operating Point")
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
	ax.grid(True, alpha=0.22, linestyle=":")
	ax.legend(frameon=False, loc="upper right")
	fig.tight_layout()
	fig.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	logger.info("DET publication plot saved to %s", path)
	return path


def _subgroup_metric_frame(initial_report: dict[str, Any], mitigated_report: dict[str, Any]) -> pd.DataFrame:
	initial_metrics = _subgroup_metrics(initial_report)
	mitigated_metrics = _subgroup_metrics(mitigated_report)
	groups = _sort_subgroups_for_display(initial_report, mitigated_report)
	rows: list[dict[str, Any]] = []
	for group in groups:
		initial = initial_metrics.get(group, {})
		mitigated = mitigated_metrics.get(group, {})
		rows.append(
			{
				"group": group,
				"initial_far": _safe_float(initial.get("far", 0.0)),
				"mitigated_far": _safe_float(mitigated.get("far", 0.0)),
				"initial_frr": _safe_float(initial.get("frr", 0.0)),
				"mitigated_frr": _safe_float(mitigated.get("frr", 0.0)),
				"far_delta": _safe_float(mitigated.get("far", 0.0)) - _safe_float(initial.get("far", 0.0)),
				"frr_delta": _safe_float(mitigated.get("frr", 0.0)) - _safe_float(initial.get("frr", 0.0)),
			}
		)
	return pd.DataFrame(rows)


def _plot_grouped_bar(metric_frame: pd.DataFrame, metric_key: str, title: str, output_path: str | Path, worst_group: str | None = None) -> Path:
	path = _prepare_output_path(output_path)
	fig, ax = plt.subplots(figsize=(14.0, 6.8))
	x = np.arange(len(metric_frame))
	width = 0.38
	initial_values = metric_frame[f"initial_{metric_key}"].to_numpy(dtype=float)
	mitigated_values = metric_frame[f"mitigated_{metric_key}"].to_numpy(dtype=float)
	initial_color = "#9ecae1"
	mitigated_color = "#3182bd"
	ax.bar(x - width / 2, initial_values, width, label="Initial", color=initial_color, edgecolor="#1f2937", linewidth=0.35)
	mitigated_bars = ax.bar(x + width / 2, mitigated_values, width, label="Mitigated", color=mitigated_color, edgecolor="#1f2937", linewidth=0.35)
	if worst_group and worst_group in set(metric_frame["group"]):
		worst_index = metric_frame.index[metric_frame["group"] == worst_group][0]
		mitigated_bars[worst_index].set_facecolor("#cb181d")
		mitigated_bars[worst_index].set_edgecolor("#7f1d1d")
		mitigated_bars[worst_index].set_linewidth(1.2)
	ax.set_xticks(x)
	ax.set_xticklabels(metric_frame["group"], rotation=60, ha="right")
	ax.set_ylabel(metric_key.upper())
	ax.set_title(title)
	ax.grid(axis="y", alpha=0.22)
	ax.legend(frameon=False, ncol=2)
	fig.tight_layout()
	fig.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	logger.info("Grouped bar chart saved to %s", path)
	return path


def plot_subgroup_far_chart(initial_report: dict[str, Any], mitigated_report: dict[str, Any], output_path: str | Path) -> Path:
	"""Plot subgroup FAR before and after mitigation."""

	metric_frame = _subgroup_metric_frame(initial_report, mitigated_report)
	worst_group = metric_frame.sort_values(["mitigated_far", "initial_far"], ascending=False).iloc[0]["group"] if not metric_frame.empty else None
	return _plot_grouped_bar(metric_frame, "far", "Subgroup FAR Before vs After Mitigation", output_path, worst_group=worst_group)


def plot_subgroup_frr_chart(initial_report: dict[str, Any], mitigated_report: dict[str, Any], output_path: str | Path) -> Path:
	"""Plot subgroup FRR before and after mitigation and highlight the worst subgroup."""

	metric_frame = _subgroup_metric_frame(initial_report, mitigated_report)
	worst_group = metric_frame.sort_values(["mitigated_frr", "initial_frr"], ascending=False).iloc[0]["group"] if not metric_frame.empty else None
	return _plot_grouped_bar(metric_frame, "frr", "Subgroup FRR Before vs After Mitigation", output_path, worst_group=worst_group)


def _cross_group_gap_frame(fairness_comparison: dict[str, Any]) -> pd.DataFrame:
	cross_group_deltas = fairness_comparison.get("cross_group_deltas", {})
	rows: list[dict[str, Any]] = []
	for pair_key, metrics in cross_group_deltas.items():
		if not isinstance(metrics, dict):
			continue
		rows.append(
			{
				"pair": str(pair_key),
				"initial_gap": _safe_float(metrics.get("initial_gap", max(_safe_float(metrics.get("initial_far")), _safe_float(metrics.get("initial_frr")))), 0.0),
				"mitigated_gap": _safe_float(metrics.get("mitigated_gap", max(_safe_float(metrics.get("mitigated_far")), _safe_float(metrics.get("mitigated_frr")))), 0.0),
				"gap_change": _safe_float(metrics.get("mitigated_gap", 0.0)) - _safe_float(metrics.get("initial_gap", 0.0)),
			}
		)
	frame = pd.DataFrame(rows)
	if not frame.empty:
		frame = frame.sort_values("gap_change")
	return frame


def plot_fairness_heatmap(initial_report: dict[str, Any], mitigated_report: dict[str, Any], fairness_comparison: dict[str, Any], output_path: str | Path) -> Path:
	"""Plot subgroup FAR/FRR metrics together with cross-group disparity information."""

	path = _prepare_output_path(output_path)
	metric_frame = _subgroup_metric_frame(initial_report, mitigated_report).set_index("group")
	cross_frame = _cross_group_gap_frame(fairness_comparison)
	cross_gap_lookup: dict[str, list[float]] = {}
	for _, row in cross_frame.iterrows():
		parts = str(row["pair"]).split("__vs__")
		for part in parts:
			cross_gap_lookup.setdefault(part, []).append(_safe_float(row["mitigated_gap"]))
	metric_frame = metric_frame.copy()
	metric_frame["cross_group_gap"] = [max(cross_gap_lookup.get(group, [0.0])) for group in metric_frame.index]
	heatmap_frame = metric_frame[["initial_far", "mitigated_far", "initial_frr", "mitigated_frr", "far_delta", "frr_delta", "cross_group_gap"]]
	fig, ax = plt.subplots(figsize=(12.5, 10.0))
	sns.heatmap(
		heatmap_frame,
		ax=ax,
		cmap="mako",
		linewidths=0.25,
		linecolor="#f0f0f0",
		cbar_kws={"label": "Rate / Gap"},
	)
	ax.set_title("Fairness Heatmap: Subgroup Rates and Cross-Group Disparities")
	ax.set_xlabel("Metric")
	ax.set_ylabel("Subgroup")
	fig.tight_layout()
	fig.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	logger.info("Fairness heatmap saved to %s", path)
	return path


def plot_mitigation_comparison(overall_metrics: dict[str, Any], output_path: str | Path) -> Path:
	"""Plot before/after mitigation overall verification metrics."""

	path = _prepare_output_path(output_path)
	initial_model = overall_metrics.get("initial_model", {})
	mitigated_model = overall_metrics.get("mitigated_model", {})
	metrics = ["Accuracy", "FAR", "FRR"]
	initial_values = [
		_safe_float(initial_model.get("accuracy", 0.0)),
		_safe_float(initial_model.get("far", 0.0)),
		_safe_float(initial_model.get("frr", 0.0)),
	]
	mitigated_values = [
		_safe_float(mitigated_model.get("accuracy", 0.0)),
		_safe_float(mitigated_model.get("far", 0.0)),
		_safe_float(mitigated_model.get("frr", 0.0)),
	]
	x = np.arange(len(metrics))
	width = 0.34
	fig, ax = plt.subplots(figsize=(8.5, 5.8))
	ax.bar(x - width / 2, initial_values, width, label="Initial", color="#9ecae1", edgecolor="#1f2937", linewidth=0.35)
	ax.bar(x + width / 2, mitigated_values, width, label="Mitigated", color="#3182bd", edgecolor="#1f2937", linewidth=0.35)
	ax.set_xticks(x)
	ax.set_xticklabels(metrics)
	ax.set_ylim(0.0, 1.05)
	ax.set_ylabel("Score")
	ax.set_title("Before vs After Mitigation: Accuracy and Error Rates")
	ax.grid(axis="y", alpha=0.22)
	ax.legend(frameon=False)
	fig.tight_layout()
	fig.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	logger.info("Mitigation comparison plot saved to %s", path)
	return path


def plot_disparity_gap(initial_report: dict[str, Any], mitigated_report: dict[str, Any], fairness_comparison: dict[str, Any], output_path: str | Path) -> Path:
	"""Plot subgroup and cross-group fairness gaps before and after mitigation."""

	path = _prepare_output_path(output_path)
	metric_frame = _subgroup_metric_frame(initial_report, mitigated_report)
	group_order = metric_frame.sort_values("frr_delta").copy()
	cross_frame = _cross_group_gap_frame(fairness_comparison)
	fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.0))

	# Panel 1: subgroup FAR deltas.
	ax = axes[0]
	metric_frame_sorted = metric_frame.sort_values("far_delta")
	ax.barh(metric_frame_sorted["group"], metric_frame_sorted["far_delta"], color="#6baed6", edgecolor="#1f2937", linewidth=0.3)
	ax.axvline(0.0, color="#374151", linestyle="--", linewidth=1.0)
	ax.set_title("Subgroup FAR Delta")
	ax.set_xlabel("Mitigated - Initial")
	ax.set_ylabel("Subgroup")

	# Panel 2: subgroup FRR deltas.
	ax = axes[1]
	metric_frame_sorted = metric_frame.sort_values("frr_delta")
	ax.barh(metric_frame_sorted["group"], metric_frame_sorted["frr_delta"], color="#74c476", edgecolor="#1f2937", linewidth=0.3)
	ax.axvline(0.0, color="#374151", linestyle="--", linewidth=1.0)
	ax.set_title("Subgroup FRR Delta")
	ax.set_xlabel("Mitigated - Initial")
	ax.set_ylabel("")

	# Panel 3: cross-group gap deltas.
	ax = axes[2]
	if not cross_frame.empty:
		cross_frame_sorted = cross_frame.sort_values("gap_change")
		show_frame = cross_frame_sorted.head(12) if len(cross_frame_sorted) > 12 else cross_frame_sorted
		ax.barh(show_frame["pair"], show_frame["gap_change"], color="#9e9ac8", edgecolor="#1f2937", linewidth=0.3)
		ax.axvline(0.0, color="#374151", linestyle="--", linewidth=1.0)
	ax.set_title("Cross-Group Gap Delta")
	ax.set_xlabel("Mitigated - Initial")
	ax.set_ylabel("")

	fig.suptitle("Fairness Disparity Gap Analysis", y=1.02, fontsize=14)
	fig.tight_layout()
	fig.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	logger.info("Disparity gap plot saved to %s", path)
	return path


def plot_fairness_dashboard(
	initial_report: dict[str, Any],
	mitigated_report: dict[str, Any],
	fairness_summary: dict[str, Any],
	fairness_comparison: dict[str, Any],
	overall_metrics: dict[str, Any],
	output_path: str | Path,
) -> Path:
	"""Create a compact dashboard for deployment-oriented fairness review."""

	path = _prepare_output_path(output_path)
	initial_overall = _overall_metrics(initial_report)
	mitigated_overall = _overall_metrics(mitigated_report)
	worst_subgroup = fairness_summary.get("worst_frr_group", {}) if isinstance(fairness_summary, dict) else {}
	worst_cross_group = fairness_summary.get("cross_group_summary", {}) if isinstance(fairness_summary, dict) else {}
	comparison_tradeoff = fairness_comparison.get("tradeoff_analysis", {}) if isinstance(fairness_comparison, dict) else {}
	initial_model = overall_metrics.get("initial_model", {})
	mitigated_model = overall_metrics.get("mitigated_model", {})

	fig = plt.figure(figsize=(14.0, 9.5))
	grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.15], wspace=0.35, hspace=0.28)
	cards = [
		("Overall FAR", f"{_safe_float(mitigated_overall.get('far', 0.0)):.3f}", "After mitigation"),
		("Overall FRR", f"{_safe_float(mitigated_overall.get('frr', 0.0)):.3f}", "After mitigation"),
		("Fairness Risk", str(fairness_summary.get("fairness_risk_level", "UNKNOWN")), "Audit assessment"),
		("Worst Subgroup", str(worst_subgroup.get("group", "n/a")), f"FRR = {_safe_float(worst_subgroup.get('frr', 0.0)):.3f}"),
		("Worst Cross-Group", str(worst_cross_group.get("worst_cross_group", "n/a")), f"FAR = {_safe_float(worst_cross_group.get('worst_cross_group_far', 0.0)):.3f}"),
		("Mitigation Status", "Improved" if comparison_tradeoff.get("fairness_improved") else "Trade-off", f"Accuracy Δ = {_safe_float(overall_metrics.get('tradeoff_analysis', {}).get('accuracy_change', 0.0)):.3f}"),
	]

	for index, (title, value, subtitle) in enumerate(cards):
		ax = fig.add_subplot(grid[index // 3, index % 3])
		ax.set_axis_off()
		panel_color = "#ecf4fb" if index < 2 else "#f7f7f7"
		ax.add_patch(
			plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=panel_color, edgecolor="#d1d5db", linewidth=1.0)
		)
		ax.text(0.04, 0.82, title, fontsize=11, fontweight="bold", color="#1f2937", transform=ax.transAxes)
		ax.text(0.04, 0.47, value, fontsize=18, fontweight="bold", color="#111827", transform=ax.transAxes)
		ax.text(0.04, 0.18, subtitle, fontsize=9, color="#4b5563", transform=ax.transAxes)

	status_ax = fig.add_subplot(grid[2, :])
	status_ax.set_axis_off()
	status_ax.add_patch(
		plt.Rectangle((0, 0), 1, 1, transform=status_ax.transAxes, facecolor="#ffffff", edgecolor="#d1d5db", linewidth=1.0)
	)
	initial_far = _safe_float(initial_model.get("far", initial_overall.get("far", 0.0)))
	mitigated_far = _safe_float(mitigated_model.get("far", mitigated_overall.get("far", 0.0)))
	initial_frr = _safe_float(initial_model.get("frr", initial_overall.get("frr", 0.0)))
	mitigated_frr = _safe_float(mitigated_model.get("frr", mitigated_overall.get("frr", 0.0)))
	status_text = (
		f"Initial FAR/FRR: {initial_far:.3f} / {initial_frr:.3f}\n"
		f"Mitigated FAR/FRR: {mitigated_far:.3f} / {mitigated_frr:.3f}\n"
		f"Trade-off summary: {comparison_tradeoff.get('summary', 'n/a')}"
	)
	status_ax.text(0.02, 0.72, "Deployment Fairness Dashboard", fontsize=13, fontweight="bold", transform=status_ax.transAxes)
	status_ax.text(0.02, 0.40, status_text, fontsize=10, color="#1f2937", transform=status_ax.transAxes, va="center")
	fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.05, wspace=0.35, hspace=0.28)
	fig.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	logger.info("Fairness dashboard saved to %s", path)
	return path


def load_artifacts(config: VisualizationConfig) -> dict[str, Any]:
	"""Load the JSON artifacts required for the visual analytics layer."""

	logger.info("Loading visualization artifacts")
	initial_report = _load_json(config.initial_audit_path)
	mitigated_report = _load_json(config.mitigated_audit_path)
	fairness_summary = _load_json(config.fairness_summary_path)
	fairness_comparison = _load_json(config.fairness_comparison_path)
	cross_group_metrics = _load_json(config.cross_group_metrics_path)
	threshold_payload = _load_json(config.threshold_path)
	overall_metrics = _load_json(config.overall_metrics_path)
	return {
		"initial_report": initial_report,
		"mitigated_report": mitigated_report,
		"fairness_summary": fairness_summary,
		"fairness_comparison": fairness_comparison,
		"cross_group_metrics": cross_group_metrics,
		"threshold_payload": threshold_payload,
		"overall_metrics": overall_metrics,
	}


def run_visualization_pipeline(config: VisualizationConfig | None = None) -> dict[str, Any]:
	"""Generate the full visualization set from existing artifacts."""

	configure_logging()
	configure_style()
	cfg = config or VisualizationConfig()
	artifacts = load_artifacts(cfg)
	cfg.plots_dir.mkdir(parents=True, exist_ok=True)

	roc_analysis, det_analysis, threshold = _build_validation_curve_data(cfg)
	plots: list[Path] = []
	plots.append(save_publication_roc_plot(roc_analysis, threshold, cfg.plots_dir / "roc_curve_publication.png"))
	plots.append(save_publication_det_plot(det_analysis, threshold, cfg.plots_dir / "det_curve_publication.png"))
	plots.append(plot_subgroup_far_chart(artifacts["initial_report"], artifacts["mitigated_report"], cfg.plots_dir / "subgroup_far_chart.png"))
	plots.append(plot_subgroup_frr_chart(artifacts["initial_report"], artifacts["mitigated_report"], cfg.plots_dir / "subgroup_frr_chart.png"))
	plots.append(plot_fairness_heatmap(artifacts["initial_report"], artifacts["mitigated_report"], artifacts["fairness_comparison"], cfg.plots_dir / "fairness_heatmap.png"))
	plots.append(plot_mitigation_comparison(artifacts["overall_metrics"], cfg.plots_dir / "mitigation_comparison.png"))
	plots.append(plot_disparity_gap(artifacts["initial_report"], artifacts["mitigated_report"], artifacts["fairness_comparison"], cfg.plots_dir / "disparity_gap_plot.png"))
	plots.append(plot_fairness_dashboard(artifacts["initial_report"], artifacts["mitigated_report"], artifacts["fairness_summary"], artifacts["fairness_comparison"], artifacts["overall_metrics"], cfg.plots_dir / "fairness_dashboard.png"))

	logger.info("Visualization pipeline completed | plots=%s", len(plots))
	print(f"TOTAL_PLOTS_GENERATED={len(plots)}")
	print(f"PLOT_DIRECTORY={cfg.plots_dir}")
	return {"plots": plots, "plot_directory": cfg.plots_dir}


def main() -> None:
	"""Executable entry point for the visualization pipeline."""

	run_visualization_pipeline()


if __name__ == "__main__":
	main()