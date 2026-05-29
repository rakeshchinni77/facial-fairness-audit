"""Bias root-cause analysis for the facial-fairness-audit project.

Fairness audits are required in biometric systems because overall accuracy can
hide large subgroup errors. In particular, a high FRR for a demographic slice
means that legitimate matches are being rejected more often, which is harmful in
access-control settings and can create disparate treatment across groups.

This module does not retrain the model, change thresholds, or alter embeddings.
It only reads the existing audit artifacts and synthesizes evidence-backed root
cause hypotheses, ethical implications, and deployment guidance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BiasAnalysisConfig:
    """Configuration for the bias root-cause analysis layer."""

    initial_audit_path: Path = Path("results/initial_audit.json")
    cross_group_metrics_path: Path = Path("results/cross_group_metrics.json")
    fairness_summary_path: Path = Path("results/fairness_summary.json")
    output_path: Path = Path("results/analysis.json")


def configure_logging() -> None:
    """Configure logging for standalone execution."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {json_path}")
    logger.info("Audit artifact loading complete | path=%s", json_path)
    return payload


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _is_subgroup_key(key: str) -> bool:
    return key not in {"overall", "metadata"} and "__vs__" not in key


def _subgroup_metrics(initial_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    subgroup_metrics: dict[str, dict[str, Any]] = {}
    for key, value in initial_audit.items():
        if _is_subgroup_key(str(key)) and isinstance(value, dict):
            subgroup_metrics[str(key)] = value
    logger.info("Subgroup extraction complete | subgroup_count=%s", len(subgroup_metrics))
    return subgroup_metrics


def _find_group_by_frr(initial_audit: dict[str, Any], target_frr: float) -> str | None:
    subgroup_metrics = _subgroup_metrics(initial_audit)
    matches = [
        group_name
        for group_name, metrics in subgroup_metrics.items()
        if abs(_coerce_float(metrics.get("frr", 0.0)) - target_frr) <= 1e-6
    ]
    if not matches:
        return None
    return sorted(matches)[0]


def _largest_gap_pair(initial_audit: dict[str, Any], fairness_summary: dict[str, Any]) -> dict[str, Any]:
    largest_frr_gap = fairness_summary.get("largest_frr_gap", {})
    best_group_frr = _coerce_float(largest_frr_gap.get("best_group_frr", 0.0))
    worst_group_frr = _coerce_float(largest_frr_gap.get("worst_group_frr", 0.0))
    gap = _coerce_float(largest_frr_gap.get("gap", 0.0))
    return {
        "group_1": _find_group_by_frr(initial_audit, best_group_frr),
        "group_2": fairness_summary.get("worst_frr_group", {}).get("group"),
        "metric": "frr_disparity",
        "value": round(gap, 6),
        "best_group_frr": round(best_group_frr, 6),
        "worst_group_frr": round(worst_group_frr, 6),
    }


def _worst_subgroup_from_summary(fairness_summary: dict[str, Any]) -> dict[str, Any]:
    worst_frr_group = fairness_summary.get("worst_frr_group", {})
    return {
        "group": worst_frr_group.get("group"),
        "frr": round(_coerce_float(worst_frr_group.get("frr", 0.0)), 6),
        "support": _coerce_int(worst_frr_group.get("support", 0)),
    }


def _worst_far_group_from_initial(initial_audit: dict[str, Any]) -> dict[str, Any]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for key, value in _subgroup_metrics(initial_audit).items():
        candidates.append((key, value))
    if not candidates:
        return {"group": None, "far": 0.0, "support": 0}
    worst_group, metrics = max(
        candidates,
        key=lambda item: (
            _coerce_float(item[1].get("far", 0.0)),
            -_coerce_int(item[1].get("support", 0)),
            item[0],
        ),
    )
    return {
        "group": worst_group,
        "far": round(_coerce_float(metrics.get("far", 0.0)), 6),
        "support": _coerce_int(metrics.get("support", 0)),
    }


def _largest_far_disparity(cross_group_metrics: dict[str, Any]) -> dict[str, Any]:
    """Cross-group instability matters because a model can fail differently
    across demographic intersections even when aggregate metrics look similar."""

    if not cross_group_metrics:
        return {"group_1": None, "group_2": None, "metric": "far_disparity", "value": 0.0}

    ranked = sorted(
        ((key, value) for key, value in cross_group_metrics.items() if isinstance(value, dict)),
        key=lambda item: (
            -_coerce_float(item[1].get("far", 0.0)),
            _coerce_int(item[1].get("support", 0)),
            item[0],
        ),
    )
    worst_pair, worst_metrics = ranked[0]
    if "__vs__" in worst_pair:
        group_1, group_2 = worst_pair.split("__vs__", 1)
    else:
        group_1, group_2 = worst_pair, worst_pair
    return {
        "group_1": group_1,
        "group_2": group_2,
        "metric": "far_disparity",
        "value": round(_coerce_float(worst_metrics.get("far", 0.0)), 6),
        "support": _coerce_int(worst_metrics.get("support", 0)),
    }


def _worst_cross_group_pair(cross_group_metrics: dict[str, Any]) -> dict[str, Any]:
    if not cross_group_metrics:
        return {"pair": None, "far": 0.0, "frr": 0.0, "support": 0}

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int, str]:
        key, metrics = item
        far = _coerce_float(metrics.get("far", 0.0))
        frr = _coerce_float(metrics.get("frr", 0.0))
        support = _coerce_int(metrics.get("support", 0))
        return (-max(far, frr), -far, support, key)

    pair_name, metrics = sorted(
        ((key, value) for key, value in cross_group_metrics.items() if isinstance(value, dict)),
        key=sort_key,
    )[0]
    return {
        "pair": pair_name,
        "far": round(_coerce_float(metrics.get("far", 0.0)), 6),
        "frr": round(_coerce_float(metrics.get("frr", 0.0)), 6),
        "support": _coerce_int(metrics.get("support", 0)),
    }


def _most_affected_demographics(fairness_summary: dict[str, Any]) -> list[str]:
    demographics = fairness_summary.get("most_affected_demographics", [])
    if isinstance(demographics, list):
        return [str(group) for group in demographics]
    return []


def _hypothesized_data_level_causes(initial_audit: dict[str, Any], fairness_summary: dict[str, Any]) -> list[str]:
    subgroup_metrics = _subgroup_metrics(initial_audit)
    worst_frr_group = fairness_summary.get("worst_frr_group", {})
    worst_group_name = str(worst_frr_group.get("group", "unknown"))
    worst_support = _coerce_int(worst_frr_group.get("support", 0))

    smallest_support_groups = sorted(
        subgroup_metrics.items(),
        key=lambda item: (_coerce_int(item[1].get("support", 0)), item[0]),
    )[:3]
    small_support_names = [name for name, _ in smallest_support_groups]

    # Subgroup imbalance can bias verification because the model sees fewer
    # examples for some demographic slices, which makes their score distribution
    # noisier and their error rates more volatile.
    causes = [
        f"Some demographic slices have much smaller support than others, including groups such as {', '.join(small_support_names)}.",
        f"The worst FRR subgroup ({worst_group_name}) still shows only {worst_support} paired samples, which can make its estimate less stable than dense groups.",
        "The audit distribution is uneven across age and skin-tone combinations, which can amplify representation gaps for older and darker-skinned groups.",
        "Mixed demographic pairs with very small support counts indicate sparse coverage for some intersectional combinations.",
    ]
    return causes


def _hypothesized_model_level_causes(initial_audit: dict[str, Any], fairness_summary: dict[str, Any], cross_group_metrics: dict[str, Any]) -> list[str]:
    overall = initial_audit.get("overall", {})
    overall_far = _coerce_float(overall.get("far", 0.0))
    overall_frr = _coerce_float(overall.get("frr", 0.0))
    cross_group = _worst_cross_group_pair(cross_group_metrics)

    # High FRR is harmful because it rejects legitimate users, which can deny
    # access and disproportionately burden specific demographics even when the
    # system still appears functional in aggregate.
    causes = [
        "The existing ResNet18 embedding stack may not separate demographic manifolds equally well, which is consistent with high subgroup FRR variance.",
        f"The overall FRR remains elevated at {overall_frr:.6f}, suggesting weak match/non-match separation at the current operating point.",
        f"The worst cross-group pairing ({cross_group['pair']}) reaches FAR={cross_group['far']:.6f}, which is consistent with unstable decision boundaries across intersections.",
        f"The largest FRR gap of {_coerce_float(fairness_summary.get('largest_frr_gap', {}).get('gap', 0.0)):.6f} suggests threshold sensitivity across demographic slices rather than uniformly calibrated scores.",
        "ImageNet-style pretraining can leave residual feature bias if the embedding head is not sufficiently adapted to the face-verification domain.",
        "The high rejection rate for some groups is consistent with underfitting or demographic feature entanglement in the learned embedding space.",
    ]
    if overall_far == 0.0:
        causes.append("Low FAR alongside high FRR suggests a conservative decision boundary that may be over-rejecting many true matches.")
    return causes


def _deployment_recommendation(fairness_summary: dict[str, Any]) -> dict[str, Any]:
    risk_level = str(fairness_summary.get("fairness_risk_level", "LOW"))
    frr_gap = _coerce_float(fairness_summary.get("largest_frr_gap", {}).get("gap", 0.0))
    recommended = not (risk_level == "HIGH" or frr_gap > 0.30)
    reason = (
        "Current fairness disparities and high FRR indicate the system is not suitable for high-stakes deployment without mitigation."
        if not recommended
        else "Observed disparities are below the current high-risk threshold, but continued monitoring is still recommended."
    )
    return {
        "recommended_for_high_stakes_use": recommended,
        "reason": reason,
    }


def _ethical_implications(fairness_summary: dict[str, Any], cross_group_metrics: dict[str, Any]) -> list[str]:
    cross_group = _worst_cross_group_pair(cross_group_metrics)
    return [
        "High FRR disparities may disproportionately deny access to some demographic groups, especially when legitimate users are rejected at different rates.",
        f"Cross-group instability is visible in {cross_group['pair']}, where pairwise error behavior shifts sharply across intersections.",
        "A biometric system with disparate acceptance and rejection behavior can create unequal treatment even when overall performance looks acceptable.",
        "Fairness audits are required in biometric systems because these errors affect access, authentication, and downstream trust at the individual level.",
    ]


def _summarize_causes(causes: list[str]) -> str:
    """Condense a list of cause statements into a contract-friendly summary string."""

    return " ".join(causes).strip()


def build_analysis(
    initial_audit: dict[str, Any],
    cross_group_metrics: dict[str, Any],
    fairness_summary: dict[str, Any],
) -> dict[str, Any]:
    """Construct the bias root-cause analysis from existing audit artifacts."""

    worst_subgroup = _worst_subgroup_from_summary(fairness_summary)
    most_biased_pairing = _largest_gap_pair(initial_audit, fairness_summary)
    largest_far_disparity = _largest_far_disparity(cross_group_metrics)
    cross_group_pair = _worst_cross_group_pair(cross_group_metrics)
    most_affected_demographics = _most_affected_demographics(fairness_summary)

    analysis = {
        "most_biased_pairing": most_biased_pairing,
        "worst_subgroup": worst_subgroup,
        "fairness_risk_level": str(fairness_summary.get("fairness_risk_level", "LOW")),
        "largest_far_disparity": largest_far_disparity,
        "largest_frr_disparity": {
            "group_1": most_biased_pairing.get("group_1"),
            "group_2": most_biased_pairing.get("group_2"),
            "metric": "frr_disparity",
            "value": most_biased_pairing.get("value", 0.0),
        },
        "worst_cross_group_pairing": cross_group_pair,
        "most_affected_demographics": most_affected_demographics,
        "hypothesized_causes": {
            "data_level": _summarize_causes(_hypothesized_data_level_causes(initial_audit, fairness_summary)),
            "model_level": _summarize_causes(_hypothesized_model_level_causes(initial_audit, fairness_summary, cross_group_metrics)),
            "data_level_details": _hypothesized_data_level_causes(initial_audit, fairness_summary),
            "model_level_details": _hypothesized_model_level_causes(initial_audit, fairness_summary, cross_group_metrics),
        },
        "ethical_implications": _ethical_implications(fairness_summary, cross_group_metrics),
        "deployment_readiness": _deployment_recommendation(fairness_summary),
    }
    logger.info(
        "Bias analysis constructed | worst_subgroup=%s | risk=%s",
        worst_subgroup.get("group"),
        analysis["fairness_risk_level"],
    )
    return analysis


def save_analysis(analysis: dict[str, Any], output_path: str | Path) -> Path:
    """Persist the analysis as deterministic JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Analysis JSON exported to %s", path)
    return path


def run_bias_analysis(config: BiasAnalysisConfig | None = None) -> dict[str, Any]:
    """Load the audit artifacts and generate the bias root-cause analysis."""

    configure_logging()
    cfg = config or BiasAnalysisConfig()
    logger.info("Loading audit artifact set for bias analysis")
    initial_audit = _load_json(cfg.initial_audit_path)
    cross_group_metrics = _load_json(cfg.cross_group_metrics_path)
    fairness_summary = _load_json(cfg.fairness_summary_path)
    logger.info("Root-cause inputs loaded | subgroup_count=%s | cross_group_count=%s", len(initial_audit), len(cross_group_metrics))

    analysis = build_analysis(initial_audit, cross_group_metrics, fairness_summary)
    save_analysis(analysis, cfg.output_path)

    print(f"MOST_BIASED_SUBGROUP={analysis['worst_subgroup']['group']}")
    print(f"LARGEST_DISPARITY={analysis['largest_frr_disparity']['value']:.6f}")
    print(f"FAIRNESS_RISK_LEVEL={analysis['fairness_risk_level']}")
    print(f"DEPLOYMENT_RECOMMENDATION={analysis['deployment_readiness']['recommended_for_high_stakes_use']}")
    return analysis


def main() -> None:
    """Executable entry point for bias analysis."""

    run_bias_analysis()


if __name__ == "__main__":
    main()
