from __future__ import annotations

from src.mitigation.mitigation_pipeline import MitigationConfig, load_mitigated_model
from src.mitigation.rebalancing import compute_subgroup_weights, load_triplets
from src.models.triplet_network import TripletNetwork
from tests.helpers import load_json_strict


def test_mitigation_artifacts_and_summary_exist(artifacts_dir, results_dir) -> None:
	assert (artifacts_dir / "mitigated_model.pth").exists(), "Mitigated checkpoint must exist"
	assert (artifacts_dir / "best_mitigated_model.pth").exists(), "Best mitigated checkpoint must exist"
	assert (results_dir / "fairness_comparison.json").exists(), "Fairness comparison JSON must exist"
	assert (artifacts_dir / "mitigation_training_summary.json").exists(), "Mitigation training summary must exist"
	summary = load_json_strict(artifacts_dir / "mitigation_training_summary.json")
	assert "checkpoint_paths" in summary and "validation_loss" in summary, "Mitigation summary must contain checkpoint and validation information"


def test_mitigation_pipeline_loads_checkpoint_and_forwards(processed_dir) -> None:
	triplets = load_triplets(processed_dir / "train_triplets.csv")
	weights = compute_subgroup_weights(triplets)
	config = MitigationConfig(
		processed_dir=processed_dir,
		device="cpu",
		pretrained=False,
		max_train_rows=2,
		max_validation_rows=2,
	)
	model = load_mitigated_model(config, weights)
	assert isinstance(model, TripletNetwork), "Mitigation loader must return a TripletNetwork"
