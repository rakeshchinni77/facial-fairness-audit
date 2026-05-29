from __future__ import annotations

import torch

from src.training.callbacks import build_checkpoint_state, save_checkpoint
from src.training.trainer import (
	PairCSVSampleDataset,
	TrainerConfig,
	TripletCSVSampleDataset,
	build_optimizer,
	build_scheduler,
	build_triplet_model,
	build_training_transform,
	build_validation_loader,
)
from tests.helpers import load_json_strict


def test_training_datasets_and_components_initialize(processed_dir) -> None:
	triplet_dataset = TripletCSVSampleDataset(processed_dir / "train_triplets.csv", transform=build_training_transform(), max_rows=2)
	pair_dataset = PairCSVSampleDataset(processed_dir / "validation_pairs.csv", max_rows=2)
	assert len(triplet_dataset) >= 1, "Triplet dataset should expose at least one row"
	assert len(pair_dataset) >= 1, "Validation pair dataset should expose at least one row"
	triplet_sample = triplet_dataset[0]
	pair_sample = pair_dataset[0]
	assert triplet_sample["anchor"].shape == (3, 224, 224), "Triplet anchors must be resized to 224x224"
	assert pair_sample["image_a"].shape == (3, 224, 224), "Validation images must be resized to 224x224"
	assert triplet_sample["subgroup"], "Triplet dataset must preserve subgroup metadata"
	assert pair_sample["label"].item() in {0.0, 1.0}, "Validation labels must be binary"


def test_trainer_builds_optimizer_scheduler_and_checkpoint(tmp_path, processed_dir) -> None:
	config = TrainerConfig(
		processed_dir=processed_dir,
		artifacts_dir=tmp_path / "artifacts",
		device="cpu",
		batch_size=2,
		validation_batch_size=2,
		epochs=1,
		max_train_rows=2,
		max_validation_rows=2,
		pretrained=False,
		freeze_backbone=False,
	)
	model = build_triplet_model(config)
	optimizer = build_optimizer(model, config.learning_rate)
	scheduler = build_scheduler(optimizer)
	assert optimizer.param_groups, "Optimizer must expose at least one parameter group"
	assert scheduler.state_dict(), "Scheduler must initialize correctly"
	state = build_checkpoint_state(model, optimizer, scheduler, epoch=1, best_validation_loss=0.5, metadata={"source": "test"})
	checkpoint_path = save_checkpoint(state, tmp_path / "artifacts" / "checkpoint.pt")
	assert checkpoint_path.exists(), "Checkpoint save path must exist after saving"
	summary = load_json_strict(processed_dir.parent.parent / "artifacts" / "training_summary.json")
	assert isinstance(summary, dict), "Training summary artifact should be valid JSON"
