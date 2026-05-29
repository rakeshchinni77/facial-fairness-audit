from __future__ import annotations

import torch

from src.models.triplet_network import TripletNetwork
from src.training.callbacks import build_checkpoint_state, save_checkpoint
from src.training.trainer import TrainerConfig, build_optimizer, build_scheduler, build_triplet_model
from tests.helpers import load_json_artifact


def test_trainer_initializes_optimizer_scheduler_and_model(project_root):
    config = TrainerConfig(processed_dir=project_root / "data" / "processed", artifacts_dir=project_root / "artifacts", pretrained=False, freeze_backbone=True, train_backbone_block=False)
    model = build_triplet_model(config)
    optimizer = build_optimizer(model, learning_rate=1e-4)
    scheduler = build_scheduler(optimizer)

    assert isinstance(model, TripletNetwork), "Trainer should construct a TripletNetwork"
    assert any(parameter.requires_grad for parameter in model.parameters()), "Trainer model should expose trainable parameters"
    assert optimizer.param_groups, "Optimizer must be initialized with at least one parameter group"
    assert scheduler is not None, "Scheduler must be initialized"


def test_checkpoint_save_path_exists(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    state = build_checkpoint_state(model, optimizer, scheduler, epoch=1, best_validation_loss=0.5)
    path = tmp_path / "checkpoint.pth"
    saved = save_checkpoint(state, path)
    assert saved.exists(), f"Checkpoint path was not created: {saved}"
    assert saved.stat().st_size > 0, "Checkpoint file should not be empty"


def test_training_summary_json_structure(project_root):
    summary_path = project_root / "artifacts" / "training_summary.json"
    summary = load_json_artifact(summary_path)
    assert summary["epochs_completed"] >= 1, "Training summary should report at least one epoch"
    assert "best_model_path" in summary, "Training summary missing best_model_path"
    assert "metrics" in summary and isinstance(summary["metrics"], list), "Training summary missing metrics list"
    assert summary["metrics"], "Training summary metrics should not be empty"
    first_epoch = summary["metrics"][0]
    for key in ("epoch", "train_loss", "validation_loss", "learning_rate", "metadata"):
        assert key in first_epoch, f"Training metric entry missing key: {key}"


def test_mitigation_training_summary_json_structure(project_root):
    summary_path = project_root / "artifacts" / "mitigation_training_summary.json"
    summary = load_json_artifact(summary_path)
    assert summary["subgroup_rebalancing_enabled"] is True, "Mitigation summary should record subgroup rebalancing"
    assert summary["weighted_triplet_loss_enabled"] is True, "Mitigation summary should record weighted loss"
    assert "checkpoint_paths" in summary and "best_mitigated_model" in summary["checkpoint_paths"], "Mitigation summary missing checkpoint paths"
    assert "rebalancing_summary" in summary, "Mitigation summary missing rebalancing summary"
