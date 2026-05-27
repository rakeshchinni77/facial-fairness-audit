"""Training callbacks for checkpointing and early stopping."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


logger = logging.getLogger(__name__)


@dataclass
class EarlyStopping:
	"""Early stopping based on validation loss."""

	patience: int = 3
	min_delta: float = 0.0
	best_loss: float = float("inf")
	bad_epochs: int = 0
	should_stop: bool = False

	def step(self, validation_loss: float) -> bool:
		if validation_loss < self.best_loss - self.min_delta:
			self.best_loss = validation_loss
			self.bad_epochs = 0
			logger.info("Early stopping improved best loss to %.6f", validation_loss)
			return False
		self.bad_epochs += 1
		if self.bad_epochs >= self.patience:
			self.should_stop = True
			logger.info("Early stopping triggered after %s bad epochs", self.bad_epochs)
		return self.should_stop


def save_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
	"""Save a torch checkpoint to disk."""

	checkpoint_path = Path(path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(state, checkpoint_path)
	logger.info("Checkpoint saved to %s", checkpoint_path)
	return checkpoint_path


def build_checkpoint_state(
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scheduler: Any,
	epoch: int,
	best_validation_loss: float,
	metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Assemble a checkpoint dictionary for model persistence."""

	return {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
		"best_validation_loss": best_validation_loss,
		"metadata": metadata or {},
	}
