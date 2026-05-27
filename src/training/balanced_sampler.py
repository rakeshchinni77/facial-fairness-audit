"""Balanced sampling utilities for subgroup-aware triplet training.

Balanced batches matter because random sampling can let majority subgroups
dominate optimization, which can destabilize representation learning and later
fairness auditing. This sampler keeps subgroup representation more even without
introducing any fairness-metric logic.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterator, Sequence

from torch.utils.data import BatchSampler


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BalancedBatchSamplerConfig:
	"""Configuration for deterministic subgroup-balanced batching."""

	batch_size: int = 32
	random_seed: int = 42
	shuffle: bool = True


class SubgroupBalancedBatchSampler(BatchSampler):
	"""Yield batches with approximate subgroup balance."""

	def __init__(
		self,
		indices: Sequence[int],
		subgroups: Sequence[str],
		batch_size: int = 32,
		random_seed: int = 42,
		shuffle: bool = True,
	) -> None:
		if len(indices) != len(subgroups):
			raise ValueError("indices and subgroups must have the same length")
		if batch_size <= 0:
			raise ValueError("batch_size must be greater than zero")

		self.batch_size = batch_size
		self.random_seed = random_seed
		self.shuffle = shuffle
		self._groups: dict[str, list[int]] = defaultdict(list)
		for index, subgroup in zip(indices, subgroups):
			self._groups[str(subgroup)].append(int(index))
		self._subgroups = sorted(self._groups)
		logger.info("Balanced batch sampler initialized with %s subgroups", len(self._subgroups))

	def __iter__(self) -> Iterator[list[int]]:
		rng = random.Random(self.random_seed)
		queues = {subgroup: deque(indices) for subgroup, indices in self._groups.items()}
		if self.shuffle:
			for subgroup in self._subgroups:
				shuffled = list(queues[subgroup])
				rng.shuffle(shuffled)
				queues[subgroup] = deque(shuffled)

		active = list(self._subgroups)
		if self.shuffle:
			rng.shuffle(active)

		batch: list[int] = []
		while active:
			for subgroup in list(active):
				queue = queues[subgroup]
				if not queue:
					active.remove(subgroup)
					continue
				batch.append(queue.popleft())
				if len(batch) == self.batch_size:
					yield batch
					batch = []
				if not queue and subgroup in active:
					active.remove(subgroup)
				if not active:
					break

		if batch:
			yield batch

	def __len__(self) -> int:
		from math import ceil

		return ceil(sum(len(indices) for indices in self._groups.values()) / self.batch_size)
