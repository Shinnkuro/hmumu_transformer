from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

@dataclass
class BalancedBatchConfig:
    per_class_batch: int
    n_classes: int = 3
    steps_per_epoch: int = 1
    seed: Optional[int] = None

class BalancedBatchSampler(Sampler[List[int]]):
    """Yield balanced batches with replacement as needed.

    Expects the dataset to be a ConcatDataset-like where indices can be mapped to class pools
    externally (we provide class_index_pools).
    """
    def __init__(self, class_index_pools: Sequence[np.ndarray], cfg: BalancedBatchConfig):
        self.class_index_pools = [np.asarray(p, dtype=np.int64) for p in class_index_pools]
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        assert len(self.class_index_pools) == cfg.n_classes

    def __iter__(self) -> Iterator[List[int]]:
        # For each epoch, yield `steps_per_epoch` batches.
        for _ in range(self.cfg.steps_per_epoch):
            batch: List[int] = []
            for c in range(self.cfg.n_classes):
                pool = self.class_index_pools[c]
                if pool.size == 0:
                    raise RuntimeError(f"Empty index pool for class {c}.")
                # Sample with replacement if needed.
                pick = self.rng.choice(pool, size=self.cfg.per_class_batch, replace=(pool.size < self.cfg.per_class_batch))
                batch.extend(pick.tolist())
            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return int(self.cfg.steps_per_epoch)
