from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

@dataclass(frozen=True)
class SplitSpec:
    n_folds: int
    train_folds: Tuple[int, ...]
    val_folds: Tuple[int, ...]
    test_folds: Tuple[int, ...]

def make_fold_indices(n: int, n_folds: int) -> np.ndarray:
    # Deterministic: fold = row_index % n_folds
    idx = np.arange(n, dtype=np.int64)
    return idx % int(n_folds)

def split_indices(n: int, spec: SplitSpec) -> Dict[str, np.ndarray]:
    folds = make_fold_indices(n, spec.n_folds)
    train = np.nonzero(np.isin(folds, np.array(spec.train_folds)))[0]
    val = np.nonzero(np.isin(folds, np.array(spec.val_folds)))[0]
    test = np.nonzero(np.isin(folds, np.array(spec.test_folds)))[0]
    return {"train": train, "val": val, "test": test}
