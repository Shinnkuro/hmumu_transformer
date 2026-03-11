from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

_EPS = 1e-12

@dataclass
class MaskedStandardScaler:
    mean: np.ndarray  # [F]
    std: np.ndarray   # [F]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MaskedStandardScaler":
        return MaskedStandardScaler(mean=np.asarray(d["mean"], dtype=np.float32),
                                    std=np.asarray(d["std"], dtype=np.float32))

def fit_masked_standard_scaler(x: np.ndarray, m: np.ndarray) -> MaskedStandardScaler:
    """Fit per-feature mean/std for x[B,N,F], using mask m[B,N] on tokens.

    CLS token is always included (m[:,0]=1). Padding tokens are excluded.
    For each feature dimension f:
      mean_f = mean over all included tokens
      std_f = std over all included tokens (min std = 1.0 if too small)
    """
    assert x.ndim == 3 and m.ndim == 2
    B, N, F = x.shape
    mask = (m.astype(bool)).reshape(B, N, 1)
    # Count included values per feature
    count = mask.sum(axis=(0, 1)).astype(np.float64)  # [1] actually, but broadcast
    # Sum across tokens
    sum_ = (x * mask).sum(axis=(0, 1)).astype(np.float64)  # [F]
    mean = sum_ / np.clip(count, 1.0, None)
    # Var
    diff = (x - mean.reshape(1, 1, F)) * mask
    var = (diff ** 2).sum(axis=(0, 1)).astype(np.float64) / np.clip(count, 1.0, None)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return MaskedStandardScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))
