from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch


@dataclass
class MaskedStandardScaler:
    mean: np.ndarray  # [F]
    std: np.ndarray   # [F]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device).view(1, 1, -1)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).view(1, 1, -1)
        return (x - mean) / std

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MaskedStandardScaler":
        return MaskedStandardScaler(
            mean=np.asarray(d["mean"], dtype=np.float32),
            std=np.asarray(d["std"], dtype=np.float32),
        )


@dataclass
class MaskedStandardScalerAccumulator:
    """Streaming masked mean/std estimator for token features."""

    feature_dim: int

    def __post_init__(self) -> None:
        self.count = np.zeros((self.feature_dim,), dtype=np.float64)
        self.sum = np.zeros((self.feature_dim,), dtype=np.float64)
        self.sumsq = np.zeros((self.feature_dim,), dtype=np.float64)

    def update(self, x: np.ndarray, m: np.ndarray) -> None:
        if x.ndim != 3 or m.ndim != 2:
            raise ValueError("Expected x[B,N,F] and m[B,N].")
        if x.shape[0] == 0:
            return
        mask = m.astype(bool)[..., None]
        x64 = x.astype(np.float64, copy=False)
        self.count += mask.sum(axis=(0, 1), dtype=np.int64)
        self.sum += (x64 * mask).sum(axis=(0, 1), dtype=np.float64)
        self.sumsq += ((x64 * x64) * mask).sum(axis=(0, 1), dtype=np.float64)

    def finalize(self) -> MaskedStandardScaler:
        count = np.clip(self.count, 1.0, None)
        mean = self.sum / count
        var = (self.sumsq / count) - np.square(mean)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        std = np.where(std < 1e-6, 1.0, std)
        return MaskedStandardScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


def fit_masked_standard_scaler(x: np.ndarray, m: np.ndarray) -> MaskedStandardScaler:
    accumulator = MaskedStandardScalerAccumulator(feature_dim=int(x.shape[-1]))
    accumulator.update(x, m)
    return accumulator.finalize()
