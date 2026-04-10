from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import DataLoader

from .scaler import MaskedStandardScaler
from .shards import build_dataloaders_from_shards


@dataclass
class BuiltData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    x_scaler: MaskedStandardScaler
    mass_train: np.ndarray
    label_names: List[str]
    class_weights: np.ndarray


def build_dataloaders(
    cfg: Dict[str, Any],
    *,
    x_scaler_override: Optional[MaskedStandardScaler] = None,
) -> BuiltData:
    built = build_dataloaders_from_shards(cfg, x_scaler_override=x_scaler_override)
    return BuiltData(
        train_loader=built.train_loader,
        val_loader=built.val_loader,
        test_loader=built.test_loader,
        x_scaler=built.x_scaler,
        mass_train=built.mass_train,
        label_names=built.label_names,
        class_weights=built.class_weights,
    )
