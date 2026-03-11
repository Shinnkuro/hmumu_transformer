from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import TokenConfig, build_tokens_from_row

LABELS = {"ggH": 0, "VBF": 1, "DY": 2}

@dataclass(frozen=True)
class EventSample:
    x: torch.Tensor     # [N,F]
    v: torch.Tensor     # [N,4]
    m: torch.Tensor     # [N]
    y: torch.Tensor     # []
    mass: torch.Tensor  # []

class HmumuTokenDataset(Dataset):
    """In-memory dataset built from numpy columns."""
    def __init__(self, arrays: Dict[str, np.ndarray], label: int, token_cfg: TokenConfig):
        self.arrays = arrays
        self.label = int(label)
        self.token_cfg = token_cfg
        # Use length from any column
        self._n = int(next(iter(arrays.values())).shape[0])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> EventSample:
        row = {k: self.arrays[k][idx] for k in self.arrays.keys()}
        x, v, m, mm = build_tokens_from_row(row, self.token_cfg)
        return EventSample(
            x=torch.from_numpy(x),
            v=torch.from_numpy(v),
            m=torch.from_numpy(m),
            y=torch.tensor(self.label, dtype=torch.long),
            mass=torch.tensor(mm, dtype=torch.float32),
        )
