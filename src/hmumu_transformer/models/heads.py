from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grl import grad_reverse

@dataclass(frozen=True)
class HeadConfig:
    hidden: int
    dropout: float

class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, cfg: HeadConfig, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, n_classes),
        )

    def forward(self, h_cls: torch.Tensor) -> torch.Tensor:
        return self.net(h_cls)

class MassAdversaryHead(nn.Module):
    def __init__(self, d_model: int, cfg: HeadConfig, n_bins: int):
        super().__init__()
        self.n_bins = int(n_bins)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, self.n_bins),
        )

    def forward(self, h_cls: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        h_rev = grad_reverse(h_cls, lambda_grl)
        return self.net(h_rev)
