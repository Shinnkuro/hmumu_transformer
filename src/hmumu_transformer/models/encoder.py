from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from .attention import AttentionConfig, MultiheadSelfAttentionWithPairwiseBias

@dataclass(frozen=True)
class EncoderConfig:
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    pairwise_dim: int
    pairwise_hidden: int

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig, token_type_ids: torch.Tensor):
        super().__init__()
        att_cfg = AttentionConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            pairwise_dim=cfg.pairwise_dim,
            pairwise_hidden=cfg.pairwise_hidden,
        )
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiheadSelfAttentionWithPairwiseBias(att_cfg, token_type_ids=token_type_ids)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, hidden=4 * cfg.d_model, dropout=cfg.dropout)

    def forward(self, h: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        h = h + self.attn(self.ln1(h), v=v, m=m)
        h = h + self.ffn(self.ln2(h))
        return h

class TransformerEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, token_type_ids: torch.Tensor):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg, token_type_ids=token_type_ids) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)

    def forward(self, h: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, v=v, m=m)
        return self.final_ln(h)
