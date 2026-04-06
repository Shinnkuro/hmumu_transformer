from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention import AttentionConfig, PairwiseBias, ParticleAttentionBlock, ClassAttentionBlock


@dataclass(frozen=True)
class EncoderConfig:
    d_model: int
    n_particle_layers: int
    n_class_layers: int
    n_heads: int
    dropout: float
    pairwise_dim: int
    pairwise_hidden: int


class ParticleTransformerEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        att_cfg = AttentionConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            pairwise_dim=cfg.pairwise_dim,
            pairwise_hidden=cfg.pairwise_hidden,
        )
        self.pair_bias = PairwiseBias(cfg.pairwise_dim, cfg.pairwise_hidden, cfg.n_heads)
        self.particle_blocks = nn.ModuleList([ParticleAttentionBlock(att_cfg) for _ in range(cfg.n_particle_layers)])
        self.class_blocks = nn.ModuleList([ClassAttentionBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_class_layers)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model), requires_grad=True)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, h: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        attn_bias = self.pair_bias(v)
        for block in self.particle_blocks:
            h = block(h, m=m, attn_bias=attn_bias)

        cls = self.cls_token.expand(h.size(0), -1, -1)
        for block in self.class_blocks:
            cls = block(h, cls, m=m)
        cls = self.final_norm(cls)
        return cls.squeeze(1)
