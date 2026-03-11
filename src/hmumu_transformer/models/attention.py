from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pairwise import build_pairwise_features

@dataclass(frozen=True)
class AttentionConfig:
    d_model: int
    n_heads: int
    dropout: float
    pairwise_dim: int
    pairwise_hidden: int

class PairwiseBias(nn.Module):
    """Shared MLP producing per-head bias for attention logits."""
    def __init__(self, pairwise_dim: int, hidden: int, n_heads: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pairwise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_heads),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        # e: [B,N,N,D] -> [B,N,N,H] -> [B,H,N,N]
        b = self.mlp(e)
        return b.permute(0, 3, 1, 2).contiguous()

class MultiheadSelfAttentionWithPairwiseBias(nn.Module):
    def __init__(self, cfg: AttentionConfig, token_type_ids: torch.Tensor):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.cfg = cfg
        self.d_head = cfg.d_model // cfg.n_heads
        self.token_type_ids = token_type_ids.clone().detach()

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        self.pair_bias = PairwiseBias(cfg.pairwise_dim, cfg.pairwise_hidden, cfg.n_heads)

    def forward(self, h: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """h: [B,N,d_model], v: [B,N,4], m: [B,N] (1 valid, 0 pad)"""
        B, N, D = h.shape
        qkv = self.qkv(h)  # [B,N,3D]
        q, k, vv = qkv.chunk(3, dim=-1)
        # [B,H,N,d_head]
        q = q.view(B, N, self.cfg.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.view(B, N, self.cfg.n_heads, self.d_head).permute(0, 2, 1, 3)
        vv = vv.view(B, N, self.cfg.n_heads, self.d_head).permute(0, 2, 1, 3)

        # logits: [B,H,N,N]
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.d_head ** 0.5)

        # pairwise bias
        e = build_pairwise_features(v, self.token_type_ids)  # [B,N,N,Dp]
        bias = self.pair_bias(e)  # [B,H,N,N]
        logits = logits + bias

        # key padding mask: mask invalid keys j
        key_invalid = (m == 0).to(torch.bool)  # [B,N]
        # expand to [B,1,1,N]
        logits = logits.masked_fill(key_invalid.view(B, 1, 1, N), float("-inf"))

        attn = F.softmax(logits, dim=-1)  # [B,H,N,N]
        attn = self.dropout(attn)

        out = torch.matmul(attn, vv)  # [B,H,N,d_head]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out(out)
        out = self.dropout(out)

        # zero-out outputs for invalid query tokens to avoid propagating NaNs if logits were -inf everywhere
        query_valid = (m != 0).to(out.dtype).view(B, N, 1)
        out = out * query_valid
        return out
