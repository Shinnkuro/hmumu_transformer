from __future__ import annotations

from dataclasses import dataclass

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
    """Shared interaction embedder producing per-head bias U for all particle blocks."""

    def __init__(self, pairwise_dim: int, hidden: int, n_heads: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pairwise_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_heads),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: [B, N, 4] -> e: [B, N, N, D] -> [B, H, N, N]
        e = build_pairwise_features(v)
        b = self.mlp(e)
        return b.permute(0, 3, 1, 2).contiguous()


class ParticleMultiheadAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.cfg = cfg
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor, m: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """h: [B, N, d_model], m: [B, N], attn_bias: [B, H, N, N]"""
        B, N, D = h.shape
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.cfg.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.cfg.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.cfg.n_heads, self.d_head).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.d_head ** 0.5)
        logits = logits + attn_bias

        key_invalid = (m == 0).to(torch.bool)
        logits = logits.masked_fill(key_invalid.view(B, 1, 1, N), float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out(out)
        out = self.dropout(out)

        query_valid = (m != 0).to(out.dtype).view(B, N, 1)
        return out * query_valid


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


class ParticleAttentionBlock(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = ParticleMultiheadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, hidden=4 * cfg.d_model, dropout=cfg.dropout)

    def forward(self, h: torch.Tensor, m: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.ln1(h), m=m, attn_bias=attn_bias)
        h = h + self.ffn(self.ln2(h))
        return h * (m != 0).to(h.dtype).unsqueeze(-1)


class ClassAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.pre_attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.pre_fc_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden=4 * d_model, dropout=dropout)

    def forward(self, h: torch.Tensor, cls: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """h: [B, N, d_model], cls: [B, 1, d_model], m: [B, N]"""
        z = torch.cat([cls, h], dim=1)
        z_norm = self.pre_attn_norm(z)
        cls_norm = z_norm[:, :1, :]

        key_padding_mask = torch.cat(
            [torch.zeros((m.size(0), 1), dtype=torch.bool, device=m.device), (m == 0).to(torch.bool)],
            dim=1,
        )
        attn_out, _ = self.attn(cls_norm, z_norm, z_norm, key_padding_mask=key_padding_mask)
        cls = cls + self.dropout(attn_out)
        cls = cls + self.ffn(self.pre_fc_norm(cls))
        return cls
