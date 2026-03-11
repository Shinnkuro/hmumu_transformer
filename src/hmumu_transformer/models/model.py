from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import EncoderConfig, TransformerEncoder
from .heads import HeadConfig, ClassifierHead, MassAdversaryHead

@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    pairwise_dim: int
    pairwise_hidden: int
    classifier_hidden: int
    classifier_dropout: float
    adversary_hidden: int
    adversary_dropout: float

class HmumuTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, x_dim: int, token_type_ids: torch.Tensor, n_mass_bins: int):
        super().__init__()
        self.x_dim = int(x_dim)
        self.token_type_ids = token_type_ids.clone().detach()  # [N]
        self.n_mass_bins = int(n_mass_bins)

        self.x_embed = nn.Linear(self.x_dim, cfg.d_model)
        self.type_embed = nn.Embedding(3, cfg.d_model)  # CLS, MUON, JET
        self.drop = nn.Dropout(cfg.dropout)

        enc_cfg = EncoderConfig(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            pairwise_dim=cfg.pairwise_dim,
            pairwise_hidden=cfg.pairwise_hidden,
        )
        self.encoder = TransformerEncoder(enc_cfg, token_type_ids=self.token_type_ids)

        self.classifier = ClassifierHead(cfg.d_model, HeadConfig(cfg.classifier_hidden, cfg.classifier_dropout), n_classes=3)
        self.adversary = MassAdversaryHead(cfg.d_model, HeadConfig(cfg.adversary_hidden, cfg.adversary_dropout), n_bins=self.n_mass_bins)

    def forward(self, x: torch.Tensor, v: torch.Tensor, m: torch.Tensor, lambda_grl: float) -> Dict[str, torch.Tensor]:
        # x: [B,N,F], v: [B,N,4], m: [B,N]
        B, N, F = x.shape
        type_ids = self.token_type_ids.to(x.device).view(1, N).expand(B, N)
        h = self.x_embed(x) + self.type_embed(type_ids)
        h = self.drop(h)
        h = self.encoder(h, v=v, m=m)

        h_cls = h[:, 0, :]  # [B,d]
        logits_cls = self.classifier(h_cls)
        logits_mass = self.adversary(h_cls, lambda_grl=lambda_grl)
        return {"logits_cls": logits_cls, "logits_mass": logits_mass}
