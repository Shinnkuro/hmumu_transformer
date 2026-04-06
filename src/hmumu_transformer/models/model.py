from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .encoder import EncoderConfig, ParticleTransformerEncoder
from .heads import HeadConfig, ClassifierHead, MassAdversaryHead


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_particle_layers: int
    n_class_layers: int
    n_heads: int
    dropout: float
    pairwise_dim: int
    pairwise_hidden: int
    classifier_hidden: int
    classifier_dropout: float
    adversary_hidden: int
    adversary_dropout: float


class ParticleFeatureEmbed(nn.Module):
    """ParT-like token embedding MLP."""

    def __init__(self, x_dim: int, d_model: int):
        super().__init__()
        hidden = 4 * d_model
        self.net = nn.Sequential(
            nn.LayerNorm(x_dim),
            nn.Linear(x_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HmumuTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, x_dim: int, token_type_ids: torch.Tensor, n_mass_bins: int):
        super().__init__()
        self.x_dim = int(x_dim)
        self.token_type_ids = token_type_ids.clone().detach()  # [N]
        self.n_mass_bins = int(n_mass_bins)

        self.x_embed = ParticleFeatureEmbed(self.x_dim, cfg.d_model)
        self.type_embed = nn.Embedding(3, cfg.d_model)  # MET, MUON, JET
        self.drop = nn.Dropout(cfg.dropout)

        enc_cfg = EncoderConfig(
            d_model=cfg.d_model,
            n_particle_layers=cfg.n_particle_layers,
            n_class_layers=cfg.n_class_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            pairwise_dim=cfg.pairwise_dim,
            pairwise_hidden=cfg.pairwise_hidden,
        )
        self.encoder = ParticleTransformerEncoder(enc_cfg)

        self.classifier = ClassifierHead(cfg.d_model, HeadConfig(cfg.classifier_hidden, cfg.classifier_dropout), n_classes=3)
        self.adversary = MassAdversaryHead(cfg.d_model, HeadConfig(cfg.adversary_hidden, cfg.adversary_dropout), n_bins=self.n_mass_bins)

    def forward(self, x: torch.Tensor, v: torch.Tensor, m: torch.Tensor, lambda_grl: float) -> Dict[str, torch.Tensor]:
        # x: [B,N,F], v: [B,N,4]=(pt, eta, phi, E), m: [B,N]
        B, N, _ = x.shape
        type_ids = self.token_type_ids.to(x.device).view(1, N).expand(B, N)
        h = self.x_embed(x) + self.type_embed(type_ids)
        h = self.drop(h)
        h = h * (m != 0).to(h.dtype).unsqueeze(-1)

        h_cls = self.encoder(h, v=v, m=m)
        logits_cls = self.classifier(h_cls)
        logits_mass = self.adversary(h_cls, lambda_grl=lambda_grl)
        return {"logits_cls": logits_cls, "logits_mass": logits_mass}
