from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

_EPS = 1e-6
_PI = 3.141592653589793
_TWOPI = 2.0 * _PI

@dataclass(frozen=True)
class PairwiseConfig:
    # Expected pairwise features:
    # [deta, sin_dphi, cos_dphi, log_dR, log_pt_ratio, log_mij_jetjet]
    pairwise_dim: int = 6

def wrap_delta_phi(dphi: torch.Tensor) -> torch.Tensor:
    # wrap to (-pi, pi]
    return (dphi + _PI) % _TWOPI - _PI

def pt_eta_phi_m_to_cartesian(v: torch.Tensor) -> torch.Tensor:
    """Convert (pt, eta, phi, m) to (E, px, py, pz).
    v: [B,N,4]
    returns: [B,N,4]
    """
    pt = v[..., 0]
    eta = v[..., 1]
    phi = v[..., 2]
    m = v[..., 3]
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    p2 = px * px + py * py + pz * pz
    E = torch.sqrt(torch.clamp(m * m + p2, min=0.0))
    return torch.stack([E, px, py, pz], dim=-1)

def invariant_mass_from_cartesian(p4: torch.Tensor) -> torch.Tensor:
    """Compute m_ij for all pairs.
    p4: [B,N,4] with (E,px,py,pz)
    returns mij: [B,N,N]
    """
    E = p4[..., 0]
    px = p4[..., 1]
    py = p4[..., 2]
    pz = p4[..., 3]
    Ei = E.unsqueeze(2)
    Ej = E.unsqueeze(1)
    pxi = px.unsqueeze(2)
    pxj = px.unsqueeze(1)
    pyi = py.unsqueeze(2)
    pyj = py.unsqueeze(1)
    pzi = pz.unsqueeze(2)
    pzj = pz.unsqueeze(1)
    Et = Ei + Ej
    pxt = pxi + pxj
    pyt = pyi + pyj
    pzt = pzi + pzj
    m2 = Et * Et - (pxt * pxt + pyt * pyt + pzt * pzt)
    m2 = torch.clamp(m2, min=0.0)
    return torch.sqrt(m2)

def build_pairwise_features(v: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
    """Build pairwise features e_ij from v.

    v: [B,N,4] (pt, eta, phi, m)
    token_type_ids: [N] with 0=CLS, 1=MUON, 2=JET
    returns e: [B,N,N,D] with D=6
    """
    B, N, _ = v.shape
    pt = v[..., 0]
    eta = v[..., 1]
    phi = v[..., 2]

    deta = eta.unsqueeze(2) - eta.unsqueeze(1)  # [B,N,N]
    dphi = wrap_delta_phi(phi.unsqueeze(2) - phi.unsqueeze(1))
    sin_dphi = torch.sin(dphi)
    cos_dphi = torch.cos(dphi)
    dR = torch.sqrt(torch.clamp(deta * deta + dphi * dphi, min=0.0))
    log_dR = torch.log(torch.clamp(dR, min=_EPS))

    pt_ratio = torch.log(torch.clamp(pt.unsqueeze(2) + _EPS, min=_EPS)) - torch.log(torch.clamp(pt.unsqueeze(1) + _EPS, min=_EPS))

    # Invariant mass for jet-jet pairs only.
    p4 = pt_eta_phi_m_to_cartesian(v)
    mij = invariant_mass_from_cartesian(p4)  # [B,N,N]
    log_mij = torch.log(torch.clamp(mij, min=_EPS))

    # Gate: only keep log_mij for jet-jet pairs; else 0.
    tt = token_type_ids.to(v.device).view(1, N)  # [1,N]
    is_jet = (tt == 2)
    jetjet = (is_jet.unsqueeze(2) & is_jet.unsqueeze(1)).expand(B, N, N)
    log_mij = torch.where(jetjet, log_mij, torch.zeros_like(log_mij))

    e = torch.stack([deta, sin_dphi, cos_dphi, log_dR, pt_ratio, log_mij], dim=-1)  # [B,N,N,6]
    return e
