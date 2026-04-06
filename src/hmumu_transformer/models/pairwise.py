from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1e-6
_PI = 3.141592653589793
_TWOPI = 2.0 * _PI


@dataclass(frozen=True)
class PairwiseConfig:
    # [ln Delta, ln kT, ln z, ln m^2]
    pairwise_dim: int = 4


def wrap_delta_phi(dphi: torch.Tensor) -> torch.Tensor:
    # wrap to (-pi, pi]
    return (dphi + _PI) % _TWOPI - _PI


def pt_eta_phi_E_to_cartesian(v: torch.Tensor) -> torch.Tensor:
    """Convert (pt, eta, phi, E) to (E, px, py, pz).

    v: [B, N, 4]
    returns: [B, N, 4]
    """
    pt = v[..., 0]
    eta = v[..., 1]
    phi = v[..., 2]
    energy = v[..., 3]
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return torch.stack([energy, px, py, pz], dim=-1)


def build_pairwise_features(v: torch.Tensor) -> torch.Tensor:
    """Build ParT-like pairwise features using eta instead of rapidity.

    v: [B, N, 4] with (pt, eta, phi, E)
    returns e: [B, N, N, 4] with
      [ln Delta, ln kT, ln z, ln m^2]

    Notes:
    - Delta uses pseudorapidity eta, per user request.
    - Self-pairs are kept.
    - Each quantity is clamped before log.
    """
    pt = v[..., 0]
    eta = v[..., 1]
    phi = v[..., 2]
    energy = v[..., 3]

    deta = eta.unsqueeze(2) - eta.unsqueeze(1)  # [B,N,N]
    dphi = wrap_delta_phi(phi.unsqueeze(2) - phi.unsqueeze(1))
    delta = torch.sqrt(torch.clamp(deta * deta + dphi * dphi, min=0.0))

    pti = pt.unsqueeze(2)
    ptj = pt.unsqueeze(1)
    ptmin = torch.minimum(pti, ptj)
    kt = ptmin * delta
    z = ptmin / torch.clamp(pti + ptj, min=_EPS)

    p4 = pt_eta_phi_E_to_cartesian(v)
    E = p4[..., 0]
    px = p4[..., 1]
    py = p4[..., 2]
    pz = p4[..., 3]

    Et = E.unsqueeze(2) + E.unsqueeze(1)
    pxt = px.unsqueeze(2) + px.unsqueeze(1)
    pyt = py.unsqueeze(2) + py.unsqueeze(1)
    pzt = pz.unsqueeze(2) + pz.unsqueeze(1)
    m2 = Et * Et - (pxt * pxt + pyt * pyt + pzt * pzt)

    ln_delta = torch.log(torch.clamp(delta, min=_EPS))
    ln_kt = torch.log(torch.clamp(kt, min=_EPS))
    ln_z = torch.log(torch.clamp(z, min=_EPS))
    ln_m2 = torch.log(torch.clamp(m2, min=_EPS))

    return torch.stack([ln_delta, ln_kt, ln_z, ln_m2], dim=-1)
