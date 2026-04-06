from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

_EPS = 1e-6


@dataclass(frozen=True)
class TokenConfig:
    n_tokens: int = 5
    max_jets: int = 2
    x_dim: int = 3


# Token type ids: 0=MET, 1=MUON, 2=JET
TYPE_IDS_5 = np.array([0, 1, 1, 2, 2], dtype=np.int64)


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _EPS, None))


def _energy_from_pt_eta_mass(pt: float, eta: float, mass: float) -> float:
    px = pt * np.cos(0.0)  # only magnitude matters here, not phi
    py = 0.0
    pz = pt * np.sinh(eta)
    p2 = px * px + py * py + pz * pz
    return float(np.sqrt(np.clip(mass * mass + p2, 0.0, None)))


def build_tokens_from_row(row: Dict[str, float], cfg: TokenConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build (x, v, m, dimuon_mass) for one event.

    Token order is fixed to [MET, mu1, mu2, jet1, jet2].

    x: [N, 3] = [log(pt), log(E), eta]
    v: [N, 4] = (pt, eta, phi, E)
    m: [N] with 1 valid, 0 padding
    """
    N = cfg.n_tokens
    F = cfg.x_dim
    x = np.zeros((N, F), dtype=np.float32)
    v = np.zeros((N, 4), dtype=np.float32)
    m = np.zeros((N,), dtype=np.int64)

    # MET token: always present. User requested eta=0 and E=0.
    met_pt = float(row["MET_pt"])
    met_phi = float(row["MET_phi"])
    met_eta = 0.0
    met_E = 0.0
    v[0, :] = (met_pt, met_eta, met_phi, met_E)
    m[0] = 1
    x[0, 0] = float(_safe_log(np.array([met_pt], dtype=np.float32))[0])
    x[0, 1] = float(_safe_log(np.array([met_E], dtype=np.float32))[0])
    x[0, 2] = float(met_eta)

    # Muon tokens: always present.
    for ti, prefix in [(1, "mu1"), (2, "mu2")]:
        pt = float(row[f"{prefix}_pt"])
        eta = float(row[f"{prefix}_eta"])
        phi = float(row[f"{prefix}_phi"])
        mass = float(row[f"{prefix}_mass"])
        energy = _energy_from_pt_eta_mass(pt, eta, mass)
        v[ti, :] = (pt, eta, phi, energy)
        m[ti] = 1
        x[ti, 0] = float(_safe_log(np.array([pt], dtype=np.float32))[0])
        x[ti, 1] = float(_safe_log(np.array([energy], dtype=np.float32))[0])
        x[ti, 2] = float(eta)

    # Jet tokens: keep slot, use padding + mask if missing.
    nj = int(row["njets_nominal"])
    for j in range(1, cfg.max_jets + 1):
        ti = 2 + j  # token index: jet1->3, jet2->4
        if nj < j:
            continue

        pt = row.get(f"jet{j}_pt_nominal", np.nan)
        eta = row.get(f"jet{j}_eta_nominal", np.nan)
        phi = row.get(f"jet{j}_phi_nominal", np.nan)
        mass = row.get(f"jet{j}_mass_nominal", np.nan)
        if not (np.isfinite(pt) and np.isfinite(eta) and np.isfinite(phi) and np.isfinite(mass)):
            continue

        pt = float(pt)
        eta = float(eta)
        phi = float(phi)
        mass = float(mass)
        energy = _energy_from_pt_eta_mass(pt, eta, mass)
        v[ti, :] = (pt, eta, phi, energy)
        m[ti] = 1
        x[ti, 0] = float(_safe_log(np.array([pt], dtype=np.float32))[0])
        x[ti, 1] = float(_safe_log(np.array([energy], dtype=np.float32))[0])
        x[ti, 2] = float(eta)

    dimuon_mass = float(row["dimuon_mass"])
    return x, v, m.astype(np.int64), dimuon_mass
