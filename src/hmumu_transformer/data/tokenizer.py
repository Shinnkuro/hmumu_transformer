from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

_EPS = 1e-6

@dataclass(frozen=True)
class TokenConfig:
    n_tokens: int = 7
    max_jets: int = 4
    x_dim: int = 16

# Token type ids: 0=CLS, 1=MUON, 2=JET
TYPE_IDS_7 = np.array([0, 1, 1, 2, 2, 2, 2], dtype=np.int64)

def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _EPS, None))

def build_tokens_from_row(row: Dict[str, float], cfg: TokenConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build (x, v, m, dimuon_mass) for one event.

    x: [N, F]
    v: [N, 4] with (pt, eta, phi, mass)
    m: [N] with 1 valid, 0 padding
    """
    N = cfg.n_tokens
    F = cfg.x_dim
    x = np.zeros((N, F), dtype=np.float32)
    v = np.zeros((N, 4), dtype=np.float32)
    m = np.zeros((N,), dtype=np.int64)

    # CLS token: mask always 1
    m[0] = 1
    met_pt = float(row["MET_pt"])
    met_phi = float(row["MET_phi"])
    njets = float(row["njets_nominal"])
    dimuon_pt_log = float(row["dimuon_pt_log"])
    # Global block at indices 11..15
    x[0, 11] = float(_safe_log(np.array([met_pt], dtype=np.float32))[0])
    x[0, 12] = float(np.sin(met_phi))
    x[0, 13] = float(np.cos(met_phi))
    x[0, 14] = float(njets)
    x[0, 15] = float(dimuon_pt_log)

    # Muons
    for mi, prefix in [(1, "mu1"), (2, "mu2")]:
        pt = float(row[f"{prefix}_pt"])
        eta = float(row[f"{prefix}_eta"])
        phi = float(row[f"{prefix}_phi"])
        mass = float(row[f"{prefix}_mass"])
        v[mi, :] = (pt, eta, phi, mass)
        m[mi] = 1

    # Jets (1..max_jets)
    nj = int(row["njets_nominal"])
    for j in range(1, cfg.max_jets + 1):
        ti = 2 + j  # token index: 3..6
        if nj >= j:
            pt = row.get(f"jet{j}_pt_nominal", np.nan)
            eta = row.get(f"jet{j}_eta_nominal", np.nan)
            phi = row.get(f"jet{j}_phi_nominal", np.nan)
            mass = row.get(f"jet{j}_mass_nominal", np.nan)
            qgl = row.get(f"jet{j}_qgl_nominal", np.nan)
            # If any of the core kinematics are not finite, treat as missing.
            if np.isfinite(pt) and np.isfinite(eta) and np.isfinite(phi) and np.isfinite(mass):
                v[ti, :] = (float(pt), float(eta), float(phi), float(mass))
                m[ti] = 1
                # Jet-specific block: f8 = qgl (if missing, set 0)
                x[ti, 8] = float(qgl) if np.isfinite(qgl) else 0.0
            else:
                # leave as padding
                pass
        else:
            # padding
            pass

    # Common block for valid non-CLS tokens (muons and valid jets)
    for i in range(1, N):
        if m[i] == 1:
            pt, eta, phi, mass = v[i, :]
            x[i, 0] = float(_safe_log(np.array([pt], dtype=np.float32))[0])
            x[i, 1] = float(eta)
            x[i, 2] = float(np.sin(phi))
            x[i, 3] = float(np.cos(phi))
            x[i, 4] = float(_safe_log(np.array([mass], dtype=np.float32))[0])

    # muon-specific feature: f5 = log(mu_mass)
    for i, prefix in [(1, "mu1"), (2, "mu2")]:
        mass = float(row[f"{prefix}_mass"])
        x[i, 5] = float(_safe_log(np.array([mass], dtype=np.float32))[0])

    dimuon_mass = float(row["dimuon_mass"])
    return x, v, m.astype(np.int64), dimuon_mass
