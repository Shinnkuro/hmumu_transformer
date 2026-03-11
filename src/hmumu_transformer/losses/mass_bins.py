from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

@dataclass(frozen=True)
class MassBinning:
    edges: np.ndarray  # [K+1], inclusive endpoints
    K: int

    def to_dict(self) -> Dict:
        return {"edges": self.edges.tolist(), "K": int(self.K)}

    @staticmethod
    def from_dict(d: Dict) -> "MassBinning":
        edges = np.asarray(d["edges"], dtype=np.float32)
        K = int(d["K"])
        return MassBinning(edges=edges, K=K)

def choose_equal_frequency_bins(
    masses: np.ndarray,
    window: Tuple[float, float],
    candidate_K: Sequence[int],
    min_bin_count: int,
) -> MassBinning:
    """Choose K and edges for equal-frequency binning in a window.

    Picks the largest K in candidate_K such that every bin has >= min_bin_count events.
    masses are assumed already within window; window is still used as a safety clamp.
    """
    lo, hi = float(window[0]), float(window[1])
    m = masses.astype(np.float64)
    m = m[(m >= lo) & (m <= hi)]
    if m.size == 0:
        raise ValueError("No masses in the requested window for binning.")
    m_sorted = np.sort(m)

    best: MassBinning | None = None
    for K in sorted(set(int(k) for k in candidate_K)):
        if K < 2:
            continue
        # quantile edges
        qs = np.linspace(0.0, 1.0, K + 1)
        edges = np.quantile(m_sorted, qs)
        # Ensure exact window endpoints (avoid numerical drift)
        edges[0] = lo
        edges[-1] = hi
        # Count per bin
        counts = np.histogram(m_sorted, bins=edges)[0]
        if counts.min() >= int(min_bin_count):
            best = MassBinning(edges=edges.astype(np.float32), K=K)

    if best is None:
        # Fall back to smallest K and accept low-count bins.
        K = int(sorted(set(int(k) for k in candidate_K))[0])
        qs = np.linspace(0.0, 1.0, K + 1)
        edges = np.quantile(m_sorted, qs)
        edges[0] = lo
        edges[-1] = hi
        best = MassBinning(edges=edges.astype(np.float32), K=K)
    return best

def masses_to_bin_indices_torch(masses: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Assign masses to bin indices [0..K-1] using edges [K+1]."""
    # torch.bucketize returns index in [0..K] for right=False.
    # We want bins: [edges[i], edges[i+1]) except last includes hi.
    idx = torch.bucketize(masses, edges, right=False) - 1
    K = edges.numel() - 1
    idx = torch.clamp(idx, 0, K - 1)
    return idx.to(torch.long)
