from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class MassSculptingResult:
    pearson_r: float

def dy_mass_sculpting(
    dy_masses: np.ndarray,
    dy_scores: np.ndarray,
    n_bins: int,
    outpath: str,
) -> MassSculptingResult:
    """Plot DY mμμ distributions in score bins, compute Pearson correlation.

    score = 1 - p(DY) by default in this project.
    """
    m = dy_masses.astype(np.float64)
    s = dy_scores.astype(np.float64)
    if m.size == 0:
        raise ValueError("No DY events provided for mass sculpting check.")

    # Pearson correlation
    if np.std(s) < 1e-12 or np.std(m) < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(s, m)[0, 1])

    # Equal-frequency bin edges in score
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(s, qs)
    # make edges strictly increasing (handle constant segments)
    edges[0] = s.min()
    edges[-1] = s.max()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            sel = (s >= lo) & (s < hi)
        else:
            sel = (s >= lo) & (s <= hi)
        ax.hist(m[sel], bins=40, histtype="step", label=f"bin {i}: [{lo:.3g},{hi:.3g}] (n={sel.sum()})")
    ax.set_xlabel("m_mumu [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(f"DY mass sculpting check (Pearson r = {r:.4f})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return MassSculptingResult(pearson_r=r)
