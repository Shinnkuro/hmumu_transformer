from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    return float((y_true == y_pred).mean()) if y_true.size else 0.0

def one_vs_rest_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> Dict[str, float]:
    """Compute one-vs-rest AUC for each class using sklearn."""
    from sklearn.metrics import roc_auc_score  # type: ignore
    out: Dict[str, float] = {}
    for c in range(n_classes):
        y_bin = (y_true == c).astype(np.int32)
        # If a class is missing in y_true, roc_auc_score will fail.
        if y_bin.min() == y_bin.max():
            out[str(c)] = float("nan")
        else:
            out[str(c)] = float(roc_auc_score(y_bin, proba[:, c]))
    return out
