from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], outpath: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_roc_curves(y_true: np.ndarray, proba: np.ndarray, labels: List[str], outpath: str) -> None:
    from sklearn.metrics import roc_curve, auc  # type: ignore
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for c, name in enumerate(labels):
        y_bin = (y_true == c).astype(np.int32)
        if y_bin.min() == y_bin.max():
            continue
        fpr, tpr, _ = roc_curve(y_bin, proba[:, c])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
