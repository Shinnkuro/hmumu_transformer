from __future__ import annotations

from typing import List

import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], outpath: str) -> None:
    cm = np.asarray(cm, dtype=np.int64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_pct, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    threshold = 0.5 * (cm_pct.max() if cm_pct.size else 0.0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm_pct[i, j] > threshold else "black"
            cell_text = f"{cm[i, j]}\n{cm_pct[i, j] * 100.0:.1f}%"
            ax.text(j, i, cell_text, ha="center", va="center", color=text_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized fraction")
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
