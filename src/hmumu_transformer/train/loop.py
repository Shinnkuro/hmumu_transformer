from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..eval.metrics import accuracy, one_vs_rest_auc
from ..losses.mass_bins import masses_to_bin_indices_torch
from ..train.early_stopping import EarlyStopping


@dataclass
class TrainState:
    epoch: int
    global_step: int


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=False) for k, v in batch.items()}


@torch.no_grad()
def run_epoch_eval(
    model,
    loader,
    device: torch.device,
    mass_edges: torch.Tensor,
    lambda_grl: float,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    model.eval()
    losses = []
    y_true = []
    proba = []
    masses = []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch["x"], batch["v"], batch["m"], lambda_grl=lambda_grl)
        logits = out["logits_cls"]
        loss = F.cross_entropy(logits, batch["y"], weight=class_weights)
        losses.append(float(loss.item()))
        p = torch.softmax(logits, dim=-1)
        proba.append(p.cpu().numpy())
        y_true.append(batch["y"].cpu().numpy())
        masses.append(batch["mass"].cpu().numpy())
    if not losses:
        return {"loss": float("nan")}
    y_true_np = np.concatenate(y_true, axis=0)
    proba_np = np.concatenate(proba, axis=0)
    y_pred = proba_np.argmax(axis=1)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy(y_true_np, y_pred),
        "auc_ovr": one_vs_rest_auc(y_true_np, proba_np, n_classes=3),
        "y_true": y_true_np,
        "proba": proba_np,
        "masses": np.concatenate(masses, axis=0),
    }


def train(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    optimizer,
    num_epochs: int,
    grad_clip_norm: float,
    mass_edges: np.ndarray,
    lambda_max: float,
    warmup_epochs: int,
    run_dir: str,
    early_stopping_cfg: Dict[str, Any],
    log_every_steps: int,
    save_best_only: bool,
    class_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)
    mass_edges_t = torch.tensor(mass_edges, dtype=torch.float32, device=device)
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    class_weights_t = None
    if class_weights is not None:
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    early = None
    if early_stopping_cfg.get("enabled", True):
        early = EarlyStopping(
            patience=int(early_stopping_cfg.get("patience", 8)),
            min_delta=float(early_stopping_cfg.get("min_delta", 0.0)),
        )

    state = TrainState(epoch=0, global_step=0)
    history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        model.train()
        if warmup_epochs <= 0:
            lambda_grl = float(lambda_max)
        else:
            lambda_grl = float(lambda_max) * min(1.0, (epoch + 1) / float(warmup_epochs))

        epoch_losses = []
        epoch_cls_losses = []
        epoch_adv_losses = []

        for batch in train_loader:
            batch = _to_device(batch, device)
            out = model(batch["x"], batch["v"], batch["m"], lambda_grl=lambda_grl)
            logits_cls = out["logits_cls"]
            logits_mass = out["logits_mass"]

            loss_cls = F.cross_entropy(logits_cls, batch["y"], weight=class_weights_t)
            mass_bins = masses_to_bin_indices_torch(batch["mass"], mass_edges_t)
            loss_adv = F.cross_entropy(logits_mass, mass_bins)
            loss = loss_cls + loss_adv

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            epoch_cls_losses.append(float(loss_cls.item()))
            epoch_adv_losses.append(float(loss_adv.item()))
            state.global_step += 1
            if log_every_steps and (state.global_step % int(log_every_steps) == 0):
                pass

        train_summary = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "loss_cls": float(np.mean(epoch_cls_losses)) if epoch_cls_losses else float("nan"),
            "loss_adv": float(np.mean(epoch_adv_losses)) if epoch_adv_losses else float("nan"),
            "lambda_grl": lambda_grl,
            "steps": len(epoch_losses),
        }
        history["train"].append(train_summary)

        val_summary = run_epoch_eval(
            model,
            val_loader,
            device,
            mass_edges_t,
            lambda_grl=0.0,
            class_weights=class_weights_t,
        )
        val_summary = {k: v for k, v in val_summary.items() if k not in ("y_true", "proba", "masses")}
        val_summary["epoch"] = epoch
        history["val"].append(val_summary)

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "lambda_grl": lambda_grl,
            "class_weights": class_weights.tolist() if class_weights is not None else None,
        }
        torch.save(payload, last_path)

        metric = float(val_summary.get("loss", float("inf")))
        if early is None:
            previous_losses = [h["loss"] for h in history["val"][:-1] if "loss" in h]
            is_best = (not previous_losses) or metric <= min(previous_losses)
        else:
            is_best = early.update(metric)

        if (not save_best_only) or is_best:
            torch.save(payload, best_path)

        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if early is not None and early.should_stop:
            break

    return history
