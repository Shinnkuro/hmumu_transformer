from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .reader import read_parquet_files
from .split import SplitSpec, split_indices
from .tokenizer import TokenConfig
from .dataset import HmumuTokenDataset, LABELS
from .scaler import fit_masked_standard_scaler, MaskedStandardScaler
from .balanced_sampler import BalancedBatchSampler, BalancedBatchConfig

@dataclass
class BuiltData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    x_scaler: MaskedStandardScaler
    mass_train: np.ndarray  # raw masses (train) for binning
    label_names: List[str]

def _filter_mass_window(arrays: Dict[str, np.ndarray], window: Tuple[float, float]) -> Dict[str, np.ndarray]:
    lo, hi = float(window[0]), float(window[1])
    m = arrays["dimuon_mass"].astype(np.float32)
    keep = (m >= lo) & (m <= hi)
    return {k: v[keep] for k, v in arrays.items()}

def _make_dataset(arrays: Dict[str, np.ndarray], label: str, token_cfg: TokenConfig) -> HmumuTokenDataset:
    return HmumuTokenDataset(arrays=arrays, label=LABELS[label], token_cfg=token_cfg)

def _collate(batch):
    x = torch.stack([b.x for b in batch], dim=0)
    v = torch.stack([b.v for b in batch], dim=0)
    m = torch.stack([b.m for b in batch], dim=0)
    y = torch.stack([b.y for b in batch], dim=0)
    mass = torch.stack([b.mass for b in batch], dim=0)
    return {"x": x, "v": v, "m": m, "y": y, "mass": mass}

def build_dataloaders(
    cfg: Dict[str, Any],
    *,
    x_scaler_override: Optional[MaskedStandardScaler] = None,
) -> BuiltData:
    data_cfg = cfg["data"]
    cols = data_cfg["columns"]
    window = tuple(data_cfg["dimuon_mass_window"])
    split_cfg = data_cfg["split"]
    spec = SplitSpec(
        n_folds=int(split_cfg["n_folds"]),
        train_folds=tuple(split_cfg["train_folds"]),
        val_folds=tuple(split_cfg["val_folds"]),
        test_folds=tuple(split_cfg["test_folds"]),
    )

    token_cfg = TokenConfig(
        n_tokens=int(cfg["schema"]["n_tokens"]),
        max_jets=int(cfg["schema"]["max_jets"]),
        x_dim=int(cfg["schema"]["x_dim"]),
    )

    # Read and filter per class
    ggH = _filter_mass_window(read_parquet_files(data_cfg["ggH_files"], cols), window)
    VBF = _filter_mass_window(read_parquet_files(data_cfg["VBF_files"], cols), window)
    DY  = _filter_mass_window(read_parquet_files(data_cfg["DY_files"], cols), window)

    # Split indices per class
    idx_ggH = split_indices(len(next(iter(ggH.values()))), spec)
    idx_VBF = split_indices(len(next(iter(VBF.values()))), spec)
    idx_DY  = split_indices(len(next(iter(DY.values()))), spec)

    def subset(arrs, idx):
        return {k: v[idx] for k, v in arrs.items()}

    datasets: Dict[str, ConcatDataset] = {}
    for split in ["train", "val", "test"]:
        ds_list = [
            _make_dataset(subset(ggH, idx_ggH[split]), "ggH", token_cfg),
            _make_dataset(subset(VBF, idx_VBF[split]), "VBF", token_cfg),
            _make_dataset(subset(DY,  idx_DY[split]),  "DY",  token_cfg),
        ]
        datasets[split] = ConcatDataset(ds_list)

    # Fit or use provided scaler
    if x_scaler_override is None:
        # Fit scaler on training set only using a moderate batch to avoid high peak memory.
        train_full_loader = DataLoader(
            datasets["train"],
            batch_size=2048,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )
        xs, ms, masses = [], [], []
        for batch in train_full_loader:
            xs.append(batch["x"].numpy())
            ms.append(batch["m"].numpy())
            masses.append(batch["mass"].numpy())
        x_train = np.concatenate(xs, axis=0)
        m_train = np.concatenate(ms, axis=0)
        mass_train = np.concatenate(masses, axis=0).astype(np.float32)
        x_scaler = fit_masked_standard_scaler(x_train, m_train)
    else:
        x_scaler = x_scaler_override
        # still need train masses for binning
        train_full_loader = DataLoader(
            datasets["train"],
            batch_size=2048,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )
        masses = []
        for batch in train_full_loader:
            masses.append(batch["mass"].numpy())
        mass_train = np.concatenate(masses, axis=0).astype(np.float32)

    def collate_scaled(batch):
        out = _collate(batch)
        x_np = out["x"].numpy()
        x_np = x_scaler.transform(x_np)
        out["x"] = torch.from_numpy(x_np)
        return out

    train_cfg = cfg["train"]
    batch_size = train_cfg.get("batch_size", None)

    if batch_size is None:
        # One large balanced batch per epoch.
        n_ggH = len(datasets["train"].datasets[0])
        n_VBF = len(datasets["train"].datasets[1])
        n_DY  = len(datasets["train"].datasets[2])
        min_n = min(n_ggH, n_VBF, n_DY)
        per_class_batch = train_cfg.get("per_class_batch", None)
        per_class_batch = int(per_class_batch) if per_class_batch is not None else int(min_n)

        offsets = np.cumsum([0, n_ggH, n_VBF, n_DY]).astype(np.int64)
        pools = [
            np.arange(0, n_ggH, dtype=np.int64) + offsets[0],
            np.arange(0, n_VBF, dtype=np.int64) + offsets[1],
            np.arange(0, n_DY,  dtype=np.int64) + offsets[2],
        ]
        sampler = BalancedBatchSampler(
            pools,
            BalancedBatchConfig(per_class_batch=per_class_batch, steps_per_epoch=1, seed=None),
        )
        train_loader = DataLoader(
            datasets["train"],
            batch_sampler=sampler,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=collate_scaled,
        )
    else:
        # Standard mini-batching with balanced batches.
        batch_size = int(batch_size)
        per_class_batch = batch_size // 3
        if per_class_batch < 1:
            raise ValueError("batch_size must be >= 3 for balanced sampling.")

        n_ggH = len(datasets["train"].datasets[0])
        n_VBF = len(datasets["train"].datasets[1])
        n_DY  = len(datasets["train"].datasets[2])
        offsets = np.cumsum([0, n_ggH, n_VBF, n_DY]).astype(np.int64)
        pools = [
            np.arange(0, n_ggH, dtype=np.int64) + offsets[0],
            np.arange(0, n_VBF, dtype=np.int64) + offsets[1],
            np.arange(0, n_DY,  dtype=np.int64) + offsets[2],
        ]
        steps = int(math.ceil((3 * max(n_ggH, n_VBF, n_DY)) / batch_size))
        sampler = BalancedBatchSampler(
            pools,
            BalancedBatchConfig(per_class_batch=per_class_batch, steps_per_epoch=steps, seed=None),
        )
        train_loader = DataLoader(
            datasets["train"],
            batch_sampler=sampler,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=collate_scaled,
        )

    val_loader = DataLoader(datasets["val"], batch_size=2048, shuffle=False, num_workers=0, collate_fn=collate_scaled)
    test_loader = DataLoader(datasets["test"], batch_size=2048, shuffle=False, num_workers=0, collate_fn=collate_scaled)

    return BuiltData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        x_scaler=x_scaler,
        mass_train=mass_train,
        label_names=["ggH", "VBF", "DY"],
    )
