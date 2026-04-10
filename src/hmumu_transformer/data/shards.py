from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .reader import iter_parquet_batches
from .scaler import MaskedStandardScaler, MaskedStandardScalerAccumulator
from .split import SplitSpec
from .tokenizer import TokenConfig, build_tokens_from_row
from ..utils.paths import expand_path_patterns


LABELS = {"ggH": 0, "VBF": 1, "DY": 2}
LABEL_NAMES = ["ggH", "VBF", "DY"]
_SPLITS = ("train", "val", "test")
_DEFAULT_SEED = 1337


@dataclass(frozen=True)
class ShardSpec:
    root_dir: str
    rows_per_shard: int
    record_batch_size: int
    rebuild: bool = False
    seed: int = _DEFAULT_SEED


@dataclass(frozen=True)
class PreparedShards:
    metadata_path: str
    metadata: Dict[str, Any]


@dataclass
class BuiltShardData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    x_scaler: MaskedStandardScaler
    mass_train: np.ndarray
    label_names: List[str]
    class_weights: np.ndarray


class _ShardBuffer:
    def __init__(self, out_dir: Path, label_name: str, rows_per_shard: int):
        self.out_dir = out_dir
        self.label_name = label_name
        self.rows_per_shard = int(rows_per_shard)
        self.parts: Dict[str, List[np.ndarray]] = {k: [] for k in ("x", "v", "m", "y", "mass")}
        self.size = 0
        self.shard_index = 0
        self.paths: List[str] = []

    def add(self, x: np.ndarray, v: np.ndarray, m: np.ndarray, y: np.ndarray, mass: np.ndarray) -> None:
        if x.shape[0] == 0:
            return
        self.parts["x"].append(x.astype(np.float32, copy=False))
        self.parts["v"].append(v.astype(np.float32, copy=False))
        self.parts["m"].append(m.astype(np.int64, copy=False))
        self.parts["y"].append(y.astype(np.int64, copy=False))
        self.parts["mass"].append(mass.astype(np.float32, copy=False))
        self.size += int(x.shape[0])
        while self.size >= self.rows_per_shard:
            self._flush_exact(self.rows_per_shard)

    def finalize(self) -> List[str]:
        if self.size > 0:
            self._flush_exact(self.size)
        return list(self.paths)

    def _stack(self) -> Dict[str, np.ndarray]:
        return {name: np.concatenate(chunks, axis=0) for name, chunks in self.parts.items()}

    def _reset(self) -> None:
        self.parts = {k: [] for k in self.parts}
        self.size = 0

    def _flush_exact(self, take: int) -> None:
        stacked = self._stack()
        shard = {name: values[:take] for name, values in stacked.items()}
        leftovers = {name: values[take:] for name, values in stacked.items()}
        path = self.out_dir / f"{self.label_name}_{self.shard_index:05d}.npz"
        np.savez(path, **shard)
        self.paths.append(str(path))
        self.shard_index += 1
        self.parts = {
            name: [values] for name, values in leftovers.items() if values.shape[0] > 0
        }
        self.size = int(next(iter(leftovers.values())).shape[0]) if leftovers else 0
        for name in ("x", "v", "m", "y", "mass"):
            self.parts.setdefault(name, [])


class TokenShardBatchDataset(IterableDataset):
    """Stream already-tokenized shard files and emit pre-collated batches."""

    def __init__(
        self,
        shard_paths: Sequence[str],
        *,
        batch_size: int,
        scaler: MaskedStandardScaler,
        shuffle_shards: bool,
        shuffle_within_shard: bool,
        drop_last: bool,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        self.shard_paths = list(shard_paths)
        self.batch_size = int(batch_size)
        self.scaler = scaler
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_shard = bool(shuffle_within_shard)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()
        if worker is None:
            shard_paths = list(self.shard_paths)
            worker_id = 0
        else:
            shard_paths = list(self.shard_paths[worker.id :: worker.num_workers])
            worker_id = worker.id

        rng = np.random.default_rng(self.seed + worker_id)
        if self.shuffle_shards:
            rng.shuffle(shard_paths)

        for shard_path in shard_paths:
            with np.load(shard_path) as data:
                x = data["x"].astype(np.float32, copy=False)
                v = data["v"].astype(np.float32, copy=False)
                m = data["m"].astype(np.int64, copy=False)
                y = data["y"].astype(np.int64, copy=False)
                mass = data["mass"].astype(np.float32, copy=False)

            indices = np.arange(y.shape[0], dtype=np.int64)
            if self.shuffle_within_shard:
                rng.shuffle(indices)

            n_full = y.shape[0] // self.batch_size
            n_take = n_full * self.batch_size
            if not self.drop_last and n_take < y.shape[0]:
                n_take = y.shape[0]

            for start in range(0, n_take, self.batch_size):
                sl = indices[start : start + self.batch_size]
                if sl.shape[0] < self.batch_size and self.drop_last:
                    continue
                batch_x = torch.from_numpy(x[sl])
                batch = {
                    "x": self.scaler.transform_torch(batch_x),
                    "v": torch.from_numpy(v[sl]),
                    "m": torch.from_numpy(m[sl]),
                    "y": torch.from_numpy(y[sl]),
                    "mass": torch.from_numpy(mass[sl]),
                }
                yield batch


def _make_data_loader(dataset: IterableDataset, *, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=None, num_workers=int(num_workers), pin_memory=False)


def _resolve_split(row_ids: np.ndarray, spec: SplitSpec) -> Dict[str, np.ndarray]:
    folds = np.mod(row_ids, spec.n_folds)
    return {
        "train": np.isin(folds, spec.train_folds),
        "val": np.isin(folds, spec.val_folds),
        "test": np.isin(folds, spec.test_folds),
    }


def _tokenize_rows(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    token_cfg: TokenConfig,
    label_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(indices.shape[0])
    x = np.zeros((n_rows, token_cfg.n_tokens, token_cfg.x_dim), dtype=np.float32)
    v = np.zeros((n_rows, token_cfg.n_tokens, 4), dtype=np.float32)
    m = np.zeros((n_rows, token_cfg.n_tokens), dtype=np.int64)
    mass = np.zeros((n_rows,), dtype=np.float32)
    y = np.full((n_rows,), int(label_id), dtype=np.int64)

    columns = list(arrays.keys())
    for out_idx, src_idx in enumerate(indices.tolist()):
        row = {name: arrays[name][src_idx] for name in columns}
        x_row, v_row, m_row, mass_row = build_tokens_from_row(row, token_cfg)
        x[out_idx] = x_row
        v[out_idx] = v_row
        m[out_idx] = m_row
        mass[out_idx] = mass_row
    return x, v, m, y, mass


def _metadata_matches(metadata: Mapping[str, Any], cfg: Mapping[str, Any], shard_spec: ShardSpec) -> bool:
    data_cfg = cfg["data"]
    split_cfg = data_cfg["split"]
    expected = {
        "ggH_files": expand_path_patterns(data_cfg["ggH_files"], strict=True, description="ggH files"),
        "VBF_files": expand_path_patterns(data_cfg["VBF_files"], strict=True, description="VBF files"),
        "DY_files": expand_path_patterns(data_cfg["DY_files"], strict=True, description="DY files"),
        "columns": list(data_cfg["columns"]),
        "dimuon_mass_window": list(data_cfg["dimuon_mass_window"]),
        "split": {
            "n_folds": int(split_cfg["n_folds"]),
            "train_folds": list(split_cfg["train_folds"]),
            "val_folds": list(split_cfg["val_folds"]),
            "test_folds": list(split_cfg["test_folds"]),
        },
        "schema": {
            "n_tokens": int(cfg["schema"]["n_tokens"]),
            "max_jets": int(cfg["schema"]["max_jets"]),
            "x_dim": int(cfg["schema"]["x_dim"]),
        },
        "rows_per_shard": int(shard_spec.rows_per_shard),
        "record_batch_size": int(shard_spec.record_batch_size),
    }
    return metadata.get("build_config") == expected


def _build_config_snapshot(cfg: Mapping[str, Any], shard_spec: ShardSpec) -> Dict[str, Any]:
    data_cfg = cfg["data"]
    split_cfg = data_cfg["split"]
    return {
        "ggH_files": expand_path_patterns(data_cfg["ggH_files"], strict=True, description="ggH files"),
        "VBF_files": expand_path_patterns(data_cfg["VBF_files"], strict=True, description="VBF files"),
        "DY_files": expand_path_patterns(data_cfg["DY_files"], strict=True, description="DY files"),
        "columns": list(data_cfg["columns"]),
        "dimuon_mass_window": list(data_cfg["dimuon_mass_window"]),
        "split": {
            "n_folds": int(split_cfg["n_folds"]),
            "train_folds": list(split_cfg["train_folds"]),
            "val_folds": list(split_cfg["val_folds"]),
            "test_folds": list(split_cfg["test_folds"]),
        },
        "schema": {
            "n_tokens": int(cfg["schema"]["n_tokens"]),
            "max_jets": int(cfg["schema"]["max_jets"]),
            "x_dim": int(cfg["schema"]["x_dim"]),
        },
        "rows_per_shard": int(shard_spec.rows_per_shard),
        "record_batch_size": int(shard_spec.record_batch_size),
    }


def get_shard_spec(cfg: Mapping[str, Any]) -> ShardSpec:
    shard_cfg = cfg["data"].get("shards", {})
    return ShardSpec(
        root_dir=str(shard_cfg.get("root_dir", "processed/default")),
        rows_per_shard=int(shard_cfg.get("rows_per_shard", 50_000)),
        record_batch_size=int(shard_cfg.get("record_batch_size", 65_536)),
        rebuild=bool(shard_cfg.get("rebuild", False)),
        seed=int(shard_cfg.get("seed", _DEFAULT_SEED)),
    )


def prepare_token_shards(cfg: Mapping[str, Any], *, force_rebuild: Optional[bool] = None) -> PreparedShards:
    shard_spec = get_shard_spec(cfg)
    root_dir = Path(shard_spec.root_dir)
    metadata_path = root_dir / "metadata.json"
    should_rebuild = shard_spec.rebuild if force_rebuild is None else bool(force_rebuild)

    if metadata_path.exists() and not should_rebuild:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if _metadata_matches(metadata, cfg, shard_spec):
            return PreparedShards(metadata_path=str(metadata_path), metadata=metadata)

    if root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    for split in _SPLITS:
        (root_dir / split).mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    spec = SplitSpec(
        n_folds=int(data_cfg["split"]["n_folds"]),
        train_folds=tuple(data_cfg["split"]["train_folds"]),
        val_folds=tuple(data_cfg["split"]["val_folds"]),
        test_folds=tuple(data_cfg["split"]["test_folds"]),
    )
    token_cfg = TokenConfig(
        n_tokens=int(cfg["schema"]["n_tokens"]),
        max_jets=int(cfg["schema"]["max_jets"]),
        x_dim=int(cfg["schema"]["x_dim"]),
    )
    rows_per_shard = int(shard_spec.rows_per_shard)
    batch_size = int(shard_spec.record_batch_size)
    lo, hi = map(float, data_cfg["dimuon_mass_window"])

    scaler_accumulator = MaskedStandardScalerAccumulator(feature_dim=token_cfg.x_dim)
    class_counts = {split: {name: 0 for name in LABEL_NAMES} for split in _SPLITS}
    shard_paths = {split: [] for split in _SPLITS}
    train_masses: List[np.ndarray] = []

    for label_name, file_key in (("ggH", "ggH_files"), ("VBF", "VBF_files"), ("DY", "DY_files")):
        label_id = LABELS[label_name]
        writers = {
            split: _ShardBuffer(root_dir / split, label_name=label_name, rows_per_shard=rows_per_shard)
            for split in _SPLITS
        }
        filtered_seen = 0
        for arrays in iter_parquet_batches(data_cfg[file_key], data_cfg["columns"], batch_size=batch_size):
            masses = arrays["dimuon_mass"].astype(np.float32, copy=False)
            keep = (masses >= lo) & (masses <= hi)
            if not np.any(keep):
                continue

            filtered = {name: values[keep] for name, values in arrays.items()}
            batch_n = int(filtered["dimuon_mass"].shape[0])
            row_ids = np.arange(filtered_seen, filtered_seen + batch_n, dtype=np.int64)
            filtered_seen += batch_n
            split_masks = _resolve_split(row_ids, spec)

            for split, split_mask in split_masks.items():
                if not np.any(split_mask):
                    continue
                split_indices = np.flatnonzero(split_mask)
                x, v, m, y, mass = _tokenize_rows(
                    filtered,
                    split_indices,
                    token_cfg=token_cfg,
                    label_id=label_id,
                )
                writers[split].add(x, v, m, y, mass)
                class_counts[split][label_name] += int(y.shape[0])
                if split == "train":
                    scaler_accumulator.update(x, m)
                    train_masses.append(mass)

        for split in _SPLITS:
            shard_paths[split].extend(writers[split].finalize())

    scaler = scaler_accumulator.finalize()
    train_counts = np.asarray([class_counts["train"][name] for name in LABEL_NAMES], dtype=np.float64)
    total_train = float(train_counts.sum())
    if np.any(train_counts <= 0):
        raise RuntimeError(f"Training split contains an empty class: {class_counts['train']}")
    class_weights = (total_train / (len(LABEL_NAMES) * train_counts)).astype(np.float32)
    train_masses_arr = np.concatenate(train_masses, axis=0).astype(np.float32) if train_masses else np.empty((0,), dtype=np.float32)
    np.save(root_dir / "train_masses.npy", train_masses_arr)

    metadata = {
        "version": 3,
        "build_config": _build_config_snapshot(cfg, shard_spec),
        "scaler": scaler.to_dict(),
        "class_weights": class_weights.tolist(),
        "class_counts": class_counts,
        "splits": {
            split: {
                "num_shards": len(shard_paths[split]),
                "paths": [os.path.relpath(path, root_dir) for path in shard_paths[split]],
                "num_events": int(sum(class_counts[split].values())),
            }
            for split in _SPLITS
        },
        "train_masses": os.path.relpath(root_dir / "train_masses.npy", root_dir),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return PreparedShards(metadata_path=str(metadata_path), metadata=metadata)


def build_dataloaders_from_shards(
    cfg: Mapping[str, Any],
    *,
    x_scaler_override: Optional[MaskedStandardScaler] = None,
) -> BuiltShardData:
    prepared = prepare_token_shards(cfg)
    metadata = prepared.metadata
    root_dir = Path(prepared.metadata_path).parent
    scaler = x_scaler_override or MaskedStandardScaler.from_dict(metadata["scaler"])
    train_masses = np.load(root_dir / metadata["train_masses"]).astype(np.float32, copy=False)
    class_weights = np.asarray(metadata["class_weights"], dtype=np.float32)

    train_cfg = cfg["train"]
    batch_size = int(train_cfg["batch_size"])
    eval_batch_size = int(train_cfg.get("eval_batch_size", batch_size))
    num_workers = int(train_cfg.get("num_workers", 0))
    eval_num_workers = int(train_cfg.get("eval_num_workers", 0))
    seed = int(train_cfg.get("seed", _DEFAULT_SEED))

    datasets = {
        split: TokenShardBatchDataset(
            [str(root_dir / rel_path) for rel_path in metadata["splits"][split]["paths"]],
            batch_size=batch_size if split == "train" else eval_batch_size,
            scaler=scaler,
            shuffle_shards=(split == "train"),
            shuffle_within_shard=(split == "train"),
            drop_last=(split == "train"),
            seed=seed,
        )
        for split in _SPLITS
    }

    return BuiltShardData(
        train_loader=_make_data_loader(datasets["train"], num_workers=num_workers),
        val_loader=_make_data_loader(datasets["val"], num_workers=eval_num_workers),
        test_loader=_make_data_loader(datasets["test"], num_workers=eval_num_workers),
        x_scaler=scaler,
        mass_train=train_masses,
        label_names=list(LABEL_NAMES),
        class_weights=class_weights,
    )
