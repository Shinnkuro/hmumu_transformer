from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmumu_transformer.data import shards as shards_mod
from hmumu_transformer.data.build import build_dataloaders
from hmumu_transformer.losses.mass_bins import choose_equal_frequency_bins
from hmumu_transformer.models.model import HmumuTransformer, ModelConfig
from hmumu_transformer.train.loop import train
from hmumu_transformer.train.optimizer import make_optimizer


def _make_fake_arrays(rng: np.random.Generator, n_rows: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "MET_pt": rng.uniform(0, 50, n_rows).astype("f4"),
        "MET_phi": rng.uniform(-np.pi, np.pi, n_rows).astype("f4"),
        "mu1_pt": rng.uniform(20, 80, n_rows).astype("f4"),
        "mu1_eta": rng.uniform(-2.4, 2.4, n_rows).astype("f4"),
        "mu1_phi": rng.uniform(-np.pi, np.pi, n_rows).astype("f4"),
        "mu1_mass": np.full(n_rows, 0.105, dtype="f4"),
        "mu1_iso": rng.uniform(0, 0.2, n_rows).astype("f4"),
        "mu2_pt": rng.uniform(20, 70, n_rows).astype("f4"),
        "mu2_eta": rng.uniform(-2.4, 2.4, n_rows).astype("f4"),
        "mu2_phi": rng.uniform(-np.pi, np.pi, n_rows).astype("f4"),
        "mu2_mass": np.full(n_rows, 0.105, dtype="f4"),
        "mu2_iso": rng.uniform(0, 0.2, n_rows).astype("f4"),
        "dimuon_mass": rng.uniform(115, 135, n_rows).astype("f4"),
        "dimuon_pt_log": np.log(rng.uniform(1, 100, n_rows)).astype("f4"),
        "dimuon_rapidity": rng.uniform(-2.5, 2.5, n_rows).astype("f4"),
        "dimuon_ebe_mass_res": rng.uniform(0.5, 3.0, n_rows).astype("f4"),
        "dimuon_phi_cs": rng.uniform(-np.pi, np.pi, n_rows).astype("f4"),
        "dimuon_cos_theta_cs": rng.uniform(-1, 1, n_rows).astype("f4"),
        "nsoftjets5_nominal": rng.integers(0, 5, n_rows).astype("f4"),
        "njets_nominal": rng.integers(0, 5, n_rows).astype("i4"),
    }
    for jet_idx in range(1, 5):
        present = arrays["njets_nominal"] >= jet_idx
        arrays[f"jet{jet_idx}_pt_nominal"] = np.where(
            present, rng.uniform(25, 150, n_rows), np.nan
        ).astype("f4")
        arrays[f"jet{jet_idx}_eta_nominal"] = np.where(
            present, rng.uniform(-4.7, 4.7, n_rows), np.nan
        ).astype("f4")
        arrays[f"jet{jet_idx}_phi_nominal"] = np.where(
            present, rng.uniform(-np.pi, np.pi, n_rows), np.nan
        ).astype("f4")
        arrays[f"jet{jet_idx}_mass_nominal"] = np.where(
            present, rng.uniform(5, 50, n_rows), np.nan
        ).astype("f4")
        arrays[f"jet{jet_idx}_qgl_nominal"] = np.where(
            present, rng.uniform(0, 1, n_rows), np.nan
        ).astype("f4")
        arrays[f"jet{jet_idx}_jetId_nominal"] = np.where(
            present, rng.integers(0, 7, n_rows), np.nan
        ).astype("f4")
        arrays[f"jet{jet_idx}_puId_nominal"] = np.where(
            present, rng.integers(0, 7, n_rows), np.nan
        ).astype("f4")
    return arrays


def main() -> None:
    out_root = Path("tmp_smoke")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    columns = [
        "MET_pt",
        "MET_phi",
        "mu1_pt",
        "mu1_eta",
        "mu1_phi",
        "mu1_mass",
        "mu1_iso",
        "mu2_pt",
        "mu2_eta",
        "mu2_phi",
        "mu2_mass",
        "mu2_iso",
        "dimuon_mass",
        "dimuon_pt_log",
        "dimuon_rapidity",
        "dimuon_ebe_mass_res",
        "dimuon_phi_cs",
        "dimuon_cos_theta_cs",
        "nsoftjets5_nominal",
        "njets_nominal",
    ]
    for jet_idx in range(1, 5):
        columns.extend(
            [
                f"jet{jet_idx}_pt_nominal",
                f"jet{jet_idx}_eta_nominal",
                f"jet{jet_idx}_phi_nominal",
                f"jet{jet_idx}_mass_nominal",
                f"jet{jet_idx}_qgl_nominal",
                f"jet{jet_idx}_jetId_nominal",
                f"jet{jet_idx}_puId_nominal",
            ]
        )

    cfg = {
        "data": {
            "ggH_files": ["ggh.parquet"],
            "VBF_files": ["vbf.parquet"],
            "DY_files": ["dy1.parquet", "dy2.parquet"],
            "columns": columns,
            "split": {"n_folds": 4, "train_folds": [0, 1], "val_folds": [2], "test_folds": [3]},
            "dimuon_mass_window": [115.0, 135.0],
            "shards": {
                "root_dir": str(out_root / "processed"),
                "rows_per_shard": 8,
                "record_batch_size": 8,
                "rebuild": True,
                "seed": 7,
            },
        },
        "schema": {"n_tokens": 7, "max_jets": 4, "x_dim": 20},
        "train": {
            "batch_size": 4,
            "eval_batch_size": 4,
            "num_workers": 0,
            "eval_num_workers": 0,
            "seed": 7,
        },
    }

    rng = np.random.default_rng(42)
    source_data = {
        "ggh.parquet": _make_fake_arrays(rng, 8),
        "vbf.parquet": _make_fake_arrays(rng, 8),
        "dy1.parquet": _make_fake_arrays(rng, 8),
        "dy2.parquet": _make_fake_arrays(rng, 8),
    }

    def fake_iter(paths, columns, *, batch_size=1):
        for path in paths:
            arrays = source_data[path]
            n_rows = len(next(iter(arrays.values())))
            for start in range(0, n_rows, batch_size):
                stop = min(start + batch_size, n_rows)
                yield {name: arrays[name][start:stop] for name in columns}

    shards_mod.iter_parquet_batches = fake_iter

    built = build_dataloaders(cfg)
    mass_bins = choose_equal_frequency_bins(built.mass_train, (115.0, 135.0), [2, 3], 2)
    model = HmumuTransformer(
        cfg=ModelConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            dropout=0.0,
            pairwise_dim=6,
            pairwise_hidden=8,
            classifier_hidden=8,
            classifier_dropout=0.0,
            adversary_hidden=8,
            adversary_dropout=0.0,
        ),
        x_dim=20,
        token_type_ids=torch.tensor([0, 1, 1, 2, 2, 2, 2]),
        n_mass_bins=mass_bins.K,
    )
    optimizer = make_optimizer("adamw", model.parameters(), lr=1e-3, weight_decay=0.0)
    history = train(
        model=model,
        train_loader=built.train_loader,
        val_loader=built.val_loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        num_epochs=1,
        grad_clip_norm=1.0,
        mass_edges=mass_bins.edges,
        lambda_max=0.2,
        warmup_epochs=1,
        run_dir=str(out_root / "run"),
        early_stopping_cfg={"enabled": False},
        log_every_steps=0,
        save_best_only=True,
        class_weights=built.class_weights,
    )
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
