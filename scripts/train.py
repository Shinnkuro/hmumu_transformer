from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import argparse
import json
import os
from typing import Any, Dict

import torch

from hmumu_transformer.preflight import check_dependencies, check_files_exist
from hmumu_transformer.utils.config import load_experiment_config
from hmumu_transformer.utils.env import write_env_report
from hmumu_transformer.utils.run import make_run_dir
from hmumu_transformer.data.build import build_dataloaders
from hmumu_transformer.losses.mass_bins import choose_equal_frequency_bins
from hmumu_transformer.models.model import HmumuTransformer, ModelConfig
from hmumu_transformer.train.optimizer import make_optimizer
from hmumu_transformer.train.loop import train

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to configs/experiment.yaml")
    p.add_argument("--run-dir", default=None, help="Optional existing run directory")
    return p.parse_args()

def _resolve_device(dev: str) -> torch.device:
    dev = str(dev).lower()
    if dev == "cpu":
        return torch.device("cpu")
    if dev == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main() -> None:
    args = parse_args()

    check_dependencies()

    cfg = load_experiment_config(args.config)

    # file existence preflight
    files = []
    files += cfg["data"]["ggH_files"]
    files += cfg["data"]["VBF_files"]
    files += cfg["data"]["DY_files"]
    check_files_exist(files)

    run_dir = args.run_dir or make_run_dir("runs")
    os.makedirs(run_dir, exist_ok=True)

    # Save merged config
    with open(os.path.join(run_dir, "config_merged.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Environment report
    write_env_report(os.path.join(run_dir, "env.json"))

    built = build_dataloaders(cfg)

    # Choose mass bins (train-only, equal-frequency, all classes)
    window = tuple(cfg["data"]["dimuon_mass_window"])
    loss_cfg = cfg["loss"]["mass_adversary"]
    mb = choose_equal_frequency_bins(
        masses=built.mass_train,
        window=window,
        candidate_K=loss_cfg["candidate_K"],
        min_bin_count=int(loss_cfg["min_bin_count"]),
    )
    with open(os.path.join(run_dir, "mass_bins.json"), "w", encoding="utf-8") as f:
        json.dump(mb.to_dict(), f, indent=2)

    # Save scaler
    with open(os.path.join(run_dir, "x_scaler.json"), "w", encoding="utf-8") as f:
        json.dump(built.x_scaler.to_dict(), f, indent=2)

    # Build model
    token_type_ids = torch.tensor([0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    model_cfg = cfg["model"]
    mc = ModelConfig(
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        dropout=float(model_cfg["dropout"]),
        pairwise_dim=int(model_cfg["pairwise_dim"]),
        pairwise_hidden=int(model_cfg["pairwise_hidden"]),
        classifier_hidden=int(model_cfg["classifier_head"]["hidden"]),
        classifier_dropout=float(model_cfg["classifier_head"]["dropout"]),
        adversary_hidden=int(model_cfg["adversary_head"]["hidden"]),
        adversary_dropout=float(model_cfg["adversary_head"]["dropout"]),
    )
    x_dim = int(cfg["schema"]["x_dim"])
    model = HmumuTransformer(cfg=mc, x_dim=x_dim, token_type_ids=token_type_ids, n_mass_bins=mb.K)

    train_cfg = cfg["train"]
    device = _resolve_device(train_cfg.get("device", "auto"))
    model.to(device)

    opt = make_optimizer(
        name=train_cfg.get("optimizer", "adamw"),
        params=model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    hist = train(
        model=model,
        train_loader=built.train_loader,
        val_loader=built.val_loader,
        device=device,
        optimizer=opt,
        num_epochs=int(train_cfg.get("num_epochs", 50)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        mass_edges=mb.edges,
        lambda_max=float(loss_cfg.get("lambda_max", 0.5)),
        warmup_epochs=int(loss_cfg.get("warmup_epochs", 5)),
        run_dir=run_dir,
        early_stopping_cfg=train_cfg.get("early_stopping", {}),
        log_every_steps=int(train_cfg.get("log_every_steps", 10)),
        save_best_only=bool(train_cfg.get("save_best_only", True)),
    )

    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    print(run_dir)

if __name__ == "__main__":
    main()
