from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hmumu_transformer.data.build import build_dataloaders
from hmumu_transformer.losses.mass_bins import choose_equal_frequency_bins
from hmumu_transformer.models.model import HmumuTransformer, ModelConfig
from hmumu_transformer.preflight import check_dependencies, check_files_exist
from hmumu_transformer.train.loop import train
from hmumu_transformer.train.optimizer import make_optimizer
from hmumu_transformer.utils.config import load_experiment_config
from hmumu_transformer.utils.env import write_env_report
from hmumu_transformer.utils.run import make_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/experiment.yaml")
    parser.add_argument("--run-dir", default=None, help="Optional existing run directory")
    return parser.parse_args()


def _resolve_device(dev: str) -> torch.device:
    dev = str(dev).lower()
    if dev == "cpu":
        return torch.device("cpu")
    if dev == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    check_dependencies()
    cfg = load_experiment_config(args.config)

    files = []
    files += cfg["data"]["ggH_files"]
    files += cfg["data"]["VBF_files"]
    files += cfg["data"]["DY_files"]
    resolved_files = check_files_exist(files)

    run_dir = args.run_dir or make_run_dir("runs")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config_merged.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)
    with open(os.path.join(run_dir, "input_files_resolved.json"), "w", encoding="utf-8") as handle:
        json.dump({"files": resolved_files}, handle, indent=2)
    write_env_report(os.path.join(run_dir, "env.json"))

    built = build_dataloaders(cfg)

    window = tuple(cfg["data"]["dimuon_mass_window"])
    loss_cfg = cfg["loss"]["mass_adversary"]
    mass_bins = choose_equal_frequency_bins(
        masses=built.mass_train,
        window=window,
        candidate_K=loss_cfg["candidate_K"],
        min_bin_count=int(loss_cfg["min_bin_count"]),
    )
    with open(os.path.join(run_dir, "mass_bins.json"), "w", encoding="utf-8") as handle:
        json.dump(mass_bins.to_dict(), handle, indent=2)
    with open(os.path.join(run_dir, "x_scaler.json"), "w", encoding="utf-8") as handle:
        json.dump(built.x_scaler.to_dict(), handle, indent=2)
    with open(os.path.join(run_dir, "class_weights.json"), "w", encoding="utf-8") as handle:
        json.dump({"label_names": built.label_names, "weights": built.class_weights.tolist()}, handle, indent=2)

    token_type_ids = torch.tensor([0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    model_cfg = cfg["model"]
    model = HmumuTransformer(
        cfg=ModelConfig(
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
        ),
        x_dim=int(cfg["schema"]["x_dim"]),
        token_type_ids=token_type_ids,
        n_mass_bins=mass_bins.K,
    )

    train_cfg = cfg["train"]
    device = _resolve_device(train_cfg.get("device", "auto"))
    model.to(device)

    optimizer = make_optimizer(
        name=train_cfg.get("optimizer", "adamw"),
        params=model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    history = train(
        model=model,
        train_loader=built.train_loader,
        val_loader=built.val_loader,
        device=device,
        optimizer=optimizer,
        num_epochs=int(train_cfg.get("num_epochs", 50)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        mass_edges=mass_bins.edges,
        lambda_max=float(loss_cfg.get("lambda_max", 0.5)),
        warmup_epochs=int(loss_cfg.get("warmup_epochs", 5)),
        run_dir=run_dir,
        early_stopping_cfg=train_cfg.get("early_stopping", {}),
        log_every_steps=int(train_cfg.get("log_every_steps", 10)),
        save_best_only=bool(train_cfg.get("save_best_only", True)),
        class_weights=built.class_weights,
    )

    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(run_dir)


if __name__ == "__main__":
    main()
