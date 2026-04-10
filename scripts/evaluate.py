from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hmumu_transformer.data.build import build_dataloaders
from hmumu_transformer.data.scaler import MaskedStandardScaler
from hmumu_transformer.eval.mass_sculpting import dy_mass_sculpting
from hmumu_transformer.eval.metrics import accuracy, one_vs_rest_auc
from hmumu_transformer.eval.plots import plot_confusion_matrix, plot_roc_curves
from hmumu_transformer.models.model import HmumuTransformer, ModelConfig
from hmumu_transformer.preflight import check_dependencies, check_files_exist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Run directory under runs/")
    return parser.parse_args()


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def inference(model, loader, device: torch.device):
    model.eval()
    ys, ps, masses = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        v = batch["v"].to(device)
        m = batch["m"].to(device)
        out = model(x, v, m, lambda_grl=0.0)
        logits = out["logits_cls"]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        ys.append(batch["y"].numpy())
        ps.append(probs)
        masses.append(batch["mass"].numpy())
    return (
        np.concatenate(ys, axis=0),
        np.concatenate(ps, axis=0),
        np.concatenate(masses, axis=0),
    )


def main() -> None:
    args = parse_args()
    check_dependencies()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    merged_path = os.path.join(run_dir, "config_merged.json")
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Missing config_merged.json in run dir: {run_dir}")
    with open(merged_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    files = []
    files += cfg["data"]["ggH_files"]
    files += cfg["data"]["VBF_files"]
    files += cfg["data"]["DY_files"]
    check_files_exist(files)

    with open(os.path.join(run_dir, "x_scaler.json"), "r", encoding="utf-8") as handle:
        scaler_dict = json.load(handle)
    x_scaler = MaskedStandardScaler.from_dict(scaler_dict)

    with open(os.path.join(run_dir, "mass_bins.json"), "r", encoding="utf-8") as handle:
        mass_bins = json.load(handle)
    n_mass_bins = int(mass_bins["K"])

    built = build_dataloaders(cfg, x_scaler_override=x_scaler)

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
        n_mass_bins=n_mass_bins,
    )

    ckpt_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, "last.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    device = _resolve_device()
    model.to(device)

    y_true, probs, masses = inference(model, built.test_loader, device)
    y_pred = probs.argmax(axis=1)

    acc = accuracy(y_true, y_pred)
    aucs = one_vs_rest_auc(y_true, probs, n_classes=3)

    cm = np.zeros((3, 3), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1

    labels = built.label_names
    plot_confusion_matrix(cm, labels, os.path.join(run_dir, "confusion_matrix.png"))
    plot_roc_curves(y_true, probs, labels, os.path.join(run_dir, "roc_ovr.png"))

    p_vbf = probs[:, 1]
    threshold = np.quantile(p_vbf, 0.90)
    selected = p_vbf >= threshold
    vbf_purity = float((y_true[selected] == 1).mean()) if selected.any() else float("nan")

    dy_mask = y_true == 2
    sculpt = dy_mass_sculpting(
        masses[dy_mask],
        1.0 - probs[dy_mask, 2],
        n_bins=int(cfg["train"].get("n_score_bins", 6)),
        outpath=os.path.join(run_dir, "dy_mass_sculpting.png"),
    )

    results = {
        "test_accuracy": acc,
        "test_auc_ovr": aucs,
        "vbf_purity_top10pct": vbf_purity,
        "dy_mass_sculpting_pearson_r": sculpt.pearson_r,
        "n_test": int(y_true.shape[0]),
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
