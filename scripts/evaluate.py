from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import torch

from hmumu_transformer.preflight import check_dependencies, check_files_exist
from hmumu_transformer.data.build import build_dataloaders
from hmumu_transformer.data.scaler import MaskedStandardScaler
from hmumu_transformer.models.model import HmumuTransformer, ModelConfig
from hmumu_transformer.eval.metrics import accuracy, one_vs_rest_auc
from hmumu_transformer.eval.plots import plot_confusion_matrix, plot_roc_curves
from hmumu_transformer.eval.mass_sculpting import dy_mass_sculpting

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Run directory under runs/")
    return p.parse_args()

def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def inference(model, loader, device):
    model.eval()
    ys, ps, masses = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        v = batch["v"].to(device)
        m = batch["m"].to(device)
        out = model(x, v, m, lambda_grl=0.0)
        logits = out["logits_cls"]
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        ys.append(batch["y"].numpy())
        ps.append(p)
        masses.append(batch["mass"].numpy())
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    mm = np.concatenate(masses, axis=0)
    return y, p, mm

def main() -> None:
    args = parse_args()
    check_dependencies()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    merged_path = os.path.join(run_dir, "config_merged.json")
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Missing config_merged.json in run dir: {run_dir}")
    with open(merged_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # preflight files exist
    files = []
    files += cfg["data"]["ggH_files"]
    files += cfg["data"]["VBF_files"]
    files += cfg["data"]["DY_files"]
    check_files_exist(files)

    # Load scaler and mass bins used in training
    with open(os.path.join(run_dir, "x_scaler.json"), "r", encoding="utf-8") as f:
        scaler_dict = json.load(f)
    x_scaler = MaskedStandardScaler.from_dict(scaler_dict)

    with open(os.path.join(run_dir, "mass_bins.json"), "r", encoding="utf-8") as f:
        mass_bins = json.load(f)
    K = int(mass_bins["K"])

    built = build_dataloaders(cfg, x_scaler_override=x_scaler)

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
    model = HmumuTransformer(cfg=mc, x_dim=x_dim, token_type_ids=token_type_ids, n_mass_bins=K)

    ckpt_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, "last.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    device = _resolve_device()
    model.to(device)

    y, p, mm = inference(model, built.test_loader, device)
    y_pred = p.argmax(axis=1)

    # Metrics
    acc = accuracy(y, y_pred)
    aucs = one_vs_rest_auc(y, p, n_classes=3)

    # Confusion matrix
    cm = np.zeros((3, 3), dtype=np.int64)
    for yt, yp in zip(y, y_pred):
        cm[int(yt), int(yp)] += 1

    labels = built.label_names
    plot_confusion_matrix(cm, labels, os.path.join(run_dir, "confusion_matrix.png"))
    plot_roc_curves(y, p, labels, os.path.join(run_dir, "roc_ovr.png"))

    # VBF purity at a chosen working point: top 10% by p(VBF)
    p_vbf = p[:, 1]
    thr = np.quantile(p_vbf, 0.90)
    sel = p_vbf >= thr
    vbf_purity = float((y[sel] == 1).mean()) if sel.any() else float("nan")

    # DY mass sculpting: score = 1 - p(DY)
    dy_sel = (y == 2)
    dy_scores = 1.0 - p[dy_sel, 2]
    dy_masses = mm[dy_sel]
    n_score_bins = int(cfg["train"].get("n_score_bins", 6))
    sculpt = dy_mass_sculpting(
        dy_masses, dy_scores, n_bins=n_score_bins,
        outpath=os.path.join(run_dir, "dy_mass_sculpting.png"),
    )

    results = {
        "test_accuracy": acc,
        "test_auc_ovr": aucs,
        "vbf_purity_top10pct": vbf_purity,
        "dy_mass_sculpting_pearson_r": sculpt.pearson_r,
        "n_test": int(y.shape[0]),
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
