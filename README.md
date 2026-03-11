# hmumu_transformer

Event-level Transformer classifier for three classes:
1) ggH -> μμ
2) VBF -> μμ
3) DY -> μμ

## Data inputs (parquet)
Configured in `configs/data.yaml`:

- ggH: `/depot/cms/hu1027/hmm_ntuples/skimmed_for_dnn_AK8jets/merged/2018/ggh_powhegPS_merged.parquet`
- VBF: `/depot/cms/hu1027/hmm_ntuples/skimmed_for_dnn_AK8jets/merged/2018/dy_VBF_filter_merged.parquet`
- DY:  
  - `/depot/cms/hu1027/hmm_ntuples/skimmed_for_dnn_AK8jets/merged/2018/dy_M-50_MiNNLO_merged.parquet`  
  - `/depot/cms/hu1027/hmm_ntuples/skimmed_for_dnn_AK8jets/merged/2018/dy_M-100To200_MiNNLO_merged.parquet`

Each row is one event. The training samples are assumed already restricted to `dimuon_mass` in [115, 135] GeV.

## Tokens and tensors
- N = 7 tokens per event: [CLS], μ1, μ2, jet1..jet4
- Node features: `x[B, N, F]` (default F=16)
- 4-vector: `v[B, N, 4]` with (pt, eta, phi, mass)
- Mask: `m[B, N]` (1 = valid token, 0 = padding)

Missing jets are represented by NaNs in the parquet; the tokenizer converts them to padding with `m=0` and sets `x=v=0` for those tokens.

## Model
Backbone is an encoder-only Transformer with **attention-with-pairwise-bias**:
- Pairwise features are computed from `v` (Δη, sinΔφ, cosΔφ, logΔR, log(pt_i/pt_j), and **log m_ij for jet–jet pairs only**).
- A shared MLP maps pairwise features to per-head bias terms that are added to attention logits before softmax.

A **mass adversary head** (binned classifier over `dimuon_mass`) is trained with a gradient reversal layer (GRL) to suppress mass sculpting.

## Splits and sampling
- Per-class 4-way split by row index modulo 4:
  - train: folds 0 & 1
  - val: fold 2
  - test: fold 3
- Balanced sampling (uniform across classes) via a custom sampler. Default behavior uses a **single large balanced batch per epoch**; you can enable standard mini-batching by setting `train.batch_size` in config.

## Running (pure PyTorch)
From the repo root:

```bash
python -m scripts.train --config configs/experiment.yaml
python -m scripts.evaluate --run-dir runs/<RUN_ID>
```

The training script performs a preflight dependency check and aborts before any work if a required dependency is missing.

## Outputs
A run directory under `runs/` contains:
- merged config
- environment report (Python, OS, torch, CUDA, and pip freeze)
- scaler statistics for feature standardization (train-only)
- mass bin edges (train-only, equal-frequency within [115, 135] GeV)
- checkpoints
- metrics JSON
- plots:
  - confusion matrix
  - one-vs-rest ROC curves
  - DY mass sculpting: DY mμμ distributions in score bins and Pearson correlation

## Notes on correctness
This project aims to be runnable and internally consistent. It cannot guarantee zero runtime issues in every external environment (filesystem permissions, missing deps, GPU driver mismatches, etc.). If a failure occurs, it should fail early with a clear error message (especially for missing dependencies).
