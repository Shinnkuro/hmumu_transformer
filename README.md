# hmumu_transformer

Event-level Transformer classifier for three classes:

1. ggH -> μμ
2. VBF -> μμ
3. DY -> μμ

The model consumes a fixed object-level token sequence per event:

- `[CLS]`
- `mu1`
- `mu2`
- `jet1`
- `jet2`
- `jet3`
- `jet4`

The current codebase is optimized for **large parquet datasets** through an offline preprocessing stage that writes token shards to disk and a streaming training path based on `IterableDataset`.

---

## What changed in this version

The original in-memory pipeline loaded full parquet tables, filtered them in memory, re-split them in memory, rebuilt token tensors on demand, and used a balanced sampler. That design is simple, but it scales poorly.

This version changes the data pipeline in six ways:

1. `train.batch_size` is now an explicit mini-batch size; the previous `batch_size: null` / giant-batch path is removed.
2. Feature standardization is fitted **online** with streaming statistics instead of concatenating the full training set into one giant array.
3. Parquet reading now uses **record-batch scanning** instead of `dataset.to_table(...)` materialization.
4. A dedicated script builds **offline token shards**.
5. Training and evaluation use **streaming shard readers** built on `IterableDataset`.
6. Class imbalance is handled with **weighted classification loss**, not a balanced sampler.

These changes cut peak host-memory usage from “entire filtered dataset + multiple derived copies” down to roughly “one parquet record batch + one shard buffer + one training batch”.

---

## Data inputs

Configured in `configs/data.yaml`:

- `data.ggH_files`
- `data.VBF_files`
- `data.DY_files`

Each entry may be either:

- a concrete parquet file path, or
- a `glob` pattern such as `*.parquet` or `**/*.parquet`

Examples:

```yaml
data:
  ggH_files:
    - /data/hmm/ggh_*.parquet
  DY_files:
    - /data/hmm/dy/**/*.parquet
```

`**` patterns are resolved recursively. Input patterns are expanded before preflight checks, parquet scanning, and shard-cache metadata comparison, so adding or removing matching files will correctly trigger a shard rebuild.

The tokenizer expects the columns listed in `data.columns`. The code still enforces the `dimuon_mass_window` during preprocessing even if the input parquet files were already filtered upstream.

---

## Tensor schema

### Tokens

`N = 7`

Token order:

- token 0: `[CLS]`
- token 1: `mu1`
- token 2: `mu2`
- token 3: `jet1`
- token 4: `jet2`
- token 5: `jet3`
- token 6: `jet4`

### Per-event tensors

- `x[B, N, F]` with `F = 20`
- `v[B, N, 4]` with `(pt, eta, phi, mass)`
- `m[B, N]` with `1 = valid token`, `0 = padding`
- `y[B]` class label
- `mass[B]` raw `dimuon_mass`

Missing jets are padded with zero features and `m = 0`.

---

## Model summary

Backbone: encoder-only Transformer with pairwise attention bias.

Pairwise features are computed from `v` using:

- `Δeta`
- `sin(Δphi)`
- `cos(Δphi)`
- `log(ΔR)`
- `log(pt_i / pt_j)`
- `log(m_ij)` for jet-jet pairs only

A small MLP maps those pairwise features to per-head additive attention biases.

The classifier output has three classes:

- `ggH`
- `VBF`
- `DY`

A separate mass-adversary head predicts binned `dimuon_mass` through a gradient reversal layer to reduce mass sculpting.

---

## Split policy

The split is still deterministic and per class:

- fold = filtered row index modulo `n_folds`
- train folds: `0, 1`
- val fold: `2`
- test fold: `3`

Important detail: the fold index is assigned **after** applying the `dimuon_mass_window`, matching the previous implementation.

---

## Offline shard preprocessing

### Why it exists

Preprocessing moves the expensive steps out of the training loop:

- parquet scanning
- mass-window filtering
- fold assignment
- tokenization
- train-split scaler fitting
- shard writing

### Output layout

By default:

```text
processed/default/
  metadata.json
  train_masses.npy
  train/
    ggH_00000.npz
    ...
  val/
    VBF_00000.npz
    ...
  test/
    DY_00000.npz
    ...
```

Each shard contains:

- `x`
- `v`
- `m`
- `y`
- `mass`

### Relevant config

In `configs/data.yaml`:

```yaml
shards:
  root_dir: processed/default
  rows_per_shard: 50000
  record_batch_size: 65536
  rebuild: false
  seed: 1337
```

- `rows_per_shard`: maximum number of events per output shard
- `record_batch_size`: parquet scanning batch size
- `rebuild`: force regeneration on the next run

If `metadata.json` already exists and matches the current config, preprocessing is skipped automatically.

---

## Training path

Training now streams token shards from disk through `TokenShardBatchDataset`, an `IterableDataset` that:

- assigns shard files across workers
- optionally shuffles shard order for training
- optionally shuffles within each shard
- standardizes `x` on the fly in torch
- yields already-collated batches

Class imbalance is handled with class weights

\[
  w_c = \frac{N}{K \cdot n_c}
\]

where:

- `N` = total number of training events
- `K` = number of classes
- `n_c` = number of training events in class `c`

Those weights are used in the classification loss during training and validation.

---

## Commands

From the repository root:

### 1. Build shards explicitly

```bash
python -m scripts.build_token_shards --config configs/experiment.yaml
```

Force rebuild:

```bash
python -m scripts.build_token_shards --config configs/experiment.yaml --force
```

### 2. Train

```bash
python -m scripts.train --config configs/experiment.yaml
```

Training will automatically build shards first if they do not exist or are stale.

### 3. Evaluate

```bash
python -m scripts.evaluate --run-dir runs/<RUN_ID>
```

---

## Outputs

A run directory under `runs/` contains:

- `config_merged.json`
- `env.json`
- `x_scaler.json`
- `mass_bins.json`
- `class_weights.json`
- `best.pt`
- `last.pt`
- `history.json`
- `test_metrics.json`
- `confusion_matrix.png`
- `roc_ovr.png`
- `dy_mass_sculpting.png`

The shard directory contains its own metadata and does not need to live under `runs/`.

---

## Configuration notes

### Batch sizes

`configs/train.yaml` now uses explicit mini-batches:

```yaml
batch_size: 768
eval_batch_size: 1024
```

There is no longer a supported `batch_size: null` giant-batch mode.

### Workers

Shard streaming works with `num_workers > 0`, but each worker reads different shard files. If you increase workers, increase them gradually and watch filesystem throughput.

---

## Development notes

### Memory behavior

The large-memory failure modes in the old pipeline were mainly caused by:

- full parquet materialization
- boolean-filtered full-column copies
- split copies
- full training-set concatenation for scaler fitting
- giant balanced batches

The new pipeline removes all of those failure modes from the main training path.

### Remaining constraints

This does **not** make the project “infinite-scale”. Practical limits still come from:

- shard size
- record batch size
- filesystem bandwidth
- CPU tokenization cost during shard building
- GPU memory during model forward/backward

If preprocessing time becomes dominant, the next optimization target is vectorized batch tokenization. That is a performance issue, not a correctness issue.
