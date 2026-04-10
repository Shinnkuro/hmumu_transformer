from __future__ import annotations

from typing import Dict, Iterator, List, Sequence

import numpy as np

from ..utils.paths import expand_path_patterns


DEFAULT_RECORD_BATCH_SIZE = 65_536


def _load_pyarrow_dataset_module():
    try:
        import pyarrow.dataset as ds  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in runtime environments
        raise RuntimeError(
            "Missing dependency: pyarrow. Install it before reading parquet inputs."
        ) from exc
    return ds


def iter_parquet_batches(
    paths: Sequence[str],
    columns: Sequence[str],
    *,
    batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield parquet data as NumPy-backed record batches.

    This avoids materializing the full dataset with ``dataset.to_table()`` and keeps
    peak host memory bounded by the record-batch size.
    """
    resolved_paths = expand_path_patterns(
        paths,
        strict=True,
        description="parquet inputs",
    )
    if not resolved_paths:
        return

    ds = _load_pyarrow_dataset_module()
    dataset = ds.dataset(resolved_paths, format="parquet")
    scanner = dataset.scanner(columns=list(columns), batch_size=int(batch_size))
    names: List[str] = list(columns)
    for record_batch in scanner.to_batches():
        yield {
            name: record_batch.column(i).to_numpy(zero_copy_only=False)
            for i, name in enumerate(names)
        }


def read_parquet_files(
    paths: Sequence[str],
    columns: Sequence[str],
    *,
    batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
) -> Dict[str, np.ndarray]:
    """Compatibility helper that concatenates streamed record batches.

    Prefer ``iter_parquet_batches`` in memory-sensitive paths.
    """
    buffered: Dict[str, List[np.ndarray]] = {name: [] for name in columns}
    for batch in iter_parquet_batches(paths, columns, batch_size=batch_size):
        for name, values in batch.items():
            buffered[name].append(values)

    out: Dict[str, np.ndarray] = {}
    for name, chunks in buffered.items():
        out[name] = np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)
    return out
