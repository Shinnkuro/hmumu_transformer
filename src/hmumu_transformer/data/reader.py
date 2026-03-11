from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

def _table_to_numpy_dict(table) -> Dict[str, np.ndarray]:
    # Use zero-copy where possible; fall back gracefully.
    out: Dict[str, np.ndarray] = {}
    for name in table.column_names:
        col = table[name]
        # Convert to numpy; PyArrow may produce object dtype for mixed types.
        arr = col.to_numpy(zero_copy_only=False)
        out[name] = arr
    return out

def read_parquet_files(files: Sequence[str], columns: Sequence[str]) -> Dict[str, np.ndarray]:
    """Read parquet files into a dict of numpy arrays (concatenated)."""
    import pyarrow.dataset as ds  # type: ignore

    dataset = ds.dataset(list(files), format="parquet")
    table = dataset.to_table(columns=list(columns))
    out = _table_to_numpy_dict(table)
    return out

@dataclass
class ClassData:
    name: str
    arrays: Dict[str, np.ndarray]

    @property
    def n(self) -> int:
        # All columns should have the same length.
        any_key = next(iter(self.arrays.keys()))
        return int(self.arrays[any_key].shape[0])
