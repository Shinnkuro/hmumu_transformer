from __future__ import annotations

import os
from typing import Dict, List, Tuple

_REQUIRED_IMPORTS: List[Tuple[str, str]] = [
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("pyarrow", "pyarrow"),
    ("pyarrow.parquet", "pyarrow"),
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),
    ("yaml", "pyyaml"),
]

def _try_import(mod: str) -> Tuple[bool, str]:
    try:
        __import__(mod)
        return True, ""
    except Exception as e:
        return False, f"{mod}: {e}"

def check_dependencies() -> None:
    missing: List[str] = []
    details: List[str] = []
    for mod, pkg in _REQUIRED_IMPORTS:
        ok, msg = _try_import(mod)
        if not ok:
            missing.append(pkg)
            details.append(msg)
    if missing:
        lines = ["Dependency check failed.", "Missing packages (via import):"]
        for m in sorted(set(missing)):
            lines.append(f"  - {m}")
        lines.append("Details:")
        for d in details:
            lines.append(f"  - {d}")
        raise RuntimeError("\n".join(lines))

def check_files_exist(paths: List[str]) -> None:
    missing: List[str] = [p for p in paths if not os.path.exists(p)]
    if missing:
        lines = ["Input file check failed. Missing parquet files:"]
        for p in missing:
            lines.append(f"  - {p}")
        raise FileNotFoundError("\n".join(lines))
