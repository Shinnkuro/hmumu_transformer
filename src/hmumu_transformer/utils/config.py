from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge b into a (a is copied)."""
    out: Dict[str, Any] = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: PyYAML. Install it in your environment (e.g., `pixi add pyyaml`)."
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_experiment_config(path: str) -> Dict[str, Any]:
    """Load an experiment YAML that can include other YAMLs via `includes`."""
    cfg = load_yaml(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    includes: List[str] = cfg.pop("includes", []) or []
    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = os.path.join(base_dir, inc)
        inc_cfg = load_yaml(inc_path)
        merged = deep_merge(merged, inc_cfg)
    merged = deep_merge(merged, cfg)
    return merged
