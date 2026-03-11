from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Any, Dict

def get_pip_freeze() -> str:
    try:
        out = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
        return out
    except Exception:
        return ""

def collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["platform"] = platform.platform()
    info["python"] = platform.python_version()
    try:
        import torch  # type: ignore
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception as e:
        info["torch_error"] = repr(e)
    info["pip_freeze"] = get_pip_freeze()
    return info

def write_env_report(path: str) -> None:
    info = collect_env_info()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)
