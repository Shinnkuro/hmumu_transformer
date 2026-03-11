from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Dict, Any

def make_run_dir(base: str = "runs") -> str:
    os.makedirs(base, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha1(now.encode("utf-8")).hexdigest()[:8]
    run_dir = os.path.join(base, f"{now}_{h}")
    os.makedirs(run_dir, exist_ok=True)
    # convenience symlink/copy
    latest = os.path.join(base, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(run_dir), latest)
    except Exception:
        # symlink may fail on some filesystems; ignore
        pass
    return run_dir
