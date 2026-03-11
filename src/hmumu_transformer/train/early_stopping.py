from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best: Optional[float] = None
    num_bad: int = 0
    should_stop: bool = False

    def update(self, value: float) -> bool:
        """Return True if new best."""
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.num_bad = 0
            return True
        self.num_bad += 1
        if self.num_bad >= self.patience:
            self.should_stop = True
        return False
