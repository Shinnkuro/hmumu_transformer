from __future__ import annotations

import glob
import os
from typing import List, Sequence

_GLOB_CHARS = ("*", "?", "[")


def is_glob_pattern(path: str) -> bool:
    return any(ch in path for ch in _GLOB_CHARS)



def expand_path_patterns(
    paths: Sequence[str],
    *,
    strict: bool = False,
    description: str = "input paths",
) -> List[str]:
    """Expand literal file paths and glob patterns into a deduplicated file list.

    Parameters
    ----------
    paths:
        A sequence containing either concrete file paths or glob patterns.
    strict:
        If ``True``, raise ``FileNotFoundError`` when any item resolves to no files.
    description:
        Human-readable label used in error messages.
    """
    resolved: List[str] = []
    unmatched: List[str] = []

    for path in paths:
        if is_glob_pattern(path):
            matches = sorted(
                candidate
                for candidate in glob.glob(path, recursive=True)
                if os.path.isfile(candidate)
            )
            if matches:
                resolved.extend(matches)
            else:
                unmatched.append(path)
            continue

        if os.path.isfile(path):
            resolved.append(path)
        else:
            unmatched.append(path)

    deduped = list(dict.fromkeys(resolved))

    if strict and unmatched:
        lines = [f"Failed to resolve some {description}:"]
        for item in unmatched:
            lines.append(f"  - {item}")
        raise FileNotFoundError("\n".join(lines))

    return deduped
