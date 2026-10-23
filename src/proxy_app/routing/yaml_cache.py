"""Process-wide cache for the routing YAML configs.

Every ``RouterCore()`` construction re-parsed the same handful of files, and
``model_rankings.yaml`` alone costs ~2s in PyYAML. The test suite builds a
router per test, so it paid that price hundreds of times; production paid it on
reload too.

The cache key includes mtime and size, so editing a config on disk invalidates
it. Callers get a deep copy, because some of them keep and mutate the parsed
structure.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from typing import Any, Optional


@lru_cache(maxsize=64)
def _read(path: str, mtime_ns: int, size: int) -> Any:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_yaml_cached(path: Optional[str], default: Any = None) -> Any:
    """Parse `path` once per (mtime, size); missing or broken files -> default."""
    if not path:
        return default
    abspath = os.path.abspath(path)
    try:
        stat = os.stat(abspath)
    except OSError:
        return default
    try:
        data = _read(abspath, stat.st_mtime_ns, stat.st_size)
    except Exception:
        return default
    if data is None:
        return default
    return copy.deepcopy(data)


def clear_yaml_cache() -> None:
    """Drop every parsed config (tests that rewrite files can call this)."""
    _read.cache_clear()
