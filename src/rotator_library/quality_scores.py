"""Catalog quality scores for models with no local ranking row (#1122).

`config/model_rankings.yaml` carries rows for 281 models; the gateway catalog
scores ~4600. Chain ordering therefore had no signal for most chain entries
and ranked them as if they were worthless. The catalog export
(`scripts/export_quality_scores.py` in the provider repo) fills that gap and is
installed alongside the catalog, so no leaderboard database is needed here.

Accepted shapes -- all normalize to ``{lowercased model id: score / 100}``:

    {"models": {"provider/model": {"quality_score": 72.4}}}
    {"models": {"provider/model": 72.4}}
    {"provider/model": {"quality_score": 72.4}}
    {"provider/model": 72.4}
    [{"id": "provider/model", "quality_score": 72.4}]          # list form

Scores are documented as 0-100. Values outside 0-100 are dropped rather than
clamped: a 4592-model export with a bad column should not silently decide
routing for every unknown model.

A missing or unreadable file returns ``{}`` -- callers fall back to the neutral
prior, which is the pre-#1122 behaviour minus the "unscored means worthless"
assumption.

Path: ``QUALITY_SCORES_PATH`` env var, default ``config/quality_scores.json``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

logger = logging.getLogger(__name__)

DEFAULT_QUALITY_SCORES_PATH = "config/quality_scores.json"

# (path, mtime_ns, size) -> normalized map. Re-export invalidates by mtime.
_cache: Dict[tuple, Dict[str, float]] = {}


def default_path() -> str:
    return os.environ.get("QUALITY_SCORES_PATH", DEFAULT_QUALITY_SCORES_PATH)


def _normalize_entry(value: Any) -> float | None:
    """Pull a 0-100 quality score out of one export entry."""
    if isinstance(value, Mapping):
        value = value.get("quality_score", value.get("score"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if score < 0.0 or score > 100.0:
        return None
    return score / 100.0


def normalize(raw: Any) -> Dict[str, float]:
    """Turn any accepted export shape into {model id: 0..1}."""
    if isinstance(raw, list):
        entries: Iterable = ((e.get("id"), e) for e in raw if isinstance(e, Mapping))
    elif isinstance(raw, Mapping):
        payload = raw.get("models")
        if isinstance(payload, list):
            entries = ((e.get("id"), e) for e in payload if isinstance(e, Mapping))
        elif isinstance(payload, Mapping):
            entries = payload.items()
        else:
            entries = raw.items()
    else:
        return {}

    result: Dict[str, float] = {}
    for key, value in entries:
        if not isinstance(key, str) or "/" not in key:
            continue
        score = _normalize_entry(value)
        if score is None:
            continue
        result[key.strip().lower()] = score
    return result


def load_quality_scores(path: str | os.PathLike | None = None) -> Dict[str, float]:
    """Load the catalog quality export; ``{}`` when absent or unparseable."""
    target = Path(path or default_path())
    try:
        stat = target.stat()
    except OSError:
        return {}

    cache_key = (str(target.resolve()), stat.st_mtime_ns, stat.st_size)
    if cache_key in _cache:
        return _cache[cache_key]

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("quality scores unreadable at %s (%s); using neutral prior", target, exc)
        return {}

    scores = normalize(raw)
    if not scores:
        logger.warning("quality scores at %s had no usable entries", target)
    _cache.clear()  # one export is live at a time; drop stale generations
    _cache[cache_key] = scores
    return scores
