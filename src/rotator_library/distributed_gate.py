"""Distributed cooldown + per-(provider,model) concurrency gating (task-board #233).

Two independent mechanisms a request path can consult before picking a provider:

1. SharedCooldownStore
   A SQLite table that records active cooldowns so SEPARATE PROCESSES (and,
   on a shared mount, separate machines) skip a provider that recently 429'd
   without each having to learn the hard way.

   GOTCHA (Opus 4.8, 2026-06-24): the #233 spec says "shared cooldown across
   machines via /dev/shm sqlite". /dev/shm is host-LOCAL tmpfs -- it is NOT
   shared between physical machines. The `source_machine` column only earns
   its keep when `db_path` points at a genuinely shared filesystem (NFS, a
   mounted volume, etc.). Default db_path=/dev/shm/gateway_cooldowns.db gives
   you fast cross-PROCESS sharing on one host. For real cross-MACHINE sharing,
   pass a shared-mount path. Do not claim cross-machine coordination while
   writing to /dev/shm; it is a per-host file.
   # ponytail: per-host tmpfs by default; shared-mount path when you actually
   # need cross-machine. Upgrade path is a path change, not a rewrite.

2. ConcurrencyGate
   An in-process, thread-safe slot counter keyed by (provider, model). It is
   deliberately PER-PROCESS (the spec says "in-memory counter"): it bounds how
   many concurrent in-flight requests THIS process sends to a given model, and
   also enforces a per-key min interval between request starts. Non-blocking:
   try_acquire() returns False immediately when full so the caller falls back
   to the next provider instead of waiting.

Stdlib only (sqlite3, threading, time). No new deps.
"""

from __future__ import annotations

import socket
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import yaml  # already a project dep
except Exception:  # pragma: no cover - yaml is present in prod
    yaml = None

DEFAULT_COOLDOWN_DB = "/dev/shm/gateway_cooldowns.db"
COOLDOWN_ROW_TTL_S = 300          # spec: purge rows older than this
_BUSY_TIMEOUT_MS = 500


# ===========================================================================
# 1. Shared cooldown store (cross-process; cross-machine only on shared mount)
# ===========================================================================
class SharedCooldownStore:
    """SQLite-backed cooldown table shared across processes.

    Schema (PK = provider+model):
        provider TEXT, model TEXT, cooldown_until_ts REAL,
        retry_after_s REAL, source_machine TEXT, updated_at REAL
    """

    def __init__(self, db_path: str = DEFAULT_COOLDOWN_DB, machine_id: Optional[str] = None):
        self.db_path = db_path
        self.machine_id = machine_id or socket.gethostname()
        self._local = threading.local()  # one connection per thread
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
            c.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    def _init_db(self) -> None:
        c = self._conn()
        c.execute(
            """CREATE TABLE IF NOT EXISTS cooldowns (
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                cooldown_until_ts REAL NOT NULL,
                retry_after_s REAL,
                source_machine TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (provider, model)
            )"""
        )
        c.commit()

    def start_cooldown(
        self,
        provider: str,
        model: str,
        retry_after_s: float,
        *,
        now: Optional[float] = None,
    ) -> None:
        """Record (or extend) a cooldown. Idempotent on the longer deadline."""
        now = time.time() if now is None else now
        until = now + max(0.0, float(retry_after_s))
        c = self._conn()
        # UPSERT keeping the LATER deadline so we never shorten an active cooldown.
        c.execute(
            """INSERT INTO cooldowns
                 (provider, model, cooldown_until_ts, retry_after_s, source_machine, updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(provider, model) DO UPDATE SET
                 cooldown_until_ts=MAX(cooldown_until_ts, excluded.cooldown_until_ts),
                 retry_after_s=excluded.retry_after_s,
                 source_machine=excluded.source_machine,
                 updated_at=excluded.updated_at""",
            (provider.lower(), model, until, float(retry_after_s), self.machine_id, now),
        )
        c.commit()

    def cooldown_remaining(
        self, provider: str, model: str, *, now: Optional[float] = None
    ) -> float:
        """Seconds remaining; 0.0 if not cooling down or row expired."""
        now = time.time() if now is None else now
        c = self._conn()
        row = c.execute(
            "SELECT cooldown_until_ts FROM cooldowns WHERE provider=? AND model=?",
            (provider.lower(), model),
        ).fetchone()
        if row is None:
            return 0.0
        return max(0.0, float(row["cooldown_until_ts"]) - now)

    def is_cooling_down(
        self, provider: str, model: str, *, now: Optional[float] = None
    ) -> bool:
        return self.cooldown_remaining(provider, model, now=now) > 0.0

    def purge_expired(self, *, now: Optional[float] = None) -> int:
        """Delete rows whose cooldown ended more than COOLDOWN_ROW_TTL_S ago.

        Returns number of rows removed. Cheap; call opportunistically.
        """
        now = time.time() if now is None else now
        cutoff = now - COOLDOWN_ROW_TTL_S
        c = self._conn()
        cur = c.execute("DELETE FROM cooldowns WHERE cooldown_until_ts < ?", (cutoff,))
        c.commit()
        return cur.rowcount

    def recent_cooldowns(self, *, limit: int = 100) -> list:
        """Return up to `limit` cooldown rows for dashboards / health probes.

        Rows are returned as sqlite3.Row dicts (.keys() is the column list).
        """
        c = self._conn()
        cur = c.execute(
            "SELECT provider, model, cooldown_until_ts, retry_after_s, source_machine, updated_at "
            "FROM cooldowns ORDER BY updated_at DESC LIMIT ?",
            (int(limit),),
        )
        return cur.fetchall()


# ===========================================================================
# 2. Per-(provider,model) concurrency gate (in-process, thread-safe)
# ===========================================================================
@dataclass
class _Slot:
    max_concurrent: int
    in_flight: int = 0
    min_interval_ms: float = 0.0
    last_start_ts: float = 0.0


class ConcurrencyGate:
    """Bounds concurrent in-flight requests per (provider, model) in THIS process.

    Policy (config/concurrency_policy.yaml):
        default:
          max_concurrent: 2
          min_interval_ms: 0
        providers:
          groq:      { max_concurrent: 3 }
          cerebras:  { max_concurrent: 5 }
          freetheai: { max_concurrent: 1, min_interval_ms: 250 }

    A per-(provider,model) override may also be given under `models:`:
        models:
          "groq/llama-3.3-70b-versatile": { max_concurrent: 2 }

    NOTE: keys are (provider, model). Two different models on the same provider
    get INDEPENDENT slots and are NOT throttled together (spec requirement).
    """

    def __init__(self, config_path: Optional[str] = None):
        self._lock = threading.Lock()
        self._slots: Dict[Tuple[str, str], _Slot] = {}
        self._default_max = 2
        self._default_interval_ms = 0.0
        self._provider_cfg: Dict[str, dict] = {}
        self._model_cfg: Dict[str, dict] = {}
        if config_path is None:
            config_path = str(
                Path(__file__).resolve().parents[2] / "config" / "concurrency_policy.yaml"
            )
        self._load(config_path)

    def _load(self, config_path: str) -> None:
        if yaml is None:
            return
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return  # ponytail: missing config -> defaults; gate stays permissive-ish
        except Exception:
            return
        default = data.get("default", {}) or {}
        self._default_max = int(default.get("max_concurrent", 2) or 2)
        self._default_interval_ms = float(default.get("min_interval_ms", 0) or 0)
        self._provider_cfg = {
            k.lower(): v for k, v in (data.get("providers", {}) or {}).items() if isinstance(v, dict)
        }
        self._model_cfg = {
            k.lower(): v for k, v in (data.get("models", {}) or {}).items() if isinstance(v, dict)
        }

    def _resolve(self, provider: str, model: str) -> Tuple[int, float]:
        """Resolve (max_concurrent, min_interval_ms): model override > provider > default."""
        prov = provider.lower()
        full = f"{prov}/{model}".lower()
        mcfg = self._model_cfg.get(full, {})
        pcfg = self._provider_cfg.get(prov, {})
        max_c = int(mcfg.get("max_concurrent", pcfg.get("max_concurrent", self._default_max)))
        interval = float(
            mcfg.get("min_interval_ms", pcfg.get("min_interval_ms", self._default_interval_ms))
        )
        return max(1, max_c), max(0.0, interval)

    def _slot(self, provider: str, model: str) -> _Slot:
        key = (provider.lower(), model)
        slot = self._slots.get(key)
        if slot is None:
            max_c, interval = self._resolve(provider, model)
            slot = _Slot(max_concurrent=max_c, min_interval_ms=interval)
            self._slots[key] = slot
        return slot

    def try_acquire(
        self, provider: str, model: str, *, now: Optional[float] = None
    ) -> bool:
        """Non-blocking. True if a slot was taken; False if full or min-interval not met.

        On False the caller should fall back to the next provider immediately.
        Each True MUST be paired with exactly one release().
        """
        now = time.time() if now is None else now
        with self._lock:
            slot = self._slot(provider, model)
            if slot.in_flight >= slot.max_concurrent:
                return False
            if slot.min_interval_ms > 0:
                elapsed_ms = (now - slot.last_start_ts) * 1000.0
                if slot.last_start_ts > 0 and elapsed_ms < slot.min_interval_ms:
                    return False
            slot.in_flight += 1
            slot.last_start_ts = now
            return True

    def release(self, provider: str, model: str) -> None:
        with self._lock:
            slot = self._slots.get((provider.lower(), model))
            if slot and slot.in_flight > 0:
                slot.in_flight -= 1

    @contextmanager
    def slot(self, provider: str, model: str):
        """Context manager: yields True if acquired (and auto-releases), else False.

            with gate.slot(p, m) as ok:
                if not ok:
                    continue  # fall back
                ... do the request ...
        """
        acquired = self.try_acquire(provider, model)
        try:
            yield acquired
        finally:
            if acquired:
                self.release(provider, model)

    def snapshot(self) -> dict:
        """Read-only view of gate state for observability.

        Returns:
            {
              "active_pairs":    [(provider, model, in_flight, max), ...],
              "total_in_flight": int,
              "tracked_pairs":   int,
            }
        """
        with self._lock:
            pairs = []
            for key, slot in self._slots.items():
                if slot.in_flight > 0:
                    pairs.append((key[0], key[1], slot.in_flight, slot.max_concurrent))
            return {
                "active_pairs": pairs,
                "total_in_flight": sum(p[2] for p in pairs),
                "tracked_pairs": len(self._slots),
            }


# ---- runnable self-test (no framework; `python distributed_gate.py`) -------
def _demo() -> None:
    import tempfile

    now = 1_000_000.0

    # --- SharedCooldownStore ---
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = SharedCooldownStore(db_path=tmp.name, machine_id="test-host")

    assert store.is_cooling_down("groq", "m1", now=now) is False
    store.start_cooldown("groq", "m1", retry_after_s=60, now=now)
    assert store.is_cooling_down("groq", "m1", now=now) is True
    assert abs(store.cooldown_remaining("groq", "m1", now=now) - 60) < 1e-6
    # different model on same provider is independent
    assert store.is_cooling_down("groq", "m2", now=now) is False
    # expires
    assert store.is_cooling_down("groq", "m1", now=now + 61) is False
    # UPSERT keeps the LATER deadline (never shortens an active cooldown)
    store.start_cooldown("groq", "m1", retry_after_s=120, now=now)
    store.start_cooldown("groq", "m1", retry_after_s=10, now=now)
    assert abs(store.cooldown_remaining("groq", "m1", now=now) - 120) < 1e-6
    # purge removes long-expired rows
    store.start_cooldown("x", "y", retry_after_s=1, now=now)
    removed = store.purge_expired(now=now + COOLDOWN_ROW_TTL_S + 5)
    assert removed >= 1
    import os as _os
    _os.unlink(tmp.name)

    # --- ConcurrencyGate (config-less => defaults: max=2, interval=0) ---
    gate = ConcurrencyGate(config_path="/nonexistent/policy.yaml")
    assert gate.try_acquire("p", "m", now=now) is True   # 1/2
    assert gate.try_acquire("p", "m", now=now) is True   # 2/2
    assert gate.try_acquire("p", "m", now=now) is False  # full -> fall back
    # a different model on the same provider has its own slots
    assert gate.try_acquire("p", "other", now=now) is True
    gate.release("p", "m")
    assert gate.try_acquire("p", "m", now=now) is True   # freed -> 2/2 again
    # release is idempotent-safe (never goes negative)
    gate.release("p", "m"); gate.release("p", "m"); gate.release("p", "m")
    gate.release("p", "m"); gate.release("p", "m")
    # context manager auto-releases
    with gate.slot("z", "m") as ok:
        assert ok is True
        with gate.slot("z", "m") as ok2:
            assert ok2 is True
            with gate.slot("z", "m") as ok3:
                assert ok3 is False  # 3rd is over default max of 2
    assert gate.try_acquire("z", "m", now=now) is True  # all released after block

    # --- min_interval gating via injected config ---
    g2 = ConcurrencyGate(config_path="/nonexistent")
    g2._provider_cfg = {"slow": {"max_concurrent": 5, "min_interval_ms": 200}}
    g2._slots.clear()
    assert g2.try_acquire("slow", "m", now=now) is True
    g2.release("slow", "m")
    # too soon (100ms < 200ms) -> blocked
    assert g2.try_acquire("slow", "m", now=now + 0.100) is False
    # enough time passed -> allowed
    assert g2.try_acquire("slow", "m", now=now + 0.250) is True

    print("distributed_gate self-test: OK")


if __name__ == "__main__":
    _demo()
