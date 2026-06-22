"""TelemetryLogger: LiteLLM CustomLogger capturing TTFT/TPS into SQLite WAL.

Zero req-path latency: async hooks enqueue to asyncio.Queue; single _writer_loop
drains in batches of 50. DB on tmpfs (/dev/shm/telemetry.db) for zero disk I/O.

Registration (in main.py after `import litellm`):
    from proxy_app.telemetry import TelemetryLogger
    litellm.callbacks = [TelemetryLogger()]

TPS formulas:
  streaming:    completion_tokens / ((total_ms - ttft_ms) / 1000)
  non-stream:   completion_tokens / (total_ms / 1000)
"""
import asyncio
import datetime
import logging
import os
import sqlite3
import time
import traceback
from typing import Optional

try:
    import litellm  # noqa: F401
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # allow import without litellm (e.g. schema inspection)
    class CustomLogger:  # type: ignore
        async def async_log_success_event(self, *a, **kw): pass
        async def async_log_failure_event(self, *a, **kw): pass
        async def async_log_stream_event(self, *a, **kw): pass
        def log_pre_api_call(self, *a, **kw): pass
        def log_post_api_call(self, *a, **kw): pass

log = logging.getLogger("telemetry")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [telemetry] %(levelname)s %(message)s"))
    log.addHandler(h)

DB_PATH = os.environ.get("TELEMETRY_DB_PATH", "/dev/shm/telemetry.db")
QUEUE_MAX = 10000
BATCH_SIZE = 50
FLUSH_INTERVAL_S = 2.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    ts_start REAL,
    ts_end REAL,
    ts_first_token REAL,
    model TEXT,
    provider TEXT,
    stream INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    ttft_ms REAL,
    total_ms REAL,
    tps REAL,
    cost_usd REAL,
    caller TEXT,
    agent_session TEXT,
    status TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_ts_start ON llm_events(ts_start);
CREATE INDEX IF NOT EXISTS idx_model ON llm_events(model);
CREATE INDEX IF NOT EXISTS idx_provider ON llm_events(provider);
"""


def init_db(db_path: str = DB_PATH) -> None:
    """Create schema + apply PRAGMAs. Call once at startup."""
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA wal_autocheckpoint=1000")
        try:
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        except sqlite3.OperationalError:
            pass  # auto_vacuum must be set before any table creation
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    log.info("telemetry DB initialized at %s", db_path)


def _now_ms() -> float:
    return time.time() * 1000.0


def _to_ms(t) -> float:
    """Convert datetime | float | int | None to epoch milliseconds."""
    if t is None:
        return _now_ms()
    if isinstance(t, datetime.datetime):
        return t.timestamp() * 1000.0
    if isinstance(t, (int, float)):
        return float(t) * 1000.0
    return _now_ms()


class TelemetryLogger(CustomLogger):
    """LiteLLM callback: capture TTFT/TPS, enqueue to async writer queue."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self._writer_task: Optional[asyncio.Task] = None
        self._start_times: dict[str, float] = {}
        self._first_token_times: dict[str, float] = {}
        init_db(db_path)
        log.info("TelemetryLogger initialized (db=%s queue_max=%d)", db_path, QUEUE_MAX)

    def _ensure_writer(self) -> None:
        if self._writer_task is None or self._writer_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._writer_task = loop.create_task(self._writer_loop())
            except RuntimeError:
                pass  # no running loop yet; will start on first async hook

    # --- pre/post call hooks (sync, fast) ---
    def log_pre_api_call(self, model, messages, kwargs):
        try:
            rid = kwargs.get("litellm_call_id") or str(id(kwargs))
            self._start_times[rid] = _now_ms()
            stream = kwargs.get("stream", False)
            if stream and rid not in self._first_token_times:
                self._first_token_times[rid] = 0.0
        except Exception:
            pass  # never raise in callback

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        try:
            rid = kwargs.get("litellm_call_id") or str(id(kwargs))
            start = self._start_times.get(rid, _to_ms(start_time))
            end = _to_ms(end_time)
            total = end - start
            ttft = self._first_token_times.get(rid, 0.0)
            if ttft == 0.0:
                ttft = total
        except Exception:
            pass

    # --- streaming first-token capture (async) ---
    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        try:
            rid = kwargs.get("litellm_call_id") or str(id(kwargs))
            if rid in self._first_token_times and self._first_token_times[rid] == 0.0:
                self._first_token_times[rid] = _now_ms()
        except Exception:
            pass

    # --- async success hook (non-blocking) ---
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self._ensure_writer()
            await self._enqueue(kwargs, response_obj, start_time, end_time, status="ok", error=None)
        except Exception:
            log.error("async_log_success_event failed: %s", traceback.format_exc())

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self._ensure_writer()
            await self._enqueue(kwargs, response_obj, start_time, end_time, status="error",
                                error=str(getattr(response_obj, "message", "unknown")))
        except Exception:
            log.error("async_log_failure_event failed: %s", traceback.format_exc())

        # ponytail: penalty record is best-effort; never fail the call path on it.
        try:
            from ..penalty_store import PenaltyStore, classify_failure
            model_id = kwargs.get("model", "unknown")
            provider, model = _split_provider_model(model_id)
            status_code = getattr(response_obj, "status_code", None)
            error_message = getattr(response_obj, "message", None) or str(response_obj)
            exception_type = type(response_obj).__name__
            failure_type = classify_failure(
                status_code=status_code,
                error_message=error_message,
                exception_type=exception_type,
            )
            await PenaltyStore.get().arecord_failure(provider, model, failure_type)
            log.info("penalty recorded: provider=%s model=%s type=%s status=%s",
                      provider, model, failure_type, status_code)
        except Exception:
            log.debug("penalty record skipped: %s", traceback.format_exc())

    async def _enqueue(self, kwargs, response_obj, start_time, end_time, status, error):
        rid = kwargs.get("litellm_call_id") or str(id(kwargs))
        start_ms = self._start_times.pop(rid, _to_ms(start_time))
        end_ms = _to_ms(end_time)
        total_ms = end_ms - start_ms
        ttft_ms = self._first_token_times.pop(rid, total_ms)
        stream = 1 if kwargs.get("stream") else 0

        usage = getattr(response_obj, "usage", None) or {}
        prompt_tokens = getattr(usage, "prompt_tokens", None) or (usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0)
        completion_tokens = getattr(usage, "completion_tokens", None) or (usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0)

        if stream and completion_tokens and (total_ms - ttft_ms) > 0:
            tps = completion_tokens / ((total_ms - ttft_ms) / 1000.0)
        elif completion_tokens and total_ms > 0:
            tps = completion_tokens / (total_ms / 1000.0)
        else:
            tps = 0.0

        model = kwargs.get("model", "unknown")
        provider = getattr(response_obj, "model", model)
        litellm_params = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata", {}) if isinstance(litellm_params, dict) else {}
        caller = metadata.get("caller", "unknown") if isinstance(metadata, dict) else "unknown"
        agent_session = metadata.get("agent_session", "") if isinstance(metadata, dict) else ""
        cost = getattr(response_obj, "_hidden_params", {}).get("response_cost", 0.0) if hasattr(response_obj, "_hidden_params") else 0.0

        event = (
            rid, start_ms / 1000.0, end_ms / 1000.0,
            (start_ms + ttft_ms) / 1000.0 if ttft_ms else None,
            model, str(provider), stream,
            int(prompt_tokens or 0), int(completion_tokens or 0),
            float(ttft_ms), float(total_ms), float(tps),
            float(cost or 0.0), caller, agent_session, status, error,
        )

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
                log.warning("telemetry queue full, dropped oldest event")
            except Exception:
                pass

    async def _writer_loop(self) -> None:
        """Drain queue in batches, bulk insert single transaction."""
        log.info("telemetry writer loop started")
        while True:
            try:
                batch = []
                try:
                    first = await asyncio.wait_for(self._queue.get(), timeout=FLUSH_INTERVAL_S)
                    batch.append(first)
                except asyncio.TimeoutError:
                    continue

                while len(batch) < BATCH_SIZE and not self._queue.empty():
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                if not batch:
                    continue

                await self._bulk_insert(batch)
            except Exception:
                log.error("writer_loop error: %s", traceback.format_exc())
                await asyncio.sleep(1.0)

    async def _bulk_insert(self, batch: list) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_bulk_insert, batch)

    def _sync_bulk_insert(self, batch: list) -> None:
        conn = sqlite3.connect(self.db_path, timeout=5)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executemany(
                """INSERT INTO llm_events
                   (request_id, ts_start, ts_end, ts_first_token, model, provider, stream,
                    prompt_tokens, completion_tokens, ttft_ms, total_ms, tps, cost_usd,
                    caller, agent_session, status, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                batch,
            )
            conn.commit()
        except Exception:
            log.error("bulk insert failed: %s", traceback.format_exc())
        finally:
            conn.close()


def _split_provider_model(model_id: str):
    """Split 'groq/llama-3.3-70b' -> ('groq', 'llama-3.3-70b').
    Plain 'llama-3.3-70b' -> ('unknown', 'llama-3.3-70b')."""
    if "/" in model_id:
        provider, _, model = model_id.partition("/")
        return provider, model
    return "unknown", model_id
