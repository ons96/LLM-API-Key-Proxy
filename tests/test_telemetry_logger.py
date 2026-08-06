"""Tests for src/proxy_app/telemetry/logger.py — task-board #484.

Covers the streaming-telemetry fixes:
1. A streamed request records total_ms > ttft_ms (true end-to-end latency),
   instead of collapsing total to the first-token time.
2. The final streamed chunk (finish_reason) marks the true stream end.

Run: python -m pytest tests/test_telemetry_logger.py -v
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from proxy_app.telemetry.logger import (  # noqa: E402
    TelemetryLogger,
    _stream_finish_reason,
)


def _dt(ms: float) -> datetime:
    """Convert an epoch-ms timestamp to the datetime LiteLLM callbacks receive."""
    return datetime.fromtimestamp(ms / 1000.0)


def _num_rows(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM llm_events").fetchone()[0]
    finally:
        conn.close()


class TestStreamFinishReason(unittest.TestCase):
    """Unit tests for the chunk finish_reason parser (dict + object forms)."""

    def test_detects_finish_reason_in_dict_chunk(self):
        chunk = {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
        self.assertEqual(_stream_finish_reason(chunk), "stop")

    def test_returns_none_for_mid_stream_chunk(self):
        chunk = {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}
        self.assertIsNone(_stream_finish_reason(chunk))

    def test_detects_finish_reason_in_object_chunk(self):
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta={"content": ""}, finish_reason="stop")]
        )
        self.assertEqual(_stream_finish_reason(chunk), "stop")

    def test_handles_nested_finish_reason_dict(self):
        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta={"content": ""}, finish_reason={"reason": "length"}
                )
            ]
        )
        self.assertEqual(_stream_finish_reason(chunk), "length")

    def test_handles_malformed_chunk(self):
        self.assertIsNone(_stream_finish_reason(None))
        self.assertIsNone(_stream_finish_reason({"choices": []}))
        self.assertIsNone(_stream_finish_reason("not a chunk"))


class TestStreamLatencyFix(unittest.TestCase):
    """AC (#484 bug 1): a simulated stream records total_ms > ttft_ms."""

    def _enqueue_and_read(self, logger, kwargs, response_obj, start_ms, end_ms):
        """Run _enqueue in an event loop, return the queued event tuple."""

        async def _run():
            await logger._enqueue(
                kwargs, response_obj, _dt(start_ms), _dt(end_ms),
                status="ok", error=None,
            )
            return logger._queue.get_nowait()

        return asyncio.run(_run())

    def test_stream_total_ms_gt_ttft_ms(self):
        """A stream where the success-event end_time == ttft (the bug) must
        still record total_ms from the true stream-end snapshot."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "telemetry.db"
            logger = TelemetryLogger(db_path=str(db))

            rid = "req-stream-484"
            start_ms = 1_000_000.0
            # Simulate the buggy success event: end_time == first-token time.
            buggy_end_ms = start_ms + 100.0

            kwargs = {
                "litellm_call_id": rid,
                "stream": True,
                "model": "groq/llama-3.3-70b",
                "litellm_params": {"metadata": {"caller": "some-agent"}},
            }
            response_obj = SimpleNamespace(
                model="llama-3.3-70b-versatile",
                usage={"prompt_tokens": 5, "completion_tokens": 200},
                _hidden_params={},
            )

            logger._start_times[rid] = start_ms
            # ttft_ms is used as a relative duration (ms since start).
            logger._first_token_times[rid] = 100.0  # TTFT = 100 ms
            # #484 fix: stream-end is an absolute epoch-ms snapshot; the final
            # chunk arrives 400 ms after TTFT (500 ms after start).
            logger._stream_end_times[rid] = start_ms + 500.0

            ev = self._enqueue_and_read(logger, kwargs, response_obj, start_ms, buggy_end_ms)

            ttft_ms = ev[9]
            total_ms = ev[10]
            self.assertGreater(total_ms, ttft_ms)
            self.assertEqual(ttft_ms, 100.0)
            self.assertEqual(total_ms, 500.0)
            # Row must actually persist to the DB.
            logger._sync_bulk_insert([ev])
            self.assertEqual(_num_rows(db), 1)
            conn = sqlite3.connect(str(db))
            row = conn.execute("SELECT ttft_ms, total_ms FROM llm_events").fetchone()
            conn.close()
            self.assertGreater(row[1], row[0])

    def test_non_stream_uses_success_end_time(self):
        """Non-stream requests are unaffected — they keep using end_time."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "telemetry.db"
            logger = TelemetryLogger(db_path=str(db))

            rid = "req-nonstream"
            start_ms = 2_000_000.0
            end_ms = start_ms + 700.0

            kwargs = {
                "litellm_call_id": rid,
                "stream": False,
                "model": "groq/llama-3.3-70b",
                "litellm_params": {"metadata": {}},
            }
            response_obj = SimpleNamespace(
                model="llama-3.3-70b-versatile",
                usage={"prompt_tokens": 10, "completion_tokens": 50},
                _hidden_params={},
            )

            logger._start_times[rid] = start_ms
            logger._first_token_times[rid] = 200.0  # TTFT = 200 ms (relative)

            ev = self._enqueue_and_read(logger, kwargs, response_obj, start_ms, end_ms)
            self.assertEqual(ev[9], 200.0)   # ttft_ms
            self.assertEqual(ev[10], 700.0)  # total_ms == end - start

    def test_stream_event_records_end_time_on_final_chunk(self):
        """AC: the final streamed chunk (finish_reason='stop') records a true
        stream-end timestamp AND the first-token timestamp."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "telemetry.db"
            logger = TelemetryLogger(db_path=str(db))

            rid = "req-final-chunk"
            kwargs = {
                "litellm_call_id": rid,
                "stream": True,
                "model": "groq/llama-3.3-70b",
            }
            logger.log_pre_api_call("groq/llama-3.3-70b", [], kwargs)
            self.assertEqual(logger._first_token_times[rid], 0.0)

            async def _run():
                # Mid-stream chunk: no finish_reason, does not mark end.
                mid = {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
                await logger.async_log_stream_event(kwargs, mid, None, None)
                self.assertNotIn(rid, logger._stream_end_times)
                # Final chunk: finish_reason='stop' marks the true end.
                final = {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
                await logger.async_log_stream_event(kwargs, final, None, None)
                self.assertIn(rid, logger._stream_end_times)

            asyncio.run(_run())
            self.assertGreater(logger._first_token_times[rid], 0.0)
            self.assertGreater(logger._stream_end_times[rid], 0.0)


if __name__ == "__main__":
    unittest.main()

