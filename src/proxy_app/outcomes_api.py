#!/usr/bin/env python3
"""
Task Outcome API.

Endpoints for the quota-aware routing brain feedback loop:
  POST /api/task-outcome         — harnesses/scripts report final task results
  GET  /api/task-outcomes/export — the brain pulls labeled outcomes nightly

Auth mirrors main.verify_api_key semantics: when PROXY_API_KEY is unset the
endpoints are open (local dev); otherwise a Bearer key is required.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from proxy_app import task_telemetry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["task-outcomes"])


async def verify_outcome_api_key(request: Request) -> None:
    proxy_api_key = os.environ.get("PROXY_API_KEY", "").strip()
    if not proxy_api_key:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {proxy_api_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"expected integer, got: {value!r}")
    if iv < 0:
        raise HTTPException(status_code=422, detail="expected non-negative integer")
    return iv


@router.post("/task-outcome")
async def post_task_outcome(
    payload: Dict[str, Any],
    _: None = Depends(verify_outcome_api_key),
) -> Dict[str, Any]:
    """Report a final task outcome.

    Body: {task_class, success, model_id?, provider?, task_id?,
           total_tokens?, turns?, notes?, observed_at?}
    task_class + success are required; model_id is strongly recommended
    (outcomes without a model cannot update P(success)).
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="expected a JSON object")

    task_class = str(payload.get("task_class") or "").strip()
    if not task_class:
        raise HTTPException(status_code=422, detail="task_class is required")

    success = payload.get("success")
    if not isinstance(success, bool):
        raise HTTPException(status_code=422, detail="success must be a boolean")

    model_id = payload.get("model_id")
    if model_id is not None and not isinstance(model_id, str):
        raise HTTPException(status_code=422, detail="model_id must be a string")

    try:
        outcome_id = task_telemetry.record_outcome(
            task_class=task_class,
            success=success,
            model_id=model_id,
            provider=payload.get("provider") if isinstance(payload.get("provider"), str) else None,
            task_id=payload.get("task_id") if isinstance(payload.get("task_id"), str) else None,
            total_tokens=_coerce_int(payload.get("total_tokens")),
            turns=_coerce_int(payload.get("turns")),
            source=str(payload.get("source") or "api"),
            notes=payload.get("notes") if isinstance(payload.get("notes"), str) else None,
            observed_at=payload.get("observed_at") if isinstance(payload.get("observed_at"), str) else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"ok": True, "id": outcome_id}


@router.get("/task-outcomes/export")
async def export_task_outcomes(
    since: Optional[str] = None,
    limit: int = 50000,
    include_requests: bool = False,
    _: None = Depends(verify_outcome_api_key),
) -> Dict[str, Any]:
    """Export labeled outcomes (and optionally the per-request task log).

    Query: ?since=ISO-8601&limit=N&include_requests=true
    """
    payload: Dict[str, Any] = {
        "outcomes": task_telemetry.export_outcomes(since=since, limit=limit),
    }
    if include_requests:
        payload["gateway_requests"] = task_telemetry.export_gateway_requests(
            since=since, limit=limit
        )
    return payload
