"""
Model Management API

Endpoints for:
- Getting current model status
- Switching models via API
- Intent-based auto-routing info
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

from .intent_detector import (
    detect_intent,
    extract_model_command,
    validate_model_name,
    should_upgrade_model,
    AVAILABLE_MODELS,
    IntentResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/model", tags=["model-management"])


class ModelSwitchRequest(BaseModel):
    model: str
    session_id: Optional[str] = None


class ModelStatusResponse(BaseModel):
    current_model: str
    available_models: List[str]
    auto_routing_enabled: bool


class IntentDetectRequest(BaseModel):
    messages: List[Dict[str, Any]]
    current_model: Optional[str] = None


class IntentDetectResponse(BaseModel):
    intent: Optional[str]
    confidence: float
    suggested_model: str
    detected_keywords: List[str]
    reasoning: str


# In-memory session model tracking (for bot sessions)
# Maps session_id -> current_model
_session_models: Dict[str, str] = {}


def get_session_model(
    session_id: Optional[str], default: str = "uncensored-chat"
) -> str:
    """Get the current model for a session."""
    if session_id and session_id in _session_models:
        return _session_models[session_id]
    return default


def set_session_model(session_id: str, model: str):
    """Set the model for a session."""
    _session_models[session_id] = model


@router.get("/status")
async def get_model_status(request: Request):
    """Get current model status and available models."""
    return ModelStatusResponse(
        current_model="uncensored-chat",  # Default
        available_models=AVAILABLE_MODELS,
        auto_routing_enabled=True,
    )


@router.get("/session/{session_id}")
async def get_session_model_status(session_id: str):
    """Get the current model for a specific session (e.g., Zeroclaw bot)."""
    model = get_session_model(session_id)
    return {
        "session_id": session_id,
        "current_model": model,
        "available_models": AVAILABLE_MODELS,
    }


@router.post("/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model for a session."""
    is_valid, canonical_name = validate_model_name(request.model)

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'. Available: {AVAILABLE_MODELS}",
        )

    session_id = request.session_id or "default"
    set_session_model(session_id, canonical_name)

    logger.info(f"Session '{session_id}' switched to model: {canonical_name}")

    return {
        "success": True,
        "session_id": session_id,
        "previous_model": get_session_model(session_id) if request.session_id else None,
        "current_model": canonical_name,
        "message": f"Model switched to {canonical_name}",
    }


@router.post("/detect-intent")
async def detect_message_intent(request: IntentDetectRequest):
    """Detect intent from messages and get model recommendation."""
    result = detect_intent(
        messages=request.messages,
        current_model=request.current_model,
    )

    return IntentDetectResponse(
        intent=result.intent.value if result.intent else None,
        confidence=result.confidence,
        suggested_model=result.suggested_model,
        detected_keywords=result.detected_keywords,
        reasoning=result.reasoning,
    )


@router.post("/parse-command")
async def parse_model_command(request: Request):
    """Parse a message for model switch commands (e.g., /model coding-elite)."""
    body = await request.json()
    message = body.get("message", "")

    model_name, remaining_message = extract_model_command(message)

    if model_name:
        is_valid, canonical_name = validate_model_name(model_name)
        if is_valid:
            return {
                "command_found": True,
                "requested_model": model_name,
                "canonical_model": canonical_name,
                "remaining_message": remaining_message,
                "valid": True,
            }
        else:
            return {
                "command_found": True,
                "requested_model": model_name,
                "valid": False,
                "error": f"Invalid model. Available: {AVAILABLE_MODELS}",
                "remaining_message": message,
            }

    return {
        "command_found": False,
        "remaining_message": message,
    }


@router.get("/list")
async def list_available_models():
    """List all available virtual models."""
    return {
        "models": AVAILABLE_MODELS,
        "descriptions": {
            "uncensored-chat": "Uncensored models with tool support - best for RP",
            "coding-elite": "Best agentic coding models - complex tasks",
            "coding-smart": "Balanced coding - quality + speed",
            "coding-fast": "Fastest coding models - quick tasks",
            "chat-elite": "Most intelligent chat - deep analysis",
            "chat-smart": "Smart chat - good balance",
            "chat-fast": "Fastest chat - simple queries",
            "chat-rp": "Roleplay optimized - NSFW capable",
        },
    }
