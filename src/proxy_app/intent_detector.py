"""
Intent Detection Module for Auto Model Routing

Detects user intent from message content and routes to appropriate model:
- UNSENSORED: RP/creative/uncensored content
- CODING: Programming/technical tasks
- COMPLEX: Deep reasoning/analysis
- SIMPLE: Fast queries

Also provides quality-based retry logic.
"""

import re
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MessageIntent(Enum):
    """Classification of user message intent."""

    UNSENSORED = "uncensored-chat"  # RP, creative, uncensored content
    CODING_COMPLEX = "coding-elite"  # Complex coding/agentic tasks
    CODING_FAST = "coding-fast"  # Quick coding tasks
    COMPLEX_REASONING = "chat-smart"  # Deep analysis/reasoning
    FAST_CHAT = "chat-fast"  # Simple queries
    ROLEPLAY = "uncensored-chat"  # Explicit roleplay markers
    UNKNOWN = None  # No clear intent


@dataclass
class IntentResult:
    """Result of intent detection."""

    intent: MessageIntent
    confidence: float
    detected_keywords: List[str]
    suggested_model: str
    reasoning: str


# Intent detection patterns
UNSENSORED_PATTERNS = [
    # Roleplay markers
    r"\*[^*]+\*",  # *actions*
    r"\([^)]+\)",  # (actions)
    r"--[^-]+--",  # --narration--
    r"{{[^}]+}}",  # {{narration}}
    # Explicit RP/uncensored keywords
    r"\b(roleplay|rp|in character|IC|OOC|out of character)\b",
    r"\b(continue the story|continue roleplay|as [^,]+,|playing as)\b",
    r"\b(uncensored|no filter|no restrictions|explicit|nsfw)\b",
    r"\b(narrate|narration|scene|setting)\b",
    r"\b(character|persona|protagonist|antagonist)\b",
    # Creative writing markers
    r"\b(once upon a time|in a land|the story begins)\b",
    r"\b(write a story|tell me a story|creative writing)\b",
    # Emotional/intimate content markers
    r"\b(passionate|intimate|sensual|romantic|affectionate)\b",
    r"\b(whisper(ed|ing)?|gaze|touch|embrace)\b",
]

CODING_PATTERNS = [
    # Programming keywords
    r"\b(function|class|method|variable|const|let|var)\b",
    r"\b(def |async def|import |from |return |if __name__)\b",
    r"\b(python|javascript|typescript|java|c\+\+|rust|go)\b",
    r"\b(debug|error|bug|fix|implement|refactor)\b",
    r"\b(code|script|program|algorithm|syntax)\b",
    r"\b(API|endpoint|request|response|JSON|HTTP)\b",
    r"\b(git|commit|push|pull|merge|branch)\b",
    r"\b(test|unit test|integration|pytest|jest)\b",
    r"\b(database|SQL|query|table|schema)\b",
    r"\b(react|vue|angular|node|express|fastapi|django)\b",
    # Code markers
    r"```[a-z]*\n",  # Code blocks
    r"`[^`]+`",  # Inline code
    r"\b(indentation|indent|bracket|brace)\b",
    # Complex coding triggers
    r"\b(architecture|design pattern|system design)\b",
    r"\b(optimize|performance|scalability)\b",
    r"\b(security|vulnerability|authentication)\b",
]

COMPLEX_REASONING_PATTERNS = [
    # Analysis keywords
    r"\b(analyze|analysis|evaluate|assessment)\b",
    r"\b(compare|contrast|difference|similarity)\b",
    r"\b(pros and cons|advantages|disadvantages)\b",
    r"\b(implications|consequences|impact)\b",
    # Deep thinking
    r"\b(why|how come|explain in detail|elaborate)\b",
    r"\b(philosophy|ethical|moral|implication)\b",
    r"\b(theory|concept|hypothesis|premise)\b",
    r"\b(consider|take into account|factor in)\b",
    # Research/exploration
    r"\b(research|investigate|explore|examine)\b",
    r"\b(comprehensive|thorough|in-depth|detailed)\b",
]

FAST_CHAT_PATTERNS = [
    # Simple queries
    r"^(hi|hello|hey|yo|sup)\s*[!.]?$",
    r"^(thanks|thank you|thx)\s*[!.]?$",
    r"^(ok|okay|k|sure|yes|no)\s*[!.]?$",
    r"^(bye|goodbye|see ya)\s*[!.]?$",
    # Short questions
    r"^(what is|who is|where is|when is|how (do|to))\s+\w+\s*[?]?$",
    # Quick tasks
    r"\b(quick(ly)?|fast|simple|short|brief)\b.*\b(question|answer|response)\b",
]


def detect_intent(
    messages: List[Dict[str, Any]],
    current_model: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> IntentResult:
    """
    Detect intent from a list of chat messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        current_model: Currently selected model (for context)
        context: Additional context (system prompt, etc.)

    Returns:
        IntentResult with detected intent and suggested model
    """
    # Combine all message content for analysis
    all_content = ""
    for msg in messages:
        if isinstance(msg.get("content"), str):
            all_content += msg["content"] + " "
        elif isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    all_content += part.get("text", "") + " "

    # Also check system prompt for hints
    system_content = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
            if isinstance(system_content, str):
                break

    detected_keywords = []
    confidence_scores = {}

    # Check for uncensored/RP intent
    unsensored_matches = []
    for pattern in UNSENSORED_PATTERNS:
        matches = re.findall(pattern, all_content, re.IGNORECASE)
        unsensored_matches.extend(matches)

    if unsensored_matches:
        detected_keywords.extend(unsensored_matches[:5])  # Limit to 5
        # Higher confidence for multiple matches or strong markers
        confidence_scores[MessageIntent.UNSENSORED] = min(
            0.9, 0.5 + len(unsensored_matches) * 0.1
        )

    # Check for coding intent
    coding_matches = []
    for pattern in CODING_PATTERNS:
        matches = re.findall(pattern, all_content, re.IGNORECASE)
        coding_matches.extend(matches)

    if coding_matches:
        detected_keywords.extend(coding_matches[:5])
        # Complex coding vs fast coding
        has_complex_markers = any(
            re.search(p, all_content, re.IGNORECASE)
            for p in [r"\b(architecture|design|system|optimize|security)\b", r"```"]
        )
        if has_complex_markers:
            confidence_scores[MessageIntent.CODING_COMPLEX] = min(
                0.9, 0.6 + len(coding_matches) * 0.05
            )
        else:
            confidence_scores[MessageIntent.CODING_FAST] = min(
                0.85, 0.5 + len(coding_matches) * 0.05
            )

    # Check for complex reasoning
    reasoning_matches = []
    for pattern in COMPLEX_REASONING_PATTERNS:
        matches = re.findall(pattern, all_content, re.IGNORECASE)
        reasoning_matches.extend(matches)

    if reasoning_matches:
        detected_keywords.extend(reasoning_matches[:5])
        confidence_scores[MessageIntent.COMPLEX_REASONING] = min(
            0.85, 0.5 + len(reasoning_matches) * 0.1
        )

    # Check for fast chat (simple queries)
    fast_matches = []
    for pattern in FAST_CHAT_PATTERNS:
        matches = re.findall(pattern, all_content.strip(), re.IGNORECASE)
        fast_matches.extend(matches)

    # Fast chat only triggers if nothing else matched strongly
    if fast_matches and not any(v > 0.7 for v in confidence_scores.values()):
        confidence_scores[MessageIntent.FAST_CHAT] = min(
            0.7, 0.4 + len(fast_matches) * 0.15
        )

    # Determine final intent
    if not confidence_scores:
        # Default to current model or uncensored for RP bots
        if current_model:
            return IntentResult(
                intent=MessageIntent.UNKNOWN,
                confidence=0.0,
                detected_keywords=[],
                suggested_model=current_model,
                reasoning="No clear intent detected, using current model",
            )
        else:
            # Check system prompt for hints
            if any(
                kw in system_content.lower()
                for kw in ["roleplay", "rp", "uncensored", "character"]
            ):
                return IntentResult(
                    intent=MessageIntent.UNSENSORED,
                    confidence=0.6,
                    detected_keywords=["system_prompt_hint"],
                    suggested_model="uncensored-chat",
                    reasoning="System prompt suggests roleplay/uncensored content",
                )
            return IntentResult(
                intent=MessageIntent.FAST_CHAT,
                confidence=0.5,
                detected_keywords=[],
                suggested_model="chat-fast",
                reasoning="No clear intent, defaulting to fast chat",
            )

    # Get highest confidence intent
    best_intent = max(confidence_scores, key=confidence_scores.get)
    best_confidence = confidence_scores[best_intent]

    return IntentResult(
        intent=best_intent,
        confidence=best_confidence,
        detected_keywords=detected_keywords[:10],
        suggested_model=best_intent.value,
        reasoning=f"Detected {best_intent.name} with {best_confidence:.0%} confidence based on {len(detected_keywords)} keywords",
    )


def get_model_upgrade_path(current_model: str) -> List[str]:
    """
    Get upgrade path for quality-based retry.

    Returns list of models to try in order of increasing quality.
    """
    UPGRADE_PATHS = {
        "chat-fast": ["chat-smart", "chat-elite"],
        "chat-smart": ["chat-elite"],
        "coding-fast": ["coding-smart", "coding-elite"],
        "coding-smart": ["coding-elite"],
        "uncensored-chat": ["uncensored-chat"],  # No upgrade
        "chat-rp": ["chat-rp"],  # No upgrade
    }
    return UPGRADE_PATHS.get(current_model, [])


def should_upgrade_model(
    response_content: str, current_model: str, original_request: Dict[str, Any]
) -> Tuple[bool, Optional[str], str]:
    """
    Determine if response quality warrants an upgrade to a smarter model.

    Returns:
        Tuple of (should_upgrade, upgrade_model, reason)
    """
    # Don't upgrade if already at top tier
    if current_model in ["coding-elite", "chat-elite"]:
        return False, None, "Already at highest tier"

    # Check for empty/very short responses
    if not response_content or len(response_content.strip()) < 10:
        upgrade_path = get_model_upgrade_path(current_model)
        if upgrade_path:
            return True, upgrade_path[0], "Empty or very short response"

    # Check for refusal/hedge patterns
    refusal_patterns = [
        r"I (cannot|can't|won't|am unable to)",
        r"I (apologize|apologise|sorry)",
        r"(inappropriate|against my|policy|guidelines)",
        r"(I'm not able|I am not able)",
    ]

    for pattern in refusal_patterns:
        if re.search(pattern, response_content, re.IGNORECASE):
            # For uncensored content refusals, try uncensored-chat
            if current_model != "uncensored-chat":
                # Check if original request had RP markers
                messages = original_request.get("messages", [])
                intent = detect_intent(messages, current_model)
                if intent.intent == MessageIntent.UNSENSORED:
                    return (
                        True,
                        "uncensored-chat",
                        "Refusal detected, routing to uncensored model",
                    )

            upgrade_path = get_model_upgrade_path(current_model)
            if upgrade_path:
                return True, upgrade_path[0], "Refusal pattern detected"

    # Check for "I don't know" or low-confidence responses
    uncertainty_patterns = [
        r"I (don't|do not) (know|understand|have)",
        r"I'm (not sure|uncertain)",
        r"(unsure|uncertain|unclear)",
    ]

    for pattern in uncertainty_patterns:
        if re.search(pattern, response_content, re.IGNORECASE):
            upgrade_path = get_model_upgrade_path(current_model)
            if upgrade_path:
                return (
                    True,
                    upgrade_path[0],
                    "Uncertainty detected, upgrading for better reasoning",
                )

    return False, None, "Response quality acceptable"


def extract_model_command(message: str) -> Tuple[Optional[str], str]:
    """
    Check if message contains a model switch command.

    Returns:
        Tuple of (model_name if command found, remaining_message)
    """
    # Match patterns like /model coding-elite or !model chat-fast
    command_patterns = [
        r"^[/!](?:model|switch)\s+(\S+)",
        r"^[/!](?:use)\s+(\S+)",
    ]

    for pattern in command_patterns:
        match = re.match(pattern, message.strip(), re.IGNORECASE)
        if match:
            model_name = match.group(1).lower()
            remaining = message[match.end() :].strip()
            return model_name, remaining

    # Special shortcuts
    shortcuts = {
        "/uncensored": "uncensored-chat",
        "/coding": "coding-elite",
        "/fast": "chat-fast",
        "/smart": "chat-smart",
        "/rp": "chat-rp",
        "!uncensored": "uncensored-chat",
        "!coding": "coding-elite",
        "!fast": "chat-fast",
        "!smart": "chat-smart",
        "!rp": "chat-rp",
    }

    for shortcut, model in shortcuts.items():
        if message.strip().lower().startswith(shortcut):
            remaining = message[len(shortcut) :].strip()
            return model, remaining

    return None, message


# Available models for command validation
AVAILABLE_MODELS = [
    "uncensored-chat",
    "coding-elite",
    "coding-smart",
    "coding-fast",
    "chat-elite",
    "chat-smart",
    "chat-fast",
    "chat-rp",
]


def validate_model_name(model_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate model name and return canonical name if valid.

    Returns:
        Tuple of (is_valid, canonical_name)
    """
    model_lower = model_name.lower().strip()

    # Direct match
    if model_lower in AVAILABLE_MODELS:
        return True, model_lower

    # Aliases
    aliases = {
        "uncensored": "uncensored-chat",
        "coding": "coding-elite",
        "smart": "chat-smart",
        "fast": "chat-fast",
        "rp": "chat-rp",
        "elite": "coding-elite",
    }

    if model_lower in aliases:
        return True, aliases[model_lower]

    return False, None
