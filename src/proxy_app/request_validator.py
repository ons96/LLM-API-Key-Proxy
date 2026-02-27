"""
Request Validation Middleware

Phase 1.3 Security Hardening - Validates incoming OpenAI-compatible API requests
for required fields, message format, parameter constraints, and model naming conventions.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Set
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RequestValidator:
    """Validates OpenAI-compatible chat completion requests."""
    
    VALID_ROLES: Set[str] = {"system", "user", "assistant", "tool", "function"}
    MAX_MESSAGES = 1000
    MAX_CONTENT_LENGTH = 1000000  # 1MB limit for single message content
    
    def validate(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate request data and return error response dict if invalid.
        Returns None if valid.
        """
        errors: List[str] = []
        
        # Check required fields
        if "model" not in data:
            errors.append("Missing required field: model")
        elif not isinstance(data["model"], str):
            errors.append("Field 'model' must be a string")
        elif "/" not in data["model"]:
            errors.append("Model must be in format 'provider/model_name'")
        else:
            # Security: Check for path traversal or special characters in model name
            model = data["model"]
            if ".." in model or model.startswith("/") or model.endswith("/"):
                errors.append("Invalid model name format")
        
        if "messages" not in data:
            errors.append("Missing required field: messages")
        elif not isinstance(data["messages"], list):
            errors.append("Field 'messages' must be a list")
        elif not data["messages"]:
            errors.append("Messages list cannot be empty")
        elif len(data["messages"]) > self.MAX_MESSAGES:
            errors.append(f"Too many messages. Maximum is {self.MAX_MESSAGES}")
        else:
            msg_errors = self._validate_messages(data["messages"])
            errors.extend(msg_errors)
        
        # Validate optional parameters
        param_errors = self._validate_parameters(data)
        errors.extend(param_errors)
        
        if errors:
            return {
                "error": {
                    "message": "Invalid request",
                    "type": "invalid_request_error",
                    "details": errors
                }
            }
        return None
    
    def _validate_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        errors = []
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"Message at index {i} must be an object")
                continue
            
            # Check role
            if "role" not in msg:
                errors.append(f"Message at index {i} missing required field: role")
            elif msg["role"] not in self.VALID_ROLES:
                errors.append(
                    f"Invalid role '{msg['role']}' at index {i}. "
                    f"Must be one of: {', '.join(sorted(self.VALID_ROLES))}"
                )
            
            # Check content or tool_calls
            has_content = "content" in msg
            has_tool_calls = "tool_calls" in msg
            
            if not has_content and not has_tool_calls:
                errors.append(
                    f"Message at index {i} must have either 'content' or 'tool_calls'"
                )
            
            if has_content:
                content = msg["content"]
                if content is not None:
                    if isinstance(content, str):
                        if len(content) > self.MAX_CONTENT_LENGTH:
                            errors.append(
                                f"Message content at index {i} exceeds maximum length "
                                f"of {self.MAX_CONTENT_LENGTH} characters"
                            )
                    elif isinstance(content, list):
                        # Multimodal content validation
                        for j, item in enumerate(content):
                            if not isinstance(item, dict):
                                errors.append(
                                    f"Content item at index {j} in message {i} must be an object"
                                )
                            elif "type" not in item:
                                errors.append(
                                    f"Content item at index {j} in message {i} missing 'type'"
                                )
                    else:
                        errors.append(
                            f"Message content at index {i} must be a string, list, or null"
                        )
            
            # Validate tool_calls structure if present
            if has_tool_calls:
                if not isinstance(msg["tool_calls"], list):
                    errors.append(f"Field 'tool_calls' at message {i} must be a list")
                else:
                    for j, tool in enumerate(msg["tool_calls"]):
                        if not isinstance(tool, dict):
                            errors.append(f"Tool call at index {j} in message {i} must be an object")
                        elif "function" not in tool:
                            errors.append(f"Tool call at index {j} in message {i} missing 'function'")
        
        return errors
    
    def _validate_parameters(self, data: Dict[str, Any]) -> List[str]:
        errors = []
        
        # Validate temperature
        if "temperature" in data:
            temp = data["temperature"]
            if not isinstance(temp, (int, float)):
                errors.append("temperature must be a number")
            elif temp < 0 or temp > 2:
                errors.append("temperature must be between 0 and 2")
        
        # Validate max_tokens
        if "max_tokens" in data:
            max_tokens = data["max_tokens"]
            if not isinstance(max_tokens, int):
                errors.append("max_tokens must be an integer")
            elif max_tokens <= 0:
                errors.append("max_tokens must be positive")
            elif max_tokens > 32000:  # Reasonable upper limit
                errors.append("max_tokens exceeds maximum allowed value")
        
        # Validate top_p
        if "top_p" in data:
            top_p = data["top_p"]
            if not isinstance(top_p, (int, float)):
                errors.append("top_p must be a number")
            elif top_p < 0 or top_p > 1:
                errors.append("top_p must be between 0 and 1")
        
        # Validate penalties
        for penalty_field in ["presence_penalty", "frequency_penalty"]:
            if penalty_field in data:
                penalty = data[penalty_field]
                if not isinstance(penalty, (int, float)):
                    errors.append(f"{penalty_field} must be a number")
                elif penalty < -2 or penalty > 2:
                    errors.append(f"{penalty_field} must be between -2 and 2")
        
        # Validate stream
        if "stream" in data and not isinstance(data["stream"], bool):
            errors.append("stream must be a boolean")
        
        # Validate n (number of completions)
        if "n" in data:
            n = data["n"]
            if not isinstance(n, int):
                errors.append("n must be an integer")
            elif n < 1 or n > 10:
                errors.append("n must be between 1 and 10")
        
        return errors


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request validation.
    Validates chat completion requests and returns 422 for invalid requests.
    """
    
    def __init__(self, app, validator: Optional[RequestValidator] = None):
        super().__init__(app)
        self.validator = validator or RequestValidator()
    
    async def dispatch(self, request: Request, call_next):
        # Only validate POST/PUT requests to chat completions endpoints
        if request.method in ["POST", "PUT"] and "/chat/completions" in request.url.path:
            try:
                # Check Content-Type
                content_type = request.headers.get("content-type", "")
                if not content_type.startswith("application/json"):
                    return JSONResponse(
                        status_code=415,
                        content={
                            "error": {
                                "message": "Unsupported media type. Expected application/json",
                                "type": "invalid_request_error"
                            }
                        }
                    )
                
                # Read and cache body
                body = await request.body()
                
                if not body:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": {
                                "message": "Request body is empty",
                                "type": "invalid_request_error"
                            }
                        }
                    )
                
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {e}")
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": {
                                "message": f"Invalid JSON: {str(e)}",
                                "type": "invalid_request_error"
                            }
                        }
                    )
                
                # Validate request structure
                error_response = self.validator.validate(data)
                if error_response:
                    logger.info(f"Request validation failed: {error_response['error']['details']}")
                    return JSONResponse(
                        status_code=422,
                        content=error_response
                    )
                
                # Reconstruct request with body for downstream handlers
                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                
                request = Request(request.scope, receive, request._send)
                
            except Exception as e:
                logger.exception("Unexpected validation error")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": "Internal validation error",
                            "type": "internal_error"
                        }
                    }
                )
        
        return await call_next(request)
