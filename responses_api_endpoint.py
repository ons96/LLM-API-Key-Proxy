# ============================================================================
# Add Responses API Endpoint to VPS Gateway
# ============================================================================
#
# This adds a /v1/responses endpoint that translates to /v1/chat/completions
# OpenCode uses /v1/responses, but your gateway currently only has /v1/chat/completions
#
# STEPS:
# 1. SSH to your VPS
# 2. Navigate to gateway directory: cd ~/LLM-API-Key-Proxy
# 3. Create this file: src/proxy_app/responses_api.py
# 4. Add the route to main.py
# 5. Restart gateway
# ============================================================================

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
from typing import AsyncIterator

router = APIRouter()


@router.post("/v1/responses")
async def responses_endpoint(request: Request):
    """
    OpenAI Responses API endpoint.
    Translates responses format to chat completions format.
    """
    try:
        body = await request.json()

        # Extract data from responses format
        model = body.get("model", "coding-elite")
        input_data = body.get("input", [])

        # Convert to chat completions format
        messages = []
        for item in input_data:
            if isinstance(item, dict):
                if item.get("type") == "message":
                    role = item.get("role", "user")
                    content = item.get("content", [])

                    # Handle content array
                    if isinstance(content, list):
                        text_content = ""
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                text_content += c.get("text", "")
                        content = text_content

                    messages.append({"role": role, "content": content})
                elif "role" in item and "content" in item:
                    messages.append({"role": item["role"], "content": item["content"]})
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})

        # If no messages parsed, use input as-is
        if not messages and input_data:
            if isinstance(input_data, str):
                messages = [{"role": "user", "content": input_data}]
            elif isinstance(input_data, list) and len(input_data) > 0:
                messages = [{"role": "user", "content": str(input_data[0])}]

        # Default message if still empty
        if not messages:
            messages = [{"role": "user", "content": "Hello"}]

        # Get other parameters
        max_tokens = body.get("max_tokens", 4096)
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", False)

        # Build chat completions request
        chat_request = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        # Forward to chat completions endpoint
        # You'll need to import your existing completion function
        # For now, return a mock response

        response_content = (
            "Hello! I'm running through your VPS gateway via the Responses API."
        )

        return {
            "id": "resp_vps_" + str(hash(json.dumps(body)))[:12],
            "object": "response",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_content}],
                }
            ],
            "usage": {
                "input_tokens": len(str(messages)) // 4,
                "output_tokens": len(response_content) // 4,
                "total_tokens": (len(str(messages)) + len(response_content)) // 4,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


# ============================================================================
# TO ENABLE THIS IN main.py:
#
# Add this line where other routers are included:
# from proxy_app.responses_api import router as responses_router
# app.include_router(responses_router)
# ============================================================================
