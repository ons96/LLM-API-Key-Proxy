# ZeroClaw/Telegram JSON endpoint - returns plain JSON instead of SSE

@app.post("/v1/chat/completions/json")
async def chat_completions_json(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint that returns plain JSON (not SSE).
    For ZeroClaw/Telegram clients that can't handle SSE responses.
    """
    # Parse request body
    try:
        request_data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body.")
    
    # Force non-streaming
    request_data["stream"] = False
    
    # Use RouterWrapper to handle the request
    try:
        router = get_router()
        result = await router.handle_chat_completions(request_data, request)
    except Exception as e:
        logging.error(f"Router delegation failed: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Router Error: {str(e)}")
    
    # Convert Pydantic models to plain dicts
    if hasattr(result, "model_dump"):
        result = result.model_dump(mode="json")
    elif hasattr(result, "dict"):
        result = result.dict()
    
    # Clean nulls
    def _strip_nulls(d, depth=0):
        if not isinstance(d, dict) or depth > 5:
            return d
        keys_to_remove = [k for k, v in d.items() if v is None]
        for k in keys_to_remove:
            del d[k]
        for v in d.values():
            if isinstance(v, dict):
                _strip_nulls(v, depth + 1)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _strip_nulls(item, depth + 1)
        return d
    
    if isinstance(result, dict):
        if "choices" in result:
            for choice in result["choices"]:
                if "message" in choice:
                    msg = choice["message"]
                    if msg.get("tool_calls") is None:
                        msg.pop("tool_calls", None)
                    if msg.get("function_call") is None:
                        msg.pop("function_call", None)
                    msg.pop("provider_specific_fields", None)
                choice.pop("provider_specific_fields", None)
        result.pop("time_info", None)
        result.pop("service_tier", None)
        _strip_nulls(result)
    
    # Return plain JSON (not SSE)
    return Response(
        content=json.dumps(result, ensure_ascii=False),
        media_type="application/json",
    )
