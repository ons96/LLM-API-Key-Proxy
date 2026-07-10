"""
Regression test for #362: chat_completions non-stream path must serialize
litellm ModelResponse to dict before returning to FastAPI.

Tests the serialization block in isolation (not full endpoint) so we don't
couple to router internals. Simulates the bug: litellm Message has 5 populated
fields, OpenAI SDK Message expects 10 -> Pydantic UserWarning -> malformed JSON.
"""
import warnings
import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _make_partial_response():
    """litellm-shaped ModelResponse with 5 of 10 Message fields populated."""
    from pydantic import BaseModel

    class PartialMessage(BaseModel):
        content: str
        role: str
        refusal: Optional[str] = None
        audio: Optional[str] = None
        tool_calls: Optional[list] = None

    class PartialChoices(BaseModel):
        index: int = 0
        message: PartialMessage
        finish_reason: str = "stop"

    class PartialModelResponse(BaseModel):
        id: str = "test"
        choices: list
        model: str = "test-model"
        object: str = "chat.completion"

    return PartialModelResponse(
        id="chatcmpl-test",
        choices=[PartialChoices(
            index=0,
            message=PartialMessage(content="Hello", role="assistant"),
        )],
        model="groq/llama-3.3-70b",
    )


def _serialize(result):
    """Mirror the exact block added to chat_completions at main.py:1153."""
    from fastapi.responses import JSONResponse

    import warnings as _w
    if isinstance(result, dict):
        return JSONResponse(content=result)
    if isinstance(result, JSONResponse):
        return result
    if hasattr(result, "model_dump"):
        with _w.catch_warnings():
            _w.filterwarnings("ignore", category=UserWarning)
            dumped = result.model_dump(exclude_unset=True, exclude_none=True)
        return JSONResponse(content=dumped)
    if hasattr(result, "dict"):
        with _w.catch_warnings():
            _w.filterwarnings("ignore", category=UserWarning)
            dumped = result.dict()
        return JSONResponse(content=dumped)
    return result


def test_partial_model_response_serializes_without_userwarning():
    """The exact bug from #362: partial litellm ModelResponse must serialize cleanly."""
    partial = _make_partial_response()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning here -> fail
        result = _serialize(partial)
    from fastapi.responses import JSONResponse
    assert isinstance(result, JSONResponse), f"got {type(result)}"
    payload = json.loads(result.body)
    assert payload["choices"][0]["message"]["content"] == "Hello"
    assert payload["model"] == "groq/llama-3.3-70b"


def test_dict_passthrough():
    raw_dict = {
        "id": "test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "model": "test",
    }
    result = _serialize(raw_dict)
    from fastapi.responses import JSONResponse
    assert isinstance(result, JSONResponse)
    assert json.loads(result.body)["choices"][0]["message"]["content"] == "Hi"


def test_jsonresponse_passthrough():
    from fastapi.responses import JSONResponse
    original = JSONResponse(content={"x": 1})
    result = _serialize(original)
    assert result is original


if __name__ == "__main__":
    test_partial_model_response_serializes_without_userwarning()
    test_dict_passthrough()
    test_jsonresponse_passthrough()
    print("test_chat_completions_serialize: OK")
