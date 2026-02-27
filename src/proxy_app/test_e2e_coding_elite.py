import pytest
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Any
from httpx import Response

@pytest.fixture
def client():
    return httpx.AsyncClient(
        base_url="http://localhost:8000",
        headers={"Authorization": "Bearer PROXY_API_KEY"}
    )

@pytest.fixture
def coding_elite_model() -> str:
    return "coding-elite"

@pytest.fixture
def valid_code_prompt() -> str:
    return "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10))"

@pytest.fixture
def invalid_code_prompt() -> str:
    return "def buggy_function():\n    return undefined_variable\n\nbuggy_function()"

@pytest.fixture
def completion_request_payload(coding_elite_model, valid_code_prompt) -> Dict[str, Any]:
    return {
        "model": coding_elite_model,
        "prompt": valid_code_prompt,
        "max_tokens": 500,
        "temperature": 0.7
    }

@pytest.fixture
def chat_completion_request_payload(coding_elite_model, valid_code_prompt) -> Dict[str, Any]:
    return {
        "model": coding_elite_model,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

class TestCodingEliteE2E:
    @pytest.mark.asyncio
    async def test_coding_elite_available(self, client: httpx.AsyncClient):
        response = await client.get("/v1/models/coding-elite")
        assert response.status_code == 200
        model_info = response.json()
        assert model_info["id"] == "coding-elite"
        assert model_info["object"] == "model"

    @pytest.mark.asyncio
    async def test_coding_elite_completion(
        self,
        client: httpx.AsyncClient,
        completion_request_payload: Dict[str, Any]
    ):
        response = await client.post("/v1/completions", json=completion_request_payload)
        assert response.status_code == 200
        completion = response.json()
        assert "choices" in completion
        assert len(completion["choices"]) > 0
        assert "text" in completion["choices"][0]

    @pytest.mark.asyncio
    async def test_coding_elite_chat_completion(
        self,
        client: httpx.AsyncClient,
        chat_completion_request_payload: Dict[str, Any]
    ):
        response = await client.post("/v1/chat/completions", json=chat_completion_request_payload)
        assert response.status_code == 200
        completion = response.json()
        assert "choices" in completion
        assert len(completion["choices"]) > 0
        assert "message" in completion["choices"][0]
        assert "content" in completion["choices"][0]["message"]

    @pytest.mark.asyncio
    async def test_coding_elite_embedding(
        self,
        client: httpx.AsyncClient,
        coding_elite_model: str
    ):
        response = await client.post("/v1/embeddings", json={
            "model": coding_elite_model,
            "input": "def add(a, b): return a + b"
        })
        assert response.status_code == 200
        embeddings = response.json()
        assert "data" in embeddings
        assert len(embeddings["data"]) > 0
        assert "embedding" in embeddings["data"][0]
        assert len(embeddings["data"][0]["embedding"]) > 0

    @pytest.mark.asyncio
    async def test_coding_elite_streaming_completion(
        self,
        client: httpx.AsyncClient,
        completion_request_payload: Dict[str, Any]
    ):
        response = await client.post("/v1/completions", json={
            **completion_request_payload,
            "stream": True
        })
        assert response.status_code == 200
        async for chunk in response.stream():
            assert "choices" in chunk
            assert "delta" in chunk["choices"][0]
            assert "text" in chunk["choices"][0]["delta"]

    @pytest.mark.asyncio
    async def test_coding_elite_streaming_chat_completion(
        self,
        client: httpx.AsyncClient,
        chat_completion_request_payload: Dict[str, Any]
    ):
        response = await client.post("/v1/chat/completions", json={
            **chat_completion_request_payload,
            "stream": True
        })
        assert response.status_code == 200
        async for chunk in response.stream():
            assert "choices" in chunk
            assert "delta" in chunk["choices"][0]
            assert "content" in chunk["choices"][0]["delta"]

    @pytest.mark.asyncio
    async def test_coding_elite_error_handling(
        self,
        client: httpx.AsyncClient,
        coding_elite_model: str,
        invalid_code_prompt: str
    ):
        # Test with invalid model name
        response = await client.post("/v1/completions", json={
            "model": "invalid-model",
            "prompt": invalid_code_prompt,
            "max_tokens": 100
        })
        assert response.status_code == 400

        # Test with invalid prompt
        response = await client.post("/v1/completions", json={
            "model": coding_elite_model,
            "prompt": "",
            "max_tokens": 100
        })
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_coding_elite_caching_behavior(
        self,
        client: httpx.AsyncClient,
        completion_request_payload: Dict[str, Any]
    ):
        # First request
        response1 = await client.post("/v1/completions", json=completion_request_payload)
        assert response1.status_code == 200

        # Repeat same request
        response2 = await client.post("/v1/completions", json=completion_request_payload)
        assert response2.status_code == 200

        # Responses should be similar but not identical (caching may vary)
        completion1 = response1.json()
        completion2 = response2.json()
        assert completion1["choices"][0]["text"] == completion2["choices"][0]["text"]

    @pytest.mark.asyncio
    async def test_coding_elite_function_calling(
        self,
        client: httpx.AsyncClient,
        coding_elite_model: str
    ):
        response = await client.post("/v1/chat/completions", json={
            "model": coding_elite_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that can call functions."},
                {"role": "user", "content": "What's the current weather?"}
            ],
            "functions": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get weather for"
                            }
                        },
                        "required": ["location"]
                    }
                }
            ],
            "max_tokens": 1000
        })
        assert response.status_code == 200
        completion = response.json()
        assert "choices" in completion
        if "function_call" in completion["choices"][0]["message"]:
            assert "name" in completion["choices"][0]["message"]["function_call"]
            assert "arguments" in completion["choices"][0]["message"]["function_call"]
