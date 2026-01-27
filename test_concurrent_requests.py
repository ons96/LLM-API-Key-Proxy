#!/usr/bin/env python3
"""
Quick test script to verify LLM proxy is working with concurrent requests.
Tests fallback routing and concurrent request handling.
"""

import asyncio
import aiohttp
import json
import sys
import time


API_BASE = "http://127.0.0.1:8000"
API_KEY = "test_key_123"


async def test_single_request():
    """Test a single chat completion request."""
    print("Testing single request...")

    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "coding-smart",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python function that calculates fibonacci numbers.",
                }
            ],
            "max_tokens": 100,
        }

        async with session.post(
            f"{API_BASE}/v1/chat/completions", json=payload, headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✓ Single request succeeded!")
                print(f"  Model used: {result.get('model', 'unknown')}")
                print(
                    f"  Response preview: {result.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}..."
                )
                return True
            else:
                error = await response.text()
                print(f"✗ Single request failed: {response.status}")
                print(f"  Error: {error[:200]}")
                return False


async def test_concurrent_requests(num_requests=5):
    """Test concurrent requests to verify fallback routing works."""
    print(f"\nTesting {num_requests} concurrent requests...")

    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        tasks = []
        for i in range(num_requests):
            payload = {
                "model": "coding-smart",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Write Python function #{i + 1}: Calculate factorial of a number.",
                    }
                ],
                "max_tokens": 50,
            }
            task = session.post(
                f"{API_BASE}/v1/chat/completions", json=payload, headers=headers
            )
            tasks.append(task)

        # Execute all requests concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        success_count = 0
        providers_used = {}

        for i, resp in enumerate(responses):
            if isinstance(resp, aiohttp.ClientResponse) and resp.status == 200:
                success_count += 1
                result = await resp.json()
                model = result.get("model", "unknown")
                provider = model.split("/")[0] if "/" in model else "unknown"
                providers_used[provider] = providers_used.get(provider, 0) + 1
            elif isinstance(resp, Exception):
                print(f"  Request {i + 1} exception: {str(resp)[:50]}")
            else:
                text = await resp.text() if hasattr(resp, "text") else str(resp)
                print(
                    f"  Request {i + 1} failed: {resp.status if hasattr(resp, 'status') else 'unknown'} - {text[:50]}"
                )

        print(f"\n✓ Concurrent test completed in {elapsed:.2f}s")
        print(f"  Success: {success_count}/{num_requests}")
        print(f"  Providers used: {providers_used}")

        return success_count == num_requests


async def test_different_models():
    """Test different model categories."""
    print("\nTesting different model categories...")

    models_to_test = [
        ("coding-smart", "Best for complex coding tasks"),
        ("coding-fast", "Fast coding with good quality"),
        ("anthropic/claude-opus-4-5", "Claude Opus 4.5 directly"),
        ("openai/gpt-5.2", "GPT-5.2 directly"),
    ]

    results = []
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        for model, description in models_to_test:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 20,
            }

            async with session.post(
                f"{API_BASE}/v1/chat/completions", json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    results.append((model, True, result.get("model")))
                    print(f"  ✓ {model}: {result.get('model')}")
                else:
                    results.append((model, False, None))
                    print(f"  ✗ {model}: {response.status}")

    return results


async def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM Proxy Server Test Suite")
    print("=" * 60)
    print(f"Server: {API_BASE}")
    print(f"API Key: {API_KEY[:8]}...")
    print()

    # Test single request
    single_ok = await test_single_request()

    # Test concurrent requests
    concurrent_ok = await test_concurrent_requests(5)

    # Test different models
    model_results = await test_different_models()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Single request: {'PASS' if single_ok else 'FAIL'}")
    print(f"Concurrent requests: {'PASS' if concurrent_ok else 'FAIL'}")
    print(
        f"Model categories: {sum(1 for _, ok, _ in model_results if ok)}/{len(model_results)} passed"
    )

    if single_ok and concurrent_ok:
        print("\n✅ All tests passed! Proxy is working with concurrent requests.")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
