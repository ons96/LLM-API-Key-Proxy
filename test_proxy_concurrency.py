import asyncio
import aiohttp
import time
import json

PROXY_URL = "http://40.233.101.233:8000/v1/chat/completions"
PROXY_KEY = "poop"

MODELS_TO_TEST = [
    "coding-elite",
    "coding-fast",
    "chat-smart",
    "chat-elite",
    "chat-fast",
]

HEADERS = {"Authorization": f"Bearer {PROXY_KEY}", "Content-Type": "application/json"}


async def send_request(session, model, request_id):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Reply with a single word: Hello (Request {request_id})",
            }
        ],
        "temperature": 0.0,
        "max_tokens": 10,
    }

    start_time = time.time()
    try:
        async with session.post(PROXY_URL, json=payload, headers=HEADERS) as response:
            latency = time.time() - start_time
            if response.status == 200:
                data = await response.json()
                content = data["choices"][0]["message"]["content"].strip()
                used_model = data.get("model", "unknown")
                print(
                    f"[OK] {model} ({request_id}) -> {content} | Upstream: {used_model} | {latency:.2f}s"
                )
                return True
            else:
                text = await response.text()
                print(
                    f"[ERR] {model} ({request_id}) -> {response.status}: {text[:100]}"
                )
                return False
    except Exception as e:
        latency = time.time() - start_time
        print(
            f"[EXC] {model} ({request_id}) -> {type(e).__name__}: {str(e)[:50]} | {latency:.2f}s"
        )
        return False


async def main():
    print("\n=== Phase 1: Sequential Test ===")
    async with aiohttp.ClientSession() as session:
        for i, model in enumerate(MODELS_TO_TEST):
            await send_request(session, model, f"seq-{i}")

    print("\n=== Phase 2: Parallel Test (12 concurrent) ===")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(3):
            for j, model in enumerate(MODELS_TO_TEST):
                tasks.append(send_request(session, model, f"par-{i}-{j}"))

        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r)
        print(f"\nResults: {success_count}/{len(results)} OK")


if __name__ == "__main__":
    asyncio.run(main())
