import httpx
import asyncio
import json


async def test_responses_api():
    url = "http://40.233.101.233:8000/v1/responses"
    payload = {
        "model": "coding-elite",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "Hello, who are you?"}],
            }
        ],
        "max_tokens": 50,
    }

    print(f"Testing Responses API at {url}...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print("Response received successfully!")
                print(json.dumps(data, indent=2))
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_responses_api())
