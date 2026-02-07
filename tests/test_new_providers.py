import os
import asyncio
import logging
from dotenv import load_dotenv
from litellm import acompletion

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_provider(provider_name, model_name):
    print(f"\n--- Testing {provider_name} ({model_name}) ---")
    try:
        response = await acompletion(
            model=f"{provider_name}/{model_name}",
            messages=[{"role": "user", "content": "Hi, what is 2+2?"}],
            max_tokens=10,
        )
        print(f"SUCCESS: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


async def main():
    # Load .env
    load_dotenv("/home/ubuntu/LLM-API-Key-Proxy/.env")

    # Test SambaNova
    await test_provider("sambanova", "Meta-Llama-3.1-8B-Instruct")

    # Test Together
    await test_provider("together_ai", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")


if __name__ == "__main__":
    asyncio.run(main())
