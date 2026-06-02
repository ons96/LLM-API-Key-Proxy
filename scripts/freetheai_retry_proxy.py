#!/usr/bin/env python3
"""Local retry proxy for freetheai API.

Sits between opencode and freetheai. Retries on "temporarily unavailable" errors
that the Vercel AI SDK doesn't automatically retry.

Usage:
    python scripts/freetheai_retry_proxy.py
    python scripts/freetheai_retry_proxy.py --port 8340 --max-retries 5

Then in opencode.json, set freetheai baseURL to:
    http://localhost:8340/v1
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("freetheai-proxy")

FREETHEAI_BASE = os.getenv("FREETHEAI_BASE_URL", "https://api.freetheai.xyz")
DEFAULT_PORT = 8340
MAX_RETRIES = 5
RETRY_DELAY = 1.0  # seconds, doubles each attempt

UPSTREAM_ERROR_PATTERNS = [
    b"temporarily unavailable",
    b"upstream provider",
    b"503 Service Unavailable",
    b"502 Bad Gateway",
    b"service unavailable",
    b"try again later",
]

API_KEY = os.getenv("FREETHEAI_API_KEY", "")
if not API_KEY:
    key_file = os.path.expanduser("~/.secrets/freetheai-key")
    try:
        with open(key_file) as f:
            API_KEY = f.read().strip()
    except FileNotFoundError:
        pass


def _should_retry(status: int, body: bytes) -> bool:
    if status >= 500:
        return True
    if status == 429:
        return True
    for pattern in UPSTREAM_ERROR_PATTERNS:
        if pattern in body.lower():
            return True
    return False


async def handle_request(
    client: aiohttp.ClientSession, request
) -> aiohttp.web.Response:
    path = request.rel_url.path_qs
    target = f"{FREETHEAI_BASE}{path}"

    headers = dict(request.headers)
    headers.pop("Host", None)
    if API_KEY and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {API_KEY}"

    body = await request.read()

    last_error: Optional[str] = None
    delay = RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with client.request(
                request.method,
                target,
                headers=headers,
                data=body,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp_body = await resp.read()

                if not _should_retry(resp.status, resp_body):
                    return aiohttp.web.Response(
                        status=resp.status,
                        headers={
                            k: v
                            for k, v in resp.headers.items()
                            if k.lower()
                            not in (
                                "content-encoding",
                                "transfer-encoding",
                                "content-length",
                            )
                        },
                        body=resp_body,
                    )

                last_error = f"HTTP {resp.status}"
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES} failed ({last_error}), retrying in {delay:.0f}s..."
                )

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = str(e)[:60]
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES} connection error ({last_error}), retrying in {delay:.0f}s..."
            )

        if attempt < MAX_RETRIES:
            await asyncio.sleep(delay)
            delay *= 2

    logger.error(f"All {MAX_RETRIES} attempts failed. Last error: {last_error}")
    return aiohttp.web.Response(
        status=503,
        body=json.dumps(
            {
                "error": {
                    "message": f"freetheai proxy: all retries exhausted after {MAX_RETRIES} attempts",
                    "type": "proxy_error",
                    "last_error": last_error,
                }
            }
        ).encode(),
        content_type="application/json",
    )


async def health_check(request) -> aiohttp.web.Response:
    return aiohttp.web.json_response({"status": "ok", "upstream": FREETHEAI_BASE})


async def main():
    parser = argparse.ArgumentParser(description="freetheai retry proxy")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Local port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Max retries (default: {MAX_RETRIES})",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    global MAX_RETRIES
    MAX_RETRIES = args.max_retries

    if not API_KEY:
        logger.warning(
            "No API key found. Set FREETHEAI_API_KEY env var or ~/.secrets/freetheai-key"
        )

    app = aiohttp.web.Application()
    app["client"] = aiohttp.ClientSession()

    app.router.add_get("/health", health_check)
    app.router.add_route("*", "/{tail:.*}", handle_request)

    logger.info(f"freetheai retry proxy listening on http://{args.host}:{args.port}")
    logger.info(f"Upstream: {FREETHEAI_BASE}")
    logger.info(f"Max retries: {MAX_RETRIES}")
    logger.info(
        f"Update opencode.json: set baseURL to http://{args.host}:{args.port}/v1"
    )

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, args.host, args.port)
    await site.start()

    try:
        await asyncio.Event().wait()
    finally:
        await app["client"].close()
        await runner.cleanup()


if __name__ == "__main__":
    import json

    asyncio.run(main())
