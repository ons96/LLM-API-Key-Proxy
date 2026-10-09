"""Provider-free concurrency smoke test for a running router."""

import argparse
import concurrent.futures
import json
from urllib import request


def route(url: str, index: int) -> int:
    body = json.dumps({"model": "auto", "messages": [{"role": "user", "content": f"question {index}"}]}).encode()
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=5) as response:
        response.read()
        return response.status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/router/route")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        statuses = list(pool.map(lambda i: route(args.url, i), range(args.requests)))
    print(json.dumps({"requests": len(statuses), "successful": statuses.count(200)}))
    return 0 if all(status == 200 for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
