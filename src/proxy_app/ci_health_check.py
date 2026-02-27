"""
CI/CD Health Check Utility

A lightweight script for verifying proxy deployment health in CI/CD pipelines.
Exits with code 0 on success, 1 on failure.
"""

import argparse
import sys
import time
import urllib.request
import urllib.error
import json
from typing import Optional


def check_health(
    base_url: str, 
    api_key: Optional[str] = None,
    timeout: int = 30,
    verify_models: bool = True
) -> bool:
    """
    Perform health check on the proxy server.
    
    Args:
        base_url: Base URL of the proxy (e.g., http://localhost:8000)
        api_key: Optional API key for authenticated endpoints
        timeout: Request timeout in seconds
        verify_models: Whether to verify /v1/models endpoint
        
    Returns:
        True if healthy, False otherwise
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Normalize URL
    base_url = base_url.rstrip("/")
    
    # Check 1: Root health endpoint (if available)
    health_url = f"{base_url}/health"
    try:
        req = urllib.request.Request(health_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                print(f"✓ Health endpoint responding (HTTP {response.status})")
            else:
                print(f"✗ Health endpoint returned HTTP {response.status}")
                return False
    except urllib.error.HTTPError as e:
        # Some proxies may not have /health, continue to check models
        if e.code != 404:
            print(f"✗ Health endpoint error: HTTP {e.code}")
            return False
        print(f"ℹ Health endpoint not found (404), continuing...")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Check 2: Models endpoint (OpenAI compatible)
    if verify_models:
        models_url = f"{base_url}/v1/models"
        try:
            req = urllib.request.Request(models_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    model_count = len(data.get('data', []))
                    print(f"✓ Models endpoint responding ({model_count} models available)")
                else:
                    print(f"✗ Models endpoint returned HTTP {response.status}")
                    return False
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print("✗ Authentication failed (401) - check API key")
            elif e.code == 403:
                print("✗ Forbidden (403) - check API key permissions")
            else:
                print(f"✗ Models endpoint error: HTTP {e.code}")
            return False
        except Exception as e:
            print(f"✗ Models check failed: {e}")
            return False
    
    # Check 3: Status API (if available)
    status_url = f"{base_url}/status"
    try:
        req = urllib.request.Request(status_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                print("✓ Status API responding")
    except:
        # Status API is optional
        pass
    
    return True


def wait_for_healthy(
    base_url: str,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    interval: int = 2,
    timeout: int = 30
) -> bool:
    """
    Wait for the service to become healthy with retries.
    
    Args:
        base_url: Base URL of the proxy
        api_key: Optional API key
        max_retries: Maximum number of attempts
        interval: Seconds between retries
        timeout: Request timeout per attempt
        
    Returns:
        True if became healthy, False otherwise
    """
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}...")
        if check_health(base_url, api_key, timeout):
            return True
        if attempt < max_retries:
            print(f"Waiting {interval} seconds before retry...")
            time.sleep(interval)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Health check for Mirro-Proxy in CI/CD pipelines"
    )
    parser.add_argument(
        "--url", 
        required=True,
        help="Base URL of the proxy server (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication (or set PROXY_API_KEY env var)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum number of retry attempts (default: 1)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=2,
        help="Seconds between retries (default: 2)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip the /v1/models endpoint check"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output errors"
    )
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key or __import__('os').getenv("PROXY_API_KEY")
    
    if not args.quiet:
        print(f"Checking health of {args.url}...")
        if api_key:
            masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
            print(f"Using API key: {masked}")
    
    success = wait_for_healthy(
        args.url,
        api_key=api_key,
        max_retries=args.max_retries,
        interval=args.interval,
        timeout=args.timeout
    )
    
    if success:
        if not args.quiet:
            print("\n✓ Health check passed!")
        sys.exit(0)
    else:
        print("\n✗ Health check failed!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
