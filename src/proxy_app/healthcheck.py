#!/usr/bin/env python3
"""
Docker healthcheck script for Mirro-Proxy.
Used by Dockerfile HEALTHCHECK instruction to verify container health.
"""
import sys
import urllib.request
import urllib.error
import os


def check_health():
    """
    Check the health endpoint of the local FastAPI server.
    Returns exit code 0 if healthy, 1 otherwise.
    """
    # Get port from environment or use default
    port = os.getenv("PORT", "8000")
    host = os.getenv("HOST", "localhost")
    
    url = f"http://{host}:{port}/health"
    
    try:
        req = urllib.request.Request(
            url, 
            method="GET",
            headers={"Accept": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if data.get("status") == "healthy":
                    print("Health check passed: Service is healthy")
                    return 0
                else:
                    print(f"Health check failed: Status is {data.get('status')}")
                    return 1
            else:
                print(f"Health check failed: HTTP {response.status}")
                return 1
                
    except urllib.error.HTTPError as e:
        print(f"Health check failed: HTTP Error {e.code}")
        return 1
    except urllib.error.URLError as e:
        print(f"Health check failed: Connection error - {e.reason}")
        return 1
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return 1


if __name__ == "__main__":
    import json  # Import here to avoid issues if run standalone
    sys.exit(check_health())
