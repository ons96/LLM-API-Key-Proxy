#!/usr/bin/env python3
"""
Test API endpoints for Provider Status Tracker
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Set environment to avoid launcher
os.environ["SKIP_LAUNCHER"] = "true"

# Import required modules
from fastapi.testclient import TestClient

# Import app directly to avoid launcher
sys.argv = ["test"]  # Prevent launcher from starting
from proxy_app.main import app

def test_api_endpoints():
    """Test all API endpoints."""
    print("Testing API endpoints...")
    
    # Create test client
    client = TestClient(app)
    
    # Test 1: GET /api/providers/status
    print("\n1. Testing GET /api/providers/status")
    response = client.get("/api/providers/status")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Response keys: {list(data.keys())}")
        print(f"Number of providers: {len(data.get('providers', {}))}")
        print("✓ PASS")
    else:
        print(f"✗ FAIL: {response.text}")
    
    # Test 2: GET /api/providers/best
    print("\n2. Testing GET /api/providers/best")
    response = client.get("/api/providers/best")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Best provider: {data.get('best_provider', 'None')}")
        print(f"Reason: {data.get('reason', '')}")
        print("✓ PASS")
    else:
        print(f"✗ FAIL: {response.text}")
    
    # Test 3: GET /api/providers/history
    print("\n3. Testing GET /api/providers/history")
    response = client.get("/api/providers/history?hours=24")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Number of providers with history: {len(data)}")
        print("✓ PASS")
    else:
        print(f"✗ FAIL: {response.text}")
    
    # Test 4: GET /api/providers/export/csv
    print("\n4. Testing GET /api/providers/export/csv")
    response = client.get("/api/providers/export/csv")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print("✓ PASS")
    else:
        print(f"✗ FAIL: {response.text}")
    
    # Test 5: GET /api/providers/health
    print("\n5. Testing GET /api/providers/health")
    response = client.get("/api/providers/health")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Tracker status: {data.get('status', 'unknown')}")
        print(f"Providers monitored: {data.get('providers_monitored', 0)}")
        print("✓ PASS")
    else:
        print(f"✗ FAIL: {response.text}")
    
    print("\n=== API Endpoint Tests Complete ===")

if __name__ == "__main__":
    test_api_endpoints()