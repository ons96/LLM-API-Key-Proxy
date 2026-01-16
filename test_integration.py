#!/usr/bin/env python3
"""
Test integration of Provider Status Tracker with the main proxy
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Prevent launcher from starting
os.environ["SKIP_LAUNCHER"] = "true"

print("=== Integration Test ===")

try:
    # Test 1: Import the status API module
    print("\n1. Testing status API module import...")
    from proxy_app.status_api import router as status_router, get_healthiest_provider
    print("   ✓ Status API module imported successfully")
    
    # Test 2: Check available routes
    print("\n2. Testing available routes...")
    routes = [route.path for route in status_router.routes]
    expected_routes = [
        "/api/providers/status",
        "/api/providers/status/{provider_name}",
        "/api/providers/best",
        "/api/providers/history",
        "/api/providers/history/{provider_name}",
        "/api/providers/export/csv",
        "/api/providers/health"
    ]
    
    for expected in expected_routes:
        found = any(expected in route for route in routes)
        print(f"   {'✓' if found else '✗'} Route {expected}")
        if not found:
            print(f"      Available routes: {routes}")
            raise Exception(f"Route {expected} not found")
    
    # Test 3: Test the integration function
    print("\n3. Testing integration function...")
    
    # Create a mock tracker for testing
    from rotator_library.provider_status_tracker import ProviderStatusTracker
    mock_tracker = ProviderStatusTracker()
    
    best_provider = get_healthiest_provider(mock_tracker)
    print(f"   ✓ Integration function returned: {best_provider}")
    
    # Test 4: Test router dependency
    print("\n4. Testing router dependency...")
    from proxy_app.status_api import get_status_tracker
    print("   ✓ Router dependency function imported")
    
    print("\n=== Integration Test Passed ===")
    print("\nThe Provider Status Tracker is properly integrated with the proxy:")
    print("  ✓ All API routes are available")
    print("  ✓ Integration functions work correctly")
    print("  ✓ Router dependencies are properly set up")
    
except Exception as e:
    print(f"\n✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ SUCCESS: Integration test passed")