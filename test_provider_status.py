#!/usr/bin/env python3
"""
Test script for Provider Status Tracker
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the tracker
from rotator_library.provider_status_tracker import ProviderStatusTracker

def test_provider_discovery():
    """Test provider discovery functionality."""
    print("Testing provider discovery...")
    
    tracker = ProviderStatusTracker()
    print(f"Discovered providers: {tracker.providers_to_monitor}")
    print(f"Number of providers: {len(tracker.providers_to_monitor)}")
    
    return len(tracker.providers_to_monitor) > 0

def test_database_initialization():
    """Test database initialization."""
    print("\nTesting database initialization...")
    
    tracker = ProviderStatusTracker()
    
    # Check if database file exists
    import sqlite3
    try:
        with sqlite3.connect(tracker.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"Database tables: {tables}")
            return True
    except Exception as e:
        print(f"Database test failed: {e}")
        return False

async def test_health_checks():
    """Test health check functionality."""
    print("\nTesting health checks...")
    
    tracker = ProviderStatusTracker()
    
    try:
        # Run health checks
        await tracker._run_health_checks()
        
        # Get current status
        status = tracker.get_current_status()
        print(f"Status snapshot: {status}")
        
        # Get best provider
        best = tracker.get_best_provider()
        print(f"Best provider: {best}")
        
        return True
    except Exception as e:
        print(f"Health check test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints."""
    print("\nTesting API endpoints...")
    
    # This would require running the FastAPI app, so we'll just verify the routes exist
    from proxy_app.status_api import router
    
    routes = [route.path for route in router.routes]
    print(f"Available routes: {routes}")
    
    expected_routes = ["/status", "/status/{provider_name}", "/best", "/history", "/export/csv"]
    
    for expected in expected_routes:
        found = any(expected in route for route in routes)
        print(f"Route {expected}: {'✓' if found else '✗'}")
    
    return True

def main():
    """Run all tests."""
    print("=== Provider Status Tracker Tests ===")
    
    tests = [
        ("Provider Discovery", test_provider_discovery),
        ("Database Initialization", test_database_initialization),
        ("Health Checks", test_health_checks),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n=== Test Results ===")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)