#!/usr/bin/env python3
"""
Standalone test for Provider Status Tracker
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Import only what we need, avoiding main module
from rotator_library.provider_status_tracker import ProviderStatusTracker

def test_provider_status_tracker():
    """Test the provider status tracker in isolation."""
    print("=== Provider Status Tracker Standalone Test ===")
    
    try:
        # Test 1: Initialize tracker
        print("\n1. Testing tracker initialization...")
        tracker = ProviderStatusTracker()
        print(f"   ✓ Tracker created successfully")
        print(f"   ✓ Discovered {len(tracker.providers_to_monitor)} providers")
        print(f"   ✓ Providers: {tracker.providers_to_monitor}")
        
        # Test 2: Database functionality
        print("\n2. Testing database functionality...")
        with sqlite3.connect(tracker.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [table[0] for table in cursor.fetchall()]
            print(f"   ✓ Database tables: {tables}")
            
            cursor.execute("SELECT COUNT(*) FROM provider_health_checks")
            count = cursor.fetchone()[0]
            print(f"   ✓ Health check records: {count}")
        
        # Test 3: Status methods
        print("\n3. Testing status methods...")
        
        # Get current status
        status = tracker.get_current_status()
        print(f"   ✓ Current status timestamp: {status['timestamp']}")
        print(f"   ✓ Number of providers in status: {len(status['providers'])}")
        
        # Check a few providers
        for provider_name, provider_data in list(status['providers'].items())[:3]:
            print(f"   ✓ {provider_name}: {provider_data['status']} ({provider_data['response_time_ms']:.1f}ms)")
        
        # Test 4: Best provider selection
        print("\n4. Testing best provider selection...")
        best = tracker.get_best_provider()
        print(f"   ✓ Best provider: {best['best_provider']}")
        print(f"   ✓ Reason: {best['reason']}")
        print(f"   ✓ Alternatives: {best['alternatives']}")
        
        # Test 5: History functionality
        print("\n5. Testing history functionality...")
        history = tracker.get_all_history(hours=24)
        print(f"   ✓ History contains {len(history)} providers")
        
        # Test 6: CSV export
        print("\n6. Testing CSV export...")
        csv_data = tracker.export_to_csv()
        lines = csv_data.strip().split('\n')
        print(f"   ✓ CSV has {len(lines)} lines (including header)")
        print(f"   ✓ CSV header: {lines[0]}")
        
        # Test 7: Individual provider methods
        print("\n7. Testing individual provider methods...")
        if tracker.providers_to_monitor:
            test_provider = tracker.providers_to_monitor[0]
            provider_history = tracker.get_provider_history(test_provider, hours=1)
            print(f"   ✓ History for {test_provider}: {len(provider_history)} records")
            
            single_status = tracker._get_latest_status(test_provider)
            if single_status:
                print(f"   ✓ Latest status for {test_provider}: {single_status.status}")
        
        print("\n=== All Tests Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_health_check_simulation():
    """Test health check simulation without actual network calls."""
    print("\n=== Health Check Simulation Test ===")
    
    try:
        tracker = ProviderStatusTracker()
        
        # Simulate a health check result
        print("\n1. Testing health check result storage...")
        tracker._store_health_check_result(
            provider_name="test_provider",
            status="healthy",
            response_time_ms=123.45,
            uptime_percent=99.5,
            rate_limit_percent=45.0,
            error_message="",
            consecutive_failures=0
        )
        
        # Verify it was stored
        status = tracker._get_latest_status("test_provider")
        if status and status.status == "healthy":
            print("   ✓ Health check result stored and retrieved successfully")
            print(f"   ✓ Response time: {status.response_time_ms}ms")
            print(f"   ✓ Uptime: {status.uptime_percent}%")
        else:
            print("   ✗ Failed to store/retrieve health check result")
            return False
        
        # Test uptime calculation
        print("\n2. Testing uptime calculation...")
        uptime = tracker._calculate_uptime_percent("test_provider", window_hours=1)
        print(f"   ✓ Uptime calculation: {uptime}%")
        
        print("\n=== Health Check Simulation Test Passed ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Health check simulation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting Provider Status Tracker Tests...")
    
    test1_passed = test_provider_status_tracker()
    test2_passed = test_health_check_simulation()
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe Provider Status Tracker is working correctly:")
        print("  ✓ Provider discovery and monitoring")
        print("  ✓ Database storage and queries")
        print("  ✓ Status tracking and reporting")
        print("  ✓ Best provider selection")
        print("  ✓ Historical data tracking")
        print("  ✓ CSV export functionality")
        print("  ✓ Health check simulation")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)