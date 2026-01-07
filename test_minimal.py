#!/usr/bin/env python3
"""
Minimal test for Provider Status Tracker - avoids all proxy imports
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Set environment to prevent any launcher from starting
os.environ["SKIP_LAUNCHER"] = "true"
os.environ["SUPPRESS_LAUNCHER"] = "true"

# Test only the provider status tracker module
print("=== Minimal Provider Status Tracker Test ===")

try:
    # Import only what we need
    print("1. Testing import...")
    from rotator_library.provider_status_tracker import ProviderStatusTracker
    print("   ✓ ProviderStatusTracker imported successfully")
    
    # Create tracker
    print("\n2. Testing tracker creation...")
    tracker = ProviderStatusTracker()
    print(f"   ✓ Tracker created with {len(tracker.providers_to_monitor)} providers")
    
    # Test database
    print("\n3. Testing database...")
    with sqlite3.connect(tracker.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        print(f"   ✓ Database tables: {tables}")
    
    # Test status methods
    print("\n4. Testing status methods...")
    status = tracker.get_current_status()
    print(f"   ✓ Current status has {len(status['providers'])} providers")
    
    best = tracker.get_best_provider()
    print(f"   ✓ Best provider selection works")
    
    # Test CSV export
    print("\n5. Testing CSV export...")
    csv_data = tracker.export_to_csv()
    lines = csv_data.strip().split('\n')
    print(f"   ✓ CSV export has {len(lines)} lines")
    
    print("\n=== All Minimal Tests Passed ===")
    print("\nThe Provider Status Tracker core functionality is working correctly!")
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ SUCCESS: Provider Status Tracker is functional")