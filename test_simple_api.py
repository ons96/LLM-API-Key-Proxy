#!/usr/bin/env python3
"""
Simple test for Provider Status Tracker API
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Test the status tracker directly
from rotator_library.provider_status_tracker import ProviderStatusTracker

def test_status_tracker():
    """Test the status tracker functionality."""
    print("=== Testing Provider Status Tracker ===")
    
    # Create tracker
    tracker = ProviderStatusTracker()
    print(f"✓ Created tracker with {len(tracker.providers_to_monitor)} providers")
    
    # Test database
    import sqlite3
    with sqlite3.connect(tracker.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM provider_health_checks")
        count = cursor.fetchone()[0]
        print(f"✓ Database has {count} health check records")
    
    # Test status methods
    status = tracker.get_current_status()
    print(f"✓ Current status has {len(status['providers'])} providers")
    
    best = tracker.get_best_provider()
    print(f"✓ Best provider: {best['best_provider']}")
    
    history = tracker.get_all_history(hours=1)
    print(f"✓ History has {len(history)} providers with data")
    
    csv_data = tracker.export_to_csv()
    lines = csv_data.strip().split('\n')
    print(f"✓ CSV export has {len(lines)} lines")
    
    print("\n=== All Tests Passed ===")

if __name__ == "__main__":
    test_status_tracker()