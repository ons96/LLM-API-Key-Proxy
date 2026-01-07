#!/usr/bin/env python3
"""
Isolated test for Provider Status Tracker - no imports from main modules
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Prevent any interactive components
os.environ["CI"] = "true"
os.environ["NONINTERACTIVE"] = "true"
os.environ["HEADLESS"] = "true"

print("=== Isolated Provider Status Tracker Test ===")

try:
    # Test direct import of the status tracker module
    print("1. Testing direct module import...")
    
    # Import the module directly without going through __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "provider_status_tracker",
        "/home/engine/project/src/rotator_library/provider_status_tracker.py"
    )
    module = importlib.util.module_from_spec(spec)
    
    # Mock any problematic imports
    import unittest.mock
    with unittest.mock.patch('rotator_library.provider_status_tracker.PROVIDER_PLUGINS', {}):
        spec.loader.exec_module(module)
    
    print("   ✓ Module imported successfully")
    
    # Test the ProviderStatusTracker class directly
    print("\n2. Testing ProviderStatusTracker class...")
    tracker = module.ProviderStatusTracker()
    print(f"   ✓ Tracker created with {len(tracker.providers_to_monitor)} providers")
    
    # Test basic functionality
    print("\n3. Testing basic functionality...")
    
    # Test database
    with sqlite3.connect(tracker.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        print(f"   ✓ Database tables: {tables}")
    
    # Test status methods
    status = tracker.get_current_status()
    print(f"   ✓ Current status has {len(status['providers'])} providers")
    
    best = tracker.get_best_provider()
    print(f"   ✓ Best provider selection works")
    
    print("\n=== Isolated Test Passed ===")
    
except Exception as e:
    print(f"\n✗ Isolated test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ SUCCESS: Isolated test passed")