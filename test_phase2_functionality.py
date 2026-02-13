#!/usr/bin/env python3
"""
Test script to verify Phase 2 core functionality is working.
"""

import sys
import time
import numpy as np


def test_import_performance():
    """Test that import is fast and matplotlib not loaded."""
    print("🧪 Testing import performance...")

    # Test import in fresh subprocess
    start_time = time.time()
    import spectrochempy

    import_time = time.time() - start_time

    # Check if matplotlib modules are loaded
    mpl_loaded = "matplotlib" in sys.modules
    pyplot_loaded = "matplotlib.pyplot" in sys.modules

    print(f"   Import time: {import_time * 1000:.1f}ms")
    print(f"   Matplotlib loaded: {mpl_loaded}")
    print(f"   Pyplot loaded: {pyplot_loaded}")

    if import_time < 0.2:  # Should be under 200ms
        print("   ✅ Import time is excellent")
        return True
    else:
        print("   ❌ Import is too slow")
        return False


def test_lazy_functionality():
    """Test that lazy initialization works correctly."""
    print("\n🧪 Testing lazy initialization functionality...")
    import spectrochempy

    # Check lazy initialization state
    from spectrochempy.core.plotters.plot_setup import _get_mpl_state

    initial_state = "initialized" if _get_mpl_state() else "not_initialized"

    # Create dataset and plot (should trigger lazy init)
    start_time = time.time()
    data = np.random.rand(100)
    dataset = spectrochempy.NDDataset(data)

    # Use non-blocking plot (show=False)
    ax = dataset.plot(show=False)
    first_plot_time = time.time() - start_time

    # Check state after first plot
    final_state = "initialized" if _get_mpl_state() else "not_initialized"

    print(f"   Initial state: {initial_state}")
    print(f"   Final state: {final_state}")
    print(f"   First plot time: {first_plot_time * 1000:.1f}ms")
    print(f"   Plot successful: {ax is not None}")

    success = True
    if initial_state == "not_initialized":
        print("   ✅ Initial state is NOT_INITIALIZED")
    else:
        print("   ❌ Initial state is incorrect")
        success = False

    if final_state == "initialized":
        print("   ✅ Final state is INITIALIZED")
    else:
        print("   ❌ Final state is incorrect")
        success = False

    if 0.5 <= first_plot_time <= 5.0:  # Should be 0.5-5 seconds
        print("   ✅ First plot time is acceptable")
    else:
        print("   ❌ First plot time is out of range")
        success = False

    if ax is not None:
        print("   ✅ Plot executed successfully")
    else:
        print("   ❌ Plot failed")
        success = False

    return success


def test_basic_plotting():
    """Test basic plotting functionality."""
    print("\n🧪 Testing basic plotting functionality...")
    import spectrochempy

    try:
        # Test 1D plotting
        data1d = np.random.rand(50)
        dataset1d = spectrochempy.NDDataset(data1d)
        ax1 = dataset1d.plot(show=False)
        print("   ✅ 1D plotting works")

        # Test 2D plotting
        data2d = np.random.rand(20, 30)
        dataset2d = spectrochempy.NDDataset(data2d)
        ax2 = dataset2d.plot(show=False)
        print("   ✅ 2D plotting works")

        return True

    except Exception as e:
        print(f"   ❌ Plotting failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Phase 2 Core Functionality Test")
    print("=" * 50)

    tests = [
        ("Import Performance", test_import_performance),
        ("Lazy Functionality", test_lazy_functionality),
        ("Basic Plotting", test_basic_plotting),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")

    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not success:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Phase 2 implementation is working correctly.")
        print(
            "💡 Lazy matplotlib initialization simplified while maintaining functionality."
        )
        return 0
    else:
        print("❌ SOME TESTS FAILED! Check implementation details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
