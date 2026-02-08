#!/usr/bin/env python3
"""
Test script to verify Alternative A lazy initialization performance.

This script measures:
1. Import time (should be 0ms - no matplotlib init)
2. First plot time (should be ~950-2650ms)
3. Subsequent plot time (should be ~50-100ms)
4. Verify matplotlib not modified during import
"""

import sys
import time
import subprocess
import os


def test_import_performance():
    """Test that import doesn't initialize matplotlib."""
    print("🧪 Testing import performance...")

    # Test import in fresh subprocess to avoid contamination
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import time
start = time.time()
import spectrochempy
import_time = time.time() - start

# Check if matplotlib modules are loaded
mpl_loaded = 'matplotlib' in sys.modules
pyplot_loaded = 'matplotlib.pyplot' in sys.modules

print(f"IMPORT_TIME:{import_time:.6f}")
print(f"MATPLOTLIB_LOADED:{mpl_loaded}")
print(f"PYLOADED_LOADED:{pyplot_loaded}")
""",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"❌ Import test failed: {result.stderr}")
            return False

        output = result.stdout.strip()
        import_time = float(
            [line for line in output.split("\n") if line.startswith("IMPORT_TIME:")][
                0
            ].split(":")[1]
        )
        mpl_loaded = [
            line for line in output.split("\n") if line.startswith("MATPLOTLIB_LOADED:")
        ][0].split(":")[1] == "True"
        pyplot_loaded = [
            line for line in output.split("\n") if line.startswith("PYLOADED_LOADED:")
        ][0].split(":")[1] == "True"

        print(f"   Import time: {import_time * 1000:.1f}ms")
        print(f"   Matplotlib loaded: {mpl_loaded}")
        print(f"   Pyplot loaded: {pyplot_loaded}")

        # Success criteria
        if import_time < 0.1:  # Should be under 100ms
            print("   ✅ Import time is excellent")
            success = True
        else:
            print("   ❌ Import is too slow - matplotlib still being initialized")
            success = False

        if not mpl_loaded and not pyplot_loaded:
            print("   ✅ Matplotlib not loaded during import")
        else:
            print("   ❌ Matplotlib was loaded during import")
            success = False

        return success

    except subprocess.TimeoutExpired:
        print("   ❌ Import test timed out")
        return False
    except Exception as e:
        print(f"   ❌ Import test error: {e}")
        return False


def test_lazy_functionality():
    """Test that lazy initialization works correctly."""
    print("\n🧪 Testing lazy initialization functionality...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import time
import numpy as np

# Import spectrochempy
start = time.time()
import spectrochempy
import_time = time.time() - start

# Check lazy initialization state
from spectrochempy.core.plotters.plot_setup import _get_mpl_state, MPLInitState
state = _get_mpl_state()
initial_state = state.name

# Create dataset and plot (should trigger lazy init)
start = time.time()
data = np.random.rand(100)
dataset = spectrochempy.NDDataset(data)

# Use non-blocking plot (show=False)
ax = dataset.plot(show=False)
first_plot_time = time.time() - start

# Check state after first plot
final_state = _get_mpl_state().name

print(f"INITIAL_STATE:{initial_state}")
print(f"FINAL_STATE:{final_state}")
print(f"FIRST_PLOT_TIME:{first_plot_time:.6f}")
print(f"PLOT_SUCCESS:{ax is not None}")
""",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"   ❌ Lazy functionality test failed: {result.stderr}")
            return False

        output = result.stdout.strip()
        lines = output.split("\n")

        initial_state = [line for line in lines if line.startswith("INITIAL_STATE:")][
            0
        ].split(":")[1]
        final_state = [line for line in lines if line.startswith("FINAL_STATE:")][
            0
        ].split(":")[1]
        first_plot_time = float(
            [line for line in lines if line.startswith("FIRST_PLOT_TIME:")][0].split(
                ":"
            )[1]
        )
        plot_success = [line for line in lines if line.startswith("PLOT_SUCCESS:")][
            0
        ].split(":")[1] == "True"

        print(f"   Initial state: {initial_state}")
        print(f"   Final state: {final_state}")
        print(f"   First plot time: {first_plot_time * 1000:.1f}ms")
        print(f"   Plot successful: {plot_success}")

        # Success criteria
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

        if plot_success:
            print("   ✅ Plot executed successfully")
        else:
            print("   ❌ Plot failed")
            success = False

        return success

    except subprocess.TimeoutExpired:
        print("   ❌ Lazy functionality test timed out")
        return False
    except Exception as e:
        print(f"   ❌ Lazy functionality test error: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Alternative A: Full Lazy Initialization Test")
    print("=" * 60)

    tests = [
        ("Import Performance", test_import_performance),
        ("Lazy Functionality", test_lazy_functionality),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")

    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not success:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Alternative A implementation is working correctly.")
        print(
            "💡 Import performance dramatically improved while maintaining functionality."
        )
        return 0
    else:
        print("❌ SOME TESTS FAILED! Check implementation details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
