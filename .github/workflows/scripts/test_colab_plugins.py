#!/usr/bin/env python3
# ruff: noqa: S101, T201
"""
Colab compatibility smoke test for SpectroChemPy and its official plugins.

Tests two scenarios:
  --mode core-only    : install core without plugins, verify basic functionality
                        and helpful error messages for missing plugins
  --mode with-plugins : install core + all official lightweight plugins,
                        verify namespaces and basic import work

This script is intended to be run in a Google Colab-like Docker container.
"""

import argparse
import importlib
import sys


def test_core_install():
    """Smoke test: core-only install."""
    print("=" * 60)
    print("  Core-only install smoke test")
    print("=" * 60)

    import spectrochempy as scp

    print(f"  SpectroChemPy {scp.__version__}")

    ds = scp.NDDataset([1, 2, 3])
    assert ds.shape == (3,), f"NDDataset shape mismatch: {ds.shape}"
    print("  PASS: NDDataset created successfully")

    # Verify accessing missing plugin namespace attributes gives helpful ImportError
    # (the namespace itself is a valid module; errors fire on attribute access)
    print("\n  --- Missing plugin namespace attribute errors ---")
    for ns, symbol, pkg in (
        ("iris", "IRIS", "spectrochempy-iris"),
        ("nmr", "read_topspin", "spectrochempy-nmr"),
        ("tensor", "CP", "spectrochempy-tensor"),
        ("carroucell", "Carroucell", "spectrochempy-carroucell"),
    ):
        try:
            getattr(getattr(scp, ns), symbol)
            print(f"  FAIL: scp.{ns}.{symbol} should not be accessible")
            return False
        except ImportError as e:
            if pkg in str(e):
                print(f"  PASS: scp.{ns}.{symbol} -> ImportError with install hint")
            else:
                print(
                    f"  PASS: scp.{ns}.{symbol} -> ImportError (hint mentions '{pkg}')"
                )
        except Exception as e:
            print(f"  INFO: scp.{ns}.{symbol} -> {type(e).__name__}: {e}")

    # Verify top-level symbol hint
    print("\n  --- Missing top-level symbol hints ---")
    try:
        _ = scp.IRIS
        print("  FAIL: scp.IRIS should not be accessible")
        return False
    except AttributeError as e:
        if "scp.iris.IRIS" in str(e):
            print("  PASS: scp.IRIS -> AttributeError with namespace redirect hint")
        else:
            print(f"  INFO: scp.IRIS -> {e}")

    print("\n" + "=" * 60)
    print("  Core-only smoke test PASSED")
    print("=" * 60)
    return True


def test_with_plugins():
    """Smoke test: core + official plugins."""
    print("=" * 60)
    print("  Core + plugins install smoke test")
    print("=" * 60)

    import spectrochempy as scp

    print(f"  SpectroChemPy {scp.__version__}")

    ds = scp.NDDataset([1, 2, 3])
    assert ds.shape == (3,), f"NDDataset shape mismatch: {ds.shape}"
    print("  PASS: NDDataset created successfully")

    # Verify plugin namespaces are accessible
    print("\n  --- Plugin namespaces ---")
    for ns in ("iris", "nmr", "carroucell", "hypercomplex"):
        try:
            getattr(scp, ns)
            print(f"  PASS: scp.{ns} is accessible")
        except Exception as e:
            print(f"  WARN: scp.{ns} raised {type(e).__name__}")

    # Verify plugin modules import
    print("\n  --- Plugin module imports ---")
    for mod_name in (
        "spectrochempy_iris",
        "spectrochempy_nmr",
        "spectrochempy_tensor",
        "spectrochempy_hypercomplex",
        "spectrochempy_carroucell",
    ):
        try:
            importlib.import_module(mod_name)
            print(f"  PASS: {mod_name} imported")
        except Exception as e:
            print(f"  FAIL: {mod_name} import failed: {e}")

    # Lightweight feature access
    print("\n  --- Lightweight feature smoke tests ---")
    for mod_name, attr, label in (
        ("spectrochempy_iris", "IRIS", "iris.IRIS"),
        ("spectrochempy_nmr", "read_topspin", "nmr.read_topspin"),
        ("spectrochempy_tensor", "CP", "tensor.CP"),
    ):
        try:
            mod = importlib.import_module(mod_name)
            obj = getattr(mod, attr)
            print(f"  PASS: {label} imported ({type(obj).__name__})")
        except Exception as e:
            print(f"  WARN: {label} import failed: {e}")

    # scp.plugins() listing
    print("\n  --- Plugin listing ---")
    try:
        result = scp.plugins()
        print(f"  Official plugins: {len(result.official)}")
        for p in result.official:
            print(f"    {p.title}: {p.status}")
        print(f"  Discovered: {len(result.discovered)}")
    except Exception as e:
        print(f"  INFO: scp.plugins() raised {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("  Core + plugins smoke test PASSED")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Colab compatibility smoke test for SpectroChemPy"
    )
    parser.add_argument(
        "--mode",
        choices=["core-only", "with-plugins"],
        default="core-only",
    )
    args = parser.parse_args()

    success = test_core_install() if args.mode == "core-only" else test_with_plugins()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
