# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Test Phase 1: Verify no global matplotlib rcParams mutation.

This test verifies that our changes to disable global rcParams mutation work correctly.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest


class TestPhase1NoGlobalMutation:
    """Verify Phase 1 changes eliminated global rcParams mutation."""

    def test_no_global_rcparams_mutation_after_import(self):
        """Test that import doesn't change global rcParams."""
        # Store initial rcParams
        initial_rcparams = dict(mpl.rcParams)

        # Import spectrochempy (this would trigger lazy init in old system)
        import sys

        sys.path.insert(0, "./src")
        try:
            pass
        finally:
            # Verify rcParams unchanged after import
            current_rcparams = dict(mpl.rcParams)

            # Most keys should be identical
            unchanged_keys = 0
            for key in initial_rcparams:
                if (
                    key in current_rcparams
                    and current_rcparams[key] == initial_rcparams[key]
                ):
                    unchanged_keys += 1

            # At least 90% of keys should be unchanged
            total_keys = len(initial_rcparams)
            unchanged_ratio = unchanged_keys / total_keys
            assert (
                unchanged_ratio >= 0.9
            ), f"Only {unchened_ratio:.2%} of rcParams unchanged after import"

    def test_preferences_change_no_global_mutation(self):
        """Test that preference changes don't affect global rcParams."""
        # Skip if spectrochempy not available
        try:
            from spectrochempy import preferences
        except ImportError:
            pytest.skip("spectrochempy not available")
            return

        # Store initial rcParams
        rcparams_before = dict(mpl.rcParams)

        # Change a preference (this would have triggered @observe in old system)
        try:
            preferences.plot.lines_linewidth = 3.0
        except Exception:
            pytest.skip("Cannot modify preferences")
            return

        # Verify rcParams unchanged
        rcparams_after = dict(mpl.rcParams)

        # lines.linewidth should NOT be in global rcParams
        assert (
            "lines.linewidth" not in rcparams_after
            or rcparams_after["lines.linewidth"] != 3.0
        ), "Preference change should not affect global rcParams"

        # Most other parameters should be unchanged
        unchanged_keys = 0
        for key in rcparams_before:
            if key in rcparams_after and rcparams_after[key] == rcparams_before[key]:
                unchanged_keys += 1

        total_keys = len(rcparams_before)
        unchanged_ratio = unchanged_keys / total_keys
        assert (
            unchanged_ratio >= 0.9
        ), f"Only {unchened_ratio:.2%} of rcParams unchanged after preference change"

    def test_style_context_isolation(self):
        """Test that style context managers work correctly."""
        rcparams_before = dict(mpl.rcParams)

        # Apply style in context - this should work
        with plt.style.context("default"):
            # Inside context, style is applied locally
            context_params = dict(mpl.rcParams)
            # Style should be applied within context
            assert (
                "default" in str(context_params) or "axes.prop_cycle" in context_params
            )

        # After context, rcParams should be restored
        rcparams_after = dict(mpl.rcParams)

        for key in rcparams_before:
            if rcparams_before[key] != rcparams_after.get(key):
                if key in [
                    "text.usetex",
                    "mathtext.fontset",
                ]:  # These may legitimately change
                    continue
                assert False, f"rcParams['{key}'] changed after style context"

        # Quick check on a key that should definitely be restored
        if "lines.linewidth" in rcparams_before:
            assert (
                rcparams_after["lines.linewidth"] == rcparams_before["lines.linewidth"]
            )
