"""
Characterization tests for history and provenance semantics.

These tests document CURRENT behavior only.
They are not normative — they capture what the code does, not what it should do.

Do not modify history behavior.  Do not add helper functions.
Do not redesign provenance architecture.

Purpose: establish a precise baseline before discussing any future changes.

See also: audit/~metadata-architecture-audit.md §4, §5
"""


import numpy as np
import pytest

import spectrochempy as scp

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

HISTORY_PREFIX_LEN = len("YYYY-MM-DD HH:MM:SS+HH:MM> ")


def _entry(value: str) -> str:
    """Return the text portion of a history entry, stripping the timestamp prefix."""
    return value[HISTORY_PREFIX_LEN:]


def _entries(ds) -> list[str]:
    """Return the text portions of all history entries."""
    return [_entry(h) for h in ds.history]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ds1():
    """1D dataset with one history entry."""
    return scp.NDDataset([1.0, 2.0, 3.0], name="ds1", title="test data")


@pytest.fixture
def ds2():
    """1D dataset with different data and one history entry."""
    return scp.NDDataset([4.0, 5.0, 6.0], name="ds2", title="test data")


@pytest.fixture
def ds_history_reset():
    """Dataset whose history has been explicitly reset for testing."""
    d = scp.NDDataset([1.0, 2.0, 3.0], name="reset_ds")
    d.history = ["Reset for test"]
    return d


@pytest.fixture
def ds2d():
    """2D dataset suitable for reductions that return NDDatasets."""
    return scp.NDDataset([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="ds2d")


# ===========================================================================
# 1. Single-source transformations
# ===========================================================================


class TestSingleSourceTransformations:
    """History behavior for operations on a single dataset."""

    def test_initial_history(self, ds1):
        """A freshly created NDDataset has an empty history list."""
        assert len(ds1.history) == 0

    def test_copy_preserves_history(self, ds1):
        """copy() preserves all history entries."""
        ds1.history = "Initial op"
        c = ds1.copy()
        assert len(c.history) == len(ds1.history)
        assert _entries(c) == _entries(ds1)

    def test_deep_copy_preserves_history(self, ds1):
        """Deep copy preserves all history entries."""
        ds1.history = "Initial op"
        c = ds1.copy(deep=True)
        assert len(c.history) == len(ds1.history)
        assert _entries(c) == _entries(ds1)

    def test_copy_with_keepname(self, ds1):
        """copy(keepname=True) preserves history (same as regular copy)."""
        ds1.history = "Initial op"
        c = ds1.copy(keepname=True)
        assert len(c.history) == len(ds1.history)

    def test_negate_appends(self, ds1):
        """Unary __neg__ appends a history entry."""
        ds1.history = "Initial op"
        result = -ds1
        assert len(result.history) == len(ds1.history) + 1
        assert _entry(result.history[-1]) == "Unary operation neg applied"

    def test_abs_appends(self, ds1):
        """__abs__ appends a history entry (uses _unary_op)."""
        ds1.history = "Initial op"
        result = abs(ds1)
        assert len(result.history) == len(ds1.history) + 1
        assert "abs" in _entry(result.history[-1])

    def test_slice_appends(self, ds1):
        """Slicing (__getitem__) appends a history entry."""
        result = ds1[0:2]
        assert len(result.history) == 1
        assert "Slice extracted" in _entry(result.history[-1])

    def test_slice_history_preserves_prior(self, ds1):
        """Slicing preserves all prior history entries."""
        ds1.history = "Prior op"
        result = ds1[0:2]
        # Prior history entry should be preserved
        assert _entry(result.history[0]) == "Prior op"
        assert "Slice extracted" in _entry(result.history[1])

    def test_transpose_appends(self, ds1):
        """Transpose appends a history entry."""
        result = ds1.T
        assert len(result.history) == 1
        assert "Data transposed" in _entry(result.history[-1])

    def test_squeeze_appends(self):
        """Squeeze preserves prior history and appends a new entry."""
        d = scp.NDDataset([[1.0, 2.0, 3.0]], name="squeeze_test")
        d.history = "Prior op"
        result = d.squeeze()
        assert len(result.history) == 2
        assert _entry(result.history[0]) == "Prior op"
        assert "Data squeezed" in _entry(result.history[-1])

    def test_reshape_appends(self, ds1):
        """Reshape appends a history entry."""
        result = ds1.reshape((3, 1))
        assert len(result.history) == 1
        assert "Data reshaped" in _entry(result.history[-1])

    def test_swapdims_appends(self):
        """Swapdims appends a history entry for 2D datasets."""
        d = scp.NDDataset([[1, 2], [3, 4]], name="swap_ds")
        result = d.swapdims("y", "x")
        assert len(result.history) == 1
        assert "Data swapped between dims" in _entry(result.history[-1])

    def test_ufunc_appends(self, ds1):
        """Ufunc (e.g., np.sin) appends a history entry."""
        ds1.history = "Initial"
        result = np.sin(ds1)
        assert len(result.history) == len(ds1.history) + 1
        assert _entry(result.history[-1]) == "Ufunc sin applied."

    def test_ufunc_multiple_calls_accumulate(self, ds1):
        """Multiple ufunc calls each append a history entry."""
        ds1.history = "Initial"
        result = np.sin(np.exp(ds1))
        assert len(result.history) == len(ds1.history) + 2

    def test_ufunc_title_wrapping(self, ds1):
        """Ufuncs not in __keep_title wrap the title."""
        ds1.title = "test data"
        result = np.sin(ds1)
        assert result.title == "sin(test data)"


# ===========================================================================
# 2. Reductions
# ===========================================================================


class TestReductions:
    """
    History behavior for reduction operations on 2D datasets.

    For 1D datasets, reductions with default axis return scalars (no history).
    Use 2D datasets with explicit dim to get NDDataset results.
    """

    def test_sum_appends(self, ds2d):
        """Sum with explicit dim appends."""
        result = ds2d.sum(dim="y")
        assert len(result.history) == 1
        assert "`sum`" in _entry(result.history[0])

    def test_mean_appends(self, ds2d):
        """Mean with explicit dim appends."""
        result = ds2d.mean(dim="y")
        assert len(result.history) == 1
        assert "`mean`" in _entry(result.history[0])

    def test_std_appends(self, ds2d):
        """Std with explicit dim appends."""
        result = ds2d.std(dim="y")
        assert len(result.history) == 1
        assert "`std`" in _entry(result.history[0])

    def test_var_appends(self, ds2d):
        """Var with explicit dim appends."""
        result = ds2d.var(dim="y")
        assert len(result.history) == 1
        assert "`var`" in _entry(result.history[0])

    def test_1d_sum_returns_scalar(self, ds1):
        """For 1D input, sum with default axis returns scalar, not NDDataset."""
        result = ds1.sum()
        assert not isinstance(result, scp.NDDataset)

    def test_trapezoid_replaces_history(self):
        """Trapezoid REPLACES history (not appends)."""
        d = scp.NDDataset([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="trapz_test")
        d.set_coordset(x=scp.Coord([0.1, 0.2, 0.3]))
        d.history = "Prior op"
        result = d.trapezoid(dim="x")
        assert len(result.history) == 1
        assert "trapezoid" in _entry(result.history[0])

    def test_simpson_replaces_history(self):
        """Simpson REPLACES history (not appends)."""
        d = scp.NDDataset([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="simp_test")
        d.set_coordset(x=scp.Coord([0.1, 0.2, 0.3]))
        d.history = "Prior op"
        result = d.simpson(dim="x")
        assert len(result.history) == 1
        assert "simpson" in _entry(result.history[0])

    def test_sum_with_keepdims_appends(self, ds2d):
        """Sum with keepdims=True still appends."""
        ds2d.history = "Prior"
        result = ds2d.sum(dim="y", keepdims=True)
        assert len(result.history) == 2
        assert "`sum`" in _entry(result.history[-1])

    def test_reduction_preserves_prior_history(self, ds2d):
        """Reduction appends to, doesn't replace, prior history."""
        ds2d.history = "Prior op"
        result = ds2d.sum(dim="y")
        assert len(result.history) == 2
        assert _entry(result.history[0]) == "Prior op"

    def test_argmin_returns_scalar(self, ds1):
        """Argmin returns scalar, not NDDataset."""
        result = ds1.argmin()
        assert isinstance(result, (int, np.integer))

    def test_argmax_returns_scalar(self, ds1):
        """Argmax returns scalar, not NDDataset."""
        result = ds1.argmax()
        assert isinstance(result, (int, np.integer))

    def test_diagonal_appends(self):
        """Diagonal preserves prior history and appends its own entry."""
        d = scp.NDDataset([[1, 2], [3, 4]], name="diag")
        d.history = "Prior"
        result = d.diagonal()
        assert len(result.history) == 2
        assert _entry(result.history[0]) == "Prior"
        assert "diagonal" in _entry(result.history[-1])

    def test_ptp_appends(self, ds2d):
        """Ptp with explicit dim appends."""
        result = ds2d.ptp(dim="y")
        assert len(result.history) == 1
        assert "`ptp`" in _entry(result.history[0])


# ===========================================================================
# 3. Multi-source operations
# ===========================================================================


class TestMultiSourceOperations:
    """History behavior for operations involving multiple datasets."""

    def test_add_appends(self, ds1, ds2):
        """Binary addition appends a history entry."""
        ds1.history = "Initial"
        result = ds1 + ds2
        assert len(result.history) == len(ds1.history) + 1
        assert "Binary operation" in _entry(result.history[-1])

    def test_add_discards_second_operand_history(self, ds1, ds2):
        """Binary addition discards the second operand's history entirely."""
        ds1.history = "From first"
        ds2.history = "From second"
        result = ds1 + ds2
        # History should come from ds1 (first operand) only
        assert _entry(result.history[0]) == "From first"
        # No entries from ds2's history
        assert "From second" not in _entry(result.history[0])

    def test_subtract_appends(self, ds1, ds2):
        """Binary subtraction appends."""
        ds1.history = "Initial"
        result = ds1 - ds2
        assert len(result.history) == len(ds1.history) + 1

    def test_multiply_appends(self, ds1, ds2):
        """Binary multiplication appends."""
        ds1.history = "Initial"
        result = ds1 * ds2
        assert len(result.history) == len(ds1.history) + 1

    def test_divide_appends(self, ds1, ds2):
        """Binary division appends."""
        ds1.history = "Initial"
        result = ds1 / ds2
        assert len(result.history) == len(ds1.history) + 1

    def test_iadd_appends_inplace(self, ds1):
        """In-place addition appends to self."""
        ds1.history = "Initial"
        original_len = len(ds1.history)
        ds1 += scp.NDDataset([4.0, 5.0, 6.0])
        assert len(ds1.history) == original_len + 1
        assert "Inplace binary op" in _entry(ds1.history[-1])

    def test_concatenate_replaces_history(self):
        """Concatenate REPLACES history with 'Created by concatenate'."""
        a = scp.NDDataset([[1.0, 2.0], [3.0, 4.0]], name="A")
        b = scp.NDDataset([[5.0, 6.0], [7.0, 8.0]], name="B")
        a.history = ["History A"]
        b.history = ["History B"]

        result = scp.concatenate(a, b)

        assert len(result.history) == 1
        assert _entry(result.history[0]) == "Created by concatenate"

    def test_importer_merge_replaces_history(self):
        """Importer merge replaces history with 'Merged from several files'."""
        a = scp.NDDataset([[1, 2], [3, 4]], name="A")
        b = scp.NDDataset([[5, 6], [7, 8]], name="B")
        a.history = ["History A"]
        b.history = ["History B"]
        a.origin = "test_a"
        b.origin = "test_b"

        from spectrochempy.core.readers.importer import merge_datasets

        result = merge_datasets([a, b], merge=True)

        assert len(result) == 1
        text = _entry(result[0].history[-1])
        assert text in ("Merged from several files", "Stacked from several files")


# ===========================================================================
# 4. Binary operations
# ===========================================================================


class TestBinaryOperations:
    """Detailed history behavior for binary operations."""

    def test_add_history_contains_operand_name(self, ds1, ds2):
        """Binary operation history includes the other operand's name."""
        ds1.history = "Initial"
        result = ds1 + ds2
        text = _entry(result.history[-1])
        assert "add" in text
        assert "ds2" in text

    def test_mul_history_contains_operand_name(self, ds1, ds2):
        """Binary multiplication includes 'mul' and the other name."""
        ds1.history = "Initial"
        result = ds1 * ds2
        text = _entry(result.history[-1])
        assert "mul" in text
        assert "ds2" in text

    def test_operations_with_scalar_have_history(self, ds1):
        """Binary op with a plain number still gets history."""
        ds1.history = "Initial"
        result = ds1 + 1.0
        assert len(result.history) == len(ds1.history) + 1


# ===========================================================================
# 5. History property edge cases
# ===========================================================================


class TestHistoryProperty:
    """History getter/setter edge cases."""

    def test_history_getter_formatted(self, ds1):
        """History entries have timestamp and value separated by '> '."""
        ds1.history = "Test entry"
        entry = ds1.history[0]
        assert "> " in entry

    def test_history_setter_list_resets(self, ds_history_reset):
        """Setting history = ['text'] resets then appends."""
        assert len(ds_history_reset.history) == 1
        assert "Reset for test" in _entry(ds_history_reset.history[0])

    def test_history_setter_string_appends(self, ds1):
        """Setting history = 'text' appends a single entry."""
        ds1.history = "Additional entry"
        assert len(ds1.history) == 1
        assert _entry(ds1.history[0]) == "Additional entry"

    def test_history_setter_none_is_noop(self, ds1):
        """Setting history = None does nothing."""
        ds1.history = "Some entry"
        original_len = len(ds1.history)
        ds1.history = None
        assert len(ds1.history) == original_len

    def test_history_setter_empty_list_resets_to_empty(self, ds1):
        """Setting history = [] resets to empty."""
        ds1.history = "Some entry"
        ds1.history = []
        assert len(ds1.history) == 0

    def test_history_entries_maintain_order(self):
        """History entries maintain append order."""
        d = scp.NDDataset([1.0], name="order_test")
        d.history = "First"
        d.history = "Second"
        entries = _entries(d)
        assert entries == ["First", "Second"]

    def test_history_setter_list_uses_first_element(self):
        """Setting history = [list] takes first element and discards rest."""
        d = scp.NDDataset([1.0], name="list_test")
        d.history = ["First", "Second"]
        assert len(d.history) == 1
        assert _entry(d.history[0]) == "First"

    def test_multiple_strings_accumulate(self):
        """Multiple string assignments accumulate."""
        d = scp.NDDataset([1.0], name="accum_test")
        d.history = "A"
        d.history = "B"
        d.history = "C"
        assert _entries(d) == ["A", "B", "C"]


# ===========================================================================
# 6. NDDataset methods
# ===========================================================================


class TestNDDatasetMethods:
    """History behavior of NDDataset methods."""

    def test_set_coordset_preserves_history(self, ds1):
        """set_coordset does not modify history."""
        new_coord = scp.Coord([10.0, 20.0, 30.0])
        ds1.set_coordset(x=new_coord)
        assert len(ds1.history) == 0

    def test_ito_does_not_append_history(self, ds1):
        """Ito (unit conversion) does not add history."""
        ds1.ito("m")
        assert len(ds1.history) == 0

    def test_inplace_operations_maintain_history_chain(self):
        """In-place operations (via setter) maintain order."""
        d = scp.NDDataset([1.0], name="inplace_test")
        d.history = "Step 1"
        d.history = "Step 2"
        assert _entries(d) == ["Step 1", "Step 2"]


# ===========================================================================
# 7. Coordinate history
# ===========================================================================


class TestCoordHistory:
    """Coord objects do not have history."""

    def test_coord_has_no_history(self):
        """Coord lacks a history property."""
        c = scp.Coord([1, 2, 3])
        assert not hasattr(c, "history")

    def test_coord_operations_no_history(self):
        """Coord operations do not reference history."""
        c = scp.Coord([1, 2, 3])
        result = -c
        assert not hasattr(result, "history")


# ===========================================================================
# 8. History chain behavior
# ===========================================================================


class TestHistoryChain:
    """History chain behavior across multiple operations."""

    def test_chained_operations_accumulate(self):
        """Multiple chained operations each append."""
        d = scp.NDDataset([1.0, 2.0, 3.0], name="chain_test")
        d.history = "Initial"
        d = d + 1.0
        d = d * 2.0
        d = np.log(d)
        # Initial + add + mul + log = 4 total
        assert len(d.history) == 4

    def test_reset_breaks_chain(self):
        """Setting history = ['...'] breaks the provenance chain."""
        d = scp.NDDataset([1.0], name="break_test")
        d.history = "Old entry"
        d.history = ["New chain"]
        assert len(d.history) == 1
        assert _entry(d.history[0]) == "New chain"

    def test_history_not_compared_in_equality(self):
        """History is excluded from __eq__."""
        d1 = scp.NDDataset([1.0, 2.0], name="x")
        d2 = scp.NDDataset([1.0, 2.0], name="x")
        d1.history = ["Different history"]
        d2.history = ["Completely different"]
        assert d1 == d2


# ===========================================================================
# 9. History patterns
# ===========================================================================


class TestHistoryPatterns:
    """Determine whether history follows consistent entry format patterns."""

    def test_binary_op_format_includes_operation_and_operand(self, ds1, ds2):
        """Binary operation entries include the operation and operand name."""
        result = ds1 + ds2
        text = _entry(result.history[-1])
        assert "add" in text.lower()
        assert "ds2" in text

    def test_unary_op_includes_operator_name(self, ds1):
        """Unary operation entries include the operator name."""
        result = -ds1
        text = _entry(result.history[-1])
        assert "neg" in text.lower()

    def test_reduction_uses_consistent_format(self, ds2d):
        """
        All reductions use the same format string.

        Format: 'Dataset resulting from application of `{method}` method'
        """
        reductions = ["sum", "mean", "std", "var"]
        for method_name in reductions:
            result = getattr(ds2d, method_name)(dim="y")
            text = _entry(result.history[-1])
            assert (
                f"`{method_name}`" in text
            ), f"Expected method name {method_name} in history, got: {text}"

    def test_ufunc_format_consistent(self, ds1):
        """All ufuncs start with 'Ufunc' prefix."""
        for func in [np.sin, np.exp, np.cos, np.negative]:
            result = func(ds1)
            assert _entry(result.history[-1]).startswith("Ufunc")

    def test_ufunc_format_contains_func_name(self):
        """Ufunc format includes the ufunc name."""
        d = scp.NDDataset([1.0, 2.0], name="ufunc_test")
        result = np.sin(d)
        assert "sin" in _entry(result.history[-1])
        result = np.exp(d)
        assert "exp" in _entry(result.history[-1])


# ===========================================================================
# 10. Operations that lose history
# ===========================================================================


class TestHistoryLoss:
    """Operations known to lose prior history."""

    def test_concatenate_loses_input_histories(self):
        """Concatenate results have no trace of input histories."""
        a = scp.NDDataset([[1.0, 2.0], [3.0, 4.0]], name="A")
        b = scp.NDDataset([[5.0, 6.0], [7.0, 8.0]], name="B")
        a.history = ["Input A history"]
        b.history = ["Input B history"]
        result = scp.concatenate(a, b)
        assert _entry(result.history[0]) == "Created by concatenate"

    def test_integration_loses_prior_history(self):
        """Integration results lose prior operation history."""
        d = scp.NDDataset([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="int_loss_test")
        d.set_coordset(x=scp.Coord([0.1, 0.2, 0.3]))
        d.history = "Prior operation"
        result = d.trapezoid(dim="x")
        assert len(result.history) == 1
        assert "trapezoid" in _entry(result.history[0])

    def test_diagonal_preserves_prior_history(self):
        """Diagonal preserves prior history."""
        d = scp.NDDataset([[1, 2], [3, 4]], name="diag_test")
        d.history = "Prior operation"
        result = d.diagonal()
        assert len(result.history) == 2
        assert _entry(result.history[0]) == "Prior operation"


# ===========================================================================
# 11. History format details
# ===========================================================================


class TestHistoryFormat:
    """Format details of history entries."""

    def test_history_timestamp_format(self):
        """History timestamps follow ISO-like format with timezone."""
        d = scp.NDDataset([1.0], name="fmt_test")
        d.history = "Test operation"
        entry = d.history[0]
        # Format: "YYYY-MM-DD HH:MM:SS+HH:MM> Test operation"
        timestamp_part = entry.split("> ")[0]
        # Should contain date and time separated by space
        parts = timestamp_part.split(" ")
        assert len(parts) >= 2  # date + time+tz

    def test_history_capitalizes_first_letter(self):
        """History getter capitalizes the first letter of each entry."""
        d = scp.NDDataset([1.0], name="cap_test")
        d.history = "lowercase entry"
        entry = d.history[0]
        text = _entry(entry)
        assert text[0].isupper()

    def test_operation_uses_format_created_using(self):
        """From_numpy_method non-reduce ops use 'Created using method' format."""
        d = scp.NDDataset([1.0, 2.0], name="created_test")
        # empty_like uses _from_numpy_method path
        result = scp.empty_like(d)
        assert len(result.history) >= 1
        text = _entry(result.history[-1])
        # Several possible formats depending on code path
        assert "Created using method" in text or "Dataset resulting" in text


# ---------------------------------------------------------------------------
# If run directly, just print a summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("History semantics characterization tests")
    print("Run with: pytest -v test_history_semantics_baseline.py")
