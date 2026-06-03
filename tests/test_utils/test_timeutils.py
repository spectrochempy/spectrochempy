# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from spectrochempy.utils.timeutils import timeit


def test_timeit_context_manager():
    """Test that timeit measures elapsed time and stores readout."""
    with timeit("test block") as timer:
        sum(range(1000))

    assert timer.time >= 0
    assert "test block" in timer.readout
    assert "seconds" in timer.readout


def test_timeit_test_only_default():
    """Test that test_only=True (default) does not print during normal runs."""
    with timeit("silent block") as timer:
        pass

    assert hasattr(timer, "readout")
    assert timer.time >= 0
