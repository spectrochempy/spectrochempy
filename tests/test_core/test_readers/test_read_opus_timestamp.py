# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Unit tests for OPUS acquisition-timestamp parsing (no test data required)."""

from types import SimpleNamespace

from spectrochempy.core.readers.read_opus import _get_timestamp_from


def test_timestamp_malformed_subsecond():
    """
    A malformed sub-second field must not break the read (#1036).

    Some OPUS files store a garbage fractional-seconds value such as
    ``10:31:19.-70``; ``_get_timestamp_from`` should fall back to whole-second
    precision instead of raising ``ValueError``.
    """
    params = SimpleNamespace(dat="25/03/2020", tim="10:31:19.-70 (GMT+1)")
    dt, timestamp = _get_timestamp_from(params)
    # GMT+1 -> UTC shifts 10:31:19 to 09:31:19, sub-second dropped
    assert (dt.year, dt.month, dt.day) == (2020, 3, 25)
    assert (dt.hour, dt.minute, dt.second) == (9, 31, 19)
    assert dt.microsecond == 0
    assert isinstance(timestamp, float)


def test_timestamp_valid_subsecond_preserved():
    """A well-formed sub-second field keeps microsecond precision."""
    params = SimpleNamespace(dat="25/03/2020", tim="10:31:19.123 (GMT+1)")
    dt, _ = _get_timestamp_from(params)
    assert (dt.hour, dt.second, dt.microsecond) == (9, 19, 123000)


def test_timestamp_iso_date_form():
    """The ``YYYY/MM/DD`` second-precision form is unaffected."""
    params = SimpleNamespace(dat="2020/03/25", tim="10:31:19 (GMT+1)")
    dt, _ = _get_timestamp_from(params)
    assert (dt.year, dt.hour, dt.second) == (2020, 9, 19)
