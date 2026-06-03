# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy.utils.coordrange import trim_ranges


# ======================================================================================
# trim_ranges
# ======================================================================================
def test_trim_ranges():
    # empty
    r = trim_ranges()
    assert r == []

    # single unordered pair
    r = trim_ranges(3, 2)
    assert r[0] == [2, 3]

    # multiple overlapping ranges
    r = trim_ranges((3, 2), (4.4, 10), (4, 5))
    assert r[-1] == [4, 10]
    assert r == [[2, 3], [4, 10]]

    # reversed output order
    r = trim_ranges((3, 2), (4.4, 10), (4, 5), reversed=True)
    assert r == [[10, 4], [3, 2]]

    # single range with float values
    r = trim_ranges((1.5, 5.5))
    assert r == [[1.5, 5.5]]

    # identical ranges
    r = trim_ranges((0, 5), (0, 5))
    assert r == [[0, 5]]

    # adjacent ranges (merged by implementation)
    r = trim_ranges((0, 2), (2, 4))
    assert r == [[0, 4]]

    # fully contained range
    r = trim_ranges((0, 10), (3, 7))
    assert r == [[0, 10]]

    # negative values
    r = trim_ranges((-5, 5), (-1, 1))
    assert r == [[-5, 5]]
