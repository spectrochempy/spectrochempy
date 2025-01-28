# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy.utils.coordrange import trim_ranges


# ======================================================================================
# trim_ranges
# ======================================================================================
def test_trim_ranges():
    r = trim_ranges()
    assert r == []

    r = trim_ranges(3, 2)
    assert r[0] == [2, 3]

    r = trim_ranges((3, 2), (4.4, 10), (4, 5))
    assert r[-1] == [4, 10]
    assert r == [[2, 3], [4, 10]]

    r = trim_ranges((3, 2), (4.4, 10), (4, 5), reversed=True)
    assert r == [[10, 4], [3, 2]]
