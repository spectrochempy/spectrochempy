# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module implements the class _CoordRange.
"""

__all__ = __slots__ = ["trim_ranges"]

from traitlets import HasTraits, List, Bool

from spectrochempy.utils.traits import Range


# ======================================================================================================================
# _CoordRange
# ======================================================================================================================
class _CoordRange(HasTraits):
    # TODO: May use also units ???
    ranges = List(Range())
    reversed = Bool()

    # ..........................................................................
    def __init__(self, *ranges, reversed=False):

        self.reversed = reversed
        if len(ranges) == 0:
            # first case: no argument passed, returns an empty range
            self.ranges = []
        elif len(ranges) == 2 and all(isinstance(elt, (int, float)) for elt in ranges):
            # second case: a pair of scalars has been passed
            # using the Interval class, we have autochecking of the interval
            # validity
            self.ranges = [list(map(float, ranges))]
        else:
            # third case: a set of pairs of scalars has been passed
            self._clean_ranges(ranges)
        if self.ranges:
            self._clean_ranges(self.ranges)

    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------
    # ..........................................................................
    def _clean_ranges(self, ranges):
        """Sort and merge overlapping ranges
        It works as follows::
        1. orders each interval
        2. sorts intervals
        3. merge overlapping intervals
        4. reverse the orders if required
        """
        # transforms each pairs into valid interval
        # should generate an error if a pair is not valid
        ranges = [list(range) for range in ranges]
        # order the ranges
        ranges = sorted(ranges, key=lambda r: min(r[0], r[1]))
        cleaned_ranges = [ranges[0]]
        for range in ranges[1:]:
            if range[0] <= cleaned_ranges[-1][1]:
                if range[1] >= cleaned_ranges[-1][1]:
                    cleaned_ranges[-1][1] = range[1]
            else:
                cleaned_ranges.append(range)
        self.ranges = cleaned_ranges
        if self.reversed:
            for range in self.ranges:
                range.reverse()
            self.ranges.reverse()


def trim_ranges(*ranges, reversed=False):
    """
    Set of ordered, non intersecting intervals.

    An ordered set of ranges is contructed from the inputs and returned.
    *e.g.,* [[a, b], [c, d]] with a < b < c < d or a > b > c > d.

    Parameters
    -----------
    *ranges :  iterable
        An interval or a set of intervals.
        set of  intervals. If none is given, the range will be a set of an empty interval [[]]. The interval
        limits do not need to be ordered, and the intervals do not need to be distincts.
    reversed : bool, optional
        The intervals are ranked by decreasing order if True or increasing order if False.

    Returns
    -------
    ordered
        list of ranges.

    Examples
    --------

    >>> scp.trim_ranges([1, 4], [7, 5], [6, 10])
    [[1, 4], [5, 10]]
    """
    return _CoordRange(*ranges, reversed=reversed).ranges


# ======================================================================================================================
if __name__ == "__main__":
    pass
