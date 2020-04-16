# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
This module implements the class |CoordRange|.
"""

__all__ = ['CoordRange']


# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------
from traitlets import (HasTraits, List, Bool, TraitType, TraitError, class_of)


# ======================================================================================================================
# Range trait type
# ======================================================================================================================

class Range(List):
    """
    Create a trait with two values defining an ordered range of values,
    with an optional sampling parameters

    Parameters
    ----------

    trait : TraitType [ optional ]
        the type for restricting the contents of the Container.
        If unspecified, types are not checked.

    default_value : SequenceType [ optional ]
        The default value for the Trait.  Must be list/tuple/set, and
        will be cast to the container type.


    Examples
    --------

    >>> class MyClass(HasTraits):
    ...     r = Range([10,5])  # Initialized with some default values

    >>> c = MyClass()
    >>> print(c.r) # the list is ordered
    [5, 10]
    >>> c.r = [1, 3, 5]
    Traceback (most recent call last):
     ...
    traitlets.traitlets.TraitError : The 'r' trait of a type instance must be of length 2 exactly, but a value of [1, 3, 5] was specified.

    """
    klass = list
    _cast_types = (tuple,)
    
    # Describe the trait type
    info_text = 'an ordered interval trait'
    allow_none = True
    
    def __init__(self, default_value=None, **kwargs):
        
        super(Range, self).__init__(trait=None, default_value=default_value,
                                    **kwargs)
        pass
    
    def length_error(self, obj, value):
        e = "The '%s' trait of %s instance must be of length 2 exactly," \
            " but a value of %s was specified." \
            % (self.name, class_of(obj), value)
        raise TraitError(e)
    
    def validate_elements(self, obj, value):
        if value is None or len(value) == 0:
            return
        length = len(value)
        if length != 2:
            self.length_error(obj, value)
        value.sort()
        value = super(Range, self).validate_elements(obj, value)
        return value
    
    def validate(self, obj, value):
        
        value = super(Range, self).validate(object, value)
        value = self.validate_elements(obj, value)
        
        return value


# ======================================================================================================================
# CoordRange
# ======================================================================================================================
class CoordRange(HasTraits):
    """Set of ordered, non intersecting intervals
    e.g. [[a, b], [c, d]] with a < b < c < d or a > b > c > d.
    """
    # TODO: May use also units ???
    ranges = List(Range)
    reversed = Bool

    # ..................................................................................................................
    def __call__(self, *ranges, **kwargs):
        """
        Parameters
        -----------
        ranges :  iterable
            An interval or a set of intervals.
            set of  intervals. If none is given, the range
            will be a set of an empty interval [[]]. The interval limits do not
            need to be ordered, and the intervals do not need to be distincts.
        reversed : bool, optional.
            The intervals are ranked by decreasing order if True
            or increasing order if False.
        """
        # super(CoordRange, self).__init__(**kwargs)
        self.reversed = kwargs.get('reversed', False)
        if len(ranges) == 0:
            # first case: no argument passed, returns an empty range
            self.ranges = []
        elif len(ranges) == 2 and all(
                isinstance(elt, (int, float)) for elt in ranges):
            # second case: a pair of scalars has been passed
            # using the Interval class, we have autochecking of the interval
            # validity
            self.ranges = [list(map(float, ranges))]
        else:
            # third case: a set of pairs of scalars has been passed
            self._cleanranges(ranges)
        if self.ranges:
            self._cleanranges(self.ranges)
        return self.ranges

    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def _cleanranges(self, ranges):
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


CoordRange = CoordRange()

# ======================================================================================================================
if __name__ == '__main__':
    pass
