# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================



"""

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ["assert_equal",
           "assert_array_equal",
           "assert_array_almost_equal",
           "assert_approx_equal",
           "raises",
           "ignore_warnings",
           "catch_warnings",
           "enable_deprecations_as_exceptions",
           "treat_deprecations_as_exceptions",
           "NumpyRNGContext",
           ]

import pytest
import functools
import sys
import types
import warnings
import six

from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_approx_equal)

from spectrochempy.utils.exceptions import SpectroChemPyDeprecationWarning
from pint.errors import DimensionalityError


# =============================================================================
# NumpyRNGContext
# =============================================================================

class NumpyRNGContext(object):
    """
    A context manager (for use with the ``with`` statement) that will seed the
    numpy random number generator (RNG) to a specific value, and then restore
    the RNG state back to whatever it was before.

    This is primarily intended for use in the astropy testing suit, but it
    may be useful in ensuring reproducibility of Monte Carlo simulations in a
    science context.

    Parameters
    ----------
    seed : int
        The value to use to seed the numpy RNG

    Examples
    --------
    A typical use case might be::

        with NumpyRNGContext(<some seed value you pick>):
            from numpy import random

            randarr = random.randn(100)
            ... run your test using `randarr` ...

        #Any code using numpy.random at this indent level will act just as it
        #would have if it had been before the with statement - e.g. whatever
        #the default seed is.


    """
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        from numpy import random

        self.startstate = random.get_state()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        from numpy import random

        random.set_state(self.startstate)


# =============================================================================
# raises and assertions (mostly copied from astropy)
# =============================================================================

def assert_equal_units(unit1, unit2):
    try:
        x = (1. * unit1) / (1. * unit2)
    except DimensionalityError:
        return False
    if x.dimensionless:
        return True
    False


class raises(object):
    """
    A decorator to mark that a test should raise a given exception.
    Use as follows::

        @raises(ZeroDivisionError)
        def test_foo():
            x = 1/0

    This can also be used a context manager, in which case it is just
    an alias for the ``pytest.raises`` context manager (because the
    two have the same name this help avoid confusion by being
    flexible).
    """

    # pep-8 naming exception -- this is a decorator class
    def __init__(self, exc):
        self._exc = exc
        self._ctx = None

    def __call__(self, func):
        @functools.wraps(func)
        def run_raises_test(*args, **kwargs):
            pytest.raises(self._exc, func, *args, **kwargs)
        return run_raises_test

    def __enter__(self):
        self._ctx = pytest.raises(self._exc)
        return self._ctx.__enter__()

    def __exit__(self, *exc_info):
        return self._ctx.__exit__(*exc_info)


_deprecations_as_exceptions = False
_include_spectrochempy_deprecations = True

def enable_deprecations_as_exceptions(include_spectrochempy_deprecations=True):
    """
    Turn on the feature that turns deprecations into exceptions.
    """
    global _deprecations_as_exceptions
    _deprecations_as_exceptions = True

    global _include_spectrochempy_deprecations
    _include_spectrochempy_deprecations = include_spectrochempy_deprecations


def treat_deprecations_as_exceptions():
    """
    Turn all DeprecationWarnings (which indicate deprecated uses of
    Python itself or Numpy, but not within SpectroChemPy, where we use our
    own deprecation warning class) into exceptions so that we find
    out about them early.

    This completely resets the warning filters and any "already seen"
    warning state.
    """
    # First, totally reset the warning state
    for module in list(six.itervalues(sys.modules)):
        # We don't want to deal with six.MovedModules, only "real"
        # modules.
        if (isinstance(module, types.ModuleType) and
            hasattr(module, '__warningregistry__')):
            del module.__warningregistry__

    if not _deprecations_as_exceptions:
        return

    warnings.resetwarnings()

    # Hide the next couple of DeprecationWarnings
    warnings.simplefilter('ignore', DeprecationWarning)
    # Here's the wrinkle: a couple of our third-party dependencies
    # (py.test and scipy) are still using deprecated features
    # themselves, and we'd like to ignore those.  Fortunately, those
    # show up only at import time, so if we import those things *now*,
    # before we turn the warnings into exceptions, we're golden.
    try:
        # A deprecated stdlib module used by py.test
        import compiler
    except ImportError:
        pass

    try:
        import scipy
    except ImportError:
        pass

    # Now, start over again with the warning filters
    warnings.resetwarnings()
    # Now, turn DeprecationWarnings into exceptions
    warnings.filterwarnings("error", ".*", DeprecationWarning)

    # Only turn spectrochempy deprecation warnings into exceptions if requested
    if _include_spectrochempy_deprecations:
        warnings.filterwarnings("error", ".*", SpectroChemPyDeprecationWarning)

    if sys.version_info[:2] == (2, 6):
        # py.test's warning.showwarning does not include the line argument
        # on Python 2.6, so we need to explicitly ignore this warning.
        warnings.filterwarnings(
            "ignore",
            r"functions overriding warnings\.showwarning\(\) must support "
            r"the 'line' argument",
            DeprecationWarning)

    if sys.version_info[:2] >= (3, 4):
        # py.test reads files with the 'ur' flag, which is now
        # deprecated in Python 3.4.
        warnings.filterwarnings(
            "ignore",
            r"'ur' mode is deprecated",
            DeprecationWarning)

        # BeautifulSoup4 triggers a DeprecationWarning in stdlib's
        # html module.x
        warnings.filterwarnings(
            "ignore",
            r"The strict argument and mode are deprecated\.",
            DeprecationWarning)
        warnings.filterwarnings(
            "ignore",
            r"The value of convert_charrefs will become True in 3\.5\. "
            r"You are encouraged to set the value explicitly\.",
            DeprecationWarning)

    if sys.version_info[:2] >= (3, 5):
        # py.test raises this warning on Python 3.5.
        # This can be removed when fixed in py.test.
        # See https://github.com/pytest-dev/pytest/pull/1009
        warnings.filterwarnings(
            "ignore",
            r"inspect\.getargspec\(\) is deprecated, use "
            r"inspect\.signature\(\) instead",
            DeprecationWarning)


class catch_warnings(warnings.catch_warnings):
    """
    A high-powered version of warnings.catch_warnings to use for testing
    and to make sure that there is no dependence on the order in which
    the tests are run.

    This completely blitzes any memory of any warnings that have
    appeared before so that all warnings will be caught and displayed.

    ``*args`` is a set of warning classes to collect.  If no arguments are
    provided, all warnings are collected.

    Use as follows::

        with catch_warnings(MyCustomWarning) as w:
            do.something.bad()
        assert len(w) > 0
    """
    def __init__(self, *classes):
        super(catch_warnings, self).__init__(record=True)
        self.classes = classes

    def __enter__(self):
        warning_list = super(catch_warnings, self).__enter__()
        treat_deprecations_as_exceptions()
        if len(self.classes) == 0:
            warnings.simplefilter('always')
        else:
            warnings.simplefilter('ignore')
            for cls in self.classes:
                warnings.simplefilter('always', cls)
        return warning_list

    def __exit__(self, type, value, traceback):
        treat_deprecations_as_exceptions()


class ignore_warnings(catch_warnings):
    """
    This can be used either as a context manager or function decorator to
    ignore all warnings that occur within a function or block of code.

    An optional category option can be supplied to only ignore warnings of a
    certain category or categories (if a list is provided).
    """

    def __init__(self, category=None):
        super(ignore_warnings, self).__init__()

        if isinstance(category, type) and issubclass(category, Warning):
            self.category = [category]
        else:
            self.category = category

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Originally this just reused self, but that doesn't work if the
            # function is called more than once so we need to make a new
            # context manager instance for each call
            with self.__class__(category=self.category):
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        retval = super(ignore_warnings, self).__enter__()
        if self.category is not None:
            for category in self.category:
                warnings.simplefilter('ignore', category)
        else:
            warnings.simplefilter('ignore')
        return retval

