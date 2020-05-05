# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
Various methods and classes used in other part of the program

"""
__all__ = [

    "TYPE_INTEGER",
    "TYPE_COMPLEX",
    "TYPE_FLOAT",
    "HAS_PANDAS",
    "HAS_XARRAY",

    "make_new_object",
    "getdocfrom",
    "dict_compare",
    'htmldoc',
    "ignored",
    "is_iterable",
    "is_sequence",
    "is_number",
    "silence",
    "makedirs",
    "multisort",
    'makestr',
    'srepr',
    "spacing",

    
    #
]

import re
from operator import itemgetter
import os
import numpy as np
import sys
from contextlib import contextmanager
import inspect
import functools
from datetime import datetime
import uuid
import warnings


#
# constants
#
TYPE_INTEGER = (int, np.int_, np.int32, np.int64)
TYPE_FLOAT = (float, np.float_, np.float32, np.float64)
TYPE_COMPLEX = (complex, np.complex_, np.complex64, np.complex128)

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

def make_new_object(obj):
    """
    Make a new object of type obj

    Parameters
    ----------
    obj : the poject type

    Returns
    -------
    new : the new object of same type.

    """

    new = type(obj)()

    # new id and date
    new._id = "{}_{}".format(type(obj).__name__, str(uuid.uuid1()).split('-')[0])
    new._date = datetime.now()

    return new


# ======================================================================================================================
# Ignored context
# ======================================================================================================================

try:
    from contextlib import ignored
except ImportError:
    @contextmanager
    def ignored(*exceptions):
        """
        A context manager for ignoring exceptions.  Equivalent to::

            try :
                <body>
            except exceptions :
                pass

        Examples
        --------

            >>> import os
            >>> with ignored(OSError):
            ...     os.remove('file-that-does-not-exist')

        """

        try:
            yield
        except exceptions:
            pass


# ======================================================================================================================
# dummy file
# ======================================================================================================================

class _DummyFile(object):
    """
    A noop writeable object.

    """

    def write(self, s):
        pass


# ======================================================================================================================
# silence
# ======================================================================================================================

@contextmanager
def silence():
    """
    A context manager that silences sys.stdout and sys.stderr.

    """

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _DummyFile()
    sys.stderr = _DummyFile()
    yield
    sys.stdout = old_stdout
    sys.stderr = old_stderr


# ======================================================================================================================
# check for a number
# ======================================================================================================================

def is_number(x):
    try:
        if isinstance(x, np.ndarray):
            return False
        x + 1
        return True
    except TypeError:
        return False


# ======================================================================================================================
# Epsilon
# ======================================================================================================================

def gt_eps(arr):
    """lambda function to check that an array has at least some values
    greater than epsilon

    Parameters
    -----------
    arr : array to check

    Returns
    --------
    bool : results ot checking
        True means that at least some values are greater than epsilon

    """
    return np.any(arr > EPSILON)


# ======================================================================================================================
# sequence check
# ======================================================================================================================

def is_iterable(arg):
    """
    Determine if an object is iterable
    """
    return hasattr(arg, "__iter__")


def is_sequence(arg):
    """
    Determine if an object is iterable but is not a string
    """
    return (not hasattr(arg, 'strip')) and hasattr(arg, "__iter__")


# ======================================================================================================================
# sorting
# ======================================================================================================================

def multisort(*args, **kargs):
    z = list(zip(*args))
    z = sorted(z, key=itemgetter(kargs.get('index', 0)))
    return list(zip(*z))


# ======================================================================================================================
# makedirs
# ======================================================================================================================

def makedirs(newdir):
    """
    works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    # from active recipes http://code.activestate.com/recipes/82465-a-friendly-mkdir/

    newdir = os.path.expanduser(newdir)
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            makedirs(head)
        # print "_mkdir %s" % repr(newdir)
        if tail:
            os.mkdir(newdir)


# ======================================================================================================================
# Dictionary comparison
# ======================================================================================================================

def dict_compare(d1, d2, check_equal_only=True):
    """
    Compare two dictionaries

    Examples
    --------

        >>> x = dict(a=1, b=2)
        >>> y = dict(a=2, b=2)
        >>> added, removed, modified, same = dict_compare(x, y, check_equal_only=False)
        >>> print(added, removed, modified, same)
        set() set() {'a'} {'b'}
        >>> dict_compare(x, y)
        False

    """
    # from http://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python
    # modified to account for the comparison of list objects

    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys

    modified = added.union(removed)
    for o in intersect_keys:
        if is_sequence(d1[o]):
            if not is_sequence(d2[o]) or len(d1[o]) != len(d2[o]):
                modified.add(o)
            else:
                # in principe we vae here two list of same length
                for i1, i2 in zip(d1[o], d2[o]):
                    if np.any(i1 != i2):
                        modified.add(o)
        else:
            if is_sequence(d2[o]) or d1[o] != d2[o]:
                modified.add(o)

    same = set(o for o in intersect_keys if o not in modified)

    if not check_equal_only:
        return added, removed, modified, same
    else:
        if modified or removed or added:
            return False
        return True


# ======================================================================================================================
# doc utilities
# ======================================================================================================================
def getdocfrom(origin):
    def decorated(func):
        func.__doc__ = origin.__doc__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            return response

        return wrapper

    return decorated


def htmldoc(text):
    """
    format docstring in html for a nice display in IPython

    Parameters
    ----------
    text : str
        The string to convert to html

    Returns
    -------
    out : str
        the html string

    """
    p = re.compile("^(?P<name>.*:)(.*)", re.MULTILINE)  # To get the keywords
    html = p.sub(r'<b>\1</b>\2', text)
    html = html.replace('-', '')
    html = html.split('\n')
    while html[0].strip() == '':
        html = html[1:]  # suppress initial blank lines

    for i in range(len(html)):
        html[i] = html[i].strip()
        if i == 0:
            html[i] = "<h3>%s</h3>" % html[i]
        html[i] = html[i].replace('Parameters', '<h4>Parameters</h4>')
        html[i] = html[i].replace('Properties', '<h4>Properties</h4>')
        html[i] = html[i].replace('Methods', '<h4>Methods</h4>')
        if html[i] != '':
            if "</h" not in html[i]:
                html[i] = html[i] + '<br/>'
            if not html[i].strip().startswith('<'):
                html[i] = "&nbsp;&nbsp;&nbsp;&nbsp;" + html[i]
    html = "".join(html)

    return html


def srepr(arg):
    if is_sequence(arg):
        return '<' + ", ".join(srepr(x) for x in arg) + '>'
    return repr(arg)


def makestr(l):
    """
    make a string from a list of string

    """

    if is_sequence(l):
        l = " ".join(map(str, l))
    l = l.replace('$', '')
    l = l.replace(' ', '\ ')
    l = r'$%s$' % l
    return l


def primefactors(n):
    from itertools import chain
    result = []
    for i in chain([2], range(3, n + 1, 2)):
        s = 0
        while n % i == 0:  # a good place for mod
            n /= i
            s += 1
        result.extend([i] * s)  # avoid another for loop
        if n == 1:
            return result


def spacing(arr):
    """
    Return a scalar for the spacing in the one-dimensional input array (if it is uniformly spaced,
    else return an array of the different spacings
    
    Parameters
    ----------
    arr : 1D np.array

    Returns
    -------
    out : float or array
    
    """
    spacings = np.diff(arr)
    # we need to take into account only the significative digits ( but round to some decimals doesn't work
    # for very small number
    mantissa, twoexp = np.frexp(spacings)
    mantissa = mantissa.round(6)
    spacings = np.ldexp(mantissa, twoexp)
    spacings = list(set(abs(spacings)))
    if len(spacings) == 1:
        # uniform spacing
        return spacings[0]
    else:
        return spacings



