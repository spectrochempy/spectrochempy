# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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


""".. _misc:

Various methods used in other part of the program

"""
__all__ = _methods = ["closer_power_of_two",
                      "create_traitsdoc",
                      "dict_compare",
                      'htmldoc',
                      "ignored",
                      "is_iterable",
                      "is_sequence",
                      "is_number",
                      "silence",
                      "makedirs",
                      "multisort",
                      "numpyprintoptions",
                      'makestr',
                      'srepr',
                      'largest_power_of_2']

import re
from operator import itemgetter
import os
import numpy as np
import sys
from contextlib import contextmanager
from spectrochempy.extern.uncertainties.core import Variable


# =============================================================================
# Ignored context
# =============================================================================

try:
    from contextlib import ignored
except ImportError:
    @contextmanager
    def ignored(*exceptions):
        """A context manager for ignoring exceptions.  Equivalent to::

            try:
                <body>
            except exceptions:
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


# =============================================================================
# dummy file
# =============================================================================

class _DummyFile(object):
    """A noop writeable object."""

    def write(self, s):
        pass


# =============================================================================
# silence
# =============================================================================

@contextmanager
def silence():
    """A context manager that silences sys.stdout and sys.stderr."""

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _DummyFile()
    sys.stderr = _DummyFile()
    yield
    sys.stdout = old_stdout
    sys.stderr = old_stderr


# =============================================================================
# check for a number
# =============================================================================

def is_number(x):
    try:
        x + 1
        return True
    except TypeError:
        return False


# =============================================================================
# sequence check
# =============================================================================

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


# =============================================================================
# sorting
# =============================================================================

def multisort(*args, **kargs):
    z = list(zip(*args))
    z = sorted(z, key=itemgetter(kargs.get('index', 0)))
    return list(zip(*z))


# =============================================================================
# makedirs
# =============================================================================

def makedirs(
        newdir):  # from active recipes http://code.activestate.com/recipes/82465-a-friendly-mkdir/
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
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


# =============================================================================
# Dictionary comparison
# =============================================================================

def dict_compare(d1, d2, check_equal_only=True):
    """ Compare two dictionaries

    Examples
    --------

        >>> x = dict(a=1, b=2)
        >>> y = dict(a=2, b=2)
        >>> added, removed, modified, same = \
        ... dict_compare(x, y, check_equal_only=False)
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


# =============================================================================
# numpy print options
# =============================================================================

def numpyprintoptions(precision=4, threshold=6, edgeitems=2, suppress=True,
                      formatter=None, spc=4, linewidth=130, **kargs):

    def _format_object(x):

        if hasattr(x, 'uncert_data'):
            x = x.uncert_data[()]

        if isinstance(x, (float, np.float_, np.float32)) :
            fmt = '{:{l}.0{prec}f}'.format(x,
                                prec=precision - 1,
                                l=precision + spc)
        elif isinstance(x, np.complex) :
            fmt =  '({:^{l}.0{prec}f}{:^+{lc}.0{prec}f}j)'.format(
                    x.real, x.imag,
                    prec=precision - 1, l=precision + spc, lc=precision)
        elif isinstance(x, np.ma.core.MaskedConstant):
            fmt = '  {}'.format(x)
        elif isinstance(x, Variable):
            fmt = '{:{l}.0{prec}f}+/-{:.0{prec}f}'.format(
                    x.nominal_value, x.std_dev,
                    prec=precision - 1, l=precision + spc)
        elif isinstance(x, np.int_) :
            fmt = '{:>{l}d}'.format(x, l=precision + spc)
        else:
            fmt = '{}'.format(x)

        return fmt

    if not formatter:
        spc = 4
        formatter = {
            'all': _format_object,
        }

    np.set_printoptions(precision=precision, threshold=threshold,
                        edgeitems=edgeitems, suppress=suppress,
                        formatter=formatter, linewidth=linewidth, **kargs)


# =============================================================================
# doc utilities
# =============================================================================

def create_traitsdoc(klass):
    # this function is for the sphinx documentation.
    # I could not make numpy trait docs working so the Property traits were
    # not autodocumented
    # this is a try to have this working
    import inspect
    import textwrap

    module = inspect.getmodule(klass)

    attrs = dict(inspect.getmembers(klass))['__base_traits__']
    attr_doc = ""
    for k, attr in sorted(attrs.items()):
        if attr.type == "property":
            kdoc = getattr(klass, "_get_{0}".format(k)).__doc__
            if kdoc is not None:
                attr_doc += "**{0}**\n\t{1}\n".format(k, kdoc)

    if not klass.__doc__.startswith(' ' * 4):
        klass.__doc__ = "    " + klass.__doc__
    doc = textwrap.dedent(klass.__doc__).strip()
    doc += "\n\nAttributes\n----------\n\n{attributes}\n"
    klass.__doc__ = doc.format(attributes=attr_doc)


def closer_power_of_two(value):
    """
    Find the nearest power of two equal to or larger than a value.

    Parameters
    ----------
    value : int
        Value to find nearest power of two equal to or larger than.

    Returns
    -------
    pw : int
        Power of 2.

    """
    return int(pow(2, np.ceil(np.log(value) / np.log(2))))


def htmldoc(text):
    """
    format docstring in html for a nice display in IPython
    :param text:
    :return:
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


def largest_power_of_2(value):
    """
    Find the largest and nearest power of two of a value

    Parameters
    ----------
    value: int

    Returns
    -------
    pw: int
        Power of 2.

    Examples
    --------

        >>> largest_power_of_2(1020)
        1024
        >>> largest_power_of_2(1025)
        2048
        >>> largest_power_of_2(0)
        2
        >>> largest_power_of_2(1)
        2


    """
    value = max(2, value)
    p = int(pow(2, np.ceil(np.log(value) / np.log(2))))
    return p