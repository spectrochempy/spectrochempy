# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import numpy as np


# ======================================================================================
# Public methods
# ======================================================================================
def get_n_decimals(n, sigdigits=3):
    """

    Parameters
    ----------
    n : float
    sigdigits : int, optional, default: 3
        Number of significant digits.

    Returns
    -------
    int
        number of significant decimals to use when rounding float.
    """
    try:
        n_decimals = sigdigits - int(np.floor(np.log10(abs(n)))) - 1
    except OverflowError:
        n_decimals = 2
    return n_decimals


def spacings(arr, sd=4):
    """
    Return a scalar for the spacing in the one-dimensional input array
    (if it is uniformly spaced, else return an array of the different spacings.

    Parameters
    ----------
    arr : 1D np.array
    sd : int, optional, default: 4
        Number of significant digits.

    Returns
    -------
    float or array
        Spacing or list of spacing in the given array.
    """
    spacings = np.diff(arr)
    # we need to take into account only the significant digits
    nd = get_n_decimals(spacings.max(), sd)
    spacings = list(set(np.around(spacings, nd)))

    if len(spacings) == 1:
        # uniform spacing
        return spacings[0]
    else:
        return spacings


def gt_eps(arr):
    """
    Check that an array has at least some values greater than epsilon.

    Parameters
    ----------
    arr : array to check

    Returns
    --------
    bool : results of checking
        True means that at least some values are greater than epsilon.
    """
    from spectrochempy.utils.constants import EPSILON

    return np.any(arr > EPSILON)


def largest_power_of_2(value):
    """
    Find the nearest power of two equal to or larger than a value.

    Parameters
    ----------
    value : int
        Value to find the nearest power of two equal to or larger than.

    Returns
    -------
    pw : int
        Power of 2.
    """
    return int(pow(2, np.ceil(np.log(value) / np.log(2))))

    #
    #
    # def make_func_from(func, first=None):
    # """


#     Create a new func with its arguments from another func and a new signature.
#     """
#     code_obj = func.__code__
#     new_varnames = list(code_obj.co_varnames)
#     if first:
#         new_varnames[0] = first
#     new_varnames = tuple(new_varnames)
#     new_code_obj = _codechange(
#         code_obj,
#         changes={
#             "co_varnames": new_varnames,
#         },
#     )
#     modified = types.FunctionType(
#         new_code_obj,
#         func.__globals__,
#         func.__name__,
#         func.__defaults__,
#         func.__closure__,
#     )
#     modified.__doc__ = func.__doc__
#     return modified
#


#
# def makedirs(newdir):
#     """
#     Works the way a good mkdir should :
#         - already exists, silently complete.
#         - regular file in the way, raise an exception.
#         - parent directory(ies) does not exist, make them as well.
#     """
#     # from active recipes http://code.activestate.com/recipes/82465-a-friendly-mkdir/
#
#     newdir = os.path.expanduser(newdir)
#     if os.path.isdir(newdir):
#         pass
#     elif os.path.isfile(newdir):
#         raise OSError(
#             "a file with the same name as the desired "
#             "dir, '%s', already exists." % newdir
#         )
#     else:
#         head, tail = os.path.split(newdir)
#         if head and not os.path.isdir(head):
#             makedirs(head)
#         # print "_mkdir %s" % repr(newdir)
#         if tail:
#             os.mkdir(newdir)
#

#
# def makestr(li):
#     """
#     Make a string from a list of string.
#     """
#
#     if is_sequence(li):
#         li = " ".join(map(str, li))
#     li = li.replace("$", "")
#     li = li.replace(" ", r"\ ")
#     li = r"$%s$" % li
#     return li
#

#
# def multisort(*args, **kargs):
#     z = list(zip(*args))
#     z = sorted(z, key=itemgetter(kargs.get("index", 0)))
#     return list(zip(*z))
#

# def primefactors(n):
#     from itertools import chain
#
#     result = []
#     for i in chain([2], range(3, n + 1, 2)):
#         s = 0
#         while n % i == 0:  # a good place for mod
#             n /= i
#             s += 1
#         result.extend([i] * s)  # avoid another for loop
#         if n == 1:
#             return result


#
# @contextmanager
# def silence():
#     """
#     A context manager that silences sys.stdout and sys.stderr.
#     """
#
#     old_stdout = sys.stdout
#     old_stderr = sys.stderr
#     sys.stdout = _DummyFile()
#     sys.stderr = _DummyFile()
#     yield
#     sys.stdout = old_stdout
#     sys.stderr = old_stderr
#


#
# def srepr(arg):
#     if is_sequence(arg):
#         return "<" + ", ".join(srepr(x) for x in arg) + ">"
#     return repr(arg)
