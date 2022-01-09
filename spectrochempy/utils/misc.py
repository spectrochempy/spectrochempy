# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Various methods and classes used in other part of the program.
"""
import re
import functools
from datetime import datetime, timezone
import uuid
import types
import warnings
import textwrap

import numpy as np
from quaternion import as_float_array, as_quat_array

__all__ = [
    "TYPE_INTEGER",
    "TYPE_COMPLEX",
    "TYPE_FLOAT",
    "TYPE_BOOL",
    "EPSILON",
    "INPLACE",
    "typequaternion",
    "make_new_object",
    "getdocfrom",
    "dict_compare",
    "htmldoc",
    "is_iterable",
    "is_sequence",
    "is_number",
    "spacing_",
    "largest_power_of_2",
    "get_component",
    "interleaved2quaternion",
    "interleaved2complex",
    "as_quaternion",
    "quat_as_complex_array",
    "get_n_decimals",
    "add_docstring",
]

#
# constants
#
TYPE_INTEGER = (int, np.int_, np.int32, np.int64, np.uint32, np.uint64)
TYPE_FLOAT = (float, np.float_, np.float32, np.float64)
TYPE_COMPLEX = (complex, np.complex_, np.complex64, np.complex128)
TYPE_BOOL = (bool, np.bool_)

EPSILON = epsilon = np.finfo(float).eps
"Minimum value before considering it as zero value."

INPLACE = "INPLACE"
"Flag used to specify inplace slicing."

typequaternion = np.dtype(np.quaternion)


def get_n_decimals(val, accuracy):
    if abs(val) > 0.0:
        nd = int(np.log10(abs(val) * accuracy))
    else:
        return 3
    if nd >= 0:
        nd = 1
    else:
        nd = -nd + 1
    return nd


# ======================================================================================================================
# Private methods
# ======================================================================================================================


def _codechange(code_obj, changes):
    code = types.CodeType
    names = [
        "co_argcount",
        "co_nlocals",
        "co_stacksize",
        "co_flags",
        "co_code",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_filename",
        "co_name",
        "co_firstlineno",
        "co_lnotab",
        "co_freevars",
        "co_cellvars",
    ]
    if hasattr(code, "co_kwonlyargcount"):
        names.insert(1, "co_kwonlyargcount")
    if hasattr(code, "co_posonlyargcount"):
        names.insert(1, "co_posonlyargcount")
    values = [changes.get(name, getattr(code_obj, name)) for name in names]
    return code(*values)


# ..............................................................................
class _DummyFile(object):
    """
    A writeable object.
    """

    def write(self, s):
        pass


# ======================================================================================================================
# Public methods
# ======================================================================================================================


def add_docstring(*args):
    """
    Decorator which add a docstring to the actual func doctring.
    """

    def new_doc(func):

        for item in args:
            item.strip()

        func.__doc__ = textwrap.dedent(func.__doc__).format(*args)
        return func

    return new_doc


def as_quaternion(*args):
    """
    Recombine the arguments to produce a numpy array with quaternion dtype.

    Parameters
    ----------
    *args : ndarray with dtype:float or complex
        The quaternion array components: If there is 4 components, then we assume it is the four compoents of the
        quaternion array: w, x, y, z. If there is only two, they are casted to complex and correspond respectively
        to w + i.x and y + j.z.
    """
    if len(args) == 4:
        # we assume here that the for components have been provided w, x, y, z
        w, x, y, z = args

    if len(args) == 2:
        r, i = args
        w, x, y, z = r.real, r.imag, i.real, i.imag

    data = as_quat_array(list(zip(w.flatten(), x.flatten(), y.flatten(), z.flatten())))
    return data.reshape(w.shape)


def quat_as_complex_array(arr):
    """
    Recombine the component of a quaternion array into a tuple of two complex array.

    Parameters
    ----------
    arr : quaternion ndarray
        The arr will be separated into (w + i.x) and (y + i.z).
    Returns
    -------
    tuple
        Tuple of two complex array.
    """
    if not arr.dtype == np.quaternion:
        # no change
        return arr

    wt, xt, yt, zt = as_float_array(arr).T
    w, x, y, z = wt.T, xt.T, yt.T, zt.T

    return (w + 1j * x), (y + 1j * z)


def dict_compare(d1, d2, check_equal_only=True):
    """
    Compare two dictionaries.

    Parameters
    ----------
    d1, d2 : dict
        Dictionaries to compare.
    check_equal_only : bool, optional default: True
        If check equal is True, then is the dictionaries are not equal, False is False
        returned. If check is False, for set of values are returned which contains
        added, removed, modified, or same values, respectively.

    Returns
    -------
    results
        Either  bool or a tupl e of for sets if check_equals_only is False.

    Examples
    --------

        >>> from spectrochempy.utils import dict_compare
        >>>
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


# ..........................................................................
def get_component(data, select="REAL"):
    """
    Take selected components of an hypercomplex array (RRR, RIR, ...).

    Parameters
    ----------
    data : ndarray
    select : str, optional, default='REAL'
        If 'REAL', only real component in all dimensions will be selected.
        Else a string must specify which real (R) or imaginary (I) component
        has to be selected along a specific dimension. For instance,
        a string such as 'RRI' for a 2D hypercomplex array indicated
        that we take the real component in each dimension except the last
        one, for which imaginary component is preferred.

    Returns
    -------
    component
        A component of the complex or hypercomplex array.

    .. warning::
        The definition is somewhat different from Bruker, as we order the component in the order of the dimensions in
        dataset:
        e.g., for dims = ['y','x'], 'IR' means that the `y` component is imaginary while the `x` is real.
    """
    if not select:
        return data

    new = data.copy()

    if select == "REAL":
        select = "R" * new.ndim

    w = x = y = z = None

    if new.dtype == typequaternion:
        w, x, y, z = as_float_array(new).T
        w, x, y, z = w.T, x.T, y.T, z.T
        if select == "R":
            new = w + x * 1j
        elif select == "I":
            new = y + z * 1j
        elif select == "RR":
            new = w
        elif select == "RI":
            new = x
        elif select == "IR":
            new = y
        elif select == "II":
            new = z
        else:
            raise ValueError(
                f"something wrong: cannot interpret `{select}` for hypercomplex (quaternion) data!"
            )

    elif new.dtype in TYPE_COMPLEX:
        w, x = new.real, new.imag
        if (select == "R") or (select == "RR"):
            new = w
        elif (select == "I") or (select == "RI"):
            new = x
        else:
            raise ValueError(
                f"something wrong: cannot interpret `{select}` for complex data!"
            )
    else:
        warnings.warn(
            f"No selection was performed because datasets with complex data have no `{select}` component. "
        )

    return new


# ..............................................................................
def getdocfrom(origin):
    def decorated(func):
        func.__doc__ = origin.__doc__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            return response

        return wrapper

    return decorated


# ..............................................................................
def gt_eps(arr):
    """
    Lambda function to check that an array has at least some values greater than epsilon.

    Parameters
    -----------
    arr : array to check

    Returns
    --------
    bool : results ot checking
        True means that at least some values are greater than epsilon.
    """
    return np.any(arr > EPSILON)


# ..............................................................................
def htmldoc(text):
    """
    Format docstring in html for a nice display in IPython.

    Parameters
    ----------
    text : str
        The string to convert to html.

    Returns
    -------
    out : str
        The html string.
    """
    p = re.compile("^(?P<name>.*:)(.*)", re.MULTILINE)  # To get the keywords
    html = p.sub(r"<b>\1</b>\2", text)
    html = html.replace("-", "")
    html = html.split("\n")
    while html[0].strip() == "":
        html = html[1:]  # suppress initial blank lines

    for i in range(len(html)):
        html[i] = html[i].strip()
        if i == 0:
            html[i] = "<h3>%s</h3>" % html[i]
        html[i] = html[i].replace("Parameters", "<h4>Parameters</h4>")
        html[i] = html[i].replace("Properties", "<h4>Properties</h4>")
        html[i] = html[i].replace("Methods", "<h4>Methods</h4>")
        if html[i] != "":
            if "</h" not in html[i]:
                html[i] = html[i] + "<br/>"
            if not html[i].strip().startswith("<"):
                html[i] = "&nbsp;&nbsp;&nbsp;&nbsp;" + html[i]
    html = "".join(html)

    return html


# ..............................................................................
def interleaved2complex(data):
    """
    Make a complex array from interleaved data.
    """
    return data[..., ::2] + 1j * data[..., 1::2]


# ..............................................................................
def interleaved2quaternion(data):
    """
    Make a complex array from interleaved data.
    """
    return data[..., ::2] + 1j * data[..., 1::2]


# ..............................................................................
def is_iterable(arg):
    """
    Determine if an object is iterable.
    """
    return hasattr(arg, "__iter__")


# ..............................................................................
def is_number(x):
    try:
        if isinstance(x, np.ndarray):
            return False
        x + 1
        return True
    except TypeError:
        return False


# ..............................................................................
def is_sequence(arg):
    """
    Determine if an object is iterable but is not a string.
    """
    return (not hasattr(arg, "strip")) and hasattr(arg, "__iter__")


# ..............................................................................
def largest_power_of_2(value):
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


#
# # ..............................................................................
# def make_func_from(func, first=None):
#     """
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

# ..............................................................................
def make_new_object(objtype):
    """
    Make a new object of type obj.

    Parameters
    ----------
    objtype : the object type

    Returns
    -------
    new : the new object of same type
    """

    new = type(objtype)()

    # new id and date
    new._id = "{}_{}".format(type(objtype).__name__, str(uuid.uuid1()).split("-")[0])
    new._date = datetime.now(timezone.utc)

    return new


# # ..............................................................................
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

# # ..............................................................................
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

# # ..............................................................................
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


# # ..............................................................................
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

# ..............................................................................
def spacing_(arr):
    """
    Return a scalar for the spacing in the one-dimensional input array (if it is uniformly spaced,
    else return an array of the different spacings.

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
    #    mantissa, twoexp = np.frexp(spacings)
    #    mantissa = mantissa.round(6)
    #    spacings = np.ldexp(mantissa, twoexp)
    #    spacings = list(set(abs(spacings)))
    nd = get_n_decimals(spacings.max(), 1.0e-3)
    spacings = list(set(np.around(spacings, nd)))

    if len(spacings) == 1:
        # uniform spacing
        return spacings[0]
    else:
        return spacings


# # ..............................................................................
# def srepr(arg):
#     if is_sequence(arg):
#         return "<" + ", ".join(srepr(x) for x in arg) + ">"
#     return repr(arg)
