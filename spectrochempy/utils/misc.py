# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
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
    "closer_power_of_two",
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
    'largest_power_of_2',
    # signature / funcs
    'change_first_func_args',
    'change_func_args',
    'make_func_from',
    #
    'display_info_string',
    'convert_to_html',
    
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
from IPython.display import HTML, publish_display_data
from jinja2 import Template

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


def convert_to_html(obj):
    
    tr = "<tr>" \
         "<td style='padding-right:5px; padding-bottom:0px; padding-top:0px; width:124px'>{0}</td>" \
         "<td style='text-align:left; padding-bottom:0px; padding-top:0px; {2} '>{1}</td><tr>\n"

    obj._html_output = True
    
    out = obj._cstr()
    
    regex = r'\0{3}[\w\W]*?\0{3}'
    # noinspection PyPep8
    subst = lambda match: "<div>{}</div>".format(match.group(0).replace('\n', '<br/>').replace('\0', ''))
    out = re.sub(regex, subst, out, 0, re.MULTILINE)
    
    regex = r"^(\W*\w+\W?\w+)(:.*$)"
    subst = r"<font color='green'>\1</font> \2"
    out = re.sub(regex, subst, out, 0, re.MULTILINE)
    
    regex = r"^(.*(DIMENSION|DATA).*)$"
    subst = r"<strong>\1</strong>"
    out = re.sub(regex, subst, out, 0, re.MULTILINE)
    
    regex = r'\0{2}[\w\W]*?\0{2}'
    # noinspection PyPep8
    subst = lambda match: "<div><font color='darkcyan'>{}</font></div>".format(
        match.group(0).replace('\n', '<br/>').replace('\0', ''))
    out = re.sub(regex, subst, out, 0, re.MULTILINE)
    
    regex = r'\0{1}[\w\W]*?\0{1}'
    # noinspection PyPep8
    subst = lambda match: "<div><font color='blue'>{}</font></div>".format(
        match.group(0).replace('\n', '<br/>').replace('\0', ''))
    out = re.sub(regex, subst, out, 0, re.MULTILINE)
    
    regex = r'\.{3}\s+\n'
    out = re.sub(regex, '', out, 0, re.MULTILINE)
    
    html = "<table style='background:transparent'>\n"
    for line in out.splitlines():
        if '</font> :' in line:
            # keep only first match
            parts = line.split(':')
            html += tr.format(parts[0], ':'.join(parts[1:]), 'border:.5px solid lightgray; ')
        elif '<strong>' in line:
            html += tr.format(line, '<hr/>', 'padding-top:10px;')
    html += "</table>"
    
    obj._html_output = False

    return html

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


# ======================================================================================================================
# function signature
# ======================================================================================================================

import types


def change_func_args(func, new_args):
    """
    Create a new func with its arguments renamed to new_args.

    """
    # based on:
    # https://stackoverflow.com/questions/20712403/creating-a-python-function-at-runtime-with-specified-argument-names
    # https://stackoverflow.com/questions/16064409/how-to-create-a-code-object-in-python

    code_obj = func.__code__
    new_varnames = tuple(list(new_args))

    new_code_obj = types.CodeType(
        code_obj.co_argcount,  # integer
        code_obj.co_kwonlyargcount,  # integer
        code_obj.co_nlocals,  # integer
        code_obj.co_stacksize,  # integer
        code_obj.co_flags,  # integer
        code_obj.co_code,  # bytes
        code_obj.co_consts,  # tuple
        code_obj.co_names,  # tuple
        new_varnames,  # tuple
        code_obj.co_filename,  # string
        code_obj.co_name,  # string
        code_obj.co_firstlineno,  # integer
        code_obj.co_lnotab,  # bytes
        code_obj.co_freevars,  # tuple
        code_obj.co_cellvars  # tuple
    )
    modified = types.FunctionType(new_code_obj, func.__globals__)
    func.__code__ = modified.__code__  # replace code portion of original


def change_first_func_args(func, new_arg):
    """ This will change the first argument of function
     to the new_arg. This is essentially useful for documentation process

    """
    code_obj = func.__code__
    new_varnames = tuple([new_arg] +
                         list(code_obj.co_varnames[
                              1:code_obj.co_argcount]))
    change_func_args(func, new_varnames)


def make_func_from(func, first=None):
    """
    Create a new func with its arguments from another func ansd a new signature

    """
    code_obj = func.__code__
    new_varnames = list(code_obj.co_varnames)
    if first:
        new_varnames[0] = first
    new_varnames = tuple(new_varnames)

    new_code_obj = types.CodeType(
        code_obj.co_argcount,  # integer
        code_obj.co_kwonlyargcount,  # integer
        code_obj.co_nlocals,  # integer
        code_obj.co_stacksize,  # integer
        code_obj.co_flags,  # integer
        code_obj.co_code,  # bytes
        code_obj.co_consts,  # tuple
        code_obj.co_names,  # tuple
        new_varnames,  # tuple
        code_obj.co_filename,  # string
        code_obj.co_name,  # string
        code_obj.co_firstlineno,  # integer
        code_obj.co_lnotab,  # bytes
        code_obj.co_freevars,  # tuple
        code_obj.co_cellvars  # tuple
    )
    modified = types.FunctionType(new_code_obj,
                                  func.__globals__,
                                  func.__name__,
                                  func.__defaults__,
                                  func.__closure__)
    modified.__doc__ = func.__doc__
    return modified


def display_info_string(**kwargs):
    _template = """
    {{widgetcss}}
    <table><tr><td>
    {% if logo %}
    <img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAAXNSR0IArs4c6QAAAAlw
    SFlzAAAJOgAACToB8GSSSgAAAetpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6
    bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8x
    OTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAg
    eG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMu
    YWRvYmUuY29tL3RpZmYvMS4wLyI+CiAgICAgICAgIDx4bXA6Q3JlYXRvclRvb2w+bWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4wLCBo
    dHRwOi8vbWF0cGxvdGxpYi5vcmcvPC94bXA6Q3JlYXRvclRvb2w+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6
    T3JpZW50YXRpb24+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgqNQaNYAAAGiUlE
    QVRIDY1We4xU1Rn/3XPuYx47u8w+hnU38hTcuoUEt/6D2y4RB0ME1BoEd9taJaKh9CFiN7YGp7appUAMNmktMZFoJTYVLVQ0smsy
    26CN0SU1QgsuFAaW3WVmx33N677O6XfuyoIxTXqSO/fec+75fd93vt/3/UbDV0aKSZmCpkFMLz3T9utuu2N+o98aDSMBKVAo89z5
    y+zEz3ZafcCOfvWdlGCalqKn1Bf71CygTd+mf1esSOnpdMpTb+vWpTZuWVfe3jLPa5tzHYNm0T5N0gpdkkHaDBeGBU6d1/t/fyS8
    +/CbqdfUvmsx1PuMgc2bNxv79u1zgd31r+7JH1jbIZKxWRXAcYUQ8IWvBfBXNjEuJWPgMA02NR7C3/pYT9fjdZ3A9tGrWF8YSJHn
    qcDz3y7q2T967PZv+gnYJdd1mEZ+62zGDQV/dQgKhmLzDNOXCEWM3j6eTT5Y3w78dOBKJLR1PQf+4ivPj76UPZnssBN+wbM9Aet/
    AV81Mf1EEULXYfOobvX2WWQk0aoioXwwSmirOlioY0mu8BIouzYl7P8GV3vpqCCEZvlFz769w08oLDWvyKIyL1asSm28d6WfzA97
    ztvvV1kexUMsmhlkULEkuGYmFYC6AvfUrITnwUKl5K79lkjeSSRRTCTbQPd95e1WzMbZSya74XoXAxctCllCnbECMOjZNGRwvzIX
    nD85wbkMmKK+U045Dtdi8Qp+SAxU2GTg2bYlC9224pgvmSb54vkVTBQYyhUt2KjAMyMmPjwRQW5Mh2WKwJhlBh6jVGagFM84wZnQ
    4bpC0Rt4pk1PbSt0NDcxDA5xryosDHWgtbM0DGZDWLSoiDMDYeQnGVrmOThxLozB0RAaahzkJzjKNqcIQBymJFMkOlN8Dqjpg0XY
    Tx5xO/QbmmUrqIjGJznq47TqTaClKYfjp+PInLMwnOdYvtQBZ2XcunQY+VwIo4U4muoFEjVEFE6lQyEUKzHYfgQG9ylCyngU+Cxj
    tOqxCDGHcCsOMCs6iQul5ZiStdATYxjMZXDLTUVwLY8Jey4uOh2IxjwsrP8UXJYxUrkZrghBahzV5iXU6gNkq0Z1EzIsUBUSCV2n
    EOHo0LVxHCpuxabJJdhi5PFnvw5vLXwXIfNZvD/+JNo/X40NegE54sUaazl+UL8XD1x+FB9Ijjt4EQfdGN6J/x131LwIV9ap/AYs
    0x1fz1ZKFbh6A7qKy/By9Dg6G36Ep91vUJJ15Cqr0Z67E8/HzmBrw1OwxWyM+3Mo6BAuSB17oyfx0Oyl2DN0Hqs/70Cx6hBCvESF
    UY1ShWXZZEE7OTAYxZzaPH4TuoiusZvRnunFy2NbiHYuBp2vB66srX4vMEjpRKPxKXmnoQ4+Mn4DPiv8CYcrs3GfNUXJLtM+alSO
    hrMj/KT+wBNW3+E/2liywNO3iSflbaFva/+stGDTxE0E9Sjaox8HBhxpEamzMGSEaFKg+mjEddzDh1MxTDq3YV1kGBsjfwW3S9Cq
    anjmko+ndlb1UR3s6K8JlfphNWq9Ew/7c61T2BB/EbcaNkb8GBaE0tANH7/M34PLdhJDzjIcL9xPbdTG6zyM72Y+wXPHmvB489No
    fm0b5HnbQ9Rgp/7DSSd29AeVvPeNyK6JcYl/yQVi5dBjuGvoV/gaJe47s45QUxrDmcYX0MBsdF7egvXZ7+O0vZA4X8QmOQWjlSK7
    RDz5wIM30gp9UbWcGjXxhzdDu1SiNSpx6kcQB57rPnr/3dlkZarWLnlRq5oPET1dOCIOk4wALib9eeS5iygfhkd09H0DWphB/+gs
    +PcOAS+ssrFmmXXgVfR0de9cpbAJfH3Q1jofW9DZk56dDcVsq9YcsoUMEd1qyLoT3BX1YiyHMJuk97hyjqIoE91t+NcTLeN0ZrfM
    oXatZbu6G0h4VG+ibqq0IJVK6cAjo6serG3vSUezCMct0yQeSOFJSUImqb2qbknUpDqlZxE0QZ+ZUpSlZx79h4Nda6zef9dlk121
    JDjbR5XggPRZlRnS6bRQRtLpn4++cuie/Yvn2svmNxuLw9WCcYIl4fEoTEGiSTUqJdfgU+8ROqf1iMkLzS389YtNPXc/PH8l8ONB
    JZkHD+4JtD04HmVEDWWErmBhzV2/2LB1bemJG6krzv2S6NOHUgtEP0Oif5pE/3fHoruP7N8RiP61GArzSwbUhJJQpXJKiKbfr/3b
    IhKq76sKPUdF9NW/LSqfSn6vjv8C45H/6FSgvZQAAAAASUVORK5CYII='
         style='height:25px; border-radius:12px; display: inline-block; float: left; vertical-align: middle'></img>
    {% endif %}
    </td><td>
    {% if message %}
    &nbsp;&nbsp;<span style='font-size:12px'>{{ message }}</span>
    {% endif %}
    </td></tr></table>
    </div>
    """

    logo = kwargs.get('logo', True)
    message = kwargs.get('message', 'info ')

    template = Template(_template)
    html = template.render({'logo': logo,
                            'message': message.strip().replace('\n', '<br/>')})
    publish_display_data(data={'text/html': html})
