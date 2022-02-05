#  -*- coding: utf-8 -*-
#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
from __future__ import annotations

__all__ = [
    "set_env",
    "assert_equal",
    "assert_array_equal",
    "assert_array_almost_equal",
    "assert_ndarray_equal",
    "assert_ndarray_almost_equal",
    "assert_coord_equal",
    "assert_coord_almost_equal",
    "assert_dataset_equal",
    "assert_dataset_almost_equal",
    "assert_project_equal",
    "assert_project_almost_equal",
    "assert_approx_equal",
    "assert_raises",
    "assert_units_equal",
    "assert_array_compare",
    "assert_script_equal",
    "RandomSeedContext",
    "assert_produces_warning",
]


import os
import operator
import warnings
import re
from contextlib import contextmanager
from typing import (
    Sequence,
    Type,
    cast,
)

import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.testing.compare import calculate_rms, ImageAssertionError
from numpy.testing import (
    assert_equal,
    assert_array_equal,
    assert_array_almost_equal,
    assert_approx_equal,
    assert_raises,
    assert_array_compare,
)


@contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.

    Parameters
    ----------
    environ : dict(str)
        Environment variables to set

    Examples
    --------
    >>> import os
    >>> from spectrochempy.utils.testing import set_env
    >>> with set_env(PLUGINS_DIR=u'test/plugins'):
    ...     "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    """
    # https://stackoverflow.com/questions/2059482/python-temporarily-modify-the-current-processs-environment/51754362
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


# ======================================================================================
# NDDataset comparison
# ======================================================================================
def gisinf(x):
    # copied from numpy.testing._private.utils
    from numpy.core import isinf, errstate

    with errstate(invalid="ignore"):
        st = isinf(x)
        if isinstance(st, type(NotImplemented)):
            raise TypeError("isinf not supported for this type")
    return st


def _compare(x, y, decimal):
    # copied from numpy.testing._private.utils
    from numpy.core import number, float_, result_type, array
    from numpy.core.numerictypes import issubdtype
    from numpy.core.fromnumeric import any as npany

    try:
        if npany(gisinf(x)) or npany(gisinf(y)):
            xinfid = gisinf(x)
            yinfid = gisinf(y)
            if not (xinfid == yinfid).all():
                return False
            # if one item, x and y is +- inf
            if x.size == y.size == 1:
                return x == y
            x = x[~xinfid]
            y = y[~yinfid]
    except (TypeError, NotImplementedError):
        pass

    # make sure y is an inexact type to avoid abs(MIN_INT); will cause
    # casting of x later.
    try:
        dtype = result_type(y, 1.0)
    except TypeError as e:
        if issubdtype(np.dtype("datetime64"), y[0]):
            dtype = y.dtype
        else:
            raise e
    y = array(y, dtype=dtype, copy=False, subok=True)
    z = abs(x - y)

    if not issubdtype(z.dtype, number) or issubdtype(z.dtype, np.dtype("timedelta64")):
        z = z.astype(float_)  # handle object arrays

    return z < 1.5 * 10.0 ** (-decimal)


def compare_ndarrays(this, other, approx=False, decimal=6, data_only=False):

    # Comparison based on attributes:
    #        data, dims, mask, labels, units, meta

    from spectrochempy.core.units import ur

    def compare(x, y):
        return _compare(x, y, decimal)

    eq = True
    thistype = this.implements()

    if other.data is None and this.data is None and data_only:
        attrs = ["labels"]
    elif data_only:
        attrs = ["data"]
    else:
        attrs = (
            "data",
            "dims",
            "mask",
            "labels",
            "units",
            "meta",
        )

    for attr in attrs:
        if attr != "units":
            sattr = getattr(this, f"_{attr}")
            if hasattr(other, f"_{attr}"):
                oattr = getattr(other, f"_{attr}")
                if sattr is None and oattr is not None:
                    raise AssertionError(f"`{attr}` of {this} is None.")
                if oattr is None and sattr is not None:
                    raise AssertionError(f"{attr} of {other} is None.")
                if (
                    hasattr(oattr, "size")
                    and hasattr(sattr, "size")
                    and oattr.size != sattr.size
                ):
                    # particular case of mask
                    if attr != "mask":
                        raise AssertionError(f"{thistype}.{attr} sizes are different.")
                    else:
                        assert_array_equal(
                            other.mask,
                            this.mask,
                            f"{this} and {other} masks are different.",
                        )
                if attr in ["data", "mask"]:
                    if approx:
                        assert_array_compare(
                            compare,
                            sattr,
                            oattr,
                            header=(
                                f"{thistype}.{attr} attributes ar"
                                f"e not almost equal to %d decimals" % decimal
                            ),
                            precision=decimal,
                        )
                    else:
                        assert_array_compare(
                            operator.__eq__,
                            sattr,
                            oattr,
                            header=f"{thistype}.{attr} "
                            f"attributes are not "
                            f"equal",
                        )
                else:
                    eq &= np.all(sattr == oattr)
                if not eq:
                    raise AssertionError(
                        f"The {attr} attributes of {this} and {other} are "
                        f"different."
                    )
            else:
                return False
        else:
            # unitless and dimensionless are supposed equal units
            sattr = this._units
            if sattr is None:
                sattr = ur.dimensionless
            if hasattr(other, "_units"):
                oattr = other._units
                if oattr is None:
                    oattr = ur.dimensionless

                eq &= np.all(sattr == oattr)
                if not eq:
                    raise AssertionError(
                        f"attributes `{attr}` are not equals or one is "
                        f"missing: \n{sattr} != {oattr}"
                    )
            else:
                raise AssertionError(f"{other} has no units")

    return True


def compare_coords(
    this, other, approx=False, decimal=6, data_only=False, quantity_only=False
):

    from spectrochempy.core.units import ur

    def compare(x, y):
        return _compare(x, y, decimal)

    eq = True
    thistype = this.implements()
    if thistype == "CoordSet":  # this may happen for multicoordinates
        for coord0, coord1 in zip(this, other):
            eq &= compare_coords(
                coord0,
                coord1,
                approx=approx,
                decimal=decimal,
                data_only=data_only,
                quantity_only=quantity_only,
            )
        return eq

    if thistype not in ["Coord", "LinearCoord"]:
        raise TypeError(
            "This function compare `Coord` or `LinearCoord` objects, "
            "not `{thistype}`"
        )

    if not data_only:
        # we must rescale the two coordinates to the same base units for correct comparison
        other = other.to(this.units)  # rescale data for common units if possible

    if quantity_only:  # important to let it after the previous check
        data_only = True

    if other.data is None and this.data is None and data_only:
        attrs = ["labels"]
    elif quantity_only:  # important to have this before check on data_only
        attrs = ["data", "units"]
    elif data_only:
        attrs = ["data"]
    else:
        attrs = ["data", "labels", "units", "meta", "long_name"]
        # if 'long_name' in attrs:  #    attrs.remove('title')  #TODO: should we use long for comparison?

    if other.linear == this.linear:
        # To còmpare linear coordinates
        attrs += ["offset", "increment", "linear", "size"]

    for attr in attrs:

        if attr != "units":
            sattr = getattr(this, f"_{attr}")
            if this.linear and attr == "data":
                # allow comparison of LinearCoord and Coord
                sattr = this.data
            if hasattr(other, f"_{attr}"):
                oattr = getattr(other, f"_{attr}")
                if other.linear and attr == "data":
                    oattr = other.data
                # to avoid deprecation warning issue for unequal array
                if sattr is None and oattr is not None:
                    raise AssertionError(f"`{attr}` of {this} is None.")
                if oattr is None and sattr is not None:
                    raise AssertionError(f"{attr} of {other} is None.")
                if (
                    hasattr(oattr, "size")
                    and hasattr(sattr, "size")
                    and oattr.size != sattr.size
                ):
                    raise AssertionError(f"{thistype}.{attr} sizes are different.")

                if attr == "data":
                    if approx:
                        assert_array_compare(
                            compare,
                            sattr,
                            oattr,
                            header=(
                                f"{thistype}.{attr} attributes ar"
                                f"e not almost equal to %d decimals" % decimal
                            ),
                            precision=decimal,
                        )
                    else:
                        assert_array_compare(
                            operator.__eq__,
                            sattr,
                            oattr,
                            header=f"{thistype}.{attr} "
                            f"attributes are not "
                            f"equal",
                        )

                elif attr == "offset" and approx:
                    assert_approx_equal(
                        sattr,
                        oattr,
                        significant=decimal,
                        err_msg=f"{thistype}.{attr} attributes "
                        f"are not almost equal to %d decimals" % decimal,
                    )

                else:
                    eq &= np.all(sattr == oattr)

                if not eq:
                    raise AssertionError(
                        f"The {attr} attributes of {this} and {other} are "
                        f"different."
                    )
            else:
                return False
        else:
            # unitless and dimensionless are supposed equals
            sattr = this._units
            if sattr is None:
                sattr = ur.dimensionless
            if hasattr(other, "_units"):
                oattr = other._units
                if oattr is None:
                    oattr = ur.dimensionless

                eq &= np.all(sattr == oattr)
                if not eq:
                    raise AssertionError(
                        f"attributes `{attr}` are not equals or one is "
                        f"missing: \n{sattr} != {oattr}"
                    )
            else:
                raise AssertionError(f"{other} has no units")

    return True


def compare_datasets(this, other, approx=False, decimal=6, data_only=False):
    from spectrochempy.core.units import ur

    def compare(x, y):
        return _compare(x, y, decimal)

    eq = True

    # if not isinstance(other, NDArray):
    #     # try to make some assumption to make useful comparison.
    #     if isinstance(other, Quantity):
    #         otherdata = other.magnitude
    #         otherunits = other.units
    #     elif isinstance(other, (float, int, np.ndarray)):
    #         otherdata = other
    #         otherunits = False
    #     else:
    #         raise AssertionError(
    #             f"{this} and {other} objects are too different to be " f"compared."
    #         )
    #
    #     if not this.has_units and not otherunits:
    #         eq = np.all(this._data == otherdata)
    #     elif this.has_units and otherunits:
    #         eq = np.all(this._data * this._units == otherdata * otherunits)
    #     else:
    #         raise AssertionError(f\"units of {this} and {other} objects does not match\")
    #     return eq

    thistype = this.implements()

    if other.data is None and this.data is None and data_only:
        attrs = ["labels"]
    elif data_only:
        attrs = ["data"]
    else:
        attrs = this.__dir__()
        exclude = (
            "filename",
            "preferences",
            "comment",
            "history",
            "date",
            "modified",
            "source",
            "roi",
            "size",
            "name",
            "show_datapoints",
            "modeldata",
            "processeddata",
            "baselinedata",
            "referencedata",
            "state",
        )
        for attr in exclude:
            # these attributes are not used for comparison (comparison based on
            # data and units!)
            if attr in attrs:
                if attr in attrs:
                    attrs.remove(attr)

        # if 'title' in attrs:  #    attrs.remove('title')  #TODO: should we use title for comparison?

    for attr in attrs:
        if attr != "units":
            sattr = getattr(this, f"_{attr}")
            if hasattr(other, f"_{attr}"):
                oattr = getattr(other, f"_{attr}")
                if sattr is None and oattr is not None:
                    raise AssertionError(f"`{attr}` of {this} is None.")
                if oattr is None and sattr is not None:
                    raise AssertionError(f"{attr} of {other} is None.")
                if (
                    hasattr(oattr, "size")
                    and hasattr(sattr, "size")
                    and oattr.size != sattr.size
                ):
                    # particular case of mask
                    if attr != "mask":
                        raise AssertionError(f"{thistype}.{attr} sizes are different.")
                    else:
                        assert_array_equal(
                            other.mask,
                            this.mask,
                            f"{this} and {other} masks are different.",
                        )
                if attr in ["data", "mask"]:
                    if approx:
                        assert_array_compare(
                            compare,
                            sattr,
                            oattr,
                            header=(
                                f"{thistype}.{attr} attributes ar"
                                f"e not almost equal to %d decimals" % decimal
                            ),
                            precision=decimal,
                        )
                    else:
                        assert_array_compare(
                            operator.__eq__,
                            sattr,
                            oattr,
                            header=f"{thistype}.{attr} "
                            f"attributes are not "
                            f"equal",
                        )

                elif attr in ["coordset"]:
                    if (sattr is None and oattr is not None) or (
                        oattr is None and sattr is not None
                    ):
                        raise AssertionError("One of the coordset is None")
                    elif sattr is None and oattr is None:
                        pass
                    else:
                        for item in zip(sattr, oattr):
                            res = compare_coords(*item, approx=approx, decimal=decimal)
                            if not res:
                                raise AssertionError(f"coords differs:\n{res}")
                else:
                    eq &= np.all(sattr == oattr)
                if not eq:
                    raise AssertionError(
                        f"The {attr} attributes of {this} and {other} are "
                        f"different."
                    )
            else:
                return False
        else:
            # unitlesss and dimensionless are supposed equals
            sattr = this._units
            if sattr is None:
                sattr = ur.dimensionless
            if hasattr(other, "_units"):
                oattr = other._units
                if oattr is None:
                    oattr = ur.dimensionless

                eq &= np.all(sattr == oattr)
                if not eq:
                    raise AssertionError(
                        f"attributes `{attr}` are not equals or one is "
                        f"missing: \n{sattr} != {oattr}"
                    )
            else:
                raise AssertionError(f"{other} has no units")

    return True


# ..............................................................................
def assert_dataset_equal(nd1, nd2, **kwargs):
    kwargs["approx"] = False
    assert_dataset_almost_equal(nd1, nd2, **kwargs)
    return True


# ..............................................................................
def assert_dataset_almost_equal(nd1, nd2, **kwargs):
    decimal = kwargs.get("decimal", 6)
    approx = kwargs.get("approx", True)
    # if data_only is True, compare only based on data (not labels and so on)
    # except if dataset is label only!.
    data_only = kwargs.get("data_only", False)
    compare_datasets(nd1, nd2, approx=approx, decimal=decimal, data_only=data_only)
    return True


# ..............................................................................
def assert_coord_equal(nd1, nd2, **kwargs):
    kwargs["approx"] = False
    assert_coord_almost_equal(nd1, nd2, **kwargs)
    return True


# ..............................................................................
def assert_coord_almost_equal(nd1, nd2, **kwargs):
    decimal = kwargs.get("decimal", 6)
    approx = kwargs.get("approx", True)
    # if data_only is True, compare only based on data (not labels and so on)
    # except if coord is label only!.
    data_only = kwargs.get("data_only", False)
    quantity_only = kwargs.get("quantity_only", False)
    compare_coords(
        nd1,
        nd2,
        approx=approx,
        decimal=decimal,
        quantity_only=quantity_only,
        data_only=data_only,
    )
    return True


# ..............................................................................
def assert_ndarray_equal(nd1, nd2, **kwargs):
    kwargs["approx"] = False
    assert_ndarray_almost_equal(nd1, nd2, **kwargs)
    return True


# ..............................................................................
def assert_ndarray_almost_equal(nd1, nd2, **kwargs):
    decimal = kwargs.get("decimal", 6)
    approx = kwargs.get("approx", True)
    # if data_only is True, compare only based on data (not labels and so on)
    # except if ndarray is label only!.
    data_only = kwargs.get("data_only", False)
    compare_ndarrays(nd1, nd2, approx=approx, decimal=decimal, data_only=data_only)
    return True


def assert_project_equal(proj1, proj2, **kwargs):
    assert_project_almost_equal(proj1, proj2, approx=False)
    return True


# ..............................................................................
def assert_project_almost_equal(proj1, proj2, **kwargs):
    assert len(proj1.datasets) == len(proj2.datasets)
    for nd1, nd2 in zip(proj1.datasets, proj2.datasets):
        compare_datasets(nd1, nd2, **kwargs)

    assert len(proj1.projects) == len(proj2.projects)
    for pr1, pr2 in zip(proj1.projects, proj2.projects):
        assert_project_almost_equal(pr1, pr2, **kwargs)

    assert len(proj1.scripts) == len(proj2.scripts)
    for sc1, sc2 in zip(proj1.scripts, proj2.scripts):
        assert_script_equal(sc1, sc2, **kwargs)

    return True


# ..............................................................................
def assert_script_equal(sc1, sc2, **kwargs):
    if sc1 != sc2:
        raise AssertionError(f"Scripts are different: {sc1.content} != {sc2.content}")


# ======================================================================================
# RandomSeedContext
# ======================================================================================

# .............................................................................
class RandomSeedContext(object):
    """
    A context manager (for use with the ``with`` statement) that will seed the
    numpy random number generator (RNG) to a specific value, and then restore
    the RNG state back to whatever it was before.

    (Copied from Astropy, licence BSD-3).

    Parameters
    ----------
    seed : int
        The value to use to seed the numpy RNG

    Examples
    --------
    A typical use case might be::

        with RandomSeedContext(<some seed value you pick>):
            from numpy import random

            randarr = random.randn(100)
            ... run your test using `randarr` ...
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


# .............................................................................
def assert_units_equal(unit1, unit2):
    from pint.errors import DimensionalityError
    from spectrochempy.core.units import ur

    if unit1 is None:
        unit1 = ur(unit1)
    if unit2 is None:
        unit2 = ur(unit2)

    try:
        x = (1.0 * unit1) / (1.0 * unit2)
    except DimensionalityError:
        return False
    if x.dimensionless:
        return True
    return False


# TODO: work on this
# #
# ------------------------------------------------------------------
# # Matplotlib testing utilities
# #
# ------------------------------------------------------------------
#
# figures_dir = os.path.join(os.path.expanduser("~"), ".spectrochempy",
# "figures")
# os.makedirs(figures_dir, exist_ok=True)
#
#
# #
# .............................................................................
# def _compute_rms(x, y):
#     return calculate_rms(x, y)
#
#
# #
# .............................................................................
# def _image_compare(imgpath1, imgpath2, REDO_ON_TYPEERROR):
#     # compare two images saved in files imgpath1 and imgpath2
#
#     from matplotlib.pyplot import imread
#     from skimage.measure import compare_ssim as ssim
#
#     # read image
#     try:
#         img1 = imread(imgpath1)
#     except IOError:
#         img1 = imread(imgpath1 + '.png')
#     try:
#         img2 = imread(imgpath2)
#     except IOError:
#         img2 = imread(imgpath2 + '.png')
#
#     try:
#         sim = ssim(img1, img2,
#                    data_range=img1.max() - img2.min(),
#                    multichannel=True) * 100.
#         rms = _compute_rms(img1, img2)
#
#     except ValueError:
#         rms = sim = -1
#
#     except TypeError as e:
#         # this happen sometimes and erratically during testing using
#         # pytest-xdist (parallele testing). This is work-around the problem
#         if e.args[0] == "unsupported operand type(s) " \
#                         "for - : 'PngImageFile' and 'int'" and not
#                         REDO_ON_TYPEERROR:
#             REDO_ON_TYPEERROR = True
#             rms = sim = -1
#         else:
#             raise
#
#     return (sim, rms, REDO_ON_TYPEERROR)
#
#
# #
# .............................................................................
# def compare_images(imgpath1, imgpath2,
#                    max_rms=None,
#                    min_similarity=None, ):
#     sim, rms, _ = _image_compare(imgpath1, imgpath2, False)
#
#     EPSILON = np.finfo(float).eps
#     CHECKSIM = (min_similarity is not None)
#     SIM = min_similarity if CHECKSIM else 100. - EPSILON
#     MESSSIM = "(similarity : {:.2f}%)".format(sim)
#     CHECKRMS = (max_rms is not None and not CHECKSIM)
#     RMS = max_rms if CHECKRMS else EPSILON
#     MESSRMS = "(rms : {:.2f})".format(rms)
#
#     if sim < 0 or rms < 0:
#         message = "Sizes of the images are different"
#     elif CHECKRMS and rms <= RMS:
#         message = "identical images {}".format(MESSRMS)
#     elif (CHECKSIM or not CHECKRMS) and sim >= SIM:
#         message = "identical/similar images {}".format(MESSSIM)
#     else:
#         message = "different images {}".format(MESSSIM)
#
#     return message
#
#
# #
# .............................................................................
# def same_images(imgpath1, imgpath2):
#     if compare_images(imgpath1, imgpath2).startswith('identical'):
#         return True
#
#
# #
# .............................................................................
# def image_comparison(reference=None,
#                      extension=None,
#                      max_rms=None,
#                      min_similarity=None,
#                      force_creation=False,
#                      savedpi=150):
#     """
#     image file comparison decorator.
#
#     Performs a comparison of the images generated by the decorated function.
#     If none of min_similarity and max_rms if set,
#     automatic similarity check is done :
#
#     Parameters
#     ----------
#     reference : list of image filename for the references
#
#         List the image filenames of the reference figures
#         (located in ``.spectrochempy/figures``) which correspond in
#         the same order to
#         the various figures created in the decorated function. if
#         these files doesn't exist an error is generated, except if the
#         force_creation argument is True. This should allow the creation
#         of a reference figures, the first time the corresponding figures are
#         created.
#
#     extension : str, optional, default=``png``
#
#         Extension to be used to save figure, among
#         (eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff)
#
#     force_creation : `bool`, optional, default=`False`.
#
#         if this flag is True, the figures created in the decorated
#         function are
#         saved in the reference figures directory (
#         ``.spectrocchempy/figures``)
#
#     min_similarity : float (percent).
#
#         If set, then it will be used to decide if an image is the same (
#         similar)
#         or not. In this case max_rms is not used.
#
#     max_rms : float
#
#         rms stands for `Root Mean Square`. If set, then it will
#         be used to decide if an image is the same
#         (less than the acceptable rms). Not used if min_similarity also set.
#
#     savedpi : int, optional, default=150
#
#         dot per inch of the generated figures
#
#     """
#     from spectrochempy.utils import is_sequence
#
#     if not reference:
#         raise ValueError('no reference image provided. Stopped')
#
#     if not extension:
#         extension = 'png'
#
#     if not is_sequence(reference):
#         reference = list(reference)
#
#     def make_image_comparison(func):
#
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#
#             # check the existence of the file if force creation is False
#             for ref in reference:
#                 filename = os.path.join(figures_dir,
#                                         '{}.{}'.format(ref, extension))
#                 if not os.path.exists(filename) and not force_creation:
#                     raise ValueError(
#                             'One or more reference file do not exist.\n'
#                             'Creation can be forced from the generated '
#                             'figure, by setting force_creation flag to True')
#
#             # get the nums of the already existing figures
#             # that, obviously,should not considered in
#             # this comparison
#             fignums = plt.get_fignums()
#
#             # execute the function generating the figures
#             # rest style to basic 'lcs' style
#             _ = func(*args, **kwargs)
#
#             # get the new fignums if any
#             curfignums = plt.get_fignums()
#             for x in fignums:
#                 # remove not newly created
#                 curfignums.remove(x)
#
#             if not curfignums:
#                 # no figure where generated
#                 raise RuntimeError(f'No figure was generated by the "{
#                 func.__name__}" function. Stopped')
#
#             if len(reference) != len(curfignums):
#                 raise ValueError("number of reference figures provided
#                 doesn't match the number of generated
#                 figures.")
#
#             # Comparison
#             REDO_ON_TYPEERROR = False
#
#             while True:
#                 errors = ""
#                 for fignum, ref in zip(curfignums, reference):
#                     referfile = os.path.join(figures_dir,
#                                              '{}.{}'.format(ref, extension))
#
#                     fig = plt.figure(fignum)  # work around to set
#                     # the correct style: we
#                     # we have saved the rcParams
#                     # in the figure attributes
#                     plt.rcParams.update(fig.rcParams)
#                     fig = plt.figure(fignum)
#
#                     if force_creation:
#                         # make the figure for reference and bypass
#                         # the rest of the test
#                         tmpfile = referfile
#                     else:
#                         # else we create a temporary file to save the figure
#                         fd, tmpfile = tempfile.mkstemp(
#                                 prefix='temp{}-'.format(fignum),
#                                 suffix='.{}'.format(extension), text=True)
#                         os.close(fd)
#
#                     fig.savefig(tmpfile, dpi=savedpi)
#
#                     sim, rms = 100.0, 0.0
#                     if not force_creation:
#                         # we do not need to loose time
#                         # if we have just created the figure
#                         sim, rms, REDO_ON_TYPEERROR = _image_compare(
#                         referfile,
#                                                                      tmpfile,
#                                                                      REDO_ON_TYPEERROR)
#                     EPSILON = np.finfo(float).eps
#                     CHECKSIM = (min_similarity is not None)
#                     SIM = min_similarity if CHECKSIM else 100. - EPSILON
#                     MESSSIM = "(similarity : {:.2f}%)".format(sim)
#                     CHECKRMS = (max_rms is not None and not CHECKSIM)
#                     RMS = max_rms if CHECKRMS else EPSILON
#                     MESSRMS = "(rms : {:.2f})".format(rms)
#
#                     if sim < 0 or rms < 0:
#                         message = "Sizes of the images are different"
#                     elif CHECKRMS and rms <= RMS:
#                         message = "identical images {}".format(MESSRMS)
#                     elif (CHECKSIM or not CHECKRMS) and sim >= SIM:
#                         message = "identical/similar images {}".format(
#                         MESSSIM)
#                     else:
#                         message = "different images {}".format(MESSSIM)
#
#                     message += "\n\t reference : {}".format(
#                             os.path.basename(referfile))
#                     message += "\n\t generated : {}\n".format(
#                             tmpfile)
#
#                     if not message.startswith("identical"):
#                         errors += message
#                     else:
#                         print(message)
#
#                 if errors and not REDO_ON_TYPEERROR:
#                     # raise an error if one of the image is different from
#                     the
#                     # reference image
#                     raise ImageAssertionError("\n" + errors)
#
#                 if not REDO_ON_TYPEERROR:
#                     break
#
#             return
#
#         return wrapper
#
#     return make_image_comparison

# from here it is copied from pandas._testing
# See License in LICENSES


@contextmanager
def assert_produces_warning(
    expected_warning: type[Warning] | bool | None = Warning,
    filter_level="always",
    check_stacklevel: bool = True,
    raise_on_extra_warnings: bool = True,
    match: str | None = None,
):
    """
    Context manager for running code expected to either raise a specific
    warning, or not raise any warnings. Verifies that the code raises the
    expected warning, and that it does not raise any other unexpected
    warnings. It is basically a wrapper around ``warnings.catch_warnings``.

    Parameters
    ----------
    expected_warning : {Warning, False, None}, default Warning
        The type of Exception raised. ``exception.Warning`` is the base
        class for all warnings. To check that no warning is returned,
        specify ``False`` or ``None``.
    filter_level : str or None, default "always"
        Specifies whether warnings are ignored, displayed, or turned
        into errors.
        Valid values are:

        * "error" - turns matching warnings into exceptions
        * "ignore" - discard the warning
        * "always" - always emit a warning
        * "default" - print the warning the first time it is generated
          from each location
        * "module" - print the warning the first time it is generated
          from each module
        * "once" - print the warning the first time it is generated

    check_stacklevel : bool, default True
        If True, displays the line that called the function containing
        the warning to show were the function is called. Otherwise, the
        line that implements the function is displayed.
    raise_on_extra_warnings : bool, default True
        Whether extra warnings not of the type `expected_warning` should
        cause the test to fail.
    match : str, optional
        Match warning message.

    Examples
    --------
    >>> import warnings
    >>> with assert_produces_warning():
    ...     warnings.warn(UserWarning())
    ...
    >>> with assert_produces_warning(False):
    ...     warnings.warn(RuntimeWarning())
    ...
    Traceback (most recent call last):
        ...
    AssertionError: Caused unexpected warning(s): ['RuntimeWarning'].
    >>> with assert_produces_warning(UserWarning):
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Did not see expected warning of class 'UserWarning'.

    ..warn:: This is *not* thread-safe.
    """
    __tracebackhide__ = True

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter(filter_level)
        yield w

        if expected_warning:
            expected_warning = cast(Type[Warning], expected_warning)
            _assert_caught_expected_warning(
                caught_warnings=w,
                expected_warning=expected_warning,
                match=match,
                check_stacklevel=check_stacklevel,
            )

        if raise_on_extra_warnings:
            _assert_caught_no_extra_warnings(
                caught_warnings=w,
                expected_warning=expected_warning,
            )


def _assert_caught_expected_warning(
    *,
    caught_warnings: Sequence[warnings.WarningMessage],
    expected_warning: type[Warning],
    match: str | None,
    check_stacklevel: bool,
) -> None:
    """Assert that there was the expected warning among the caught warnings."""
    saw_warning = False
    matched_message = False
    unmatched_messages = []

    for actual_warning in caught_warnings:
        if issubclass(actual_warning.category, expected_warning):
            saw_warning = True

            if check_stacklevel and issubclass(
                actual_warning.category, (FutureWarning, DeprecationWarning)
            ):
                _assert_raised_with_correct_stacklevel(actual_warning)

            if match is not None:
                if re.search(match, str(actual_warning.message)):
                    matched_message = True
                else:
                    unmatched_messages.append(actual_warning.message)

    if not saw_warning:
        raise AssertionError(
            f"Did not see expected warning of class "
            f"{repr(expected_warning.__name__)}"
        )

    if match and not matched_message:
        raise AssertionError(
            f"Did not see warning {repr(expected_warning.__name__)} "
            f"matching '{match}'. The emitted warning messages are "
            f"{unmatched_messages}"
        )


def _assert_caught_no_extra_warnings(
    *,
    caught_warnings: Sequence[warnings.WarningMessage],
    expected_warning: type[Warning] | bool | None,
) -> None:
    """Assert that no extra warnings apart from the expected ones are caught."""
    extra_warnings = []

    for actual_warning in caught_warnings:
        if _is_unexpected_warning(actual_warning, expected_warning):
            unclosed = "unclosed transport <asyncio.sslproto._SSLProtocolTransport"
            if actual_warning.category == ResourceWarning and unclosed in str(
                actual_warning.message
            ):
                # FIXME: kludge because pytest.filterwarnings does not
                #  suppress these, xref GH#38630
                continue

            extra_warnings.append(
                (
                    actual_warning.category.__name__,
                    actual_warning.message,
                    actual_warning.filename,
                    actual_warning.lineno,
                )
            )

    if extra_warnings:
        raise AssertionError(f"Caused unexpected warning(s): {repr(extra_warnings)}")


def _is_unexpected_warning(
    actual_warning: warnings.WarningMessage,
    expected_warning: type[Warning] | bool | None,
) -> bool:
    """Check if the actual warning issued is unexpected."""
    if actual_warning and not expected_warning:
        return True
    expected_warning = cast(Type[Warning], expected_warning)
    return bool(not issubclass(actual_warning.category, expected_warning))


def _assert_raised_with_correct_stacklevel(
    actual_warning: warnings.WarningMessage,
) -> None:
    from inspect import (
        getframeinfo,
        stack,
    )

    caller = getframeinfo(stack()[4][0])
    msg = (
        "Warning not set with correct stacklevel. "
        f"File where warning is raised: {actual_warning.filename} != "
        f"{caller.filename}. Warning message: {actual_warning.message}"
    )
    assert actual_warning.filename == caller.filename, msg


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
