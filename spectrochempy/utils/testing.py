#! python3

# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================


"""

"""
__all__ = ["assert_equal",
           "assert_array_equal",
           "assert_array_almost_equal",
           "assert_approx_equal",
           "assert_raises",
           "raises",
           "catch_warnings",
           "RandomSeedContext",
           "EPSILON",
           "is_sequence",
           "preferences",
           "datadir"
           ]

import os
import sys
import functools
import tempfile
import warnings

import nbformat
import pytest

import matplotlib.pyplot as plt
from matplotlib.testing.compare import calculate_rms, ImageComparisonFailure
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_approx_equal, assert_raises)


#  we defer import in order to avoid importing all the spectroscopy namespace
def preferences():
    from spectrochempy.core import app
    return app


preferences = preferences()


def datadir():
    from spectrochempy.core import app
    return app.datadir.path


datadir = datadir()

figures_dir = os.path.join(os.path.expanduser("~"), ".spectrochempy", "figures")
os.makedirs(figures_dir, exist_ok=True)

# utilities
is_sequence = lambda arg: (not hasattr(arg, 'strip')) and hasattr(arg,
                                                                  "__iter__")

import numpy as np

EPSILON = epsilon = np.finfo(float).eps


# ======================================================================================================================
# RandomSeedContext
# ======================================================================================================================

# .............................................................................
class RandomSeedContext(object):
    """
    A context manager (for use with the ``with`` statement) that will seed the
    numpy random number generator (RNG) to a specific value, and then restore
    the RNG state back to whatever it was before.

    (Copied from Astropy, licence BSD-3)

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


# ======================================================================================================================
# raises and assertions (mostly copied from astropy)
# ======================================================================================================================

# .............................................................................
def assert_equal_units(unit1, unit2):
    from pint import DimensionalityError
    try:
        x = (1. * unit1) / (1. * unit2)
    except DimensionalityError:
        return False
    if x.dimensionless:
        return True
    return False


# .............................................................................
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

    (Copied from Astropy, licence BSD-3)

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


# .............................................................................
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

        with catch_warnings(MyCustomWarning) as w :
            do.something.bad()
        assert len(w) > 0

    (Copied from Astropy, licence BSD-3)

    """

    def __init__(self, *classes):
        super(catch_warnings, self).__init__(record=True)
        self.classes = classes

    def __enter__(self):
        warning_list = super(catch_warnings, self).__enter__()
        if len(self.classes) == 0:
            warnings.simplefilter('always')
        else:
            warnings.simplefilter('ignore')
            for cls in self.classes:
                warnings.simplefilter('always', cls)
        return warning_list

    def __exit__(self, type, value, traceback):
        pass


# ----------------------------------------------------------------------------------------------------------------------
# Testing examples and notebooks in docs
# ----------------------------------------------------------------------------------------------------------------------

# .............................................................................
def notebook_run(path):
    """
    Execute a notebook via nbconvert and collect output.

    returns
    -------

     results : (parsed nb object, execution errors)

    """
    import sys
    import subprocess

    print(sys.version_info)
    kernel_name = 'python%d' % sys.version_info[0]
    this_file_directory = os.path.dirname(__file__)
    errors = []

    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
        nb.metadata.get('kernelspec', {})['name'] = kernel_name
        ep = ExecutePreprocessor(kernel_name=kernel_name,
                                 timeout=10, allow_errors=True)

        try:
            ep.preprocess(nb, {'metadata': {'path': this_file_directory}})

        except CellExecutionError as e:
            if "SKIP" in e.traceback:
                print(str(e.traceback).split("\n")[-2])
            else:
                raise e

    return nb, errors


# .............................................................................
def example_run(path):
    import subprocess

    try:
        print('env', os.environ['CONDA_DEFAULT_ENV'])
    except:
        pass
        #debug_('no conda env')
    pipe = None
    try:
        pipe = subprocess.Popen(
            ["python", path, ],
            stdout=subprocess.PIPE)
        (so, serr) = pipe.communicate()
    except:
        pass

    return pipe.returncode, so, serr


# ----------------------------------------------------------------------------------------------------------------------
# Matplotlib testing utilities
# ----------------------------------------------------------------------------------------------------------------------

# .............................................................................
def _compute_rms(x, y):
    return calculate_rms(x, y)


# .............................................................................
def _image_compare(imgpath1, imgpath2, REDO_ON_TYPEERROR):
    # compare two images saved in files imgpath1 and imgpath2

    from matplotlib.pyplot import imread
    from skimage.measure import compare_ssim as ssim

    # read image
    try:
        img1 = imread(imgpath1)
    except IOError:
        img1 = imread(imgpath1 + '.png')
    try:
        img2 = imread(imgpath2)
    except IOError:
        img2 = imread(imgpath2 + '.png')

    try:
        sim = ssim(img1, img2,
                   data_range=img1.max() - img2.min(),
                   multichannel=True) * 100.
        rms = _compute_rms(img1, img2)

    except ValueError as e:
        rms = sim = -1

    except TypeError as e:
        # this happen sometimes and erratically during testing using
        # pytest-xdist (parallele testing). This is work-around the problem
        if e.args[0] == "unsupported operand type(s) " \
                        "for - : 'PngImageFile' and 'int'" and not REDO_ON_TYPEERROR:
            REDO_ON_TYPEERROR = True
            rms = sim = -1
        else:
            raise

    return (sim, rms, REDO_ON_TYPEERROR)


# .............................................................................
def compare_images(imgpath1, imgpath2,
                   max_rms=None,
                   min_similarity=None, ):
    sim, rms, _ = _image_compare(imgpath1, imgpath2, False)

    CHECKSIM = (min_similarity is not None)
    SIM = min_similarity if CHECKSIM else 100. - EPSILON
    MESSSIM = mess = "(similarity : {:.2f}%)".format(sim)
    CHECKRMS = (max_rms is not None and not CHECKSIM)
    RMS = max_rms if CHECKRMS else EPSILON
    MESSRMS = "(rms : {:.2f})".format(rms)

    if sim < 0 or rms < 0:
        message = "Sizes of the images are different"
    elif CHECKRMS and rms <= RMS:
        message = "identical images {}".format(MESSRMS)
    elif (CHECKSIM or not CHECKRMS) and sim >= SIM:
        message = "identical/similar images {}".format(MESSSIM)
    else:
        message = "different images {}".format(MESSSIM)

    return message


# .............................................................................
def same_images(imgpath1, imgpath2):
    if compare_images(imgpath1, imgpath2).startswith('identical'):
        return True


# .............................................................................
def image_comparison(reference=None,
                     extension=None,
                     max_rms=None,
                     min_similarity=None,
                     force_creation=False,
                     savedpi=150):
    """
    image file comparison decorator.

    Performs a comparison of the images generated by the decorated function.
    If none of min_similarity and max_rms if set,
    automatic similarity check is done :

    Parameters
    ----------
    reference : list of image filename for the references

        List the image filenames of the reference figures
        (located in ``.spectrochempy/figures``) which correspond in
        the same order to
        the various figures created in the decorated fonction. if
        these files doesn't exist an error is generated, except if the
        force_creation argument is True. This should allow the creation
        of a reference figures, the first time the corresponding figures are
        created.

    extension : str, optional, default=``png``

        Extension to be used to save figure, among
        (eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff)

    force_creation : `bool`, optional, default=`False`.

        if this flag is True, the figures created in the decorated function are
        saved in the reference figures directory (``.spectrocchempy/figures``)

    min_similarity : float (percent).

        If set, then it will be used to decide if an image is the same (similar)
        or not. In this case max_rms is not used.

    max_rms : float

        rms stands for `Root Mean Square`. If set, then it will
        be used to decide if an image is the same
        (less than the acceptable rms). Not used if min_similarity also set.

    savedpi : int, optional, default=150

        dot per inch of the generated figures

    """

    if not reference:
        raise ValueError('no reference image provided. Stopped')

    if not extension:
        extension = 'png'

    if not is_sequence(reference):
        reference = list(reference)

    def make_image_comparison(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # check the existence of the file if force creation is False
            for ref in reference:
                filename = os.path.join(figures_dir,
                                        '{}.{}'.format(ref, extension))
                if not os.path.exists(filename) and not force_creation:
                    raise ValueError(
                        'One or more reference file do not exist.\n'
                        'Creation can be forced from the generated '
                        'figure, by setting force_creation flag to True')

            # get the nums of the already existing figures
            # that, obviously,should not considered in
            # this comparison
            fignums = plt.get_fignums()

            # execute the function generating the figures
            # rest style to basic 'lcs' style
            res = func(*args, **kwargs)

            # get the new fignums if any
            curfignums = plt.get_fignums()
            for x in fignums:
                # remove not newly created
                curfignums.remove(x)

            if not curfignums:
                # no figure where generated
                raise RuntimeError('No figure was generated '
                                   'by the "{}" function. Stopped'.format(
                    func.__name__))

            if len(reference) != len(curfignums):
                raise ValueError('number of reference figures provided desn\'t'
                                 ' match the number of generated figures.')

            # Comparison
            REDO_ON_TYPEERROR = False

            while True:
                errors = ""
                for fignum, ref in zip(curfignums, reference):
                    referfile = os.path.join(figures_dir,
                                             '{}.{}'.format(ref, extension))

                    fig = plt.figure(fignum)  # work around to set
                    # the correct style: we
                    # we have saved the rcParams
                    # in the figure attributes
                    plt.rcParams.update(fig.rcParams)
                    fig = plt.figure(fignum)

                    if force_creation:
                        # make the figure for reference and bypass
                        # the rest of the test
                        tmpfile = referfile
                    else:
                        # else we create a temporary file to save the figure
                        fd, tmpfile = tempfile.mkstemp(
                            prefix='temp{}-'.format(fignum),
                            suffix='.{}'.format(extension), text=True)
                        os.close(fd)

                    fig.savefig(tmpfile, dpi=savedpi)

                    sim, rms = 100.0, 0.0
                    if not force_creation:
                        # we do not need to loose time
                        # if we have just created the figure
                        sim, rms, REDO_ON_TYPEERROR = _image_compare(referfile,
                                                                     tmpfile,
                                                                     REDO_ON_TYPEERROR)

                    CHECKSIM = (min_similarity is not None)
                    SIM = min_similarity if CHECKSIM else 100. - EPSILON
                    MESSSIM = mess = "(similarity : {:.2f}%)".format(sim)
                    CHECKRMS = (max_rms is not None and not CHECKSIM)
                    RMS = max_rms if CHECKRMS else EPSILON
                    MESSRMS = "(rms : {:.2f})".format(rms)

                    if sim < 0 or rms < 0:
                        message = "Sizes of the images are different"
                    elif CHECKRMS and rms <= RMS:
                        message = "identical images {}".format(MESSRMS)
                    elif (CHECKSIM or not CHECKRMS) and sim >= SIM:
                        message = "identical/similar images {}".format(MESSSIM)
                    else:
                        message = "different images {}".format(MESSSIM)

                    message += "\n\t reference : {}".format(
                        os.path.basename(referfile))
                    message += "\n\t generated : {}\n".format(
                        tmpfile)

                    if not message.startswith("identical"):
                        errors += message
                    else:
                        print(message)

                if errors and not REDO_ON_TYPEERROR:
                    # raise an error if one of the image is different from the
                    # reference image
                    raise ImageComparisonFailure("\n" + errors)

                if not REDO_ON_TYPEERROR:
                    break

            return

        return wrapper

    return make_image_comparison


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
