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

"""

"""
__all__ = ["assert_equal",
           "assert_array_equal",
           "assert_array_almost_equal",
           "assert_approx_equal",
           "raises",
           "catch_warnings",
           "RandomSeedContext",
           "SpectroChemPyWarning",
           "SpectroChemPyDeprecationWarning",
           ]

import pytest
import functools
import sys
import types
import warnings

import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_approx_equal)
from matplotlib.testing.compare import calculate_rms, ImageComparisonFailure
import matplotlib.pyplot as plt

from spectrochempy.utils import SpectroChemPyWarning, \
    SpectroChemPyDeprecationWarning, is_sequence
from spectrochempy.extern.pint.errors import DimensionalityError, \
    UndefinedUnitError
from spectrochempy.application import scpdata, log
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndarray import masked

from spectrochempy.application import plotoptions



# =============================================================================
# RandomSeedContext
# =============================================================================

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


# =============================================================================
# raises and assertions (mostly copied from astropy)
# =============================================================================

# .............................................................................
def assert_equal_units(unit1, unit2):

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

        with catch_warnings(MyCustomWarning) as w:
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


# -----------------------------------------------------------------------------
# Testing examples and notebooks in docs
# -----------------------------------------------------------------------------

# .............................................................................
def notebook_run(path):
    """
    Execute a notebook via nbconvert and collect output.

    returns
    -------

     results : (parsed nb object, execution errors)

    """
    kernel_name = 'python%d' % sys.version_info[0]
    this_file_directory = os.path.dirname(__file__)
    errors = []

    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
        nb.metadata.get('kernelspec', {})['name'] = kernel_name
        ep = ExecutePreprocessor(kernel_name=kernel_name,
                                 timeout=10)  # , allow_errors=True

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

    pipe = None
    try:
        pipe = subprocess.Popen(
                ["python", path, " --test=True"],
                stdout=subprocess.PIPE)
        (so, serr) = pipe.communicate()
    except:
        pass

    return pipe.returncode, so, serr


# -----------------------------------------------------------------------------
# Matplotlib testing utilities
# -----------------------------------------------------------------------------

# .............................................................................
def show_do_not_block(func):
    """
    A decorator to allow non blocking testing of matplotlib figures-
    set the plotoption.do_not_block

    This doesn't work with pytest in parallel mode because the sys.argv
    contain only the flag -c and that's all!

    To make it work, the only way I found for now is to remove the option
    -nauto in pythest.ini adopts= nauto etc...

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not '-c' in sys.argv[0] and func.__name__ in sys.argv[1]:
            # The individual test has been called - then we show figures
            # we do not show for full tests
            plotoptions.do_not_block = False
        else:
            plotoptions.do_not_block = True
        return func(*args, **kwargs)

    return wrapper

# .............................................................................
def _compute_rms(x, y):
    #return np.linalg.norm(x - y) / x.size ** 2
    return calculate_rms(x,y)

# .............................................................................
def _image_compare(imgpath1, imgpath2):
    # compare two images saved in files imgpath1 and imgpath2

    from scipy.misc import imread
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
                   multichannel=True)
        rms = _compute_rms(img1, img2)

    except ValueError as e:
        rms = sim = -1

    return sim, rms

# .............................................................................
def image_comparison(reference=None, extension=None, tol=1e-6,
                     force_creation=False, **kws):
    """
    image file comparison decorator

    Parameters
    ----------
    reference : list of image filename for the references

        List the image filenames of the reference figures
        (located in ``scpdata/figures``) which correspond in the same order to
        the various figures created in the decorated fonction. if
        these files doesn't exist an error is generated, except if the
        force_creation argument is True. This should allow the creation
        of a reference figures, the first time the corresponding figures are
        created.

    extension : `str`, optional, default=``png``

        Extension to be used to save figure, among
        (eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff)

    force_creation : `bool`, optional, default=``False``.

        if this flag is True, the figures created in the decorated function are
        saved in the reference figures directory (``scpdata/figures``)

    kwargs : other keyword arguments


    Returns
    -------

    """
    if not reference:
        raise ValueError('no reference image provided. Stopped')

    if not extension:
        extension = 'png'

    if not is_sequence(reference):
        reference = list(reference)

    figures = os.path.join(scpdata, "figures")
    os.makedirs(figures, exist_ok=True)

    # check the existence of the file if force creation is False
    for ref in reference:
        filename = os.path.join(figures, '{}.{}'.format(ref,extension))
        if not os.path.exists(filename) and not force_creation:
            raise ValueError('One or more reference file do not exist.'
                             'Creation can be force from the generated'
                             'figure, by setting force_creation flag to True')

    def make_image_comparison(func):

        import tempfile

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # get the nums of the already existing figures
            # that, obviously,should not considered in
            # this comparison
            fignums = plt.get_fignums()

            # execute the function generating the figures
            res = func(*args, **kwargs)

            # get the new fignums if any
            curfignums = plt.get_fignums()
            for x in fignums:
                # remove not newly created
                curfignums.remove(x)

            if not curfignums:
                # no figure where generated
                raise RuntimeError('No figure was generated '
                         'by the "{}" function. Stopped'.format(func.__name__))

            if len(reference)!=len(curfignums):
                raise ValueError('number of reference figures provided desn\'t'
                                 ' match the number of generated figures.')

            # Comparison

            errors = ""
            for fignum, ref in zip(curfignums, reference):
                fileref = os.path.join(figures,
                                       '{}.{}'.format(ref,extension))
                filetemp = os.path.join(figures,
                                        '~temp{}.{}'.format(fignum, extension))

                fd, tmpfile = tempfile.mkstemp(suffix='-spectrochempy.tmp')
                os.close(fd)

                fig = plt.figure(fignum)

                if force_creation :
                    # make the figure for,reference and bypass the rest of the test
                    filetemp = fileref

                fig.savefig(filetemp)

                sim, rms = 1.0, 0.0
                if not force_creation:
                    # we do not need to loose time
                    # if we have jsut created the figure
                    sim, rms = _image_compare(fileref, filetemp)

                sim = sim * 100.
                mess = "(similarity: {:.2f}%, rms: {:.2f})".format(sim, rms)
                if sim < 0 or rms < 0:
                    message = "Sizes of the images are different"
                elif sim >= 100.-tol and rms <= tol:
                    message = "identical images {}".format(mess)
                elif sim > 100.-5.*tol and rms <= 5.*tol:
                    message = "almost identical {}".format(mess)
                else:
                    message = "probably very different {}".format(mess)

                message += "\n\t reference: {}".format(os.path.basename(fileref))
                message += "\n\t generated: {}\n".format(
                    os.path.basename(filetemp))

                if not message.startswith("identical"):
                    errors += message
                else:
                    log.info(message)

            if errors:
                # raise an error if one of the image is different from the
                # reference image
                raise ImageComparisonFailure("\n"+errors)

            return

        return wrapper


    return make_image_comparison



# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    from glob import glob
    from spectrochempy.api import *

    @image_comparison(reference=['essai1','essai2'], force_creation=True)
    def test_compare():
        source = NDDataset.read_omnic(
                os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))
        source.plot()
        source.plot_image()

    @image_comparison(reference=['essai1','essai2'], force_creation=False)
    def test_compare_exact():
        source = NDDataset.read_omnic(
                os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))
        source.plot()
        source.plot_image()

    @image_comparison(reference=['essai2','essai2','essai2'],
                      force_creation=False)
    def test_compare_different():
        source = NDDataset.read_omnic(
                os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))

        source[10:11,3000.:3001] = masked
        source.plot_image()
        source[10:20,3000.:3020.] = masked
        source.plot_image()
        source.plot_image()
        fig = plt.figure(plt.get_fignums()[-1])
        xlim = source.axes['main'].get_xlim()
        # change a little bit the limits
        source.axes['main'].set_xlim(np.array(xlim) * .98)


    #test_compare()
    options.log_level = INFO
    log.info("exact:")
    test_compare_exact()
    log.info("different:")
    test_compare_different()

    plt.close('all')

