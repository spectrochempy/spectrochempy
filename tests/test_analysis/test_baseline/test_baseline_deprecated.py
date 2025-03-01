# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

# DEPRECATED TEST
# This test for the deprecated baseline correction
# (it should work until version 0.7)

import os

import pytest

import spectrochempy as scp
from spectrochempy import BaselineCorrection
from spectrochempy.core.units import ur

# noinspection PyUnresolvedReferences
from spectrochempy.utils.plots import show
from spectrochempy.utils.testing import (
    assert_dataset_almost_equal,
    assert_dataset_equal,
)

path = os.path.dirname(os.path.abspath(__file__))


def test_basecor_sequential(IR_dataset_2D):
    dataset = IR_dataset_2D[5]
    blc = BaselineCorrection(dataset)

    s = blc(
        [6000.0, 3500.0], [2200.0, 1500.0], method="sequential", interpolation="pchip"
    )
    s.plot()

    s1 = blc(
        [6000.0, 3500.0],
        [2200.0, 1500.0],
        method="sequential",
        interpolation="polynomial",
    )
    s1.plot(clear=False, color="red")

    s2 = blc(
        [6000.0, 3500.0], [2200.0, 1500.0], method="sequential", interpolation="pchip"
    )
    assert_dataset_almost_equal(s, s2, decimal=5)
    s2.plot(clear=False, color="green")

    s3 = blc(
        [6000.0, 3500.0],
        [2200.0, 1500.0],
        method="sequential",
        interpolation="polynomial",
    )
    assert_dataset_almost_equal(s1, s3, decimal=5)
    s3.plot(clear=False, color="cyan")

    dataset = IR_dataset_2D[:15]
    blc = BaselineCorrection(dataset)
    s = blc(
        [6000.0, 3500.0], [2200.0, 1500.0], method="sequential", interpolation="pchip"
    )
    s.plot()

    s = blc(
        [6000.0, 3500.0],
        [2200.0, 1500.0],
        method="sequential",
        interpolation="polynomial",
        order=5,
    )
    s.plot(cmap="copper")

    show()


def test_basecor_multivariate(IR_dataset_2D):
    dataset = IR_dataset_2D[5]

    basc = BaselineCorrection(dataset)
    s = basc(
        [6000.0, 3500.0], [1800.0, 1500.0], method="multivariate", interpolation="pchip"
    )
    s.plot()

    s1 = basc(
        [6000.0, 3500.0],
        [1800.0, 1500.0],
        method="multivariate",
        interpolation="polynomial",
        order=5,
    )
    s1.plot(clear=False, color="red")

    show()


def test_notebook_basecor_bug(IR_dataset_2D):
    dataset = IR_dataset_2D

    s = dataset[:, 1260.0:5999.0]
    s = s - s[-1]

    # Important note that we use floating point number
    # integer would mean points, not wavenumbers!

    basc = BaselineCorrection(s)

    ranges = [
        [1261.86, 1285.89],
        [1556.30, 1568.26],
        [1795.00, 1956.75],
        [3766.03, 3915.81],
        [4574.26, 4616.04],
        [4980.10, 4998.01],
        [5437.52, 5994.70],
    ]  # predefined ranges

    _ = basc.run(
        *ranges,
        method="multivariate",
        interpolation="pchip",
        npc=5,
        figsize=(6, 6),
        zoompreview=4,
    )

    # The regions used to set the baseline are accessible using the `ranges`
    #  attribute:
    ranges = basc.ranges
    print(ranges)

    basc.corrected.plot_stack()


# old plot_baseline_correction.py
def test_old_plot_baseline_correction():
    # ======================================================================================
    # Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
    # CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
    # See full LICENSE agreement in the root directory.
    # ======================================================================================
    # ruff: noqa
    """
    NDDataset baseline correction
    ==============================

    In this example, we perform a baseline correction of a 2D NDDataset
    interactively, using the `multivariate` method and a `pchip` interpolation.

    """

    # %%
    # As usual we start by importing the useful library, and at least  the
    # spectrochempy library.

    # %%

    import spectrochempy as scp

    # %%
    # Load data:

    datadir = scp.preferences.datadir
    nd = scp.NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")

    # %%
    # Do some slicing to keep only the interesting region:

    ndp = (nd - nd[-1])[:, 1291.0:5999.0]
    # Important:  notice that we use floating point number
    # integer would mean points, not wavenumbers!

    # %%
    # Define the BaselineCorrection object:

    ibc = scp.BaselineCorrection(ndp)

    # %%
    # Launch the interactive view, using the `BaselineCorrection.run` method:

    ranges = [
        [1556.30, 1568.26],
        [1795.00, 1956.75],
        [3766.03, 3915.81],
        [4574.26, 4616.04],
        [4980.10, 4998.01],
        [5437.52, 5994.70],
    ]  # predefined ranges
    span = ibc.run(
        *ranges, method="multivariate", interpolation="pchip", npc=5, zoompreview=3
    )

    # %%
    # Print the corrected dataset:

    print(ibc.corrected)
    _ = ibc.corrected.plot()

    # %%
    # This ends the example ! The following line can be uncommented if no plot shows when
    # running the .py script with python

    # scp.show()


def test_userguide_example():
    # %% [markdown]
    # # Baseline corrections
    #
    # This tutorial shows how to make baseline corrections with spectrochempy.
    # As prerequisite,
    # the user is expected to have read the [Import](../importexport/import.ipynb)
    # and [Import IR](../importexport/importIR.ipynb) tutorials.

    # %%
    import spectrochempy as scp

    # %% [markdown]
    # Now let's import and plot a typical IR dataset which was recorded during the
    # removal of ammonia from a NH4-Y
    # zeolite:
    # %%
    X = scp.read_omnic("irdata/nh4y-activation.spg")
    X[:, 1290.0:890.0] = scp.MASKED

    # %% [markdown]
    # After setting some plotting preferences and plot it

    # %%
    prefs = X.preferences
    prefs.figure.figsize = (7, 3)
    prefs.colormap = "magma"
    X.plot()

    # %% [markdown]
    # ## Background subtraction
    #
    # Often, particularly for surface species, the baseline is first corrected
    # by subtracting a reference spectrum. In this
    # example, it could be, for instance, the last spectrum (index -1). Hence:

    # %%
    Xdiff = X - X[-1]
    _ = Xdiff.plot()

    # %% [markdown]
    # ## Detrend
    #
    # Other simple baseline corrections - often use in preprocessing prior chemometric
    # analysis - constist in shifting
    # the spectra or removing a linear trend. This is done using the detrend() method,
    # which is a wrapper of the [
    # detrend() method]
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html)
    # from the [
    # scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html)
    # module to which we refer the interested reader.

    # %% [markdown]
    # ### Linear trend
    # Subtract the linear trend of each spectrum (type='linear', default)

    # %%
    _ = X.detrend().plot()

    # %% [markdown]
    # ### Constant trend
    # Subtract the average absorbance to each spectrum

    # %%
    _ = X.detrend(type="constant").plot()

    # %% [markdown]
    # ## Automatic linear baseline correction `abc`

    # %% [markdown]
    # When the baseline to remove is a simple linear correction, one can use `abc` .
    # This performs an automatic baseline correction.

    # %%
    _ = scp.abc(X).plot()

    # %% [markdown]
    # ## Advanced baseline correction
    #
    # 'Advanced' baseline correction basically consists for the user to choose:
    #
    # - spectral ranges which s/he considers as belonging to the baseline - the type of
    # polynomial(s) used to model the
    # baseline in and between these regions (keyword: `interpolation` ) - the method used
    # to apply the correction to
    # spectra: sequentially to each spectrum, or using a multivariate approach
    # (keyword: `method` ).
    #
    # ### Range selection
    #
    # Each spectral range is defined by a list of two values indicating the limits of the
    # spectral ranges, e.g. `[4500.,
    # 3500.]` to
    # select the 4500-3500 cm$^{-1}$ range. Note that the ordering has no importance and
    # using `[3500.0, 4500.]` would
    # lead to exactly the same result. It is also possible to formally pick a single
    # wavenumber `3750.` .
    #
    # The first step is then to select the various regions that we expect to belong to
    # the baseline

    # %%
    ranges = (
        [5900.0, 5400.0],
        4550.0,
        [4500.0, 4000.0],
        [2100.0, 2000.0],
        [1550.0, 1555.0],
    )

    # %% [markdown]
    # After selection of the baseline ranges, the baseline correction can be made using a
    # sequence of 2 commands:
    #
    # 1. Initialize an instance of BaselineCorrection

    # %%
    blc = scp.BaselineCorrection(X)

    # %% [markdown]
    # 2. compute baseline other the ranges

    # %%
    Xcorr = blc.compute(ranges)
    Xcorr

    # %% [markdown]
    # * plot the result (blc.corrected.plot() would lead to the same result)

    # %%
    _ = Xcorr.plot()

    # %% [markdown]
    # ### Interpolation method
    #
    #
    # The previous correction was made using the default parameters for the interpolation
    # ,i.e. an interpolation using cubic Hermite spline interpolation:
    # `interpolation='pchip'` (`pchip` stands for
    # **P**iecewise **C**ubic **H**ermite
    # **I**nterpolating **P**olynomial). This option triggers the use of
    # [scipy.interpolate.PchipInterpolator()](
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
    # to which we refer the interested readers. The other interpolation method is the
    # classical polynomial interpolation (`interpolation='polynomial'` ) in which case the
    # order can also be set (e.g. `order=3` , the default value being 6).
    # In this case, the base methods used for the interpolation are those of the
    # [polynomial module](
    # https://numpy.org/doc/stable/reference/routines.polynomials.polynomial.html)
    # of spectrochempy, in particular the
    # [polyfit()](
    # https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfit.html#numpy.polynomial.polynomial.polyfit) method.
    #
    # For instance:

    # %% [markdown]
    # First, we put the ranges in a list

    # %%
    ranges = [[5900.0, 5400.0], [4000.0, 4500.0], [2100.0, 2000.0], [1550.0, 1555.0]]

    # %% [markdown]
    # <div class='alert alert-warning'>
    # <b>Warning</b>
    #
    # if you use a tuple to define the sequences of ranges:
    #
    # ```ipython3
    # ranges = [5900.0, 5400.0], [4000., 4500.], [2100., 2000.0], [1550., 1555.]
    # ```
    #
    # or
    #
    # ```ipython3
    # ranges = ([5900.0, 5400.0], [4000., 4500.], [2100., 2000.0], [1550., 1555.])
    # ```
    #
    # then you can call `compute` by directly pass the ranges tuple, or you can unpack
    # it as below.
    #
    # ```ipython3
    # blc.compute(ranges, ....)
    # ```
    #
    #
    # if you use a list instead of tuples:
    #
    # ```ipython3
    # ranges = [[5900.0, 5400.0], [4000., 4500.], [2100., 2000.0], [1550., 1555.]]
    # ```
    #
    # then you **MUST UNPACK** the element when calling `compute`:
    #
    # ```ipython3
    # blc.compute(*ranges, ....)
    # ```
    #
    #
    # </div>

    # %%
    blc = scp.BaselineCorrection(X)
    blc.compute(*ranges, interpolation="polynomial", order=6)

    # %% [markdown]
    # The `corrected` attribute contains the corrected NDDataset.

    # %%
    _ = blc.corrected.plot()

    # %% [markdown]
    # ### Multivariate method
    #
    # The `method` option defines whether the selected baseline regions of the spectra
    # should be taken 'as is'
    # this is the default `method='sequential'` ), or modeled using a multivariate
    # approach (`method='multivariate'` ).
    #
    # The `'multivariate'` option is useful when the signal‐to‐noise ratio is low
    # and/or when the baseline changes in
    # various regions of the spectrum are correlated. It consist in (i) modeling the
    # baseline regions by a principal
    # component analysis (PCA), (ii) interpolate the loadings of the first principal
    # components over the whole spectral
    # and (iii) modeling the spectra baselines from the product of the PCA scores and
    # the interpolated loadings.
    # (for detail: see [Vilmin et al. Analytica Chimica Acta 891
    # (2015)](http://dx.doi.org/10.1016/j.aca.2015.06.006)).
    #
    # If this option is selected, the user should also choose `npc` , the number of
    # principal components used to model the
    # baseline. In a sense, this parameter has the same role as the `order` parameter,
    # except that it will affect how well
    # the baseline fits the selected regions, but on *both dimensions: wavelength
    # and acquisition time*. In particular a
    # large value of `npc` will lead to overfit of baseline variation with time and will
    # lead to the same result as the
    # `sequential` method while a too small `value` would miss important principal
    # component underlying the baseline change
    # over time. Typical optimum values are `npc=2` or `npc=3` (see Exercises below).

    # %%
    blc = scp.BaselineCorrection(X)
    blc.compute(*ranges, interpolation="pchip", method="multivariate", npc=2)
    _ = blc.corrected.plot()

    # %% [markdown]
    # ### Code snippet for 'advanced' baseline correction
    # The following code in which the user can change any of the parameters and look at
    # the changes after re-running
    # the cell:

    # %%
    # user defined parameters
    # -----------------------
    ranges = (
        [5900.0, 5400.0],
        [4000.0, 4500.0],
        4550.0,
        [2100.0, 2000.0],
        [1550.0, 1555.0],
        [1250.0, 1300.0],
        [800.0, 850.0],
    )
    interpolation = "pchip"  # choose 'polynomial' or 'pchip'
    order = 5  # only used for 'polynomial'
    method = "sequential"  # choose 'sequential' or 'multivariate'
    npc = 3  # only used for 'multivariate'

    # code: compute baseline, plot original and corrected NDDatasets and ranges
    # --------------------------------------------------------------------------------------
    blc = scp.BaselineCorrection(X)
    Xcorr = blc.compute(
        *ranges, interpolation=interpolation, order=order, method=method, npc=npc
    )

    axes = scp.multiplot(
        [X, Xcorr],
        labels=["Original", "Baseline corrected"],
        sharex=True,
        nrow=2,
        ncol=1,
        figsize=(7, 6),
        dpi=96,
    )
    blc.show_regions(axes["axe21"])

    # %% [markdown]
    # <div class='alert alert-info'>
    #     <b>Exercises</b>
    #
    # **basic:**
    # - write commands to subtract (i) the first spectrum from a dataset and (ii)
    # the mean spectrum from a dataset
    # - write a code to correct the baseline of the last 10 spectra of the above dataset
    # in the 4000-3500 cm$^{-1}$ range
    #
    # **intermediate:**
    # - what would be the parameters to use in 'advanced' baseline correction to mimic
    # 'detrend' ? Write a code to check
    # your answer.
    #
    # **advanced:**
    # - simulate noisy spectra with baseline drifts and compare the performances of
    # `multivariate` vs `sequential` methods
    # </div>
