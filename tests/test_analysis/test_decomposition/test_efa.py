# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
from os import environ

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._models import asymmetricvoigtmodel
from spectrochempy.utils import docstrings as chd
from spectrochempy.utils.constants import MASKED
from spectrochempy.utils.plots import show


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_EFA_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.efa"
    chd.check_docstrings(
        module,
        obj=scp.EFA,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"],
    )


def test_example():
    # Init the model
    model = scp.EFA()
    # Read an experimental 2D spectra (N x M )
    X = scp.read("irdata/nh4y-activation.spg")
    # Fit the model
    model.fit(X)
    # Display components spectra (2 x M)
    model.n_components = 2
    _ = model.components.plot(title="Components")
    # Get the abstract concentration profile based on the FIFO EFA analysis
    c = model.transform()
    # Plot the transposed concentration matrix  (2 x N)
    _ = c.T.plot(title="Concentration")
    scp.show()


def test_EFA(IR_dataset_2D):
    ####################################################################################
    # Generate a test dataset
    # ----------------------------------------------------------------------------------
    # 1) simulated chromatogram
    # *************************

    ntimes = 250
    ncomponents = 2

    t = scp.Coord.arange(ntimes, units="minutes", title="time")  # time coordinates
    c = scp.Coord(range(ncomponents), title="components")  # component coordinates

    data = np.zeros((ncomponents, ntimes), dtype=np.float64)

    data[0] = asymmetricvoigtmodel().f(
        t, ampl=4, width=20, ratio=0.5, asym=0.4, pos=50.0
    )  # compound 1
    data[1] = asymmetricvoigtmodel().f(
        t, ampl=5, width=40, ratio=0.2, asym=0.9, pos=120.0
    )  # compound 2

    dsc = scp.NDDataset(data=data, coords=[c, t])
    dsc.plot(title="concentration")

    ####################################################################################
    # 2) absorption spectra
    # **********************

    spec = np.array([[2.0, 3.0, 4.0, 2.0], [3.0, 4.0, 2.0, 1.0]])
    w = scp.Coord(np.arange(1, 5, 1), units="nm", title="wavelength")

    dss = scp.NDDataset(data=spec, coords=[c, w])
    dss.plot(title="spectra")

    ####################################################################################
    # 3) simulated data matrix
    # ************************

    dataset = scp.dot(dsc.T, dss)
    dataset.data = np.random.normal(dataset.data, 0.03)
    dataset.title = "intensity"

    dataset.plot_map()
    show()

    ####################################################################################
    # 4) evolving factor analysis (EFA)
    # *********************************

    efa = scp.EFA()
    efa.fit(dataset)

    ####################################################################################
    # Plots of the log(EV) for the forward and backward analysis
    #

    efa.f_ev.T.plot(yscale="log", legend=efa.f_ev.k.labels)

    efa.b_ev.T.plot(yscale="log", legend=efa.b_ev.k.labels)

    ####################################################################################
    # Looking at these EFA curves, it is quite obvious that only two components
    # are really significant, and this corresponds to the data that we have in
    # input.
    # We can consider that the third EFA components is mainly due to the noise,
    # and so we can use it to set a cut of values

    n_pc = efa.n_components = 2  # what is important here is to set n_components
    efa.cutoff = np.max(efa.f_ev[:, n_pc].data)

    f2 = efa.f_ev[:, :n_pc]
    b2 = efa.b_ev[:, :n_pc]

    # we concatenate the datasets to plot them in a single figure
    both = scp.concatenate(f2, b2)
    both.T.plot(yscale="log")

    # ##################################################################################
    # # Get the abstract concentration profile based on the FIFO EFA analysis
    # #

    c = efa.transform()
    c.T.plot()

    scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)

    ds = IR_dataset_2D.copy()
    #
    # columns masking
    ds[:, 1230.0:920.0] = MASKED  # do not forget to use float in slicing
    ds[:, 5900.0:5890.0] = MASKED
    #
    # difference spectra
    ds -= ds[-1]
    #
    # row masking
    ds[10:12] = MASKED

    efa = scp.EFA(n_components=4)
    efa.fit(ds)

    C = efa.transform()
    C.T.plot()

    show()
