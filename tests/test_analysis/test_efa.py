# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
import numpy as np

import spectrochempy as scp
from spectrochempy.analysis.models import asymmetricvoigtmodel
from spectrochempy.utils import show


def test_EFA(IR_dataset_2D):

    ####################################################################################
    # Generate a test dataset
    # ----------------------------------------------------------------------------------
    # 1) simulated chromatogram
    # *************************

    ntimes = 250
    ncomponents = 2

    t = scp.LinearCoord.arange(
        ntimes, units="minutes", title="time"
    )  # time coordinates
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

    efa.f_ev.T.plot(yscale="log", legend=efa.f_ev.x.labels)

    efa.b_ev.T.plot(yscale="log", legend=efa.b_ev.x.labels)

    ####################################################################################
    # Looking at these EFA curves, it is quite obvious that only two components
    # are really significant, and this corresponds to the data that we have in
    # input.
    # We can consider that the third EFA components is mainly due to the noise,
    # and so we can use it to set a cut of values

    n_pc = efa.used_components = 2  # what is important here is to set used_components
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

    efa = scp.EFA(used_components=4)
    efa.fit(ds)

    C = efa.transform()
    C.T.plot()

    show()
