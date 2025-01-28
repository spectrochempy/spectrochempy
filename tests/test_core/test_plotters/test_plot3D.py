# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import os

from spectrochempy import Coord, NDDataset, show


def test_plot2D_as_3D():
    data = NDDataset.read_matlab(os.path.join("matlabdata", "als2004dataset.MAT"))

    X = data[0]

    X.plot_surface()

    X.set_coordset(
        y=Coord(title="elution time", units="s"),
        x=Coord(title="wavenumbers", units="cm^-1"),
    )
    X.title = "intensity"
    X.plot_surface()

    X.plot_surface(colorbar=True)

    show()
