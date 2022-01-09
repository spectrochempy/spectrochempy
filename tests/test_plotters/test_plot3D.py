# -*- coding: utf-8 -*-
# flake8: noqa


import os

from spectrochempy import NDDataset, Coord, show


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

    # show()

    pass


# =============================================================================
if __name__ == "__main__":
    pass
