# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy import Coord, show


def test_plot2D_as_3D(sample_3d_dataset):
    X = sample_3d_dataset.copy()

    X.plot_surface()

    X.set_coordset(
        y=Coord(title="elution time", units="s"),
        x=Coord(title="wavenumbers", units="cm^-1"),
    )
    X.title = "intensity"
    X.plot_surface()

    X.plot_surface(colorbar=True)

    show()
