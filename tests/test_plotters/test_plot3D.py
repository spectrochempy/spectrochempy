# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os

from spectrochempy import NDDataset, Coord, show


def test_plot2D_as_3D():
    data = NDDataset.read_matlab(os.path.join("matlabdata", "als2004dataset.MAT"))

    X = data[0]
    # X.plot_3D()

    X.plot_surface()

    X.set_coordset(y=Coord(title="elution time"), x=Coord(title="wavenumbers"))
    X.title = "intensity"
    X.plot_surface()

    X.plot_surface(colorbar=True)

    show()

    pass


# =============================================================================
if __name__ == "__main__":
    pass
