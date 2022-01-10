# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
In this module, methods are provided to download external datasets
from public database.
"""
__all__ = ["download_iris"]
__dataset_methods__ = __all__

from io import StringIO

import numpy as np
import requests

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core import error_
from spectrochempy.optional import import_optional_dependency


# ..............................................................................
def download_iris():
    """
    Upload the classical `IRIS` dataset.

    The `IRIS` dataset is a classical example for machine learning.It is downloaded from
    the [UCI distant repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

    Returns
    -------
    dataset
        The `IRIS` dataset.

    See Also
    --------
    read : Read data from experimental data.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    try:
        connection = True
        response = requests.get(url, stream=True, timeout=10)
    except OSError:
        error_("OSError: Cannot connect to the UCI repository. Try Scikit-Learn")
        connection = False

    if connection:  # Download data
        txtdata = ""
        for rd in response.iter_content():
            txtdata += rd.decode("utf8")

        fil = StringIO(txtdata)
        try:
            data = np.loadtxt(fil, delimiter=",", usecols=range(4))
            fil.seek(0)
            labels = np.loadtxt(fil, delimiter=",", usecols=(4,), dtype="|S")
            labels = list((lab.decode("utf8") for lab in labels))
        except Exception:
            raise OSError("Wrong CSV file")

        coordx = Coord(
            labels=["sepal_length", "sepal width", "petal_length", "petal_width"],
            title="features",
        )
        coordy = Coord(labels=labels, title="samples")

        new = NDDataset(
            data,
            coordset=[coordy, coordx],
            title="size",
            name="`IRIS` Dataset",
            units="cm",
        )

        new.history = "Loaded from UC Irvine machine learning repository"

        return new

    else:
        # Cannot download - use the scikit-learn dataset (if scikit-learn is installed)

        sklearn = import_optional_dependency("sklearn", errors="ignore")
        if sklearn is None:
            raise OSError("Failed in uploading the `IRIS` dataset!")
        from sklearn import datasets

        data = datasets.load_iris()

        coordx = Coord(
            labels=["sepal_length", "sepal width", "petal_length", "petal_width"],
            title="features",
        )
        labels = [data.target_names[i] for i in data.target]
        coordy = Coord(labels=labels, title="samples")

        new = NDDataset(
            data.data,
            coordset=[coordy, coordx],
            title="size",
            name="`IRIS` Dataset",
            units="cm",
        )

        new.history = "Loaded from scikit-learn datasets"

        return new


# ======================================================================================================================
if __name__ == "__main__":
    pass
