# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
In this module, methods are provided to download external datasets
from public database.
"""
__all__ = ["download_iris", "download_nist_ir"]
__dataset_methods__ = __all__

from io import StringIO
import numpy as np
import requests
from datetime import datetime, timezone
from pathlib import Path

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.read_jcamp import read_jcamp
from spectrochempy.core import error_, info_
from spectrochempy.optional import import_optional_dependency
from spectrochempy.utils import is_iterable


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
            raise OSError("can't read JCAMP file")

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
        else:
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


def download_nist_ir(CAS, index="all"):
    """
    Upload IR spectra from NIST webbook

    Parameters
    ----------
    CAS : int or str
        the CAS number, can be given as "XXXX-XX-X" (str), "XXXXXXX" (str), XXXXXXX (int)

    index : str or int or tuple of ints
        If set to 'all' (default, import all available spectra for the compound corresponding to the index, or a single spectrum,
        or selected spectra.

    Returns
    -------
    list of NDDataset or NDDataset
        The dataset(s).

    See Also
    --------
    read : Read data from experimental data.
    """

    if isinstance(CAS, str) and "-" in CAS:
        CAS = CAS.replace("-", "")

    if index == "all":
        # test urls and return list if any...
        index = []
        i = 0
        while "continue":
            url = (
                f"https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C{CAS}&Index={i}&Type=IR"
            )
            try:
                response = requests.get(url, timeout=10)
                if b"Spectrum not found" in response.content[:30]:
                    break
                else:
                    index.append(i)
                    i += 1
            except OSError:
                error_("OSError: could not connect to NIST")
                return None

        if len(index) == 0:
            error_("NIST IR: no spectrum found")
            return
        elif len(index) == 1:
            info_("NIST IR: 1 spectrum found")
        else:
            info_("NISTR IR: {len(index)} spectra found")

    elif isinstance(index, int):
        index = [index]
    elif not is_iterable(index):
        raise ValueError("index must be 'all', int or iterable of int")

    out = []
    for i in index:
        # sample adress (water, spectrum 1)
        # https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C7732185&Index=1&Type=IR
        url = f"https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C{CAS}&Index={i}&Type=IR"
        try:
            response = requests.get(url, stream=True, timeout=10)
            if b"Spectrum not found" in response.content[:30]:
                error_(f"NIST IR: Spectrum {i} does not exist... please check !")
                if i == index[-1] and out == []:
                    return None
                else:
                    break

        except OSError:
            error_("OSError: Cannot connect... ")
            return None

        # Load data
        txtdata = ""
        for rd in response.iter_content():
            txtdata += rd.decode("utf8")

        with open("temp.jdx", "w") as f:
            f.write(txtdata)
        try:
            ds = read_jcamp("temp.jdx")

            # replace the default entry ":imported from jdx file":
            ds.history[0] = ds.history[0][: len(str(datetime.now(timezone.utc)))] + (
                f" : downloaded from NIST: {url}\n"
            )
            out.append(ds)
            (Path(".") / "temp.jdx").unlink()

        except Exception:
            raise OSError(
                "Can't read this JCAMP file: please report the issue to Spectrochempy developpers"
            )

    if len(out) == 1:
        return out[0]
    else:
        return out


# ======================================================================================================================
if __name__ == "__main__":
    pass
