# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import sys
from unittest import mock

import numpy as np
import pytest
import requests

import spectrochempy as scp


class requests_response:

    resp = "5.1,3.5,1.4,0.2,Iris-setosa\n4.9,3.0,1.4,0.2,Iris-setosa"

    def __init__(self, valid=True):
        if not valid:
            self.resp = self.resp + "\nxxxx "

    def iter_content(self):
        return (rd.encode("utf-8") for rd in self.resp)


class sklearn_data:

    data = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]])
    target = [0, 0]
    target_names = ("Iris-setosa", "Iris-versicolor")


def test_mock_download_iris(mocker):

    # Download file
    mocker.patch("requests.get", mock.Mock(return_value=requests_response()))
    ds1 = scp.download_iris()
    assert str(ds1) == "NDDataset: [float64] cm (shape: (y:2, x:4))"

    # Alternatively use sklearn
    mocker.patch(
        "requests.get", side_effect=OSError("Failed in uploading the `IRIS` dataset!")
    )
    sys.modules["sklearn"] = mock.MagicMock(__name__="sklearn", __VERSION__="0.0.0")
    sys.modules["datasets"] = mock.MagicMock(__name__="datasets", __VERSION__="0.0.0")
    mocker.patch("sklearn.datasets.load_iris", return_value=sklearn_data())
    ds2 = scp.download_iris()
    assert ds2 == ds1

    # no-sklearn?
    mocker.patch(
        "spectrochempy.core.readers.download.import_optional_dependency",
        return_value=None,
    )
    mocker.patch(
        "requests.get", side_effect=OSError("Failed in uploading the `IRIS` dataset!")
    )
    with pytest.raises(OSError):
        scp.download_iris()

    # wrong imported csv file
    mocker.patch("requests.get", mock.Mock(return_value=requests_response(valid=False)))
    with pytest.raises(OSError):
        scp.download_iris()


def test_download_nist():
    CAS = "7732-18-5"  # WATER

    ds = scp.download_nist_ir(CAS)
    assert len(ds) == 2

    ds = scp.download_nist_ir(CAS, index=0)
    assert ds.name == "Water"

    ds = scp.download_nist_ir(CAS, index=[0, 1])
    assert len(ds) == 2

    ds = scp.download_nist_ir(CAS, index=2)
    assert ds is None

    ds = scp.download_nist_ir(CAS, index=[0, 1, 2])
    assert len(ds) == 2

    CAS = 2146363  # Acenaphthylene, dodecahydro-
    ds = scp.download_nist_ir(CAS)
    assert ds is None


def test_download():
    ds1 = scp.read("http://www.eigenvector.com/data/Corn/corn.mat")
    assert len(ds1) == 7

    ds1 = scp.read_mat("http://www.eigenvector.com/data/Corn/corn.mat")
    assert len(ds1) == 7

    ds1 = scp.read_remote("http://www.eigenvector.com/data/Corn/corn.mat")
    assert len(ds1) == 7

    ds2 = scp.read("https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip")
    assert len(ds2) == 7

    with pytest.raises(FileNotFoundError):
        scp.read_remote("http://www.eigenvector.com/does_not_exist.mat")

    with pytest.raises(TypeError):
        scp.read_remote("https://www.spectrochempy.fr/latest/index.html")
