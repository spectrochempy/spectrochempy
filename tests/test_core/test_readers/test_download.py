# -*- coding: utf-8 -*-
# flake8: noqa

from unittest import mock
import sys

import pytest
import numpy as np

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
