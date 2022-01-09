# -*- coding: utf-8 -*-
# flake8: noqa


import pytest

import spectrochempy as scp

from tests.test_readers_writers.main import _download_iris


@pytest.fixture(scope="module")
def iris_dataset():
    return scp.load("../data/iris_dataset.scp")


@pytest.skip("mocking is much faster")
def test_download_iris(iris_dataset):

    actual = _download_iris()
    expected = iris_dataset
    assert expected == actual


def test_mock_download_iris(mocker, iris_dataset):

    mocker.patch(
        "tests.test_readers_writers.main.download_iris", return_value=iris_dataset
    )
    actual = _download_iris()
    expected = iris_dataset
    assert expected == actual
