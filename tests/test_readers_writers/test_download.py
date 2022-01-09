# -*- coding: utf-8 -*-
# flake8: noqa


import pytest
import pathlib

import spectrochempy as scp

from tests.test_readers_writers.main import _download_iris, get_path, save_iris_dataset


@pytest.fixture(scope="module")
def iris_dataset():
    path = get_path()
    f = path / "tests/data/iris_dataset.scp"
    if not f.exists():
        print(f"file {f} not found- create one")
        save_iris_dataset()
    return scp.load(f)


@pytest.mark.skip("mocking is much faster")
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
