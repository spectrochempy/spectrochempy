# -*- coding: utf-8 -*-
# flake8: noqa
import pathlib
import pytest

# initialize a ipython session before calling spectrochempy
# ---------------------------------------------------------


@pytest.fixture(scope="session")
def session_ip():
    try:
        from IPython.testing.globalipapp import start_ipython

        return start_ipython()
    except ImportError:
        return None


@pytest.fixture(scope="module")
def ip(session_ip):
    yield session_ip


def pytest_sessionfinish(session, exitstatus):  # pragma: no cover
    """whole test run finishes."""

    # cleaning
    cwd = pathlib.Path(__file__).parent.parent

    for f in list(cwd.glob("**/*.?scp")):
        f.unlink()
    for f in list(cwd.glob("**/*.jdx")):
        f.unlink()
    for f in list(cwd.glob("**/*.json")):
        if f.name() != ".zenodo.json":
            f.unlink()
    for f in list(cwd.glob("**/*.log")):
        f.unlink()
    for f in list(cwd.glob("**/*.nc")):
        f.unlink()
    docs = cwd / "docs"
    for f in list(docs.glob("**/*.ipynb")):
        f.unlink()


try:
    # work only if spectrochempy is installed
    import spectrochempy
except ModuleNotFoundError:  # pragma: no cover
    raise ModuleNotFoundError(
        "You must install spectrochempy and its dependencies before executing tests!"
    )

# -----------------------------------------
# SOME FIXTURES TO SHARE BETWEEN TEST FILES
# -----------------------------------------

from spectrochempy import preferences as prefs, NDDataset
from spectrochempy.utils import pathclean

datadir = pathclean(prefs.datadir)
dataset = NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")


@pytest.fixture(scope="function")
def IR_dataset_2D():
    nd = dataset.copy()
    nd.name = "IR_2D"
    return nd


@pytest.fixture(scope="function")
def IR_dataset_1D():
    nd = dataset[0].squeeze().copy()
    nd.name = "IR_1D"
    return nd


@pytest.fixture(scope="function")
def NMR_dataset_1D():
    path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_1d" / "1" / "fid"
    dataset = NDDataset.read_topspin(path, remove_digital_filter=True, name="NMR_1D")
    return dataset.copy()


@pytest.fixture(scope="function")
def NMR_dataset_2D():
    path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d" / "1" / "ser"
    dataset = NDDataset.read_topspin(
        path, expno=1, remove_digital_filter=True, name="NMR_2D"
    )
    return dataset.copy()
