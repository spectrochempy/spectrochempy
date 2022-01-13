# -*- coding: utf-8 -*-
# flake8: noqa


# suppress test for PEP8 in this file
# flake8: noqa

from os import environ
import pathlib

import numpy as np
import pytest

try:
    # work only if spectrochempy is installed
    import spectrochempy
except ModuleNotFoundError:  # pragma: no cover
    raise ModuleNotFoundError(
        "You must install spectrochempy and its dependencies before executing tests!"
    )

from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.ndcomplex import NDComplexArray
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.scripts.script import Script
from spectrochempy.core.project.project import Project
from spectrochempy.utils import pathclean
from spectrochempy.utils.testing import RandomSeedContext

# ======================================================================================================================
# FIXTURES
# ======================================================================================================================

# initialize a ipython session before calling spectrochempy
# ------------------------------------------------------------------
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
        f.unlink()
    for f in list(cwd.glob("**/*.log")):
        f.unlink()

    docs = cwd / "docs"
    for f in list(docs.glob("**/*.ipynb")):
        f.unlink()


# ------------------------------------------------------------------
# create reference arrays
# ------------------------------------------------------------------

with RandomSeedContext(12345):
    ref_data = 10.0 * np.random.random((10, 8)) - 5.0
    ref3d_data = 10.0 * np.random.random((10, 100, 3)) - 5.0
    ref3d_2_data = np.random.random((9, 50, 4))

ref_mask = ref_data < -4
ref3d_mask = ref3d_data < -3
ref3d_2_mask = ref3d_2_data < -2


# ------------------------------------------------------------------
# Fixtures: some NDArray's
# ------------------------------------------------------------------


@pytest.fixture(scope="function")
def refarray():
    return ref_data.copy()


@pytest.fixture(scope="function")
def refmask():
    return ref_mask.copy()


@pytest.fixture(scope="function")
def ndarray():
    # return a simple ndarray with some data
    return NDArray(ref_data, desc="An array", copy=True).copy()


@pytest.fixture(scope="function")
def ndarrayunit():
    # return a simple ndarray with some data and units
    return NDArray(ref_data, units="m/s", copy=True).copy()


@pytest.fixture(scope="function")
def ndarraymask():
    # return a simple ndarray with some data and units
    return NDArray(
        ref_data, mask=ref_mask, units="m/s", history="Creation with mask", copy=True
    ).copy()


# ------------------------------------------------------------------
# Fixtures: Some NDComplex's array
# ------------------------------------------------------------------


@pytest.fixture(scope="function")
def ndarraycplx():
    # return a complex ndarray
    return NDComplexArray(ref_data, units="m/s", dtype=np.complex128, copy=True).copy()


@pytest.fixture(scope="function")
def ndarrayquaternion():
    # return a quaternion ndarray
    return NDComplexArray(ref_data, units="m/s", dtype=np.quaternion, copy=True).copy()


# ------------------------------------------------------------------
# Fixtures: Some NDDatasets
# ------------------------------------------------------------------

coord0_ = Coord(
    data=np.linspace(4000.0, 1000.0, 10),
    labels=list("abcdefghij"),
    units="cm^-1",
    title="wavenumber",
)


@pytest.fixture(scope="function")
def coord0():
    return coord0_.copy()


coord1_ = Coord(data=np.linspace(0.0, 60.0, 100), units="s", title="time-on-stream")


@pytest.fixture(scope="function")
def coord1():
    return coord1_.copy()


coord2_ = Coord(
    data=np.linspace(200.0, 300.0, 3),
    labels=["cold", "normal", "hot"],
    units="K",
    title="temperature",
)


@pytest.fixture(scope="function")
def coord2():
    return coord2_.copy()


coord2b_ = Coord(
    data=np.linspace(1.0, 20.0, 3),
    labels=["low", "medium", "high"],
    units="tesla",
    title="magnetic field",
)


@pytest.fixture(scope="function")
def coord2b():
    return coord2b_.copy()


coord0_2_ = Coord(
    data=np.linspace(4000.0, 1000.0, 9),
    labels=list("abcdefghi"),
    units="cm^-1",
    title="wavenumber",
)


@pytest.fixture(scope="function")
def coord0_2():
    return coord0_2_.copy()


coord1_2_ = Coord(data=np.linspace(0.0, 60.0, 50), units="s", title="time-on-stream")


@pytest.fixture(scope="function")
def coord1_2():
    return coord1_2_.copy()


coord2_2_ = Coord(
    data=np.linspace(200.0, 1000.0, 4),
    labels=["cold", "normal", "hot", "veryhot"],
    units="K",
    title="temperature",
)


@pytest.fixture(scope="function")
def coord2_2():
    return coord2_2_.copy()


@pytest.fixture(scope="function")
def nd1d():
    # a simple ddataset
    return NDDataset(ref_data[:, 1].squeeze()).copy()


@pytest.fixture(scope="function")
def nd2d():
    # a simple 2D ndarrays
    return NDDataset(ref_data).copy()


@pytest.fixture(scope="function")
def ref_ds():
    # a dataset with coordinates
    return ref3d_data.copy()


@pytest.fixture(scope="function")
def ds1():
    # a dataset with coordinates
    return NDDataset(
        ref3d_data,
        coordset=[coord0_, coord1_, coord2_],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def ds2():
    # another dataset
    return NDDataset(
        ref3d_2_data,
        coordset=[coord0_2_, coord1_2_, coord2_2_],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def dsm():
    # dataset with coords containing several axis and a mask

    coordmultiple = CoordSet(coord2_, coord2b_)
    return NDDataset(
        ref3d_data,
        coordset=[coord0_, coord1_, coordmultiple],
        mask=ref3d_mask,
        title="absorbance",
        units="absorbance",
    ).copy()


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


# ------------------------------------------------------------------
# Fixture: NMR spectra
# ------------------------------------------------------------------


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


@pytest.fixture(scope="function")
def JDX_2D():

    jdx = """##TITLE=IR_2D
    ##JCAMP-DX=5.01
    ##DATA TYPE=LINK
    ##BLOCKS=3
    ##TITLE=vz0466.spa, Wed Jul 06 21:00:38 2016 (GMT+02:00)
    ##JCAMP-DX=5.01
    ##ORIGIN=omnic
    ##OWNER=christian@MacCF.local
    ##LONGDATE=2016/07/06
    ##TIME=19:03:14
    ##XUNITS=1/CM
    ##YUNITS=ABSORBANCE
    ##FIRSTX=5999.555664
    ##LASTX=5981.234938
    ##MAXX=5999.555664
    ##MINX=5981.234938
    ##XFACTOR=1.0
    ##FIRSTY=2.057242
    ##LASTY=2.047518
    ##MAXY=2.060812
    ##MINY=2.002743
    ##YFACTOR=1e-08
    ##NPOINTS=20
    ##XYDATA=(X++(Y..Y))
    5999.555664 205724239.000000 206081247.000000 206077337.000000 205761122.000000 
    5995.698669 205383539.000000 205176067.000000 205163621.000000 205173397.000000 
    5991.841674 205024647.000000 204717803.000000 204435229.000000 204356765.000000 
    5987.984679 204492545.000000 204698991.000000 204843306.000000 204916620.000000 
    5984.127684 204970645.000000 204992580.000000 204916238.000000 204751849.000000 
    ##END
    ##TITLE=vz0467.spa, Wed Jul 06 21:10:38 2016 (GMT+02:00)
    ##JCAMP-DX=5.01
    ##ORIGIN=omnic
    ##OWNER=christian@MacCF.local
    ##LONGDATE=2016/07/06
    ##TIME=19:13:14
    ##XUNITS=1/CM
    ##YUNITS=ABSORBANCE
    ##FIRSTX=5999.555664
    ##LASTX=5981.234938
    ##MAXX=5999.555664
    ##MINX=5981.234938
    ##XFACTOR=1.0
    ##FIRSTY=2.057242
    ##LASTY=2.047518
    ##MAXY=2.060812
    ##MINY=2.002743
    ##YFACTOR=1e-08
    ##NPOINTS=20
    ##XYDATA=(X++(Y..Y))
    5999.555664 203330016.000000 203735828.000000 203964066.000000 203920483.000000 
    5995.698669 203658008.000000 203311467.000000 203041100.000000 202970385.000000 
    5991.841674 203081583.000000 203193688.000000 203127312.000000 202899098.000000 
    5987.984679 202673625.000000 202544808.000000 202475214.000000 202450990.000000 
    5984.127684 202554702.000000 202821636.000000 203135919.000000 203363680.000000 
    ##END
    ##TITLE=vz0468.spa, Wed Jul 06 21:20:38 2016 (GMT+02:00)
    ##JCAMP-DX=5.01
    ##ORIGIN=omnic
    ##OWNER=christian@MacCF.local
    ##LONGDATE=2016/07/06
    ##TIME=19:23:14
    ##XUNITS=1/CM
    ##YUNITS=ABSORBANCE
    ##FIRSTX=5999.555664
    ##LASTX=5981.234938
    ##MAXX=5999.555664
    ##MINX=5981.234938
    ##XFACTOR=1.0
    ##FIRSTY=2.057242
    ##LASTY=2.047518
    ##MAXY=2.060812
    ##MINY=2.002743
    ##YFACTOR=1e-08
    ##NPOINTS=20
    ##XYDATA=(X++(Y..Y))
    5999.555664 200489759.000000 200328993.000000 200274324.000000 200440287.000000 
    5995.698669 200729274.000000 200897192.000000 200821328.000000 200623416.000000 
    5991.841674 200488710.000000 200454282.000000 200441122.000000 200419855.000000 
    5987.984679 200450921.000000 200571250.000000 200716447.000000 200771117.000000 
    5984.127684 200681734.000000 200519037.000000 200432109.000000 200508308.000000 
    ##END
    ##END=

    """
    return jdx


# ------------------------------------------------------------------
# fixture Project
# ------------------------------------------------------------------


@pytest.yield_fixture(scope="function")
def simple_project():

    proj = Project(
        # subprojects
        Project(name="P350", label=r"$\mathrm{M_P}\,(623\,K)$"),
        Project(name="A350", label=r"$\mathrm{M_A}\,(623\,K)$"),
        Project(name="B350", label=r"$\mathrm{M_B}\,(623\,K)$"),
        # attributes
        name="project_1",
        label="main project",
    )

    assert proj.projects_names == ["P350", "A350", "B350"]

    ir = NDDataset([1.1, 2.2, 3.3], coordset=[[1, 2, 3]])
    tg = NDDataset([1, 3, 4], coordset=[[1, 2, 3]])
    proj.A350["IR"] = ir
    proj.A350["TG"] = tg
    script_source = (
        "set_loglevel(INFO)\n"
        'info_(f"samples contained in the project are {proj.projects_names}")'
    )

    proj["print_info"] = Script("print_info", script_source)
    return proj
