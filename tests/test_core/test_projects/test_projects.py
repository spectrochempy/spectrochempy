# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa
import pytest
import numpy as np

from spectrochempy.core.project.project import Project
from spectrochempy.core.scripts.script import Script, run_script
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import preferences, INPLACE
from spectrochempy.utils.testing import RandomSeedContext
import spectrochempy as scp

prefs = preferences

# ------------------------------------------------------------------
# fixture Project
# ------------------------------------------------------------------
with RandomSeedContext(12345):
    ref_data = 10.0 * np.random.random((10, 8)) - 5.0
    ref3d_data = 10.0 * np.random.random((10, 100, 3)) - 5.0
    ref3d_2_data = np.random.random((9, 50, 4))

ref_mask = ref_data < -4
ref3d_mask = ref3d_data < -3
ref3d_2_mask = ref3d_2_data < -2


@pytest.fixture(scope="function")
def ds1():
    # a dataset with coordinates
    return scp.NDDataset(
        ref3d_data,
        coordset=[coord0_, coord1_, coord2_],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def ds2():
    # another dataset
    return scp.NDDataset(
        ref3d_2_data,
        coordset=[coord0_2_, coord1_2_, coord2_2_],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def dsm():
    # dataset with coords containing several axis and a mask

    coordmultiple = scp.CoordSet(coord2_, coord2b_)
    return scp.NDDataset(
        ref3d_data,
        coordset=[coord0_, coord1_, coordmultiple],
        mask=ref3d_mask,
        title="absorbance",
        units="absorbance",
    ).copy()


coord0_ = scp.Coord(
    data=np.linspace(4000.0, 1000.0, 10),
    labels=list("abcdefghij"),
    units="cm^-1",
    title="wavenumber",
)


@pytest.fixture(scope="function")
def coord0():
    return coord0_.copy()


coord1_ = scp.Coord(data=np.linspace(0.0, 60.0, 100), units="s", title="time-on-stream")


@pytest.fixture(scope="function")
def coord1():
    return coord1_.copy()


coord2_ = scp.Coord(
    data=np.linspace(200.0, 300.0, 3),
    labels=["cold", "normal", "hot"],
    units="K",
    title="temperature",
)


@pytest.fixture(scope="function")
def coord2():
    return coord2_.copy()


coord2b_ = scp.Coord(
    data=np.linspace(1.0, 20.0, 3),
    labels=["low", "medium", "high"],
    units="tesla",
    title="magnetic field",
)


@pytest.fixture(scope="function")
def coord2b():
    return coord2b_.copy()


coord0_2_ = scp.Coord(
    data=np.linspace(4000.0, 1000.0, 9),
    labels=list("abcdefghi"),
    units="cm^-1",
    title="wavenumber",
)


@pytest.fixture(scope="function")
def coord0_2():
    return coord0_2_.copy()


coord1_2_ = scp.Coord(
    data=np.linspace(0.0, 60.0, 50), units="s", title="time-on-stream"
)


@pytest.fixture(scope="function")
def coord1_2():
    return coord1_2_.copy()


coord2_2_ = scp.Coord(
    data=np.linspace(200.0, 1000.0, 4),
    labels=["cold", "normal", "hot", "veryhot"],
    units="K",
    title="temperature",
)


@pytest.fixture(scope="function")
def coord2_2():
    return coord2_2_.copy()


# Basic
# --------------------------------------------------------------------------------------------------------


def test_project(ds1, ds2, dsm):
    myp = Project(name="AGIR processing", method="stack")

    ds1.name = "toto"
    ds2.name = "tata"
    dsm.name = "titi"

    ds = ds1[:, 10, INPLACE]
    assert ds1.shape == ds.shape
    assert ds is ds1

    myp.add_datasets(ds1, ds2, dsm)

    print(myp.datasets_names)
    assert myp.datasets_names[-1] == "titi"
    assert ds1.parent == myp

    # iteration
    d = []
    for item in myp:
        d.append(item)

    assert d[1][0] == "tata"

    ##
    # add sub project
    msp1 = Project(name="AGIR ATG")
    msp1.add_dataset(ds1)
    assert ds1.parent == msp1  # ds1 has changed of project
    assert ds1.name not in myp.datasets_names

    msp2 = Project(name="AGIR IR")

    myp.add_projects(msp1, msp2)

    print(myp)
    # an object can be accessed by it's name whatever it's type
    assert "tata" in myp.allnames
    myp["titi"]
    assert myp["titi"] == dsm

    # import multiple objects in Project
    myp2 = Project(msp1, msp2, ds1, ds2)  # multi dataset and project and no
    # names

    print(myp2)


def test_empty_project():
    proj = Project(name="XXX")
    assert proj.name == "XXX"
    assert str(proj).strip() == "Project XXX:\n    (empty project)"


def test_project_with_script():
    # Example from tutorial agir notebook
    proj = Project(
        Project(name="P350", label=r"$\mathrm{M_P}\,(623\,K)$"),
        Project(name="A350", label=r"$\mathrm{M_A}\,(623\,K)$"),
        Project(name="B350", label=r"$\mathrm{M_B}\,(623\,K)$"),
        name="HIZECOKE_TEST",
    )

    assert proj.projects_names == ["P350", "A350", "B350"]

    # add a dataset to a subproject
    ir = NDDataset([1, 2, 3])
    tg = NDDataset([1, 3, 4])
    proj.A350["IR"] = ir
    proj["TG"] = tg

    print(proj.A350)
    print(proj)
    print(proj.A350.label)

    f = proj.save()

    newproj = Project.load("HIZECOKE_TEST")
    # print(newproj)
    assert str(newproj) == str(proj)
    assert newproj.A350.label == proj.A350.label

    # proj = Project.load('HIZECOKE')
    # assert proj.projects_names == ['A350', 'B350', 'P350']

    script_source = (
        "set_loglevel(INFO)\n"
        'info_("samples contained in the project are : '
        '%s"%proj.projects_names)'
    )

    proj["print_info"] = Script("print_info", script_source)

    # print(proj)

    # save but do not change the original data
    proj.save_as("HIZECOKE_TEST", overwrite_data=False)

    newproj = Project.load("HIZECOKE_TEST")

    # execute
    run_script(newproj.print_info, locals())
    newproj.print_info.execute(locals())

    newproj.print_info(locals())

    # attempts to resolve locals
    newproj.print_info()

    proj.save_as("HIZECOKE_TEST")
    newproj = Project.load("HIZECOKE_TEST")


def test_save_and_load_project(ds1, ds2):
    myp = Project(name="process")

    ds1.name = "toto"
    ds2.name = "tata"

    myp.add_datasets(ds1, ds2)

    fn = myp.save()

    proj = Project.load(fn)

    assert str(proj["toto"]) == "NDDataset: [float64] a.u. (shape: (z:10, y:100, x:3))"
