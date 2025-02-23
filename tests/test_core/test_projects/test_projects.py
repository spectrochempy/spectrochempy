# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa


from spectrochempy.core import INPLACE, preferences
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.project.project import Project
from spectrochempy.core.script import Script, run_script

prefs = preferences

try:
    from spectrochempy.core.common import dialogs
except ImportError:
    import pytest

    pytest.skip("dialogs not available with act", allow_module_level=True)

# Basic
# --------------------------------------------------------------------------------------


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
