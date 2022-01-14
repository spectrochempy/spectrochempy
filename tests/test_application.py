# -*- coding: utf-8 -*-
# flake8: noqa

from subprocess import Popen, PIPE


import pytest

from spectrochempy import APIref, set_loglevel, get_loglevel, warning_, version

set_loglevel("WARNING")


def test_api():
    assert "EFA" in APIref
    assert "CRITICAL" in APIref
    assert "np" in APIref
    assert "NDDataset" in APIref
    assert "abs" in APIref


def test_datadir():
    # test print a listing of the testdata directory

    from spectrochempy.application import DataDir

    print(DataDir().listing())

    # or simply
    print(DataDir())

    assert str(DataDir()).startswith("testdata")


def test_version():
    # test version
    assert len(version.split(".")) >= 3


def test_log():
    # test log
    set_loglevel("WARNING")
    assert get_loglevel() == 30
    warning_("Ok, this is nicely executing!")
    set_loglevel(10)
    assert get_loglevel() == 10


def test_magic_addscript(ip):
    if ip is None:
        warning_("ip is None - pss this test ")
        return

    from spectrochempy.application import SpectroChemPyMagics

    ip.register_magics(SpectroChemPyMagics)

    ip.run_cell("from spectrochempy import *")

    assert "preferences" in ip.user_ns.keys()

    ip.run_cell("print(preferences.available_styles)", store_history=True)
    ip.run_cell("project = Project()", store_history=True)
    x = ip.run_line_magic("addscript", "-p project -o prefs -n preferences 2")

    print("x", x)
    assert x.strip() == "Script prefs created."

    # with cell contents
    x = ip.run_cell(
        "%%addscript -p project -o essai -n preferences\n"
        "print(preferences.available_styles)"
    )

    print("result\n", x.result)
    assert x.result.strip() == "Script essai created."


@pytest.mark.skip()
def test_console_subprocess():
    # to test this, the scripts must be installed so the spectrochempy
    # package must be installed: use pip install -e .

    res = Popen(["scpy"], stdout=PIPE, stderr=PIPE)
    output, error = res.communicate()
    assert "nh4y-activation.spg'" in error.decode("utf-8")
    assert "A.Travert & C.Fernandez @ LCS" in output.decode("utf-8")


def test_config(capsys):
    from spectrochempy.application import SpectroChemPy

    app = SpectroChemPy(show_config_json=True)
    app.start()
    out, err = capsys.readouterr()
    assert "SpectroChemPy" in out
