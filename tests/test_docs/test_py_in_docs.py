# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
# --------------------------------------------------------------------------------------
# Testing examples and notebooks (Py version) in docs
# --------------------------------------------------------------------------------------
import subprocess
import sys
from os import environ
from pathlib import Path

import pytest
from traitlets import import_item

if environ.get("USER", None) is None and (
    sys.platform.startswith("win") or sys.platform == "darwin"
):
    pytest.skip("example testing on windows and macos", allow_module_level=True)
else:
    print(f"USER : {environ.get('USER', None)}")

pytestmark = pytest.mark.slow

repo = Path(__file__).parent.parent.parent

# get nbsphinx scripts located mainly in the userguide
scripts = list((repo / "docs").glob("**/*.py"))

# remove some scripts
for item in scripts[:]:
    if (
        "checkpoints" in str(item)
        or "make.py" in str(item)
        or "conf.py" in str(item)
        or "apigen.py" in str(item)
        or "gallery" in str(item)
        or ".py" not in str(item)
    ):
        scripts.remove(item)


def nbsphinx_script_run(path):
    pipe = None
    so = None
    serr = None
    try:
        pipe = subprocess.Popen(
            ["python", str(path), "--nodisplay"],
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
        (so, serr) = pipe.communicate()
    except Exception:
        pass

    return pipe.returncode, so, serr


@pytest.mark.parametrize("script", scripts)
def test_nbsphinx_script_(script):
    # some test will failed due to the magic commands or for other known reasons
    # SKIP THEM
    name = script.name
    if name in []:
        print(script, " ---> test skipped - DO IT MANUALLY")
        return

    print("Testing ", script)
    if script.suffix == ".py":
        e, message, err = nbsphinx_script_run(script)
        # this give unicoderror on workflow with window
        print(e, message, err)
        assert not e, message


examples = list((repo / "spectrochempy" / "examples").glob("**/*.py"))
for example in examples[:]:
    if example.stem == "__init__" or "checkpoint" in str(example):
        examples.remove(example)

import matplotlib as mpl

import spectrochempy as scp  # to avoid imporing it in example test (already impported)


@pytest.mark.parametrize("example", examples)
def test_examples(example):

    scp.NO_DISPLAY = True
    scp.NO_DIALOG = True
    mpl.use("agg", force=True)
    from os import environ

    # set test file and folder in environment
    # set a test file in environment
    DATADIR = scp.pathclean(scp.preferences.datadir)
    environ["TEST_FILE"] = str(DATADIR / "irdata" / "nh4y-activation.spg")
    environ["TEST_FOLDER"] = str(DATADIR / "irdata" / "subdir")
    environ["TEST_NMR_FOLDER"] = str(
        DATADIR / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d"
    )

    print("*" * 80 + "\nTesting " + str(example))
    parts = example.parts
    module = ".".join(parts[parts.index("spectrochempy") :])[0:-3]
    import_item(module)
