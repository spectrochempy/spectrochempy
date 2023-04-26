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
import sys

import pytest

if sys.platform.startswith("win") or sys.platform == "darwin":
    pytest.skip("example testing on windows and macos", allow_module_level=True)

pytestmark = pytest.mark.slow

from pathlib import Path

repo = Path(__file__).parent.parent.parent

scripts = list((repo / "docs").glob("**/*.py"))

for item in scripts[:]:
    if (
        "checkpoints" in str(item)
        or "make.py" in str(item)
        or "conf.py" in str(item)
        or "apigen.py" in str(item)
        or "gallery" in str(item)
    ):
        scripts.remove(item)


def example_run(path):
    import subprocess

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


@pytest.mark.parametrize("example", scripts)
def test_example(example):
    # some test will failed due to the magic commands or for other known reasons
    # SKIP THEM
    name = example.name
    if name in []:
        print(example, " ---> test skipped - DO IT MANUALLY")
        return

    print("Testing ", example)
    if example.suffix == ".py":
        e, message, err = example_run(example)
        # this give unicoderror on workflow with window
        print(e, message, err)
        assert not e, message
