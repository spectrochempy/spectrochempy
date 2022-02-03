# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa


# --------------------------------------------------------------------------------------
# Testing examples and notebooks (Py version) in docs
# --------------------------------------------------------------------------------------

import sys
import pytest

# Uncomment to avoid these long tests which are also done in docs
# pytestmark = pytest.mark.skip(reason="check when building docs in CI")
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or sys.version_info < (3, 9),
    reason="1) Does not work on windows, 2) Execute this long run only one time on github workflow",
)

from pathlib import Path

repo = Path(__file__).parent.parent.parent

scripts = list((repo / "docs" / "gettingstarted" / "examples").glob("**/*.py"))
for item in scripts[:]:
    if "checkpoints" in str(item):
        scripts.remove(item)


# ......................................................................................
def example_run(path):
    import subprocess

    pipe = None
    so = None
    serr = None
    try:
        pipe = subprocess.Popen(
            ["python", str(path), "--nodisplay"], stdout=subprocess.PIPE
        )
        (so, serr) = pipe.communicate()
    except Exception:
        pass

    return pipe.returncode, so, serr


# ......................................................................................
@pytest.mark.parametrize("example", scripts)
def test_example(example):
    # some test will failed due to the magic commands or for other known reasons
    # SKIP THEM
    name = example.name
    if name in []:
        print(example, " ---> test skipped - DO IT MANUALLY")
        return

    if example.suffix == ".py":
        e, message, err = example_run(example)
        print(e, message.decode("utf8"), err)
        assert not e, message.decode("utf8")
