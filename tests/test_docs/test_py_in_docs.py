# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Testing examples and notebooks (Py version) in docs.
This test is skipped by default as it's too slow and redundant with docs building process.
To run it explicitly, use: pytest tests/test_docs/test_py_in_docs.py --override-skip.
"""

import subprocess
import sys
from os import environ
from pathlib import Path

import matplotlib as mpl
import pytest
from traitlets import import_item

import spectrochempy as scp

pytestmark = [
    pytest.mark.slow,
    #    pytest.mark.skip(reason="Too slow and redundant with docs building process"),
]

repo = Path(__file__).parent.parent.parent

# Get example files list at module level
example_files = list((repo / "src" / "spectrochempy" / "examples").glob("**/*.py"))
example_files = [
    example
    for example in example_files
    if example.stem != "__init__" and "checkpoint" not in str(example)
]

# Get nbsphinx scripts
scripts = list((repo / "docs").glob("**/*.py"))
scripts = [
    script
    for script in scripts
    if not any(
        x in str(script)
        for x in ["checkpoint", "make.py", "conf.py", "apigen.py", "gallery"]
    )
]


def nbsphinx_script_run(path):
    import matplotlib

    matplotlib.use("Agg")  # Force non-interactive backend before any other imports

    pipe = None
    so = None
    serr = None
    try:
        import os

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Set backend through environment as well
        pipe = subprocess.Popen(  # noqa: S603
            [sys.executable, str(path), "--nodisplay"],
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
        (so, serr) = pipe.communicate()
    except Exception as e:
        print(f"An error occurred while running the script {path}: {e}")

    return pipe.returncode, so, serr


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="does not run well on windows - to be investigated",
)
@pytest.mark.parametrize(
    "script",
    sorted(scripts, key=lambda script: script.stem),
    ids=lambda x: x.stem,
)
def test_nbsphinx_script_(script):
    # some test will failed due to the magic commands or for other known reasons
    # SKIP THEM
    name = script.name
    if name in []:
        return

    e, message, err = nbsphinx_script_run(script)
    # this give unicoderror on workflow with window
    if e:
        error_msg = f"Error in script: {script}\n"
        error_msg += f"Return code: {e}\n"
        error_msg += f"Standard output:\n{message}\n"
        if err:
            error_msg += f"Error output:\n{err}\n"
        pytest.fail(error_msg)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="does not run well on windows - to be investigated",
)
@pytest.mark.parametrize(
    "example", sorted(example_files, key=lambda x: x.stem), ids=lambda x: x.stem
)
def test_examples(example):
    """Test example files."""
    scp.NO_DISPLAY = True
    mpl.use("agg", force=True)

    # set test file and folder in environment
    DATADIR = scp.pathclean(scp.preferences.datadir)
    environ["TEST_FILE"] = str(DATADIR / "irdata" / "nh4y-activation.spg")
    environ["TEST_FOLDER"] = str(DATADIR / "irdata" / "subdir")
    environ["TEST_NMR_FOLDER"] = str(
        DATADIR / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d"
    )

    parts = list(example.parts)
    parts[-1] = parts[-1][0:-3]
    sel = parts[-parts[::-1].index("spectrochempy") - 1 :]
    module = ".".join(sel)

    try:
        import_item(module)
    except Exception as e:
        error_msg = f"Error in example: {example}\n"
        error_msg += f"Module: {module}\n"
        error_msg += f"Exception: {type(e).__name__}: {e}\n"
        import traceback

        error_msg += f"Traceback:\n{traceback.format_exc()}\n"
        pytest.fail(error_msg)
