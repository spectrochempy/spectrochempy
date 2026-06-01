# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Testing examples and notebooks (Py version) in docs.
This test is skipped by default as it's too slow and redundant with docs building process.
To run it explicitly, use: pytest tests/test_docs/test_py_in_docs.py --override-skip.
"""

import ast
import subprocess
import sys
from importlib.util import find_spec
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

NETWORK_URL_PATTERNS = [
    "eigenvector.com",
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


def _plugin_available(name):
    return find_spec(name) is not None


def _plugin_required_by(path):
    """Determine which (if any) optional plugin an example or script needs."""
    text = path.read_text(encoding="utf8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "OPTIONAL_PLUGIN"
                for target in node.targets
            ):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                return node.value.value

    markers = {
        "spectrochempy_nmr": "read_topspin",
        "spectrochempy_iris": "spectrochempy_iris",
        "spectrochempy_cantera": "spectrochempy_cantera",
    }
    for plugin_name, marker in markers.items():
        if marker in text:
            return plugin_name
    return None


def _requires_external_network(path):
    """Check if a script requires external network access to run."""
    text = path.read_text(encoding="utf8")
    for pattern in NETWORK_URL_PATTERNS:
        if pattern in text:
            return pattern
    return None


NETWORK_ERROR_PATTERNS = [
    "URLError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "TimeoutError",
    "timed out",
    "Temporary failure in name resolution",
    "getaddrinfo failed",
    "Name or service not known",
    "No route to host",
    "Network is unreachable",
    "Cannot connect to host",
    "Remote end closed connection",
    "EOF occurred in violation of protocol",
    "SSL: CERTIFICATE_VERIFY_FAILED",
    "Name or service not known",
    "Failed to establish a new connection",
]


def _looks_like_external_network_failure(error_text, network_marker):
    """
    Determine if a failure is likely due to external network unavailability.

    Returns ``True`` only when *error_text* contains patterns consistent with
    an external network failure **and** the script is known to require external
    network access (*network_marker* is not ``None``).

    Parameters
    ----------
    error_text : str or None
        The error output (stderr or exception string).
    network_marker : str or None
        The network URL pattern detected in the script, or ``None``.

    Returns
    -------
    bool
    """
    if not network_marker:
        return False
    error_text = error_text or ""

    # FileNotFoundError is ambiguous (local file vs network URL)
    # Only treat as network failure if the path looks like a URL
    if "FileNotFoundError" in error_text:
        if "http://" in error_text.lower() or "https://" in error_text.lower():
            return True
        if network_marker.lower() in error_text.lower():
            return True
        return False

    for pattern in NETWORK_ERROR_PATTERNS:
        if pattern.lower() in error_text.lower():
            return True

    # As a last保守 check: if the network marker domain appears in the error
    # text alongside a failure context, treat as network failure
    if network_marker.lower() in error_text.lower():
        return True

    return False


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
            stderr=subprocess.PIPE,
            encoding="utf8",
            env=env,
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
    required = _plugin_required_by(script)
    if required and not _plugin_available(required):
        pytest.skip(f"requires the optional {required} plugin")

    network_marker = _requires_external_network(script)

    e, message, err = nbsphinx_script_run(script)
    # this give unicoderror on workflow with window
    if e:
        error_text = (err or "") + "\n" + (message or "")
        if _looks_like_external_network_failure(error_text, network_marker):
            pytest.skip(f"external network ({network_marker}) not reachable")
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
    required = _plugin_required_by(example)
    if required and not _plugin_available(required):
        pytest.skip(f"requires the optional {required} plugin")

    network_marker = _requires_external_network(example)

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
        error_text = f"{type(e).__name__}: {e}"
        if _looks_like_external_network_failure(error_text, network_marker):
            pytest.skip(f"external network ({network_marker}) not reachable")
        error_msg = f"Error in example: {example}\n"
        error_msg += f"Module: {module}\n"
        error_msg += f"Exception: {type(e).__name__}: {e}\n"
        import traceback

        error_msg += f"Traceback:\n{traceback.format_exc()}\n"
        pytest.fail(error_msg)


# --------------------------------------------------------------------------------------
# Unit tests for helper functions (not marked slow — run directly)
# --------------------------------------------------------------------------------------


def test_unit__requires_external_network(tmp_path):
    """_requires_external_network detects eigenvector.com in script text."""
    p = tmp_path / "script.py"
    p.write_text('scp.read("http://www.eigenvector.com/data/Corn/corn.mat")')
    assert _requires_external_network(p) == "eigenvector.com"

    p2 = tmp_path / "no_network.py"
    p2.write_text("x = 1 + 1")
    assert _requires_external_network(p2) is None


def test_unit__looks_like_external_network_failure_FileNotFoundError_with_url():
    """FileNotFoundError containing http/https URL is a network failure."""
    assert _looks_like_external_network_failure(
        'FileNotFoundError: [Errno 2] No such file or directory: "http://www.eigenvector.com/data/Corn/corn.mat"',
        "eigenvector.com",
    )


def test_unit__looks_like_external_network_failure_FileNotFoundError_no_url():
    """FileNotFoundError without URL is NOT a network failure."""
    assert not _looks_like_external_network_failure(
        'FileNotFoundError: [Errno 2] No such file or directory: "local_data.mat"',
        "eigenvector.com",
    )


def test_unit__looks_like_external_network_failure_no_marker():
    """Without a network marker, no error is considered network failure."""
    assert not _looks_like_external_network_failure(
        "FileNotFoundError: http://example.com/data.mat",
        None,
    )


def test_unit__looks_like_external_network_failure_connection_errors():
    """Connection/Timeout/DNS errors are network failures (with marker)."""
    for err_text in [
        "URLError: <urlopen error [Errno -2] Name or service not known>",
        "TimeoutError: timed out",
        "ConnectionRefusedError: [Errno 111] Connection refused",
        "Temporary failure in name resolution",
        "getaddrinfo failed",
        "No route to host",
        "Network is unreachable",
        "Failed to establish a new connection",
    ]:
        assert _looks_like_external_network_failure(
            err_text, "eigenvector.com"
        ), f"expected network failure for: {err_text}"


def test_unit__looks_like_external_network_failure_non_network():
    """AssertionError, ValueError, ShapeError are NOT network failures."""
    for err_text in [
        "AssertionError: assert 1 == 2",
        "ValueError: operands could not be broadcast together with shapes (3,) (4,)",
        "TypeError: 'NoneType' object is not iterable",
    ]:
        assert not _looks_like_external_network_failure(
            err_text, "eigenvector.com"
        ), f"expected NOT network failure for: {err_text}"


def test_unit__looks_like_external_network_failure_none_text():
    """None error text does not crash and returns False."""
    assert not _looks_like_external_network_failure(None, "eigenvector.com")
    assert not _looks_like_external_network_failure(None, None)
