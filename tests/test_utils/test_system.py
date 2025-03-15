# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import os
import platform
import sys
from unittest import mock

import pytest

from spectrochempy.utils.system import (
    get_node,
    get_user,
    get_user_and_node,
    sh,
)


def test_get_user():
    res = get_user()
    assert res is not None
    assert isinstance(res, str)
    assert len(res) > 0


def test_get_node():
    res = get_node()
    assert res is not None
    assert isinstance(res, str)
    assert len(res) > 0
    # Should match platform.node() result
    assert res == platform.node()


def test_get_user_and_node():
    res = get_user_and_node()
    assert res is not None
    assert isinstance(res, str)

    # Should contain both user and node information
    user = get_user()
    node = get_node()
    assert user in res
    assert node in res

    # Common format is "user@node"
    expected = f"{user}@{node}"
    assert res == expected


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="does not run well on windows (seems to be linked to some commit message)",
)
def test_sh():
    res = sh.git("show", "HEAD")
    assert res is not None
    assert isinstance(res, str)

    # Test with a simple command that should work everywhere
    res_echo = sh.echo("test")
    assert res_echo == "test\n" or res_echo == "test"

    # Test with arguments
    res_echo_multi = sh.echo("hello", "world")
    assert "hello" in res_echo_multi
    assert "world" in res_echo_multi


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="does not run well on windows",
)
def test_sh_error_handling():
    # Instead of raising exceptions, the sh command might just capture the error output
    # Let's check for error indicators in the output
    nonexistent_output = sh.ls("/definitely/not/a/real/directory")
    assert (
        "No such file or directory" in nonexistent_output
        or "cannot access" in nonexistent_output
    )

    # For commands that should fail completely, we'll mock subprocess.run to raise an exception
    with mock.patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("Command not found")
        with pytest.raises(Exception):
            sh.definitely_not_a_real_command()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="I do not have the OS to test on windows and it fails on github actions",
)
def test_sh_with_environment():
    # Create a temporary test script that outputs environment variables
    import tempfile
    import os

    # Create a simple script that outputs an environment variable
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        script_path = f.name
        f.write(
            """
import os
import sys
print(os.environ.get('TEST_VAR', 'NOT_FOUND'))
        """
        )

    try:
        # Run the script with sh with our test environment variable
        with mock.patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            output = sh.python(script_path)
            assert "test_value" in output
    finally:
        # Clean up the temporary file
        if os.path.exists(script_path):
            os.unlink(script_path)


@pytest.mark.parametrize(
    "username, hostname, expected",
    [
        ("testuser", "testmachine", "testuser@testmachine"),
        ("user", "host", "user@host"),
        ("", "host", "@host"),
        ("user", "", "user@"),
    ],
)
def test_get_user_and_node_with_mocks(username, hostname, expected):
    with mock.patch("spectrochempy.utils.system.get_user", return_value=username):
        with mock.patch("spectrochempy.utils.system.get_node", return_value=hostname):
            result = get_user_and_node()
            assert result == expected
