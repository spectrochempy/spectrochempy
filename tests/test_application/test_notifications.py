# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Native notification routing, argument safety, and failure handling."""

import base64
import json
import subprocess

import pytest

from spectrochempy.application import _notifications


@pytest.fixture
def desktop(monkeypatch):
    for key in ("SSH_CONNECTION", "SSH_TTY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/test/bus")
    monkeypatch.setattr(_notifications.shutil, "which", lambda name: f"/tools/{name}")
    calls = []
    monkeypatch.setattr(
        _notifications.subprocess,
        "run",
        lambda args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_native_delivery_keeps_message_out_of_executable_code(
    monkeypatch, desktop, platform
):
    monkeypatch.setattr(_notifications.sys, "platform", platform)
    message = 'Version "2.0"\n$(touch /tmp/injection) <&> `quote` é'
    assert _notifications._notify_desktop(message)
    command, options = desktop[0]
    assert options["timeout"] == 5
    assert options["check"] is True
    assert not options.get("shell")
    if platform == "win32":
        script = base64.b64decode(command[-1]).decode("utf-16-le")
        assert message not in script
        assert json.loads(options["input"])["message"] == message
        assert "InnerText" in script
    else:
        assert command[-1] == message
        if platform == "darwin":
            assert message not in command[2]
        else:
            assert "--" in command


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32", "unsupported"])
def test_missing_system_tool_is_a_quiet_failure(monkeypatch, desktop, platform):
    monkeypatch.setattr(_notifications.sys, "platform", platform)
    monkeypatch.setattr(_notifications.shutil, "which", lambda name: None)
    assert not _notifications._notify_desktop("Update")
    assert not desktop


@pytest.mark.parametrize("remote", ["SSH_CONNECTION", "SSH_TTY"])
def test_ssh_does_not_notify_server_desktop(monkeypatch, desktop, remote):
    monkeypatch.setenv(remote, "remote session")
    assert not _notifications._notify_desktop("Update")
    assert not desktop


def test_linux_without_a_session_bus_uses_fallback(monkeypatch, desktop):
    monkeypatch.setattr(_notifications.sys, "platform", "linux")
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS")
    assert not _notifications._notify_desktop("Update")
    assert not desktop


@pytest.mark.parametrize(
    "error",
    [
        OSError("Unavailable"),
        subprocess.TimeoutExpired("notify", 5),
        subprocess.CalledProcessError(1, "notify"),
    ],
)
def test_system_errors_do_not_escape(monkeypatch, desktop, error):
    monkeypatch.setattr(_notifications.sys, "platform", "linux")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(_notifications.subprocess, "run", fail)
    assert not _notifications._notify_desktop("Update")
