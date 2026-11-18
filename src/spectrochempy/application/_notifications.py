# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Deliver update notices through native desktop tools without loading a GUI."""

import base64
import json
import shutil
import subprocess
import sys
from os import environ


def _notify_desktop(message):
    if environ.get("SSH_CONNECTION") or environ.get("SSH_TTY"):
        return False

    payload = None
    if sys.platform.startswith("linux"):
        executable = shutil.which("notify-send")
        if not executable or not environ.get("DBUS_SESSION_BUS_ADDRESS"):
            return False
        command = [
            executable,
            "--app-name=SpectroChemPy",
            "--icon=dialog-information",
            "--",
            "SpectroChemPy updates",
            message,
        ]
    elif sys.platform == "darwin":
        executable = shutil.which("osascript")
        if not executable:
            return False
        script = (
            "on run argv\n"
            "display notification (item 1 of argv) "
            'with title "SpectroChemPy updates"\n'
            "end run"
        )
        command = [executable, "-e", script, "--", message]
    elif sys.platform == "win32":
        executable = shutil.which("powershell.exe")
        if not executable:
            return False
        script = """
$ErrorActionPreference = 'Stop'
$data = [Console]::In.ReadToEnd() | ConvertFrom-Json
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml('<toast><visual><binding template="ToastGeneric"><text/><text/></binding></visual></toast>')
$nodes = $xml.GetElementsByTagName('text')
$nodes.Item(0).InnerText = 'SpectroChemPy updates'
$nodes.Item(1).InnerText = $data.message
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show($toast)
"""
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        command = [
            executable,
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ]
        payload = json.dumps({"message": message}, ensure_ascii=True)
    else:
        return False

    try:
        subprocess.run(
            command,
            input=payload,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True
