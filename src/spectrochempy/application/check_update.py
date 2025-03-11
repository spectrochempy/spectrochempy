# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Check SpectroChemPy updates."""

import json
import time
from contextlib import suppress
from datetime import date
from datetime import timedelta
from os import environ
from pathlib import Path
from warnings import warn

import requests
from IPython import get_ipython
from IPython.display import Javascript
from IPython.display import display
from packaging.version import Version
from packaging.version import parse as parse_version

__all__ = ["check_update"]


# --------------------------------------------------------------------------------------
# Exception for this module
# --------------------------------------------------------------------------------------
class NeedsUpdateWarning(UserWarning):
    """Warning raised when Spectrochempy needs update."""


# --------------------------------------------------------------------------------------
# Pypi version checking
# --------------------------------------------------------------------------------------
def _get_pypi_version():
    """Get the last released pypi version."""
    url = "https://pypi.python.org/pypi/spectrochempy/json"

    connection_timeout = 120  # secondss
    start_time = time.time()
    while True:
        try:
            response = requests.get(url, timeout=connection_timeout)
            if response.status_code != 200:  # pragma: no cover
                return None
            break  # exit the while loop in case of success

        except (
            ConnectionError,
            requests.exceptions.RequestException,
        ):  # pragma: no cover
            if time.time() > start_time + connection_timeout:
                # 'Unable to get updates after {} seconds of ConnectionErrors'
                return None
            time.sleep(1)  # attempting once every second

    releases = json.loads(response.text)["releases"]
    versions = sorted(releases, key=parse_version)
    last_version = versions[-1]
    release_date = date.fromisoformat(
        releases[last_version][0]["upload_time_iso_8601"].split("T")[0],
    )
    # import datetime
    # # for testing message
    # return Version("2.0.0"), datetime.date(2021, 10, 1)
    return Version(last_version), release_date


# --------------------------------------------------------------------------------------
# Display update message
# --------------------------------------------------------------------------------------
def _display_needs_update_message(frequency):
    fil = Path.home() / ".scpy_needs_update"
    message = None
    if fil.exists():
        with suppress(Exception) as e:
            try:
                msg = fil.read_text()
                check_date, status, message = msg.split("%%")
                if status == "NOT_YET_DISPLAYED":  # pragma: no cover
                    fil.write_text(
                        f"{date.isoformat(date.today())}%%DISPLAYED%%{message}",
                    )
                else:
                    # don't notice again if the message was already displayed
                    # in the n last days
                    n_days = {"day": 1, "week": 7, "month": 30}[frequency]
                    last_view_delay = date.today() - date.fromisoformat(check_date)
                    if last_view_delay < timedelta(days=n_days):
                        message = None
            except Exception as e:
                warn(
                    f"An error occurred while reading the update message: {e}",
                    category=NeedsUpdateWarning,
                    stacklevel=2,
                )

    if message:  # pragma: no cover
        # Clean message for JavaScript
        clean_message = message.replace("\n", "<br>").replace("'", "\\'")
        if (
            get_ipython().__class__.__name__ == "ZMQInteractiveShell"
        ):  # Jupyter notebook or qtconsole
            js_code = f"""
                (() => {{
                    // Remove existing notification if any
                    const existing = document.getElementById('scp-update-notification');
                    if (existing) existing.remove();
                    // Create new notification
                    const notification = document.createElement('div');
                    notification.id = 'scp-update-notification';
                    notification.style.cssText = `
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        background: white;
                        padding: 15px;
                        border-radius: 8px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        z-index: 9999;
                        max-width: 400px;
                        font-family: -apple-system, system-ui, sans-serif;
                    `;
                    notification.innerHTML = `
                        <div style="margin-bottom: 10px;">
                            <strong style="color: #1a73e8;">SpectroChemPy Update</strong>
                        </div>
                        <div style="margin-bottom: 15px; color: #333;">
                            {clean_message}
                        </div>
                        <button onclick="this.parentElement.remove()"
                                style="
                                    background: #1a73e8;
                                    color: white;
                                    border: none;
                                    padding: 6px 12px;
                                    border-radius: 4px;
                                    cursor: pointer;
                                    float: right;
                                "
                        >Dismiss</button>
                    `;
                    document.body.appendChild(notification);
                    // Auto-hide after 10 seconds
                    setTimeout(() => {{
                        if (notification.parentElement) {{
                            notification.remove();
                        }}
                    }}, 10000);
                }})();
            """
            try:
                display(Javascript(js_code))
            except Exception as e:
                warn(f"Failed to display notification: {e}", stacklevel=2)
        else:
            warn(message, category=NeedsUpdateWarning, stacklevel=2)


# ======================================================================================
# Update checking
# ======================================================================================
def check_update(version, frequency):
    old = Version(version)
    res = _get_pypi_version()
    if res is not None:
        version, _ = res
    else:  # pragma: no cover
        # probably a ConnectionError
        return

    new_release = None
    if version > old:  # pragma: no cover
        new_version = version.public
        if not version.is_devrelease:
            new_release = new_version

    fil = Path.home() / ".scpy_needs_update"
    if new_release and environ.get("DOC_BUILDING") is None:  # pragma: no cover
        if not fil.exists():  # This new version is checked for the first time
            # write the information: date of writing, status, message
            fil.write_text(
                f"{date.isoformat(date.today())}%%NOT_YET_DISPLAYED%%"
                f"SpectroChemPy v{new_release} is available.\n"
                f"{' ' * 11}Please consider updating, using pip or conda, for bug fixes "
                f"and new features!\n",
            )
    elif fil.exists():
        fil.unlink()

    # finally display the message if necessary every day, week or month
    _display_needs_update_message(frequency)


# ======================================================================================
if __name__ == "__main__":
    """ """
