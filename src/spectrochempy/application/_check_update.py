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
from packaging.version import Version
from packaging.version import parse as parse_version


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

    connection_timeout = 30  # secondss
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
    return Version(last_version), release_date


# --------------------------------------------------------------------------------------
# Display update message
# --------------------------------------------------------------------------------------
def _display_needs_update_message():
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
                    # in the 3 last days
                    last_view_delay = date.today() - date.fromisoformat(check_date)
                    if last_view_delay < timedelta(days=3):
                        message = None
            except Exception as e:
                warn(
                    f"An error occurred while reading the update message: {e}",
                    category=NeedsUpdateWarning,
                    stacklevel=2,
                )

    if message:  # pragma: no cover
        # TODO : find how to make a non blocking dialog (GUI?)
        warn(message, category=NeedsUpdateWarning, stacklevel=2)


# ======================================================================================
# Update checking
# ======================================================================================
def check_update(version):
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

    # finally display the message if necessary
    _display_needs_update_message()


# ======================================================================================
if __name__ == "__main__":
    """ """
