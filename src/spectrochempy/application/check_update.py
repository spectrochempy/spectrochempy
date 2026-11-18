# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Check Core and installed plugin releases and deliver a consolidated notice."""

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from contextlib import suppress
from datetime import date
from importlib import metadata
from os import environ
from pathlib import Path
from warnings import warn

import requests
from packaging.specifiers import InvalidSpecifier
from packaging.specifiers import SpecifierSet
from packaging.utils import InvalidName
from packaging.utils import InvalidSdistFilename
from packaging.utils import InvalidWheelFilename
from packaging.utils import canonicalize_name
from packaging.utils import parse_sdist_filename
from packaging.utils import parse_wheel_filename
from packaging.version import InvalidVersion
from packaging.version import Version

from spectrochempy.application._notifications import _notify_desktop

__all__ = ["check_update"]

PYPI_INDEX_URL = "https://pypi.org/simple/{package}/"
_PERIODS = {"day": 86400, "week": 7 * 86400, "month": 30 * 86400}


class NeedsUpdateWarning(UserWarning):
    """Warning emitted when desktop notification delivery is unavailable."""


def _select_latest_release(releases, current_version):
    """Select the latest non-yanked release for the current channel and Python."""
    current = Version(str(current_version))
    python = ".".join(map(str, sys.version_info[:3]))
    candidates = []
    for raw_version, files in releases.items():
        try:
            candidate = Version(raw_version)
        except InvalidVersion:
            continue
        if candidate.is_devrelease or (
            candidate.is_prerelease and not current.is_prerelease
        ):
            continue
        usable = []
        for item in files:
            if item.get("yanked", False):
                continue
            try:
                if not SpecifierSet(item.get("requires-python") or "").contains(python):
                    continue
            except InvalidSpecifier:
                continue
            usable.append(item)
        if usable:
            candidates.append((candidate, usable))
    if not candidates:
        return None
    version, files = max(candidates, key=lambda item: item[0])
    dates = [
        date.fromisoformat(item["upload_time_iso_8601"].split("T")[0])
        for item in files
        if item.get("upload_time_iso_8601")
    ]
    return version, min(dates) if dates else None


def _get_pypi_version(current_version, package="spectrochempy", timeout=5):
    """Read the public PyPI Simple JSON API with one bounded request."""
    try:
        name = canonicalize_name(package, validate=True)
        response = requests.get(
            PYPI_INDEX_URL.format(package=name),
            headers={"Accept": "application/vnd.pypi.simple.v1+json"},
            timeout=timeout,
        )
        if response.status_code != 200:
            return None
        releases = {}
        for item in response.json()["files"]:
            filename = item["filename"]
            try:
                if filename.endswith(".whl"):
                    file_package, version, _, _ = parse_wheel_filename(filename)
                else:
                    file_package, version = parse_sdist_filename(filename)
            except (InvalidWheelFilename, InvalidSdistFilename):
                continue
            if file_package != name:
                continue
            releases.setdefault(str(version), []).append(
                {**item, "upload_time_iso_8601": item.get("upload-time")}
            )
        return _select_latest_release(releases, current_version)
    except (
        requests.exceptions.RequestException,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ):
        return None


def _installed_packages(version):
    packages = {"spectrochempy": str(Version(version))}
    for ep in metadata.entry_points(group="spectrochempy.plugins"):
        dist = ep.dist
        if dist is None:
            continue
        try:
            if dist.read_text("direct_url.json"):
                continue
            name = canonicalize_name(dist.metadata["Name"], validate=True)
            installed = Version(dist.version)
        except (OSError, ValueError, TypeError, KeyError, InvalidName):
            continue
        if name != "spectrochempy":
            packages[name] = str(installed)
    return packages


def _cache_path():
    config = Path(environ.get("SCP_CONFIG_HOME", Path.home() / ".spectrochempy/config"))
    environment = f"{sys.prefix}\n{sys.executable}"
    key = hashlib.sha256(environment.encode()).hexdigest()[:16]
    return config / "updates" / f"{key}.json"


def _load_cache(path, now):
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["schema"] != 1:
            return {}
        packages = document["packages"]
        valid = {}
        for name, record in packages.items():
            try:
                Version(record["installed"])
                if record.get("latest") is not None:
                    Version(record["latest"])
                if record["status"] not in ("current", "update", "unavailable"):
                    continue
                if record["status"] != "unavailable":
                    latest = Version(record["latest"])
                    record["status"] = (
                        "update" if latest > Version(record["installed"]) else "current"
                    )
                for key in ("last_attempt", "last_checked", "last_notified"):
                    value = record.get(key, 0)
                    if not isinstance(value, (float, int)) or not math.isfinite(value):
                        raise ValueError
                    if not 0 <= value <= now:
                        raise ValueError
                valid[name] = record
            except (KeyError, TypeError, ValueError):
                continue
        return valid
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def _save_cache(path, packages):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump({"schema": 1, "packages": packages}, stream)
        temporary.replace(path)
    except OSError:
        pass
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _announce(message, mode):
    if mode == "system" and _notify_desktop(message):
        return
    warn(message, category=NeedsUpdateWarning, stacklevel=3)


def _check_packages(version, frequency, mode="system", *, force=False, notify=True):
    if (
        environ.get("DOC_BUILDING") is not None
        or environ.get("SCP_CHECK_UPDATES", "1").lower() in ("0", "false", "off", "no")
        or frequency == "never"
        or mode == "off"
    ):
        return {}
    period = _PERIODS[frequency]
    now = time.time()
    path = _cache_path()
    previous = _load_cache(path, now)
    packages = _installed_packages(version)
    records = {}
    pending = []
    deadline = time.monotonic() + 15
    for name, installed in packages.items():
        record = previous.get(name, {})
        delay = 3600 if record.get("status") == "unavailable" else period
        due = (
            force
            or record.get("installed") != installed
            or now - record.get("last_attempt", 0) >= delay
        )
        if due:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                records[name] = {"installed": installed, "status": "deferred"}
                continue
            result = _get_pypi_version(installed, name, timeout=min(5, remaining))
            record = {
                **record,
                "installed": installed,
                "last_attempt": now,
                "status": "unavailable",
            }
            if result is not None:
                latest, _ = result
                record.update(
                    latest=str(latest),
                    last_checked=now,
                    status="update" if latest > Version(installed) else "current",
                )
        records[name] = record
        if record.get("status") != "update":
            continue
        if (
            record.get("notified_version") != record["latest"]
            or now - record.get("last_notified", 0) >= period
        ):
            pending.append(name)
    if pending and notify:
        lines = ["Updates available:"]
        lines.extend(
            f"{name}: {records[name]['installed']} → {records[name]['latest']}"
            for name in pending
        )
        lines.append("Use your package manager to review and install updates.")
        _announce("\n".join(lines), mode)
        for name in pending:
            records[name].update(
                last_notified=now, notified_version=records[name]["latest"]
            )
    _save_cache(path, records)
    return records


def check_update(version, frequency, notification_mode="system"):
    """Check Core and installed plugins, respecting the configured interval."""
    _check_packages(version, frequency, notification_mode)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notify", action="store_true", help="Also send a desktop notification"
    )
    arguments = parser.parse_args()
    from spectrochempy.application.info import version

    records = _check_packages(version, "week", force=True, notify=arguments.notify)
    for name, record in records.items():
        latest = (
            record.get("latest") if record["status"] in ("current", "update") else "?"
        )
        sys.stdout.write(
            f"{name}: {record['installed']} -> {latest} ({record['status']})\n"
        )


if __name__ == "__main__":
    _main()
