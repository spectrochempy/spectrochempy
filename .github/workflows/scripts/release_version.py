#!/usr/bin/env python3
# ruff: noqa: S603, S607, T201
"""
Shared PEP 440 release-version semantics for the SpectroChemPy release tooling.

This module is the single authoritative source for deriving every release
metadata from a project version.  It is used by the release workflows and by
``update_version_and_release_notes.py`` so that stable releases and release
candidates (PEP 440 ``rc`` pre-releases) are interpreted consistently.

Accepted input
--------------
* final release such as ``1.0.0`` or ``0.12.8``
* release candidate such as ``1.0.0rc1`` (canonical form: no separator)

Everything else is rejected: ``v``-prefixed strings, incomplete versions
(``1.0``), alphabetic/beta pre-releases, development, post or local releases,
and any non canonical spelling (``1.0.0-rc1``).

Usage
-----
.. code-block:: console

    $ python release_version.py describe 1.0.0rc1
    version=1.0.0rc1
    tag=spectrochempy-v1.0.0rc1
    prerelease=true
    kind=rc
    release_notes=v1.0.0rc1.rst
    docs_version=1.0.0rc1
    next_dev=1.0.0rc2
    pypi_classifier=4 - Beta

The ``latest-tag`` command selects the highest supported core release tag
using final/RC semantics, independently of Git's version-sort configuration.
It can run before dependencies are installed. Tags must have been fetched.

The ``describe`` output is emitted as ``key=value`` lines so that GitHub
Actions steps can forward them to ``$GITHUB_OUTPUT`` or ``$GITHUB_ENV``.
"""

from __future__ import annotations

import re
import subprocess

try:
    from packaging.version import InvalidVersion
    from packaging.version import Version
except ImportError:  # pragma: no cover - exercised in a dependency-free subprocess
    InvalidVersion = ValueError

    class Version:  # type: ignore[no-redef]
        """Minimal release-version parser for bootstrap use before installation."""

        def __init__(self, version: str):
            match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?", version)
            if match is None:
                raise InvalidVersion(version)
            major, minor, micro, rc = match.groups()
            self.major = int(major)
            self.minor = int(minor)
            self.micro = int(micro)
            self.pre = ("rc", int(rc)) if rc is not None else None
            self.is_prerelease = self.pre is not None

        def __str__(self) -> str:
            base = f"{self.major}.{self.minor}.{self.micro}"
            if self.pre is None:
                return base
            return f"{base}{self.pre[0]}{self.pre[1]}"


TAG_PREFIX = "spectrochempy-v"

# Canonical accepted forms only: final X.Y.Z or release candidate X.Y.ZrcN.
RELEASE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:rc\d+)?$")


class ReleaseVersionError(ValueError):
    """Raised when a version string is not a supported release version."""


def parse_release(version: str) -> Version:
    """Parse ``version`` and reject any non release-candidate release form."""
    if not isinstance(version, str) or not RELEASE_VERSION_RE.fullmatch(version):
        raise ReleaseVersionError(
            f"Unsupported release version {version!r}: expected X.Y.Z or X.Y.ZrcN "
            "(canonical form without separator, e.g. 1.0.0rc1)."
        )
    try:
        return Version(version)
    except InvalidVersion as exc:  # pragma: no cover - regex already guards
        raise ReleaseVersionError(f"Unsupported release version {version!r}.") from exc


def canonical_version(version: str) -> str:
    """Return the canonical (normalized) spelling of a supported version."""
    return str(parse_release(version))


def is_prerelease(version: str) -> bool:
    """Return True for release candidates, False for final releases."""
    return parse_release(version).is_prerelease


def is_stable(version: str) -> bool:
    """Return True for final releases, False for release candidates."""
    return not is_prerelease(version)


def kind(version: str) -> str:
    """Return ``'stable'`` or ``'rc'``."""
    return "rc" if is_prerelease(version) else "stable"


def tag_name(version: str) -> str:
    """Return the git tag name associated with ``version``."""
    return TAG_PREFIX + canonical_version(version)


def release_notes_name(version: str) -> str:
    """Return the what's-new file name associated with ``version``."""
    return f"v{canonical_version(version)}.rst"


def docs_version(version: str) -> str:
    """
    Return the documentation version identity for ``version``.

    The full version identity (including any ``rcN`` suffix) is preserved
    for metadata purposes. The choice between the root/``latest`` channel
    and a stable archived directory is made by the documentation
    publication logic (``docs/make.py``), not by this helper: release
    candidates are published on ``latest`` and are never archived.
    """
    return canonical_version(version)


def next_dev_version(version: str) -> str:
    """
    Return the base version of the next development series.

    * final ``X.Y.Z``  -> ``X.Y.(Z+1)``  (same behavior as the historical
      ``IFS=.`` + ``$((CORE_PATCH + 1))`` code kept in the workflows)
    * candidate ``X.Y.ZrcN`` -> ``X.Y.Zrc(N+1)``

    Workflows append ``.dev<commit count>`` to build the fake dev version fed
    to ``setuptools_scm`` (``SETUPTOOLS_SCM_PRETEND_VERSION``).
    """
    parsed = parse_release(version)
    if parsed.is_prerelease:
        pre, num = parsed.pre or ("rc", 1)
        return f"{parsed.major}.{parsed.minor}.{parsed.micro}{pre}{num + 1}"
    return f"{parsed.major}.{parsed.minor}.{parsed.micro + 1}"


def expected_pypi_classifier(version: str) -> str:
    """
    Return the ``Development Status`` classifier expected for ``version``.

    Release candidates keep ``4 - Beta``; only a final ``>= 1.0.0`` release is
    ``5 - Production/Stable``.  Earlier final releases (0.x line) stay Beta.
    """
    parsed = parse_release(version)
    if parsed.major >= 1 and not parsed.is_prerelease:
        return "5 - Production/Stable"
    return "4 - Beta"


def describe(version: str) -> dict[str, str]:
    """Return every derived metadata field for ``version`` (key=value map)."""
    parsed = parse_release(version)
    canonical = str(parsed)
    return {
        "version": canonical,
        "tag": TAG_PREFIX + canonical,
        "prerelease": str(parsed.is_prerelease).lower(),
        "kind": "rc" if parsed.is_prerelease else "stable",
        "release_notes": release_notes_name(canonical),
        "docs_version": docs_version(canonical),
        "next_dev": next_dev_version(canonical),
        "pypi_classifier": expected_pypi_classifier(canonical),
    }


def latest_core_tag() -> str:
    """
    Select the highest supported core release, with finals after their RCs.

    Git's version sort does not implement prerelease ordering. Keep the
    existing repository-wide tag scope, excluding plugin and unsupported tags.
    The numeric key implements PEP 440 ordering for our final/RC-only grammar
    and also works during bootstrap without site packages.
    """
    # Fixed Git command and arguments; no shell or user-provided command.
    tags = subprocess.check_output(
        ["git", "tag", "--list", f"{TAG_PREFIX}*"],
        text=True,
    ).splitlines()
    candidates = []
    for tag in tags:
        try:
            version = parse_release(tag.removeprefix(TAG_PREFIX))
        except ReleaseVersionError:
            continue
        key = (
            version.major,
            version.minor,
            version.micro,
            version.pre is None,
            version.pre[1] if version.pre else 0,
        )
        candidates.append((key, tag))
    if not candidates:
        raise ReleaseVersionError("No supported core release tag found.")
    return max(candidates)[1]


def main(argv: list[str]) -> int:
    if argv == ["latest-tag"]:
        try:
            print(latest_core_tag())
        except (ReleaseVersionError, subprocess.CalledProcessError) as exc:
            print(str(exc), file=__import__("sys").stderr)
            return 1
        return 0

    if len(argv) < 2:
        print(
            "usage: release_version.py latest-tag | {validate|describe|next-dev} <version>",
            file=__import__("sys").stderr,
        )
        return 2
    command, version = argv[0], argv[1]

    if command == "next-dev":
        try:
            print(next_dev_version(version))
        except ReleaseVersionError as exc:
            print(str(exc), file=__import__("sys").stderr)
            return 1
        return 0

    if command in ("validate", "describe"):
        try:
            data = describe(version)
        except ReleaseVersionError as exc:
            print(str(exc), file=__import__("sys").stderr)
            return 1
        if command == "validate":
            print(f"{version}: ok ({data['kind']})")
        else:
            for key, value in data.items():
                print(f"{key}={value}")
        return 0

    print(f"unknown command: {command}", file=__import__("sys").stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
