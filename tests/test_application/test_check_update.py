# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for SpectroChemPy update notifications."""

from __future__ import annotations

from datetime import date

from packaging.version import Version

from spectrochempy.application import check_update


def _release(upload_date="2026-09-16", *, yanked=False):
    return [
        {
            "upload_time_iso_8601": f"{upload_date}T12:00:00Z",
            "yanked": yanked,
        }
    ]


def test_stable_installation_ignores_prereleases():
    releases = {
        "0.12.8": _release("2026-09-09"),
        "1.0.0rc1": _release(),
    }

    latest, _ = check_update._select_latest_release(releases, "0.12.8")

    assert latest == Version("0.12.8")


def test_release_candidate_tracks_new_candidates_and_final_release():
    releases = {
        "0.12.8": _release("2026-09-09"),
        "1.0.0rc1": _release(),
        "1.0.0rc2": _release("2026-09-20"),
        "1.0.0": _release("2026-09-25"),
    }

    latest, release_date = check_update._select_latest_release(
        releases, "1.0.0rc1"
    )

    assert latest == Version("1.0.0")
    assert release_date == date(2026, 9, 25)


def test_unusable_releases_are_ignored():
    releases = {
        "1.0.0rc1": _release(),
        "1.0.0rc2.dev1": _release("2026-09-18"),
        "1.0.0rc2": _release("2026-09-20", yanked=True),
        "invalid": _release("2026-09-21"),
        "1.0.0": [],
    }

    latest, _ = check_update._select_latest_release(releases, "1.0.0rc1")

    assert latest == Version("1.0.0rc1")


def test_get_pypi_version_uses_current_release_channel(monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "releases": {
                    "0.12.8": _release("2026-09-09"),
                    "1.0.0rc1": _release(),
                }
            }

    calls = []

    def get(url, timeout):
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(check_update.requests, "get", get)

    latest, _ = check_update._get_pypi_version("0.12.8")

    assert latest == Version("0.12.8")
    assert calls == [(check_update.PYPI_JSON_URL, 120)]
