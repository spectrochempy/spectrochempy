# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for SpectroChemPy update notifications."""

from __future__ import annotations

import io
import json
from datetime import date
from types import SimpleNamespace

import pytest
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

    latest, release_date = check_update._select_latest_release(releases, "1.0.0rc1")

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


def _index_file(version, **kwargs):
    return {
        "filename": f"spectrochempy-{version}-py3-none-any.whl",
        "upload-time": "2026-09-16T12:00:00Z",
        "yanked": False,
        **kwargs,
    }


def test_get_pypi_version_uses_simple_api_and_current_channel(monkeypatch):
    response = SimpleNamespace(
        status_code=200,
        json=lambda: {"files": [_index_file("1.0"), _index_file("2.0rc1")]},
    )
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(check_update.requests, "get", get)
    assert check_update._get_pypi_version("1.0")[0] == Version("1.0")
    assert calls == [
        (
            "https://pypi.org/simple/spectrochempy/",
            {
                "headers": {"Accept": "application/vnd.pypi.simple.v1+json"},
                "timeout": 5,
            },
        )
    ]


def test_index_ignores_wrong_packages_and_unusable_files(monkeypatch):
    files = [
        _index_file("1.0"),
        _index_file("2.0", **{"requires-python": ">=99"}),
        _index_file("3.0", yanked=True),
        _index_file("4.0", **{"requires-python": "invalid"}),
        {"filename": "different-9.0.tar.gz"},
        {"filename": "README.txt"},
        {"filename": "spectrochempy-1.1.tar.gz"},
    ]
    monkeypatch.setattr(
        check_update.requests,
        "get",
        lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"files": files}),
    )
    assert check_update._get_pypi_version("1.0")[0] == Version("1.1")


@pytest.mark.parametrize("failure", ["timeout", "json", "schema", "status", "empty"])
def test_pypi_failure_is_bounded_and_silent(monkeypatch, failure):
    calls = []

    def response_json():
        if failure == "json":
            raise ValueError("Invalid JSON")
        if failure == "empty":
            return {"files": []}
        return {"files": None}

    def get(url, **kwargs):
        calls.append(kwargs)
        if failure == "timeout":
            raise check_update.requests.exceptions.Timeout
        return SimpleNamespace(
            status_code=503 if failure == "status" else 200, json=response_json
        )

    monkeypatch.setattr(check_update.requests, "get", get)
    assert check_update._get_pypi_version("1.0") is None
    assert len(calls) == 1
    assert calls[0]["timeout"] == 5


@pytest.fixture
def checker(monkeypatch, tmp_path):
    clock = SimpleNamespace(now=1800000000.0)
    cache = tmp_path / "updates.json"
    packages = {"spectrochempy": "1.0", "spectrochempy-nmr": "0.4"}
    calls = []
    notices = []

    def lookup(installed, package, timeout):
        calls.append((package, timeout))
        return Version("2.0"), date.today()

    monkeypatch.delenv("DOC_BUILDING", raising=False)
    monkeypatch.delenv("SCP_CHECK_UPDATES", raising=False)
    monkeypatch.setattr(check_update.time, "time", lambda: clock.now)
    monkeypatch.setattr(check_update, "_cache_path", lambda: cache)
    monkeypatch.setattr(check_update, "_installed_packages", lambda _: packages.copy())
    monkeypatch.setattr(check_update, "_get_pypi_version", lookup)
    monkeypatch.setattr(
        check_update, "_notify_desktop", lambda msg: notices.append(msg) or True
    )
    return SimpleNamespace(
        clock=clock, cache=cache, packages=packages, calls=calls, notices=notices
    )


def test_core_and_plugins_share_one_native_notice_and_cached_check(checker):
    check_update.check_update("1.0", "week")
    assert len(checker.notices) == 1
    assert "spectrochempy: 1.0 → 2.0" in checker.notices[0]
    assert "spectrochempy-nmr: 0.4 → 2.0" in checker.notices[0]
    assert len(checker.calls) == 2
    check_update.check_update("1.0", "week")
    assert len(checker.calls) == 2
    assert len(checker.notices) == 1
    cache = json.loads(checker.cache.read_text())["packages"]["spectrochempy"]
    assert cache["last_checked"] == cache["last_notified"] == checker.clock.now


@pytest.mark.parametrize(
    ("frequency", "days"), [("day", 1), ("week", 7), ("month", 30)]
)
def test_reminders_renew_date_without_repeating_checks(checker, frequency, days):
    check_update.check_update("1.0", frequency)
    checker.clock.now += days * 86400 - 1
    check_update.check_update("1.0", frequency)
    assert len(checker.notices) == 1
    checker.clock.now += 1
    check_update.check_update("1.0", frequency)
    assert len(checker.notices) == 2
    assert len(checker.calls) == 4
    check_update.check_update("1.0", frequency)
    assert len(checker.notices) == 2
    assert len(checker.calls) == 4


def test_no_update_is_cached_too(monkeypatch, checker):
    monkeypatch.setattr(
        check_update,
        "_get_pypi_version",
        lambda installed, *a, **k: (Version(installed), None),
    )
    check_update.check_update("1.0", "week")
    monkeypatch.setattr(
        check_update, "_get_pypi_version", lambda *a, **k: pytest.fail("HTTP")
    )
    check_update.check_update("1.0", "week")
    assert not checker.notices


def test_changed_installation_is_rechecked_and_obsolete_notice_is_removed(checker):
    check_update.check_update("1.0", "week")
    checker.packages["spectrochempy"] = "2.0"
    del checker.packages["spectrochempy-nmr"]
    result = check_update._check_packages("2.0", "week")
    assert result["spectrochempy"]["status"] == "current"
    assert "spectrochempy-nmr" not in result
    assert len(checker.calls) == 3
    assert len(checker.notices) == 1


def test_newer_release_replaces_previous_announcement(monkeypatch, checker):
    check_update.check_update("1.0", "week")
    monkeypatch.setattr(
        check_update, "_get_pypi_version", lambda *a, **k: (Version("3.0"), None)
    )
    check_update._check_packages("1.0", "week", force=True)
    assert len(checker.notices) == 2
    assert "→ 3.0" in checker.notices[-1]
    check_update.check_update("1.0", "week")
    assert len(checker.notices) == 2


@pytest.mark.parametrize("setting", ["frequency", "mode", "docs", "environment"])
def test_disabled_checks_do_not_touch_network_or_cache(monkeypatch, checker, setting):
    frequency, mode = "week", "system"
    if setting == "frequency":
        frequency = "never"
    elif setting == "mode":
        mode = "off"
    elif setting == "docs":
        monkeypatch.setenv("DOC_BUILDING", "1")
    else:
        monkeypatch.setenv("SCP_CHECK_UPDATES", "0")
    check_update.check_update("1.0", frequency, mode)
    assert not checker.calls
    assert not checker.notices
    assert not checker.cache.exists()


def test_one_package_failure_does_not_hide_other_updates(monkeypatch, checker):
    calls = []

    def lookup(installed, package, **kwargs):
        calls.append(package)
        return None if package == "spectrochempy" else (Version("1.0"), None)

    monkeypatch.setattr(check_update, "_get_pypi_version", lookup)
    result = check_update._check_packages("1.0", "week")
    assert result["spectrochempy"]["status"] == "unavailable"
    assert "spectrochempy-nmr" in checker.notices[0]
    assert "spectrochempy:" not in checker.notices[0]
    check_update.check_update("1.0", "week")
    assert len(calls) == 2
    checker.clock.now += 3600
    check_update.check_update("1.0", "week")
    assert calls == ["spectrochempy", "spectrochempy-nmr", "spectrochempy"]


def test_budget_defers_remaining_requests(monkeypatch, checker):
    times = iter([0, 0, 16])
    monkeypatch.setattr(check_update.time, "monotonic", lambda: next(times))
    result = check_update._check_packages("1.0", "week")
    assert len(checker.calls) == 1
    assert result["spectrochempy-nmr"]["status"] == "deferred"


@pytest.mark.parametrize("cached", ["invalid", "[]", '{"schema":1,"packages":[]}'])
def test_corrupt_cache_is_repaired(checker, cached):
    checker.cache.write_text(cached)
    check_update.check_update("1.0", "week")
    assert len(checker.notices) == 1
    assert json.loads(checker.cache.read_text())["schema"] == 1


def test_invalid_cache_record_does_not_discard_valid_packages(checker):
    check_update.check_update("1.0", "week")
    data = json.loads(checker.cache.read_text())
    data["packages"]["spectrochempy"]["last_attempt"] = "invalid"
    checker.cache.write_text(json.dumps(data))
    check_update.check_update("1.0", "week")
    assert len(checker.calls) == 3


def test_unwritable_cache_does_not_prevent_notification(monkeypatch, checker):
    def fail(*args, **kwargs):
        raise PermissionError("Read-only config")

    monkeypatch.setattr(check_update.Path, "mkdir", fail)
    check_update.check_update("1.0", "week")
    assert len(checker.notices) == 1


def test_failed_atomic_replace_removes_temporary_file(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise PermissionError("Cannot replace")

    monkeypatch.setattr(check_update.Path, "replace", fail)
    check_update._save_cache(tmp_path / "state.json", {})
    assert list(tmp_path.iterdir()) == []


def test_manual_check_forces_http_without_consuming_reminder(checker):
    check_update._check_packages("1.0", "week", force=True, notify=False)
    assert not checker.notices
    check_update.check_update("1.0", "week")
    assert len(checker.notices) == 1
    assert len(checker.calls) == 2


def test_system_failure_uses_text_warning(monkeypatch):
    monkeypatch.setattr(check_update, "_notify_desktop", lambda _: False)
    with pytest.warns(check_update.NeedsUpdateWarning, match="Available update"):
        check_update._announce("Available update", "system")


def test_text_mode_never_calls_desktop(monkeypatch):
    monkeypatch.setattr(
        check_update, "_notify_desktop", lambda _: pytest.fail("desktop")
    )
    with pytest.warns(check_update.NeedsUpdateWarning, match="Available update"):
        check_update._announce("Available update", "terminal")


def test_cache_is_scoped_to_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SCP_CONFIG_HOME", str(tmp_path))
    first = check_update._cache_path()
    monkeypatch.setattr(check_update.sys, "prefix", str(tmp_path / "other-env"))
    assert check_update._cache_path() != first
    assert first.parent == tmp_path / "updates"


def test_plugin_discovery_uses_distribution_metadata_without_loading(monkeypatch):
    class Distribution:
        def __init__(self, name, version="0.4", direct_url=None):
            self.metadata = {"Name": name}
            self.version = version
            self.direct_url = direct_url

        def read_text(self, filename):
            return self.direct_url

    dist = Distribution("SpectroChemPy_NMR")
    entries = [
        SimpleNamespace(dist=dist),
        SimpleNamespace(dist=dist),
        SimpleNamespace(
            dist=Distribution("local-plugin", direct_url='{"dir_info":{}}')
        ),
        SimpleNamespace(dist=Distribution("invalid name")),
        SimpleNamespace(dist=Distribution("bad-version", "invalid")),
        SimpleNamespace(dist=None),
    ]
    monkeypatch.setattr(check_update.metadata, "entry_points", lambda **k: entries)
    assert check_update._installed_packages("1.0") == {
        "spectrochempy": "1.0",
        "spectrochempy-nmr": "0.4",
    }


def test_cli_prints_status_without_sending_a_notification(monkeypatch, checker, capsys):
    monkeypatch.setattr(check_update.sys, "argv", ["check_update"])
    check_update._main()
    output = capsys.readouterr().out
    assert "spectrochempy: 1.0 -> 2.0 (update)" in output
    assert "spectrochempy-nmr" in output
    assert not checker.notices


@pytest.mark.parametrize(("frequency", "mode"), [("never", "system"), ("week", "off")])
def test_disabled_preferences_do_not_start_background_thread(
    monkeypatch, frequency, mode
):
    from spectrochempy.application import application

    preferences = SimpleNamespace(
        check_update_frequency=frequency, update_notifications=mode
    )
    monkeypatch.setattr(
        application, "app", SimpleNamespace(general_preferences=preferences)
    )
    monkeypatch.setattr(
        application.threading,
        "Thread",
        lambda **kwargs: pytest.fail("background thread"),
    )
    application._start_update_check()


def test_background_thread_receives_notification_preferences(monkeypatch):
    from spectrochempy.application import application

    preferences = SimpleNamespace(
        check_update_frequency="month", update_notifications="terminal"
    )
    monkeypatch.setattr(
        application, "app", SimpleNamespace(general_preferences=preferences)
    )
    calls = []
    started = []

    def thread(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(start=lambda: started.append(True))

    monkeypatch.setattr(application.threading, "Thread", thread)
    application._start_update_check()
    assert calls[0]["args"][1:] == ("month", "terminal")
    assert calls[0]["target"] is check_update.check_update
    assert calls[0]["daemon"] is True
    assert started == [True]


def test_notification_preferences_are_configurable():
    from spectrochempy.application._preferences.general_preferences import (
        GeneralPreferences,
    )

    traits = GeneralPreferences.class_traits()
    assert traits["update_notifications"].default_value == "system"
    assert traits["update_notifications"].metadata["config"] is True
    assert "off" in traits["update_notifications"].values
    assert "never" in traits["check_update_frequency"].values


def test_cli_supports_redirected_windows_legacy_encoding(monkeypatch, checker):
    monkeypatch.setattr(check_update.sys, "argv", ["check_update"])
    output = io.BytesIO()
    stream = io.TextIOWrapper(output, encoding="cp1252")
    with monkeypatch.context() as context:
        context.setattr(check_update.sys, "stdout", stream)
        check_update._main()
        stream.flush()
    assert b"spectrochempy: 1.0 -> 2.0 (update)" in output.getvalue()
    stream.detach()
