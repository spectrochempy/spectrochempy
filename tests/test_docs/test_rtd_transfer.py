"""Check RTD transfers without sharing credentials with pull request builds."""

import importlib.util
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from docs import rtd_transfer as reader

PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/transfer_rtd_results.py"
)
SPEC = importlib.util.spec_from_file_location("transfer_rtd_results", PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
SHA = "a" * 40
PROJECT = "spectrochempy-artifacts-test"
URL = "https://results.blob.core.windows.net/artifact/bundle.zip?sig=temporary"


@pytest.mark.parametrize(
    ("enabled", "version", "kind", "previews", "expected"),
    [
        ("", "latest", "branch", "", False),
        ("1", "latest", "branch", "", True),
        ("1", "1927", "external", "", False),
        ("1", "1927", "external", "1", True),
        ("1", "v0.9", "tag", "1", False),
        ("1", "stable", "tag", "1", False),
    ],
)
def test_new_mode_leaves_old_releases_native(
    monkeypatch, enabled, version, kind, previews, expected
):
    monkeypatch.delenv("SCPY_RTD_DOCS_ARTIFACTS", raising=False)
    monkeypatch.setenv("SCPY_RTD_GITHUB_RESULTS", enabled)
    monkeypatch.setenv("SCPY_RTD_GITHUB_RESULTS_PR", previews)
    monkeypatch.setenv("READTHEDOCS_VERSION", version)
    monkeypatch.setenv("READTHEDOCS_VERSION_TYPE", kind)
    assert reader.selected() is expected


@pytest.mark.parametrize(
    "url",
    [
        "http://results.blob.core.windows.net/x",
        "https://github.com/x",
        "https://results.blob.core.windows.net.attacker.example/x",
        "https://user:password@results.blob.core.windows.net/x",
        "https://results.blob.core.windows.net:8080/x",
    ],
)
def test_storage_url_rejects_unexpected_hosts_and_credentials(url):
    with pytest.raises(ValueError):
        reader.storage_url(url)


def source_run(event="push"):
    return {
        "id": 12,
        "head_sha": SHA,
        "head_branch": "master",
        "head_repository": {"full_name": "contributor/spectrochempy"},
        "event": event,
        "repository": {"full_name": reader.REPOSITORY},
        "path": reader.WORKFLOW,
        "conclusion": "success",
        "html_url": "https://github.com/spectrochempy/spectrochempy/actions/runs/12",
    }


def public_check(*, expires=1100, app=reader.GITHUB_ACTIONS_APP):
    return {
        "id": 5,
        "name": reader.check_name(PROJECT),
        "head_sha": SHA,
        "conclusion": "success",
        "app": {"id": app},
        "output": {
            "text": json.dumps(
                {
                    "commit": SHA,
                    "project": PROJECT,
                    "run_id": 12,
                    "expires_at": expires,
                    "artifacts": {
                        kind: {"id": i, "digest": "sha256:" + "f" * 64, "url": URL}
                        for i, kind in enumerate(reader.KINDS, 1)
                    },
                }
            )
        },
    }


def responses(monkeypatch, *, check=None, run=None, altered=False):
    monkeypatch.setenv("READTHEDOCS_PROJECT", PROJECT)
    monkeypatch.setattr(reader.time, "time", lambda: 1000)
    items = [
        {
            "id": i,
            "name": f"docs-{kind}-{SHA}",
            "digest": "sha256:" + ("e" if altered else "f") * 64,
            "expired": False,
        }
        for i, kind in enumerate(reader.KINDS, 1)
    ]

    def get(path):
        if "check-runs?" in path:
            return {"check_runs": [check or public_check()]}
        if "artifacts?" in path:
            return {"artifacts": items}
        return run or source_run()

    monkeypatch.setattr(reader, "_json", get)


def test_public_inputs_match_run_commit_and_artifact_digests(monkeypatch):
    responses(monkeypatch)
    run, artifacts = reader.verified_inputs(SHA)
    assert run == 12
    assert set(artifacts) == set(reader.KINDS)
    assert all(item["transfer_url"] == URL for item in artifacts.values())


@pytest.mark.parametrize("check", [public_check(expires=990), public_check(app=1)])
def test_expired_or_foreign_checks_are_not_used(monkeypatch, check):
    responses(monkeypatch, check=check)
    assert reader.verified_inputs(SHA) is None


def test_tampered_archive_identity_is_rejected(monkeypatch):
    responses(monkeypatch, altered=True)
    with pytest.raises(RuntimeError, match="checksum"):
        reader.verified_inputs(SHA)


@pytest.mark.parametrize(
    "field,value",
    [("head_sha", "b" * 40), ("path", "other.yml"), ("conclusion", "failure")],
)
def test_unexpected_source_run_is_rejected(monkeypatch, field, value):
    run = source_run()
    run[field] = value
    responses(monkeypatch, run=run)
    with pytest.raises(RuntimeError, match="unexpected workflow"):
        reader.verified_inputs(SHA)


def test_only_current_master_is_published(monkeypatch):
    monkeypatch.setattr(driver, "github", lambda *_: {"commit": {"sha": SHA}})
    assert driver.version_for(source_run(), "test") == "latest"
    monkeypatch.setattr(driver, "github", lambda *_: {"commit": {"sha": "b" * 40}})
    assert driver.version_for(source_run(), "test") is None


def test_fork_preview_requires_open_pr_and_exact_head(monkeypatch):
    pr = {
        "number": 1927,
        "state": "open",
        "head": {
            "sha": SHA,
            "ref": "master",
            "repo": {"full_name": "contributor/spectrochempy"},
        },
        "base": {"ref": "master", "repo": {"full_name": reader.REPOSITORY}},
    }

    def github(path, token):
        assert path.startswith("pulls?")
        assert parse_qs(path.split("?", 1)[1]) == {
            "state": ["open"],
            "base": ["master"],
            "head": ["contributor:master"],
            "per_page": ["100"],
        }
        return [pr]

    monkeypatch.setattr(driver, "github", github)
    assert driver.version_for(source_run("pull_request"), "test") == "1927"
    pr["head"]["sha"] = "b" * 40
    assert driver.version_for(source_run("pull_request"), "test") is None


@pytest.mark.parametrize(
    "field,value",
    [("ref", "other"), ("repo", None), ("repo", {"full_name": "other/spectrochempy"})],
)
def test_fork_lookup_rejects_unrelated_branch_or_repository(monkeypatch, field, value):
    pr = {
        "number": 1929,
        "state": "open",
        "head": {
            "sha": SHA,
            "ref": "master",
            "repo": {"full_name": "contributor/spectrochempy"},
        },
        "base": {"ref": "master", "repo": {"full_name": reader.REPOSITORY}},
    }
    pr["head"][field] = value
    monkeypatch.setattr(driver, "github", lambda *_: [pr])
    assert driver.version_for(source_run("pull_request"), "test") is None


def test_fork_lookup_encodes_branch_and_rejects_missing_origin(monkeypatch):
    paths = []
    monkeypatch.setattr(driver, "github", lambda path, *_: paths.append(path) or [])
    run = source_run("pull_request")
    run["head_branch"] = "ci/rtd-prepared-previews"
    assert driver.version_for(run, "test") is None
    assert parse_qs(paths[0].split("?", 1)[1])["head"] == [
        "contributor:ci/rtd-prepared-previews"
    ]
    run.pop("head_repository")
    assert driver.version_for(run, "test") is None
    assert len(paths) == 1
    run["head_repository"] = None
    assert driver.version_for(run, "test") is None
    assert len(paths) == 1


def test_transfer_starts_rtd_after_links_and_verifies_published_commit(monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "github", lambda *_: source_run())
    monkeypatch.setattr(driver, "version_for", lambda *_: "latest")
    monkeypatch.setattr(driver, "artifact_pair", lambda *_: {})
    monkeypatch.setattr(
        driver, "publish_links", lambda *args: calls.append("links") or 5
    )

    def rtd(path, token, method="GET", data=None):
        calls.append(method)
        assert token == "rtd-test"  # noqa: S105
        if method == "POST":
            return {"build": {"id": 9}}
        return {"state": {"code": "finished"}, "success": True, "commit": SHA}

    monkeypatch.setattr(driver, "rtd", rtd)
    driver.transfer(12, PROJECT, "github-test", "rtd-test")
    assert calls == ["links", "POST", "links", "GET"]


def test_driver_rejects_arbitrary_rtd_project():
    with pytest.raises(ValueError):
        driver.transfer(12, "other-project", "github-test", "rtd-test")


def test_first_webhook_build_cancels_before_installation(monkeypatch):
    monkeypatch.setenv("SCPY_RTD_GITHUB_RESULTS", "1")
    monkeypatch.setenv("READTHEDOCS_VERSION", "latest")
    monkeypatch.setattr(sys, "argv", ["rtd_transfer", "ready"])
    monkeypatch.setattr(reader.subprocess, "check_output", lambda *args, **kwargs: SHA)
    monkeypatch.setattr(reader, "envelope", lambda *_: None)
    assert reader.main() == 183
    monkeypatch.setattr(reader, "envelope", lambda *_: {"ready": True})
    assert reader.main() == 0


@pytest.mark.parametrize("success,commit", [(False, SHA), (True, "b" * 40)])
def test_failed_or_wrong_revision_rtd_build_is_not_accepted(
    monkeypatch, success, commit
):
    monkeypatch.setattr(driver, "github", lambda *_: source_run())
    monkeypatch.setattr(driver, "version_for", lambda *_: "latest")
    monkeypatch.setattr(driver, "artifact_pair", lambda *_: {})
    monkeypatch.setattr(driver, "publish_links", lambda *args: 5)
    monkeypatch.setattr(
        driver,
        "rtd",
        lambda *args: {"build": {"id": 9}}
        if len(args) > 2
        else {"state": "finished", "success": success, "commit": commit},
    )
    with pytest.raises(RuntimeError, match="expected revision"):
        driver.transfer(12, PROJECT, "github-test", "rtd-test")


def test_ready_hook_runs_after_python_is_available():
    import yaml

    root = Path(__file__).resolve().parents[2]
    jobs = yaml.safe_load((root / ".readthedocs.yaml").read_text())["build"]["jobs"]
    assert all("python" not in command for command in jobs["post_checkout"])
    assert jobs["pre_install"][0] == "python -m docs.rtd_transfer ready"
