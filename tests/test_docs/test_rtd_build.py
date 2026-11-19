"""Test provenance and build selection for the Read the Docs entry point."""

import hashlib
import importlib
from contextlib import redirect_stderr
from io import BytesIO
from io import StringIO
from urllib.error import HTTPError
from zipfile import ZipFile

import pytest

pytest.importorskip("nbformat")
pytest.importorskip("nbconvert")
pytest.importorskip("jupytext")
pytest.importorskip("sphinx")

rtd_build = importlib.import_module("docs.rtd_build")


def test_selects_both_artifacts_from_one_successful_run(monkeypatch):
    commit = "a" * 40
    runs = [
        {"id": 1, "head_sha": "b" * 40, "event": "push"},
        {"id": 2, "head_sha": commit, "event": "push"},
        {"id": 3, "head_sha": commit, "event": "push"},
        {"id": 4, "head_sha": commit, "event": "workflow_dispatch"},
    ]
    for run in runs:
        run.update(status="completed", conclusion="success")

    def artifact(kind, artifact_id, expired=False):
        return {"name": f"docs-{kind}-{commit}", "id": artifact_id, "expired": expired}

    responses = {
        2: [artifact("tutorials", 20)],
        3: [artifact("gallery", 30), artifact("tutorials", 31, expired=True)],
        4: [artifact("gallery", 40), artifact("tutorials", 41)],
    }

    def github(path, token):
        assert token == "test-token"  # noqa: S105
        if path.startswith("workflows/"):
            assert f"head_sha={commit}" in path and "status=" not in path
            return {"workflow_runs": runs}
        return {"artifacts": responses[int(path.split("/")[1])]}

    monkeypatch.setattr(rtd_build, "_json", github)
    run_id, selected = rtd_build._find_artifacts(commit, "test-token", set())
    assert run_id == 4
    assert {kind: item["id"] for kind, item in selected.items()} == {
        "gallery": 40,
        "tutorials": 41,
    }


def test_zip_checksum_is_checked_before_extraction(tmp_path, monkeypatch):
    data = BytesIO()
    with ZipFile(data, "w") as bundle:
        bundle.writestr("manifest.json", "{}")
    content = data.getvalue()
    monkeypatch.setattr(rtd_build, "_request", lambda *_: BytesIO(content))
    destination = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        rtd_build._download({"id": 1, "digest": "wrong"}, "token", destination)
    assert not destination.exists()
    digest = hashlib.sha256(content).hexdigest()
    rtd_build._download({"id": 1, "digest": f"sha256:{digest}"}, "token", destination)
    assert (destination / "manifest.json").read_text() == "{}"


def test_artifact_mode_removes_token_from_environment(monkeypatch):
    monkeypatch.setenv("SCPY_RTD_GITHUB_TOKEN", "test-token")
    called = []

    def render(token):
        assert token == "test-token"  # noqa: S105
        assert rtd_build.os.environ.get("SCPY_RTD_GITHUB_TOKEN") is None
        called.append("artifact")

    monkeypatch.setattr(rtd_build, "_render_from_actions", render)
    rtd_build.main()
    assert called == ["artifact"]


@pytest.mark.parametrize("expired", [True, False], ids=["expired", "version-mismatch"])
def test_unusable_artifact_retries_the_entire_pair(monkeypatch, expired):
    monkeypatch.setattr(rtd_build.ci_artifacts, "_commit", lambda: "commit")
    rendered = []

    def find(_commit, _token, unavailable):
        if unavailable:
            assert unavailable == ({2} if expired else {1, 2})
            return 20, {"tutorials": {"id": 3}, "gallery": {"id": 4}}
        return 10, {"tutorials": {"id": 1}, "gallery": {"id": 2}}

    def download(artifact, _token, path):
        if expired and artifact["id"] == 2:
            raise HTTPError("https://api.github.com", 410, "Gone", {}, None)
        path.write_text(str(artifact["id"]))

    def render(root):
        pair = tuple((root / kind).read_text() for kind in ("tutorials", "gallery"))
        rendered.append(pair)
        if pair == ("1", "2"):
            raise rtd_build.ci_artifacts.ArtifactVersionMismatchError("wrong version")

    monkeypatch.setattr(rtd_build, "_find_artifacts", find)
    monkeypatch.setattr(rtd_build, "_download", download)
    monkeypatch.setattr(rtd_build.ci_artifacts, "render", render)
    rtd_build._render_from_actions("test-token")
    assert rendered == ([("3", "4")] if expired else [("1", "2"), ("3", "4")])


def test_same_commit_with_different_installed_version_is_rejected(tmp_path):
    commit = "a" * 40
    (tmp_path / "manifest.json").write_text(
        f'{{"commit": "{commit}", "kind": "tutorials", "version": "0.dev0"}}'
    )
    with pytest.raises(rtd_build.ci_artifacts.ArtifactVersionMismatchError):
        rtd_build.ci_artifacts._verified_files(tmp_path, "tutorials", commit)


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_failed_producer_stops_waiting_with_run_link(monkeypatch, conclusion):
    monkeypatch.setattr(
        rtd_build,
        "_json",
        lambda *_: {
            "workflow_runs": [
                {
                    "id": 42,
                    "head_sha": "commit",
                    "event": "push",
                    "status": "completed",
                    "conclusion": conclusion,
                }
            ]
        },
    )
    with pytest.raises(RuntimeError, match=r"actions/runs/42") as error:
        rtd_build._find_artifacts("commit", "token", set())
    assert conclusion in str(error.value)


def test_active_retry_keeps_waiting_after_failed_run(monkeypatch):
    monkeypatch.setattr(
        rtd_build,
        "_json",
        lambda *_: {
            "workflow_runs": [
                {
                    "id": 43,
                    "head_sha": "commit",
                    "event": "workflow_dispatch",
                    "status": "in_progress",
                    "conclusion": None,
                },
                {
                    "id": 42,
                    "head_sha": "commit",
                    "event": "push",
                    "status": "completed",
                    "conclusion": "failure",
                },
                {
                    "id": 44,
                    "head_sha": "other",
                    "event": "push",
                    "status": "completed",
                    "conclusion": "failure",
                },
            ]
        },
    )
    assert rtd_build._find_artifacts("commit", "token", set()) is None


def test_older_complete_pair_is_used_despite_newer_failure(monkeypatch):
    def github(path, _token):
        if path.startswith("workflows/"):
            return {
                "workflow_runs": [
                    {
                        "id": 43,
                        "head_sha": "commit",
                        "event": "push",
                        "status": "completed",
                        "conclusion": "failure",
                    },
                    {
                        "id": 42,
                        "head_sha": "commit",
                        "event": "push",
                        "status": "completed",
                        "conclusion": "success",
                    },
                ]
            }
        return {
            "artifacts": [
                {"id": i, "name": f"docs-{kind}-commit", "expired": False}
                for i, kind in enumerate(("gallery", "tutorials"))
            ]
        }

    monkeypatch.setattr(rtd_build, "_json", github)
    assert rtd_build._find_artifacts("commit", "token", set())[0] == 42


def test_gallery_failure_traceback_escapes_captured_stderr(
    monkeypatch, tmp_path, capfd
):
    ci = rtd_build.ci_artifacts
    monkeypatch.setattr(ci, "_files", lambda *_: [])

    def fail(**_kwargs):
        raise RuntimeError("gallery diagnostic sentinel")

    monkeypatch.setattr(ci, "_produce_gallery", fail)
    captured = StringIO()
    with (
        redirect_stderr(captured),
        pytest.raises(RuntimeError, match="gallery diagnostic sentinel"),
    ):
        ci.produce("gallery", tmp_path / "bundle")
    stderr = capfd.readouterr().err
    assert "Traceback (most recent call last)" in stderr
    assert "RuntimeError: gallery diagnostic sentinel" in stderr
    assert "FAILED execute gallery" in stderr
    assert captured.getvalue() == ""


def test_watchdog_is_cancelled_after_failure(monkeypatch):
    from docs import build_diagnostics as diagnostics

    calls = []
    monkeypatch.setattr(
        diagnostics.faulthandler,
        "dump_traceback_later",
        lambda delay, **kwargs: calls.append((delay, kwargs)),
    )
    monkeypatch.setattr(
        diagnostics.faulthandler,
        "cancel_dump_traceback_later",
        lambda: calls.append("cancel"),
    )
    with pytest.raises(RuntimeError), diagnostics._diagnostics():
        raise RuntimeError("test")
    assert calls == [
        (120, {"repeat": True, "file": diagnostics.sys.__stderr__}),
        "cancel",
    ]
