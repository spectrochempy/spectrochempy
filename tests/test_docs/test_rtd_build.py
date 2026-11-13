"""Test provenance and build selection for the Read the Docs entry point."""

import hashlib
from io import BytesIO
from zipfile import ZipFile

import pytest

from docs import rtd_build


def test_selects_both_artifacts_from_one_successful_run(monkeypatch):
    commit = "a" * 40
    runs = [
        {"id": 1, "head_sha": "b" * 40, "event": "push"},
        {"id": 2, "head_sha": commit, "event": "push"},
        {"id": 3, "head_sha": commit, "event": "push"},
        {"id": 4, "head_sha": commit, "event": "workflow_dispatch"},
    ]

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
            assert f"head_sha={commit}" in path and "status=success" in path
            return {"workflow_runs": runs}
        return {"artifacts": responses[int(path.split("/")[1])]}

    monkeypatch.setattr(rtd_build, "_json", github)
    selected = rtd_build._find_artifacts(commit, "test-token")
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


@pytest.mark.parametrize("version_type", ["external", "branch"])
def test_artifact_mode_is_private_and_excludes_external_builds(
    monkeypatch, version_type
):
    monkeypatch.setenv("SCPY_RTD_DOCS_ARTIFACTS", "1")
    monkeypatch.setenv("READTHEDOCS_VERSION_TYPE", version_type)
    monkeypatch.setenv("SCPY_RTD_GITHUB_TOKEN", "test-token")
    called = []

    def render(token):
        assert token == "test-token"  # noqa: S105
        assert rtd_build.os.environ.get("SCPY_RTD_GITHUB_TOKEN") is None
        called.append("artifact")

    monkeypatch.setattr(rtd_build, "_render_from_actions", render)
    monkeypatch.setattr(rtd_build, "_render_on_rtd", lambda: called.append("local"))
    rtd_build.main()
    assert called == ["local" if version_type == "external" else "artifact"]
