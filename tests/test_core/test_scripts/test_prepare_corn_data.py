"""Test integrity, isolation and trust boundaries for the private Corn fixture."""

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/workflows/scripts/prepare_corn_data.py"
spec = importlib.util.spec_from_file_location("prepare_corn_data", SCRIPT)
prepare_corn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_corn)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    content = b"synthetic data for the staging test, not the Corn archive"
    source = tmp_path / "input.mat"
    source.write_bytes(content)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        )
    )
    monkeypatch.setattr(prepare_corn, "MANIFEST", manifest)
    return source, tmp_path / "runner-temp/corn", tmp_path / "env"


def test_stages_verified_raw_and_zip_without_publishing(fixture):
    source, directory, env = fixture
    prepare_corn.prepare(source, directory, env)
    assert (directory / "corn.mat").read_bytes() == source.read_bytes()
    with ZipFile(directory / "corn.zip") as bundle:
        assert bundle.namelist() == ["corn.mat"]
        assert bundle.read("corn.mat") == source.read_bytes()
    assert env.read_text().splitlines() == [
        f"SCPY_CORN_DATA={directory / 'corn.mat'}",
        f"SCPY_CORN_ZIP={directory / 'corn.zip'}",
    ]


@pytest.mark.parametrize("same_size", [False, True])
def test_rejects_corruption_before_staging(fixture, same_size):
    source, directory, env = fixture
    source.write_bytes(b"x" * source.stat().st_size if same_size else b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_corn.prepare(source, directory, env)
    assert not directory.exists()
    assert not env.exists()


def test_rejects_environment_file_injection(fixture):
    source, directory, env = fixture
    with pytest.raises(ValueError, match="newlines"):
        prepare_corn.prepare(source, directory / "\nINJECT=1", env)
    assert not env.exists()


@pytest.mark.skipif(
    os.name == "nt", reason="The composite action uses GitHub's bash shell"
)
@pytest.mark.parametrize(
    ("repository", "event", "ref", "allowed"),
    [
        ("spectrochempy/spectrochempy", "push", "refs/heads/master", True),
        ("spectrochempy/spectrochempy", "push", "refs/tags/v1.3.2", True),
        ("spectrochempy/spectrochempy", "workflow_dispatch", "refs/heads/master", True),
        ("spectrochempy/spectrochempy", "pull_request", "refs/heads/master", False),
        (
            "spectrochempy/spectrochempy",
            "pull_request_target",
            "refs/heads/master",
            False,
        ),
        (
            "spectrochempy/spectrochempy",
            "workflow_dispatch",
            "refs/heads/feature",
            False,
        ),
        ("spectrochempy/spectrochempy", "release", "refs/tags/v1.4.0", True),
        ("spectrochempy/spectrochempy", "release", "refs/heads/master", False),
        ("someone/fork", "release", "refs/tags/v1.4.0", False),
        ("someone/fork", "push", "refs/heads/master", False),
    ],
)
def test_private_checkout_requires_trusted_context(repository, event, ref, allowed):
    action = yaml.safe_load(
        (ROOT / ".github/actions/private-corn-data/action.yml").read_text()
    )
    guard = action["runs"]["steps"][0]["run"]
    result = subprocess.run(
        ["bash", "-c", guard],
        env={
            **os.environ,
            "GITHUB_REPOSITORY": repository,
            "GITHUB_EVENT_NAME": event,
            "GITHUB_REF": ref,
            "DATA_REVISION": "a" * 40,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == allowed


@pytest.mark.parametrize(
    "workflow",
    ["test_package.yml", "docs-artifact-prototype.yml", "docs-pages-build.yml"],
)
def test_private_fixture_is_outside_cache_and_artifact_paths(workflow):
    config = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    jobs = config["jobs"].values()
    steps = [step for job in jobs for step in job.get("steps", [])]
    private = next(
        step
        for step in steps
        if step.get("uses") == "./.github/actions/private-corn-data"
    )
    assert "github.repository == 'spectrochempy/spectrochempy'" in private["if"]
    assert "github.event_name == 'push'" in private["if"]
    assert "github.event_name == 'workflow_dispatch'" in private["if"]
    if workflow == "docs-pages-build.yml":
        assert "github.event_name == 'release'" in private["if"]
    for step in steps:
        if "upload-artifact@" in step.get("uses", "") or "cache@" in step.get(
            "uses", ""
        ):
            paths = step.get("with", {}).get("path", "")
            assert "spectrochempy-corn" not in paths
            assert "RUNNER_TEMP" not in paths and "runner.temp }}\n" not in paths
    assert any(
        step.get("name") == "Remove private Corn working copies"
        and step["if"] == "always()"
        for step in steps
    )
