"""
Tests for .github/workflows/scripts/release_version.py.

This module is the single authoritative source for deriving release metadata
(version, tag, prerelease flag, release notes, docs version, next dev
version, PyPI classifier) from a project version, so that stable releases and
release candidates (PEP 440 ``rc``) are interpreted consistently everywhere.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[3]
    / ".github"
    / "workflows"
    / "scripts"
    / "release_version.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("release_version", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rv = load_module()


# ---------------------------------------------------------------------------
# Accepted / rejected inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version",
    ["1.0.0", "0.12.8", "2.0.0", "1.0.0rc1", "1.0.0rc2", "0.5.0rc3"],
)
def test_accepted_release_versions(version):
    assert str(rv.parse_release(version)) == version


@pytest.mark.parametrize(
    "version",
    [
        "1.0",
        "v1.0.0",
        "spectrochempy-v1.0.0",
        "1.0.0-rc1",
        "1.0.0a1",
        "1.0.0b1",
        "1.0.0beta1",
        "1.0.0.dev0",
        "1.0.0post1",
        "1.0.0+localsuffix",
        "",
        "abc",
        None,
        1.0,
    ],
)
def test_rejected_release_versions(version):
    with pytest.raises(rv.ReleaseVersionError):
        rv.parse_release(version)


def test_rejection_message_mentions_canonical_form():
    with pytest.raises(rv.ReleaseVersionError, match="canonical form"):
        rv.parse_release("1.0.0-rc1")


# ---------------------------------------------------------------------------
# Semantic classification
# ---------------------------------------------------------------------------


def test_kind():
    assert rv.kind("1.0.0") == "stable"
    assert rv.kind("0.12.8") == "stable"
    assert rv.kind("1.0.0rc1") == "rc"


def test_prerelease_flags():
    assert rv.is_prerelease("1.0.0rc1") is True
    assert rv.is_prerelease("1.0.0") is False
    assert rv.is_stable("1.0.0") is True
    assert rv.is_stable("1.0.0rc1") is False


# ---------------------------------------------------------------------------
# Derived metadata
# ---------------------------------------------------------------------------


def test_tag_name():
    assert rv.tag_name("1.0.0") == "spectrochempy-v1.0.0"
    assert rv.tag_name("1.0.0rc1") == "spectrochempy-v1.0.0rc1"


def test_release_notes_name():
    assert rv.release_notes_name("1.0.0") == "v1.0.0.rst"
    assert rv.release_notes_name("1.0.0rc1") == "v1.0.0rc1.rst"


def test_docs_version():
    assert rv.docs_version("1.0.0") == "1.0.0"
    assert rv.docs_version("1.0.0rc1") == "1.0.0rc1"


def test_next_dev_version():
    assert rv.next_dev_version("1.0.0") == "1.0.1"
    assert rv.next_dev_version("0.12.8") == "0.12.9"
    assert rv.next_dev_version("2.0.0") == "2.0.1"
    assert rv.next_dev_version("1.0.0rc1") == "1.0.0rc2"
    assert rv.next_dev_version("1.0.0rc2") == "1.0.0rc3"


def test_expected_pypi_classifier():
    assert rv.expected_pypi_classifier("1.0.0") == "5 - Production/Stable"
    assert rv.expected_pypi_classifier("2.0.0") == "5 - Production/Stable"
    assert rv.expected_pypi_classifier("1.0.0rc1") == "4 - Beta"
    assert rv.expected_pypi_classifier("0.12.8") == "4 - Beta"


def test_describe():
    data = rv.describe("1.0.0rc1")
    assert data == {
        "version": "1.0.0rc1",
        "tag": "spectrochempy-v1.0.0rc1",
        "prerelease": "true",
        "kind": "rc",
        "release_notes": "v1.0.0rc1.rst",
        "docs_version": "1.0.0rc1",
        "next_dev": "1.0.0rc2",
        "pypi_classifier": "4 - Beta",
    }

    data = rv.describe("1.0.0")
    assert data["prerelease"] == "false"
    assert data["kind"] == "stable"
    assert data["next_dev"] == "1.0.1"
    assert data["pypi_classifier"] == "5 - Production/Stable"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_validate(capsys):
    assert rv.main(["validate", "1.0.0rc1"]) == 0
    out = capsys.readouterr().out.strip()
    assert "1.0.0rc1: ok (rc)" in out


def test_cli_validate_rejects_invalid(capsys):
    assert rv.main(["validate", "1.0.0-rc1"]) == 1
    capsys.readouterr()


def test_cli_describe_key_value(capsys):
    assert rv.main(["describe", "1.0.0rc2"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    pairs = dict(line.split("=", 1) for line in out)
    assert pairs["tag"] == "spectrochempy-v1.0.0rc2"
    assert pairs["prerelease"] == "true"
    assert pairs["next_dev"] == "1.0.0rc3"


def test_cli_next_dev(capsys):
    assert rv.main(["next-dev", "0.12.8"]) == 0
    assert capsys.readouterr().out.strip() == "0.12.9"


def test_cli_next_dev_invalid(capsys):
    assert rv.main(["next-dev", "bogus"]) == 1
    capsys.readouterr()


def test_cli_unknown_command(capsys):
    assert rv.main(["frobnicate", "1.0.0"]) == 2
    capsys.readouterr()


def test_cli_missing_arguments(capsys):
    assert rv.main([]) == 2
    capsys.readouterr()


def test_cli_run_as_subprocess():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "describe", "1.0.0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "tag=spectrochempy-v1.0.0" in result.stdout


def test_cli_next_dev_bootstraps_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT_PATH), "next-dev", "1.0.0rc1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "1.0.0rc2"


# ---------------------------------------------------------------------------
# PEP 440 ordering
# ---------------------------------------------------------------------------


def test_pep440_ordering():
    from packaging.version import Version

    values = ["0.12.8", "1.0.0rc1", "1.0.0", "1.0.1"]
    assert sorted(values, key=Version) == ["0.12.8", "1.0.0rc1", "1.0.0", "1.0.1"]


# ---------------------------------------------------------------------------
# Workflow wiring: the release workflows must derive prerelease/tag metadata
# from the shared helper rather than hard-coding stable-only values.
# ---------------------------------------------------------------------------

WORKFLOWS = Path(__file__).parents[3] / ".github" / "workflows"


def test_publish_draft_release_derives_tag_and_prerelease():
    content = (WORKFLOWS / "publish_draft_new_release.yml").read_text(encoding="utf-8")
    # The Create Release step must use the helper-derived values, not a
    # hard-coded stable tag / prerelease=false.
    assert "prerelease: false" not in content
    assert "${{ steps." in content
    assert "release_version.py" in content
    assert "validate" in content
    assert "describe" in content


def test_publish_draft_release_has_no_hardcoded_stable_tag():
    content = (WORKFLOWS / "publish_draft_new_release.yml").read_text(encoding="utf-8")
    assert "spectrochempy-v${{ github.event.inputs.versionString }}" not in content


def test_publish_draft_release_links_reviewed_release_notes():
    content = (WORKFLOWS / "publish_draft_new_release.yml").read_text(encoding="utf-8")
    assert "body: |" in content
    assert "steps.vsem.outputs.release_notes" in content
    assert "pull_request.merge_commit_sha" in content


def test_prepare_release_validates_version_before_proceeding():
    content = (WORKFLOWS / "prepare_new_release.yml").read_text(encoding="utf-8")
    assert "release_version.py" in content
    assert "validate" in content
    assert "packaging" in content


def test_build_package_uses_helper_next_dev():
    content = (WORKFLOWS / "build_package.yml").read_text(encoding="utf-8")
    assert "release_version.py next-dev" in content
    # The historical IFS=. + $((CORE_PATCH + 1)) arithmetic must be gone.
    assert "IFS=. read -r CORE_MAJOR CORE_MINOR CORE_PATCH" not in content
    assert "CORE_PATCH + 1" not in content


def test_validate_release_artifacts_uses_helper_next_dev():
    content = (WORKFLOWS / "validate_release_artifacts.yml").read_text(encoding="utf-8")
    assert "release_version.py next-dev" in content
    assert "IFS=. read -r CORE_MAJOR CORE_MINOR CORE_PATCH" not in content
    assert "CORE_PATCH + 1" not in content


def test_package_uses_helper_next_dev():
    content = (WORKFLOWS / "test_package.yml").read_text(encoding="utf-8")
    assert "release_version.py next-dev" in content
    assert "IFS=. read -r MAJOR MINOR PATCH" not in content
    assert "PATCH + 1" not in content


def test_archived_docs_skips_release_candidate_tags():
    content = (WORKFLOWS / "build_docs_archived_versions.yml").read_text(
        encoding="utf-8"
    )
    assert "release candidate" in content.lower() or "rc" in content.lower()
    # An RC tag (spectrochempy-v1.0.0rc1) must not be built as an archived
    # stable docs directory.
    assert "keep_count" in content


def test_update_script_no_longer_truncates_rc_versions():
    content = (WORKFLOWS / "scripts" / "update_version_and_release_notes.py").read_text(
        encoding="utf-8"
    )
    # Zenodo / CITATION must keep the full version, including any rcN suffix.
    assert "'.'.join(version.split('.')[:3])" not in content
    assert "from release_version import" in content
    assert "v(\\d+\\.\\d+\\.\\d+(?:rc\\d+)?)\\.rst" in content


@pytest.fixture
def tagged_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    git("init", "--quiet")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "--quiet",
        "--allow-empty",
        "-m",
        "Initial",
    )
    return git


@pytest.mark.parametrize(
    ("tags", "expected", "next_dev"),
    [
        (["1.0.0rc1"], "1.0.0rc1", "1.0.0rc2"),
        (["1.0.0rc1", "1.0.0"], "1.0.0", "1.0.1"),
        (["1.0.0", "1.0.1rc1"], "1.0.1rc1", "1.0.1rc2"),
        (["1.0.0rc2", "1.0.0rc10"], "1.0.0rc10", "1.0.0rc11"),
        (["1.9.0", "1.10.0"], "1.10.0", "1.10.1"),
    ],
)
def test_latest_tag_semantic_order(tagged_repository, tags, expected, next_dev):
    for version in tags:
        tagged_repository("tag", rv.TAG_PREFIX + version)
    # Plugin and unsupported core tags must never determine the version.
    tagged_repository("tag", "spectrochempy-nmr-v99.0.0")
    tagged_repository("tag", "spectrochempy-v99.0.0.dev1")
    assert rv.latest_core_tag() == rv.TAG_PREFIX + expected
    assert rv.next_dev_version(expected) == next_dev
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT_PATH), "latest-tag"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == rv.TAG_PREFIX + expected


def test_latest_tag_handles_annotated_release(tagged_repository):
    tagged_repository("tag", "spectrochempy-v1.0.0rc1")
    tagged_repository(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "tag",
        "-a",
        "spectrochempy-v1.0.0",
        "-m",
        "Release",
    )
    assert rv.latest_core_tag() == "spectrochempy-v1.0.0"


def test_latest_tag_fails_without_supported_tags(tagged_repository):
    tagged_repository("tag", "spectrochempy-nmr-v1.0.0")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "latest-tag"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert not result.stdout
    assert "No supported core release tag" in result.stderr


@pytest.mark.parametrize(
    "workflow",
    [
        "build_package.yml",
        "test_package.yml",
        "pre-commit.yml",
        "validate_release_artifacts.yml",
    ],
)
def test_core_workflows_select_tags_semantically(workflow):
    content = (WORKFLOWS / workflow).read_text()
    assert "release_version.py latest-tag" in content
    assert "--sort=-v:refname" not in content


def test_post_final_conda_recipe_version(tagged_repository, tmp_path, monkeypatch):
    """Exercise the real recipe generator with the post-final dev version."""
    import os
    import shutil

    import yaml

    tagged_repository("tag", "spectrochempy-v1.0.0rc1")
    tagged_repository("tag", "spectrochempy-v1.0.0")
    tagged_repository(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "--quiet",
        "--allow-empty",
        "-m",
        "Development",
    )
    tag = rv.latest_core_tag()
    count = tagged_repository("rev-list", "--count", f"{tag}..HEAD")
    version = f"{rv.next_dev_version(tag.removeprefix(rv.TAG_PREFIX))}.dev{count}"
    assert version == "1.0.1.dev1"
    source_root = SCRIPT_PATH.parents[3]
    for relative in [
        "pyproject.toml",
        "README.md",
        ".github/workflows/scripts/generate_conda_recipe.py",
        ".github/workflows/scripts/templates/recipe.tmpl",
    ]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, target)
    result = subprocess.run(
        [sys.executable, ".github/workflows/scripts/generate_conda_recipe.py"],
        env={**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": version},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    recipe = yaml.safe_load((tmp_path / "recipe/recipe.yaml").read_text())
    assert recipe["package"]["version"] == "1.0.1"
    assert recipe["build"]["string"] == "dev1"
    assert "SETUPTOOLS_SCM_PRETEND_VERSION=1.0.1 " in recipe["build"]["script"]
