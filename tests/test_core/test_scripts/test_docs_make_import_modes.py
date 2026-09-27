# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Regressions for the two execution modes of ``docs/make.py``.

The documentation workflows both run ``python3 docs/make.py`` and import the same
module from the repository root with ``from docs.make import ...``. The sibling
``tools`` package is only reachable as a top-level name in the first mode, so the
unqualified import broke every ``docs.make`` import used by ``build_docs.yml`` and
``build_docs_archived_versions.yml``.

Every check below runs a fresh interpreter started at the repository root with
``PYTHONPATH`` removed, so neither mode can be masked by an inherited path and no
test depends on another test having mutated ``sys.path``.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MAKE_SCRIPT = REPO_ROOT / "docs" / "make.py"
SNAPSHOT_HELPER = (
    REPO_ROOT / ".github" / "workflows" / "scripts" / "stable_docs_snapshot.py"
)

FINAL_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

STABLE_VERSION = "1.0.0"
STALE_MARKER = "0.12.9"
ARCHIVED_VERSIONS = ("0.12.5", "0.12.6", "0.12.7", "0.12.8", "0.12.9", "0.12.10")
RETAINED_VERSIONS = ["1.0.0", "0.12.10", "0.12.9", "0.12.8", "0.12.7"]
PRUNED_VERSIONS = ["0.12.6", "0.12.5"]

WORKFLOWS = ("build_docs.yml", "build_docs_archived_versions.yml")


def _isolated_env():
    """Copy the environment without an inherited ``PYTHONPATH``."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def _run_from_repo_root(*argv, stdin=None):
    """Start a fresh interpreter at the repository root."""
    return subprocess.run(
        [sys.executable, *argv],
        cwd=REPO_ROOT,
        env=_isolated_env(),
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def _sync_command(published_root):
    """The stable root synchronisation used by ``build_docs.yml``."""
    return (
        "from pathlib import Path; from docs.make import "
        "_sync_stable_docs_to_root, refresh_versions_index; "
        f"root=Path({str(published_root)!r}); "
        "manifest=refresh_versions_index(root); "
        "selected=_sync_stable_docs_to_root(root, manifest['stable']); "
        "refresh_versions_index(root); "
        "print('Synchronized root from stable version', selected)"
    )


def _prune_script(published_root):
    """The version retention heredoc used by both documentation workflows."""
    return (
        "from pathlib import Path\n"
        "\n"
        "from docs.make import prune_stable_versions\n"
        "\n"
        f"removed, manifest = prune_stable_versions(Path({str(published_root)!r}), "
        "keep_count=5)\n"
        "print('Removed stable documentation versions:', removed)\n"
        "print('Published version catalogue:', manifest['versions'])\n"
    )


def _write_stable_snapshot(root, version, marker):
    """Write a snapshot accepted by ``stable_docs_snapshot.validate_snapshot``."""
    (root / "whatsnew").mkdir(parents=True)
    generated = root / "reference" / "generated"
    generated.mkdir(parents=True)
    (root / ".spectrochempy-doc-version").write_text(marker, encoding="utf-8")
    (root / "whatsnew" / "latest.html").write_text(
        f"<h1>What's New in Revision {marker}</h1>", encoding="utf-8"
    )

    pages = (
        "spectrochempy.NDDataset.html",
        "spectrochempy.read_omnic.html",
        "spectrochempy.fft.html",
    )
    links = "".join(f'<a href="generated/{page}">{page}</a>' for page in pages)
    (root / "reference" / "index.html").write_text(links, encoding="utf-8")
    for page in pages:
        (generated / page).write_text(
            f'<h1>{page}</h1><a href="../index.html">Reference</a>', encoding="utf-8"
        )

    (root / "index.html").write_text(f"stable root {marker}", encoding="utf-8")
    (root / "stable-only.html").write_text(f"stable only {marker}", encoding="utf-8")


def _build_published_root(root):
    """Populate a directory that mirrors the published gh-pages layout."""
    root.mkdir(parents=True)
    (root / "CNAME").write_text("www.spectrochempy.fr", encoding="utf-8")

    for version in ARCHIVED_VERSIONS:
        (root / version).mkdir()
        (root / version / "index.html").write_text(
            f"archived {version}", encoding="utf-8"
        )

    _write_stable_snapshot(root / STABLE_VERSION, STABLE_VERSION, STALE_MARKER)

    (root / "1.2.0rc1").mkdir()

    (root / "latest").mkdir()
    (root / "latest" / "index.html").write_text("development", encoding="utf-8")
    (root / "latest" / "development-only.html").write_text(
        "development page", encoding="utf-8"
    )

    (root / "docs-navigation").mkdir()
    (root / "docs-navigation" / "index.html").write_text("preview", encoding="utf-8")

    (root / "index.html").write_text("previous root", encoding="utf-8")
    (root / "development-only.html").write_text("development page", encoding="utf-8")


def _read_manifest(published_root):
    return json.loads((published_root / "versions.json").read_text(encoding="utf-8"))


def _final_version_dirs(published_root):
    """Return the final ``X.Y.Z`` directories, ignoring previews and releases."""
    return sorted(
        item.name
        for item in published_root.iterdir()
        if item.is_dir() and FINAL_VERSION_RE.match(item.name)
    )


# ======================================================================================
# Import mode regressions
# ======================================================================================
def test_docs_make_imports_from_the_repository_root():
    result = _run_from_repo_root(
        "-c",
        "from docs.make import "
        "prune_stable_versions, refresh_versions_index, _sync_stable_docs_to_root; "
        "print('IMPORT_OK')",
    )

    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout
    assert "No module named 'tools'" not in result.stderr


def test_importing_docs_make_does_not_inject_the_docs_directory():
    result = _run_from_repo_root(
        "-c",
        "import os, sys\n"
        "before = list(sys.path)\n"
        "import docs.make\n"
        "docs_dir = os.path.dirname(os.path.abspath(docs.make.__file__))\n"
        "print('ADDED', [p for p in sys.path if p not in before])\n"
        "print('DOCS_ON_PATH', [p for p in sys.path "
        "if os.path.abspath(p or os.getcwd()) == docs_dir])\n",
    )

    assert result.returncode == 0, result.stderr
    assert "DOCS_ON_PATH []" in result.stdout
    assert "ADDED []" in result.stdout


def test_docs_make_help_runs_as_a_direct_script():
    result = _run_from_repo_root("docs/make.py", "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "Build documentation for SpectroChemPy" in result.stdout
    assert "--warning-is-error" in result.stdout


def test_docs_make_help_runs_as_a_module():
    result = _run_from_repo_root("-m", "docs.make", "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("mode", ["script", "package"])
def test_internal_helper_import_errors_are_not_masked(tmp_path, mode):
    package = tmp_path / "probe"
    (package / "tools").mkdir(parents=True)
    shutil.copy2(MAKE_SCRIPT, package / "make.py")
    (package / "tools" / "helpers.py").write_text(
        "import spectrochempy_module_that_does_not_exist\n" "\n" "sh = None\n",
        encoding="utf-8",
    )

    if mode == "script":
        result = subprocess.run(
            [sys.executable, str(package / "make.py")],
            cwd=tmp_path,
            env=_isolated_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        result = subprocess.run(
            [sys.executable, "-c", "import probe.make"],
            cwd=tmp_path,
            env=_isolated_env(),
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert "spectrochempy_module_that_does_not_exist" in result.stderr


def test_docs_workflows_import_docs_make_without_a_pythonpath_workaround():
    for name in WORKFLOWS:
        workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text(
            encoding="utf-8"
        )

        assert "from docs.make import" in workflow
        assert "PYTHONPATH" not in workflow
        assert "sys.path" not in workflow


# ======================================================================================
# Documentation publication sequence, in fresh interpreters
# ======================================================================================
def test_stable_snapshot_sync_to_root_in_a_fresh_interpreter(tmp_path):
    published = tmp_path / "gh-pages"
    _build_published_root(published)

    result = _run_from_repo_root("-c", _sync_command(published))

    assert result.returncode == 0, result.stderr
    assert f"Synchronized root from stable version {STABLE_VERSION}" in result.stdout
    assert (published / "index.html").read_text(encoding="utf-8") == (
        f"stable root {STALE_MARKER}"
    )
    assert (published / "stable-only.html").is_file()
    assert not (published / "development-only.html").exists()
    assert (published / "latest" / "development-only.html").is_file()


def test_stable_version_retention_and_catalogue_in_a_fresh_interpreter(tmp_path):
    published = tmp_path / "gh-pages"
    _build_published_root(published)

    result = _run_from_repo_root("-", stdin=_prune_script(published))

    assert result.returncode == 0, result.stderr
    for version in PRUNED_VERSIONS:
        assert f"'{version}'" in result.stdout
        assert not (published / version).exists()
    assert _final_version_dirs(published) == sorted(RETAINED_VERSIONS)
    assert (published / "latest").is_dir()
    assert (published / "docs-navigation").is_dir()
    assert (published / "1.2.0rc1").is_dir()

    manifest = _read_manifest(published)
    assert manifest["versions"] == RETAINED_VERSIONS
    assert manifest["stable"] == STABLE_VERSION
    static = json.loads(
        (published / "_static" / "versions.json").read_text(encoding="utf-8")
    )
    assert static == manifest


def test_post_rebuild_sequence_in_a_gh_pages_like_directory(tmp_path):
    published = tmp_path / "gh-pages"
    _build_published_root(published)
    candidate = tmp_path / "candidate" / STABLE_VERSION
    _write_stable_snapshot(candidate, STABLE_VERSION, STABLE_VERSION)

    validated = _run_from_repo_root(
        str(SNAPSHOT_HELPER),
        "validate",
        "--snapshot",
        str(candidate),
        "--version",
        STABLE_VERSION,
    )
    assert validated.returncode == 0, validated.stderr

    promoted = _run_from_repo_root(
        str(SNAPSHOT_HELPER),
        "promote",
        "--snapshot",
        str(candidate),
        "--published-root",
        str(published),
        "--version",
        STABLE_VERSION,
    )
    assert promoted.returncode == 0, promoted.stderr
    assert (published / STABLE_VERSION / "index.html").read_text(encoding="utf-8") == (
        f"stable root {STABLE_VERSION}"
    )
    assert not (published / f".{STABLE_VERSION}.backup").exists()
    assert not candidate.exists()

    synced = _run_from_repo_root("-c", _sync_command(published))
    assert synced.returncode == 0, synced.stderr
    assert f"Synchronized root from stable version {STABLE_VERSION}" in synced.stdout

    pruned = _run_from_repo_root("-", stdin=_prune_script(published))
    assert pruned.returncode == 0, pruned.stderr

    refreshed = _run_from_repo_root(
        "-c",
        f"from pathlib import Path; from docs.make import refresh_versions_index; "
        f"print(refresh_versions_index(Path({str(published)!r})))",
    )
    assert refreshed.returncode == 0, refreshed.stderr

    assert (published / "index.html").read_text(encoding="utf-8") == (
        f"stable root {STABLE_VERSION}"
    )
    assert (published / "stable-only.html").is_file()
    assert (published / "whatsnew" / "latest.html").read_text(encoding="utf-8") == (
        f"<h1>What's New in Revision {STABLE_VERSION}</h1>"
    )
    assert not (published / "development-only.html").exists()

    assert (published / "latest" / "index.html").read_text(encoding="utf-8") == (
        "development"
    )
    assert (published / "docs-navigation" / "index.html").read_text(
        encoding="utf-8"
    ) == ("preview")
    assert (published / "CNAME").read_text(encoding="utf-8") == "www.spectrochempy.fr"
    assert (published / "1.2.0rc1").is_dir()

    assert _final_version_dirs(published) == sorted(RETAINED_VERSIONS)
    for version in PRUNED_VERSIONS:
        assert not (published / version).exists()

    manifest = _read_manifest(published)
    static = json.loads(
        (published / "_static" / "versions.json").read_text(encoding="utf-8")
    )
    assert manifest == static
    assert manifest["versions"] == RETAINED_VERSIONS
    assert manifest["stable"] == STABLE_VERSION
    catalogue = json.dumps(manifest)
    for version in PRUNED_VERSIONS:
        assert version not in catalogue
