# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]


def _load_docs_make():
    """Import ``docs.make`` the way the workflows do, from the repository root."""
    return importlib.import_module("docs.make")


def test_versions_manifest_distinguishes_stable_and_development(tmp_path):
    docs_make = _load_docs_make()
    (tmp_path / "1.0.0").mkdir()
    (tmp_path / "0.12.8").mkdir()
    (tmp_path / "latest").mkdir()

    manifest = docs_make._write_versions_manifest(tmp_path)

    assert manifest == {
        "schema_version": 2,
        "latest": "latest",
        "development": "latest",
        "stable": "1.0.0",
        "versions": ["1.0.0", "0.12.8"],
    }


def test_stable_root_sync_preserves_development_versions_and_previews(tmp_path):
    docs_make = _load_docs_make()
    stable = tmp_path / "1.0.0"
    development = tmp_path / "latest"
    preview = tmp_path / "docs-navigation"
    for directory in (stable, development, preview):
        directory.mkdir()

    (tmp_path / "index.html").write_text("old development", encoding="utf-8")
    (tmp_path / "CNAME").write_text("www.spectrochempy.fr", encoding="utf-8")
    (stable / "index.html").write_text("stable", encoding="utf-8")
    (stable / "stable-only.html").write_text("stable page", encoding="utf-8")
    (development / "index.html").write_text("development", encoding="utf-8")
    (development / "development-only.html").write_text(
        "development page", encoding="utf-8"
    )
    (preview / "index.html").write_text("preview", encoding="utf-8")

    selected = docs_make._sync_stable_docs_to_root(tmp_path, "1.0.0")

    assert selected == "1.0.0"
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "stable"
    assert (tmp_path / "stable-only.html").is_file()
    assert not (tmp_path / "development-only.html").exists()
    assert (development / "index.html").read_text(encoding="utf-8") == "development"
    assert (preview / "index.html").read_text(encoding="utf-8") == "preview"
    assert (tmp_path / "CNAME").read_text(encoding="utf-8") == "www.spectrochempy.fr"


def test_navigation_labels_contexts_and_page_fallback_are_present():
    layout = (ROOT / "docs" / "_templates" / "layout.html").read_text(encoding="utf-8")
    script = (ROOT / "docs" / "_static" / "js" / "versions.js").read_text(
        encoding="utf-8"
    )
    workflow = (ROOT / ".github" / "workflows" / "build_docs.yml").read_text(
        encoding="utf-8"
    )

    assert "Documentation version" in layout
    assert "This is development documentation." in layout
    assert "pull request documentation preview" in layout
    assert "Stable — ${manifest.stable}" in script
    assert "Development — unreleased" in script
    assert 'group.label = "Previous versions"' in script
    assert "relativePathFromBase" in script
    assert "pageExists(candidate)" in script
    assert 'new URL("index.html"' in script
    assert "SCPY_DOCS_CONTEXT" in workflow
    assert 'echo "docs_context=stable"' in workflow
    assert 'echo "docs_context=development"' in workflow
    assert 'echo "docs_context=preview"' in workflow


def test_version_selector_keeps_its_label_and_drops_the_current_version_line():
    layout = (ROOT / "docs" / "_templates" / "layout.html").read_text(encoding="utf-8")
    script = (ROOT / "docs" / "_static" / "js" / "versions.js").read_text(
        encoding="utf-8"
    )
    style = (ROOT / "docs" / "_static" / "css" / "spectrochempy.css").read_text(
        encoding="utf-8"
    )

    # The selected option already names the consulted version.
    assert "docs-current-version" not in layout
    assert "docs-current-version" not in script
    assert "docs-current-version" not in style
    assert "Currently viewing" not in script

    # The accessible label and the context banners are part of the contract.
    assert '<label for="versions-dropdown">Documentation version</label>' in layout
    assert 'aria-label="Documentation version"' in layout
    assert "This is development documentation." in layout
    assert "pull request documentation preview" in layout
    assert "Stable — ${manifest.stable}" in script
    assert "Development — unreleased" in script
    assert "Preview — ${selector.dataset.previewName" in script


def test_development_context_never_writes_into_the_stable_version(
    monkeypatch,
):
    docs_make = _load_docs_make()
    monkeypatch.setenv("SCPY_DOCS_CONTEXT", "development")
    monkeypatch.setattr(
        docs_make.BuildDocumentation, "_get_previous_tag", lambda self: "1.0.0"
    )
    monkeypatch.setattr("spectrochempy.version", "1.1.0")

    build = docs_make.BuildDocumentation.__new__(docs_make.BuildDocumentation)
    build.tagname = None

    assert build._determine_version() == ("1.1.0", "1.0.0", "latest")


def test_stable_context_uses_the_published_release_tag(monkeypatch):
    docs_make = _load_docs_make()
    monkeypatch.setenv("SCPY_DOCS_CONTEXT", "stable")
    monkeypatch.setenv("SCPY_DOCS_RELEASE_TAG", "spectrochempy-v1.0.0")
    monkeypatch.setattr(
        docs_make.BuildDocumentation, "_get_previous_tag", lambda self: "1.0.0"
    )
    monkeypatch.setattr("spectrochempy.version", "1.1.0")

    build = docs_make.BuildDocumentation.__new__(docs_make.BuildDocumentation)
    build.tagname = None

    assert build._determine_version() == ("1.0.0", "1.0.0", "1.0.0")


def test_docs_workflow_limits_stable_publication_to_final_core_releases():
    workflow = (ROOT / ".github" / "workflows" / "build_docs.yml").read_text(
        encoding="utf-8"
    )

    assert "stable-tag-version" in workflow
    assert "steps.release_scope.outputs.is_stable_core" in workflow
    assert "Stable core documentation release installs exact version" in workflow
    assert "SCPY_DOCS_RELEASE_TAG: ${{ steps.release_scope.outputs.tag }}" in workflow


def test_tag_build_propagates_strict_sphinx_options():
    docs_make = (ROOT / "docs" / "make.py").read_text(encoding="utf-8")

    assert "warningiserror=args.warning_is_error" in docs_make
    assert "noapi=args.no_api" in docs_make
    assert "noexec=args.no_exec" in docs_make
    assert "Sphinx build failed with status code" in docs_make


def test_failed_sphinx_build_cannot_run_post_build_or_write_marker(
    tmp_path, monkeypatch
):
    docs_make = _load_docs_make()
    monkeypatch.setattr(docs_make, "HTML", tmp_path)

    stable = tmp_path / "1.0.0"
    stable.mkdir()
    (stable / "index.html").write_text("partial build", encoding="utf-8")

    build = docs_make.BuildDocumentation.__new__(docs_make.BuildDocumentation)
    build.settings = {"noexec": True}
    build._prepare_build = lambda: None

    def fail_build():
        raise RuntimeError("Sphinx build failed with status code 1")

    build._run_sphinx_build = fail_build
    build._post_build = lambda: pytest.fail("post-build ran after a failed build")

    with pytest.raises(RuntimeError, match="status code 1"):
        build._make_docs()

    assert not (stable / ".spectrochempy-doc-version").exists()


def test_stable_retention_keeps_five_newest_numeric_versions_and_refreshes_catalogue(
    tmp_path,
):
    docs_make = _load_docs_make()
    versions = [
        "0.6.10",
        "0.12.7",
        "0.12.8",
        "0.12.9",
        "0.12.10",
        "1.0.0",
        "1.1.0",
    ]
    for version in versions:
        (tmp_path / version).mkdir()
    for distinct in ("latest", "1.2.0rc1", "docs-navigation"):
        (tmp_path / distinct).mkdir()

    removed, manifest = docs_make.prune_stable_versions(tmp_path, keep_count=5)

    assert removed == ["0.12.7", "0.6.10"]
    assert manifest["versions"] == [
        "1.1.0",
        "1.0.0",
        "0.12.10",
        "0.12.9",
        "0.12.8",
    ]
    assert not (tmp_path / "0.6.10").exists()
    assert (tmp_path / "latest").is_dir()
    assert (tmp_path / "1.2.0rc1").is_dir()
    assert (tmp_path / "docs-navigation").is_dir()
    assert (tmp_path / "versions.json").is_file()


def test_both_docs_workflows_use_the_same_five_version_retention_rule():
    workflow_dir = ROOT / ".github" / "workflows"
    for name in ("build_docs.yml", "build_docs_archived_versions.yml"):
        workflow = (workflow_dir / name).read_text(encoding="utf-8")
        assert "prune_stable_versions" in workflow
        assert "keep_count=5" in workflow
        assert "oldest_supported" not in workflow
