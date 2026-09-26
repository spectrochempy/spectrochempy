# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
DOCS_MAKE = ROOT / "docs" / "make.py"


def _load_docs_make():
    sys.path.insert(0, str(ROOT / "docs"))
    spec = importlib.util.spec_from_file_location("spectrochempy_docs_make", DOCS_MAKE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
