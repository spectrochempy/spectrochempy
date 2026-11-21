"""Publication tests independent from the scientific runtime."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pages_assemble", Path(__file__).parents[2] / ".github/pages/assemble.py"
)
pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pages)


@pytest.fixture
def trees(tmp_path):
    site, source = tmp_path / "site", tmp_path / "input"
    site.mkdir()
    source.mkdir()
    for name in ("index.html", "search.html", "searchindex.js", "objects.inv"):
        (source / name).write_text("<html><head></head><body>example</body></html>")
    (source / "guide").mkdir()
    (source / "guide/page.html").write_text(
        "<html><head></head><body>guide</body></html>"
    )
    return site, source


def test_preview_update_preserves_production_and_close_removes_only_preview(trees):
    site, source = trees
    pages.assemble(site, source, "latest")
    original = (site / "latest/index.html").read_bytes()
    pages.assemble(site, source, "previews/pr-42")
    preview = site / "previews/pr-42/guide/page.html"
    assert "../../../pages-navigation.js" in preview.read_text()
    assert "noindex,nofollow" in preview.read_text()
    assert (site / "latest/index.html").read_bytes() == original
    pages.assemble(site, source, "previews/pr-42", remove=True)
    assert not (site / "previews/pr-42").exists()
    assert (site / "latest/index.html").read_bytes() == original


def test_stable_does_not_regress_when_older_release_or_rc_finishes_later(trees):
    site, source = trees
    for tag in (
        "spectrochempy-v1.10.0",
        "spectrochempy-v1.9.0",
        "spectrochempy-v2.0.0rc1",
    ):
        pages.assemble(site, source, tag)
    manifest = json.loads((site / "versions.json").read_text())
    assert manifest["stable"] == "spectrochempy-v1.10.0"
    assert (site / "stable/index.html").exists()
    assert not (site / "stable/guide/page.html").exists()
    assert "url=spectrochempy-v1.10.0/" in (site / "index.html").read_text()


def test_only_current_stable_is_retained_online(trees):
    site, source = trees
    for minor in range(12):
        pages.assemble(site, source, f"v1.{minor}.0")
    assert not (site / "v1.1.0").exists()
    assert (site / "v1.11.0").exists()
    manifest = json.loads((site / "versions.json").read_text())
    assert len(manifest["versions"]) == 2
    assert not (site / "v1.10.0").exists()


@pytest.mark.parametrize(
    "target", ["../latest", "previews/pr-0", "previews/../../latest", ".git", "stable"]
)
def test_rejects_untrusted_destination(trees, target):
    with pytest.raises(ValueError, match="Invalid publication target"):
        pages.assemble(*trees, target)


@pytest.mark.parametrize("attack", ["symlink", "git", "missing-index"])
def test_rejects_artifact_before_replacing_live_site(trees, attack):
    site, source = trees
    pages.assemble(site, source, "latest")
    original = (site / "latest/index.html").read_bytes()
    if attack == "symlink":
        (source / "bad").symlink_to(site / "latest/index.html")
    elif attack == "git":
        (source / ".git").mkdir()
        (source / ".git/config").write_text("malicious")
    else:
        (source / "index.html").unlink()
    with pytest.raises(ValueError):
        pages.assemble(site, source, "latest")
    assert (site / "latest/index.html").read_bytes() == original


def test_cross_project_links_use_the_configured_sites(trees):
    site, source = trees
    (source / "index.html").write_text(
        '<html><head></head><body><a href="https://www.spectrochempy.fr/'
        'projects/nmr/latest/guide.html">NMR</a></body></html>'
    )
    pages.assemble(site, source, "latest", nmr_url="https://example.org/nmr/latest/")
    assert (
        "https://example.org/nmr/latest/guide.html"
        in (site / "latest/index.html").read_text()
    )


def test_metadata_prerelease_validates_archive_without_advancing_stable(trees):
    site, source = trees
    pages.assemble(site, source, "v1.0.0")
    original = (site / "versions.json").read_bytes()
    pages.assemble(site, source, "v9.0.0", archive_only=True)
    assert not (site / "v9.0.0").exists()
    assert (site / "v1.0.0/index.html").exists()
    assert (site / "versions.json").read_bytes() == original
