# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
HELPER = ROOT / ".github" / "workflows" / "scripts" / "stable_docs_snapshot.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("stable_docs_snapshot", HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_snapshot(root, version="1.0.0", *, broken_link=False):
    (root / "whatsnew").mkdir(parents=True)
    generated = root / "reference" / "generated"
    generated.mkdir(parents=True)
    (root / ".spectrochempy-doc-version").write_text(version, encoding="utf-8")
    (root / "whatsnew" / "latest.html").write_text(
        f"<h1>What's New in Revision {version}</h1>", encoding="utf-8"
    )

    pages = (
        "spectrochempy.NDDataset.html",
        "spectrochempy.read_omnic.html",
        "spectrochempy.fft.html",
    )
    links = "".join(f'<a href="generated/{page}">{page}</a>' for page in pages)
    (root / "reference" / "index.html").write_text(links, encoding="utf-8")
    target = "missing.html" if broken_link else "../index.html"
    for page in pages:
        (generated / page).write_text(
            f'<h1>{page}</h1><a href="{target}">Reference</a>',
            encoding="utf-8",
        )


def test_complete_snapshot_with_representative_api_links_is_valid(tmp_path):
    helper = _load_helper()
    _write_snapshot(tmp_path)

    helper.validate_snapshot(tmp_path, "1.0.0")


@pytest.mark.parametrize(
    "missing",
    [
        "reference/generated/spectrochempy.NDDataset.html",
        "reference/generated/spectrochempy.read_omnic.html",
        "reference/generated/spectrochempy.fft.html",
    ],
)
def test_missing_representative_api_page_is_rejected(tmp_path, missing):
    helper = _load_helper()
    _write_snapshot(tmp_path)
    (tmp_path / missing).unlink()

    with pytest.raises(RuntimeError, match="missing"):
        helper.validate_snapshot(tmp_path, "1.0.0")


def test_broken_representative_api_link_is_rejected(tmp_path):
    helper = _load_helper()
    _write_snapshot(tmp_path, broken_link=True)

    with pytest.raises(RuntimeError, match="Broken documentation link"):
        helper.validate_snapshot(tmp_path, "1.0.0")


def test_invalid_candidate_does_not_replace_published_snapshot(tmp_path):
    helper = _load_helper()
    published = tmp_path / "published"
    candidate = tmp_path / "candidate"
    old = published / "1.0.0"
    old.mkdir(parents=True)
    (old / "identity").write_text("old", encoding="utf-8")
    _write_snapshot(candidate)
    (candidate / "reference" / "generated" / "spectrochempy.fft.html").unlink()

    with pytest.raises(RuntimeError, match="Broken documentation link"):
        helper.promote_snapshot(candidate, published, "1.0.0")

    assert (old / "identity").read_text(encoding="utf-8") == "old"
    assert candidate.is_dir()


def test_valid_candidate_atomically_replaces_published_snapshot(tmp_path):
    helper = _load_helper()
    published = tmp_path / "published"
    candidate = published / ".repair" / "html" / "1.0.0"
    old = published / "1.0.0"
    old.mkdir(parents=True)
    (old / "identity").write_text("old", encoding="utf-8")
    _write_snapshot(candidate)

    destination = helper.promote_snapshot(candidate, published, "1.0.0")

    assert destination == old
    assert (destination / ".spectrochempy-doc-version").is_file()
    assert not (destination / "identity").exists()
    assert not (published / ".1.0.0.backup").exists()


def test_workflow_uses_isolated_compatible_environment_and_validated_promotion():
    workflow = (ROOT / ".github" / "workflows" / "build_docs.yml").read_text(
        encoding="utf-8"
    )
    helper = HELPER.read_text(encoding="utf-8")
    repair = workflow.split(
        "- name: Repair published stable documentation snapshot", maxsplit=1
    )[1].split("- name: CI diagnostics summary", maxsplit=1)[0]

    assert "stable_docs_snapshot.py rebuild" in repair
    assert '--tag "$STABLE_TAG" --promote' in repair
    assert "--no-api" not in repair
    assert chr(91) + 'uv, "venv"' in helper
    assert 'f"spectrochempy=={version}"' in helper
    for plugin in (
        "spectrochempy-carroucell",
        "spectrochempy-hypercomplex",
        "spectrochempy-iris",
        "spectrochempy-nmr",
        "spectrochempy-perkinelmer",
        "spectrochempy-tensor",
    ):
        assert plugin in helper
    assert "validate_snapshot(candidate, version)" in helper
    assert "promote_snapshot(candidate, published_root, version)" in helper
